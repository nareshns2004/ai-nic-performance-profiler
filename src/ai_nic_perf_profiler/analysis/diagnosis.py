"""Turn feature evidence into root causes, culprits and recommendations.

The regression says *how much* time each feature explains; the rules here
decide *what that feature means* in context. The same symptom can have
different causes:

* ``tx_paused_frac`` on a rank means the rank was paused. If another rank is
  *sending* pauses, the pause started at that receiver and spread upstream
  (host backpressure). If ECN is also firing, it's congestion. If neither,
  PFC is firing without DCQCN reacting first (HOL blocking / threshold
  misconfiguration).
* Retransmissions with physical-layer errors come from a bad link; without
  them, the fabric is dropping packets.

Seconds attributed to a feature are split among its candidate causes in
proportion to each cause's evidence score. Seconds that land on causes with
negligible evidence are moved to "unexplained" rather than reported as a
diagnosis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..advisor.playbook import PLAYBOOK, Recommendation
from ..topology import Topology
from .causes import DECISION_DOMAIN, SUMMARY
from .causes import RootCause as RC
from .features import FEATURE_INDEX, SPEC

Z_SCALE = 8.0
MIN_EVIDENCE = 0.2

# Features used to localise each cause to specific ranks.
LOCALIZE: dict[RC, tuple[str, ...]] = {
    RC.FABRIC_CONGESTION: ("ecn_mark_ratio", "cnp_per_kpkt"),
    RC.PFC_HOL_BLOCKING: ("tx_paused_frac", "xmit_wait_rate"),
    RC.HOST_RX_BACKPRESSURE: ("pause_sent_rate", "drop_rate"),
    RC.LOSSY_RETRANSMISSION: ("retrans_rate", "ack_timeout_rate"),
    RC.LINK_DEGRADATION: ("phy_err_rate", "link_flap_rate", "link_speed_deficit", "fec_corrected_rate"),
    RC.SRIOV_RATE_CAP: ("vf_cap_util",),
    RC.RAIL_IMBALANCE: ("rail_skew",),
}


def saturate(z: float) -> float:
    """Map a z-score onto [0, 1) evidence strength."""
    return 1.0 - math.exp(-max(z, 0.0) / Z_SCALE)


def score_causes(e: dict[str, float]) -> dict[RC, float]:
    """Evidence score per cause from the strongest per-feature anomaly across ranks."""

    def m(*names: str) -> float:
        return max(e.get(n, 0.0) for n in names)

    physical = max(m("phy_err_rate", "link_flap_rate", "link_speed_deficit"), 0.5 * e.get("fec_corrected_rate", 0.0))
    pause_origin = m("pause_sent_rate", "drop_rate")
    congestion = m("ecn_mark_ratio", "cnp_per_kpkt")
    return {
        RC.LINK_DEGRADATION: saturate(physical),
        RC.LOSSY_RETRANSMISSION: saturate(m("retrans_rate", "ack_timeout_rate")) * (1 - saturate(physical)),
        RC.HOST_RX_BACKPRESSURE: saturate(pause_origin),
        RC.PFC_HOL_BLOCKING: saturate(m("tx_paused_frac", "xmit_wait_rate")) * (1 - saturate(pause_origin)) * (1 - 0.8 * saturate(congestion)),
        RC.FABRIC_CONGESTION: saturate(congestion),
        RC.SRIOV_RATE_CAP: saturate(e.get("vf_cap_util", 0.0)),
        RC.RAIL_IMBALANCE: saturate(e.get("rail_skew", 0.0)),
    }


def allocate(contributions: dict[str, float], scores: dict[RC, float]) -> tuple[dict[RC, float], float]:
    """Split per-feature seconds across causes by evidence. Returns (per-cause seconds, orphaned seconds)."""
    out: dict[RC, float] = defaultdict(float)
    orphaned = 0.0
    for feature, seconds in contributions.items():
        causes = SPEC[feature].causes
        weights = [scores.get(c, 0.0) for c in causes]
        total = sum(weights)
        if total < MIN_EVIDENCE:
            orphaned += seconds
            continue
        for cause, weight in zip(causes, weights, strict=True):
            out[cause] += seconds * weight / total
    return dict(out), orphaned


@dataclass
class Diagnosis:
    cause: RC
    evidence_score: float
    attributed_s: float
    share: float
    culprit_ranks: list[int]
    scope: str
    suspect: str
    evidence: list[str]
    recommendations: list[Recommendation] = field(default_factory=list)
    victim_ranks: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause.value,
            "decision_domain": DECISION_DOMAIN[self.cause],
            "summary": SUMMARY[self.cause],
            "evidence_score": round(self.evidence_score, 3),
            "attributed_s": round(self.attributed_s, 4),
            "share": round(self.share, 3),
            "culprit_ranks": self.culprit_ranks,
            "victim_ranks": self.victim_ranks,
            "scope": self.scope,
            "suspect": self.suspect,
            "evidence": self.evidence,
            "recommendations": [{"action": r.action, "verify": r.verify} for r in self.recommendations],
        }


@dataclass
class EpisodeEvidence:
    """Per-rank evidence for one degradation episode."""

    ranks: list[int]
    z_mean: np.ndarray  # (R, NF) mean z-score over the episode
    value_mean: np.ndarray  # (R, NF) mean raw feature value over the episode
    value_base: np.ndarray  # (R, NF) baseline median feature value
    contributions: dict[str, float]  # seconds per model feature
    excess_s: float
    unexplained_s: float


def _culprits(ev: EpisodeEvidence, cause: RC, z_min: float) -> list[tuple[int, str, float]]:
    cols = [FEATURE_INDEX[f] for f in LOCALIZE[cause]]
    out = []
    for j, rank in enumerate(ev.ranks):
        zs = ev.z_mean[j, cols]
        k = int(np.argmax(zs))
        if zs[k] >= z_min:
            out.append((rank, LOCALIZE[cause][k], float(zs[k])))
    return sorted(out, key=lambda t: -t[2])


def _scope(ranks: list[int], topology: Topology) -> str:
    if not ranks:
        return "job"
    bindings = [topology.binding(r) for r in ranks]
    hosts = {b.host for b in bindings}
    leaves = {b.leaf for b in bindings}
    if len(ranks) == 1:
        return "rank"
    if len(hosts) == 1:
        return "host"
    if len(leaves) == 1 and None not in leaves:
        return "leaf"
    return "fabric"


def _suspect(cause: RC, ranks: list[int], scope: str, topology: Topology) -> str:
    if not ranks:
        return "compute, input pipeline or a straggler rank" if cause is RC.NON_NETWORK else "undetermined"
    b = [topology.binding(r) for r in ranks]
    ports = ", ".join(f"{x.host}:{x.device}" for x in b[:3]) + (" ..." if len(b) > 3 else "")
    hosts = ", ".join(sorted({x.host for x in b}))
    leaf = b[0].leaf or "its leaf"
    if cause is RC.FABRIC_CONGESTION:
        if scope == "leaf":
            return f"uplinks of {leaf} (all congested ports sit under it)"
        if scope in ("rank", "host"):
            return f"switch downlinks toward {hosts}"
        return "spine layer / ECMP hashing (congestion spans multiple leaves)"
    return {
        RC.PFC_HOL_BLOCKING: f"switch egress queue toward {ports}",
        RC.HOST_RX_BACKPRESSURE: f"receive path on {ports} (PCIe link, GPUDirect, receive queue depth)",
        RC.LOSSY_RETRANSMISSION: f"dropping hop on the path to {ports}",
        RC.LINK_DEGRADATION: f"cable / transceiver / port on {ports}",
        RC.SRIOV_RATE_CAP: f"VF rate policy on {hosts}",
        RC.RAIL_IMBALANCE: f"NIC-to-GPU mapping on {hosts}",
    }.get(cause, "undetermined")


def _evidence_line(ev: EpisodeEvidence, topology: Topology, col: dict[int, int], rank: int, feat: str) -> str:
    b = topology.binding(rank)
    i, j = FEATURE_INDEX[feat], col[rank]
    return (
        f"rank {rank} ({b.host}:{b.device}{', ' + b.leaf if b.leaf else ''}): {feat}={_fmt(ev.value_mean[j, i])} "
        f"vs baseline {_fmt(ev.value_base[j, i])} (z={ev.z_mean[j, i]:.1f})"
    )


def _fmt(v: float) -> str:
    if math.isnan(v):
        return "n/a"
    return f"{v:.4g}"


def diagnose(ev: EpisodeEvidence, topology: Topology, culprit_z: float) -> list[Diagnosis]:
    strongest = {name: float(ev.z_mean[:, i].max()) if len(ev.ranks) else 0.0 for name, i in FEATURE_INDEX.items()}
    scores = score_causes(strongest)
    seconds, orphaned = allocate(ev.contributions, scores)
    unexplained = ev.unexplained_s + orphaned
    excess = max(ev.excess_s, 1e-12)
    col = {r: j for j, r in enumerate(ev.ranks)}

    out: list[Diagnosis] = []
    for cause, score in scores.items():
        attributed = seconds.get(cause, 0.0)
        share = attributed / excess
        if score < MIN_EVIDENCE or (share < 0.05 and score < 0.5):
            unexplained += attributed
            continue
        culprits = _culprits(ev, cause, culprit_z)
        ranks = [r for r, _, _ in culprits]
        scope = _scope(ranks, topology)
        evidence = []
        for rank in ranks[:3]:
            for feat in LOCALIZE[cause]:
                if ev.z_mean[col[rank], FEATURE_INDEX[feat]] >= culprit_z:
                    evidence.append(_evidence_line(ev, topology, col, rank, feat))
        if len(ranks) > 3:
            evidence.append(f"... and {len(ranks) - 3} more ranks")
        victims: list[int] = []
        if cause is RC.HOST_RX_BACKPRESSURE:
            # PFC is hop-by-hop: the receiver's pauses back up through the switch
            # and stall the *senders*. Their pause counters are the loudest in the
            # fleet, but they are victims, not the cause.
            paused = FEATURE_INDEX["tx_paused_frac"]
            victims = [r for j, r in enumerate(ev.ranks) if r not in ranks and ev.z_mean[j, paused] >= culprit_z]
            for r in victims[:3]:
                evidence.append(_evidence_line(ev, topology, col, r, "tx_paused_frac") + " (victim of pause propagation, not a cause)")
        out.append(Diagnosis(cause, score, attributed, share, ranks, scope, _suspect(cause, ranks, scope, topology), evidence, list(PLAYBOOK[cause]), victims))

    network_evidence = max(scores.values(), default=0.0)
    if unexplained / excess >= 0.2:
        out.append(
            Diagnosis(
                RC.NON_NETWORK,
                1.0 - network_evidence,
                unexplained,
                unexplained / excess,
                [],
                "job",
                _suspect(RC.NON_NETWORK, [], "job", topology),
                [f"{unexplained / excess:.0%} of the excess step time has no matching NIC counter anomaly"],
                list(PLAYBOOK[RC.NON_NETWORK]),
            )
        )
    out.sort(key=lambda d: -d.attributed_s)
    return out
