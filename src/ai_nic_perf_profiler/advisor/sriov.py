"""SR-IOV partition sizing from measured per-VF demand.

The core question: how many VFs of a given workload fit on one PF, and what
rate policy should they get?

Naive sizing adds up per-VF peaks (sum of p99s), which is safe but wasteful
when tenants burst at different times. Sizing by the mean ignores bursts
entirely. The right number is the p99 of the *aggregate* demand, measured on
a common time grid, which captures how synchronised the bursts actually are:

    synchrony index = p99(sum of VF demand) / sum(p99 of each VF)

* close to 1: bursts line up (e.g. VFs serving ranks of the same data-parallel
  job, all hitting all-reduce together). There is no multiplexing gain, so
  size for the sum of peaks.
* well below 1: bursts are independent (different jobs), so guarantees
  (``min_tx_rate``) plus work-conserving sharing beat hard caps.

Demand observed on a capped VF is *censored*: throughput can't exceed the cap,
so the measured p99 underestimates true demand. Such VFs are flagged; raise
the cap temporarily to measure real demand before resizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..analysis.align import RateIndex
from ..counters.rates import rate_series
from ..model import CounterSnapshot

AT_CAP = 0.95
ACTIVE_BPS = 1e9


@dataclass
class VfProfile:
    endpoint: str
    p50_bps: float
    p99_bps: float
    peak_bps: float
    cap_bps: float
    time_at_cap: float
    active_frac: float

    @property
    def censored(self) -> bool:
        return self.cap_bps > 0 and self.time_at_cap > 0.05


@dataclass
class SriovPlan:
    line_rate_bps: float
    headroom: float
    profiles: list[VfProfile]
    sum_p99_bps: float
    p99_of_sum_bps: float
    synchrony_index: float
    max_vfs_per_pf: int
    policy: str
    recommended_min_tx_bps: float
    recommended_max_tx_bps: float | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        gbps = lambda v: None if v is None else round(v / 1e9, 2)  # noqa: E731
        return {
            "line_rate_gbps": gbps(self.line_rate_bps),
            "headroom": self.headroom,
            "sum_p99_gbps": gbps(self.sum_p99_bps),
            "p99_of_sum_gbps": gbps(self.p99_of_sum_bps),
            "synchrony_index": round(self.synchrony_index, 3),
            "max_vfs_per_pf": self.max_vfs_per_pf,
            "policy": self.policy,
            "recommended_min_tx_gbps": gbps(self.recommended_min_tx_bps),
            "recommended_max_tx_gbps": gbps(self.recommended_max_tx_bps),
            "vfs": [
                {
                    "endpoint": p.endpoint,
                    "p50_gbps": gbps(p.p50_bps),
                    "p99_gbps": gbps(p.p99_bps),
                    "peak_gbps": gbps(p.peak_bps),
                    "cap_gbps": gbps(p.cap_bps) if p.cap_bps else None,
                    "time_at_cap": round(p.time_at_cap, 3),
                    "active_frac": round(p.active_frac, 3),
                    "censored": p.censored,
                }
                for p in self.profiles
            ],
            "notes": self.notes,
        }


def plan_partitions(
    snapshots: list[CounterSnapshot],
    line_rate_bps: float,
    select: str = "",
    headroom: float = 0.1,
    quantile: float = 0.99,
    grid_s: float | None = None,
) -> SriovPlan:
    """Size VF rate policy from counter snapshots of VFs sharing one PF.

    ``select`` is a substring matched against endpoint keys (``host/device/port``),
    e.g. ``"ens1f0/vf"`` to pick the VFs of one PF.
    """
    series = {k: v for k, v in rate_series(snapshots).items() if select in k and v}
    if not series:
        raise ValueError(f"no endpoints match {select!r}")
    indexes = {k: RateIndex(v) for k, v in series.items()}
    t_lo = max(float(ix.t0[0]) for ix in indexes.values())
    t_hi = min(float(ix.t1[-1]) for ix in indexes.values())
    dt = grid_s or float(np.median([np.median(np.diff(ix.t0)) for ix in indexes.values() if len(ix.t0) > 1]))
    edges = np.arange(t_lo, t_hi, dt)
    if len(edges) < 10:
        raise ValueError("not enough overlapping samples across VFs to size partitions")

    demand = np.zeros((len(edges) - 1, len(indexes)))
    caps = np.zeros(len(indexes))
    for j, ix in enumerate(indexes.values()):
        for i in range(len(edges) - 1):
            rates, gauges, _ = ix.window(edges[i], edges[i + 1])
            demand[i, j] = rates.get("tx_bytes", 0.0) * 8
            caps[j] = max(caps[j], gauges.get("vf_max_tx_rate_bps", 0.0))

    profiles = []
    for j, key in enumerate(indexes):
        d = demand[:, j]
        active = d > ACTIVE_BPS
        at_cap = float(np.mean(d[active] >= AT_CAP * caps[j])) if caps[j] > 0 and active.any() else 0.0
        profiles.append(
            VfProfile(key, float(np.percentile(d, 50)), float(np.percentile(d, 100 * quantile)), float(d.max()), float(caps[j]), at_cap, float(active.mean()))
        )

    sum_p99 = sum(p.p99_bps for p in profiles)
    p99_sum = float(np.percentile(demand.sum(axis=1), 100 * quantile))
    synchrony = p99_sum / sum_p99 if sum_p99 > 0 else 0.0
    usable = line_rate_bps * (1 - headroom)
    n = len(profiles)
    per_vf_need = p99_sum / n if n else 0.0
    max_vfs = int(usable // per_vf_need) if per_vf_need > 0 else 0

    notes = []
    censored = [p for p in profiles if p.censored]
    if censored:
        notes.append(f"{len(censored)} of {n} VFs are pinned at their caps, so the demand figures below are lower bounds.")
    tolerance = 1.001  # capped VFs can land exactly on the usable budget
    if sum_p99 <= usable * tolerance:
        policy = "no contention: per-VF peaks fit together"
        min_tx, max_tx = float(np.mean([p.p50_bps for p in profiles])), None
        notes.append("Hard caps aren't needed; set min_tx_rate guarantees at the median demand for isolation.")
    elif p99_sum <= usable * tolerance:
        policy = "statistical multiplexing: guarantees + work-conserving sharing"
        min_tx, max_tx = usable / n, None
        notes.append(
            f"Bursts are not synchronised (synchrony index {synchrony:.2f}): peaks sum to {sum_p99 / 1e9:.0f} Gb/s "
            f"but only {p99_sum / 1e9:.0f} Gb/s is demanded at once. Hard max caps would waste capacity."
        )
    else:
        policy = "oversubscribed: reduce VFs per PF or cap fairly"
        min_tx, max_tx = usable / n, usable / n
        notes.append(
            f"Concurrent p99 demand {p99_sum / 1e9:.0f} Gb/s exceeds usable {usable / 1e9:.0f} Gb/s. "
            f"At this workload profile the PF supports {max_vfs} VFs, not {n}."
        )
    for p in profiles:
        if p.censored:
            notes.append(
                f"{p.endpoint} sits at its {p.cap_bps / 1e9:.0f} Gb/s cap {p.time_at_cap:.0%} of active time: "
                "its demand is censored. Raise the cap temporarily to measure true demand before resizing."
            )
    return SriovPlan(line_rate_bps, headroom, profiles, sum_p99, p99_sum, synchrony, max_vfs, policy, min_tx, max_tx, notes)
