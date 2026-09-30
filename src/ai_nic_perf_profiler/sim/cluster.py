"""Synthetic GPU cluster producing NIC counters and step events with injected faults.

Model: synchronous data-parallel training. Each step is a compute phase followed
by a ring all-reduce whose duration is set by the *slowest* NIC in the ring
(alpha-beta model, :mod:`training.collectives`). Faults change per-NIC
bandwidth, per-rank compute time and the counter signatures they leave
behind. Counters are integrated from piecewise-constant rates and sampled on
each host's own (skewed) clock with random phase, like a real agent.

The simulator encodes the same domain knowledge as the diagnosis rules, so it
tests the *pipeline* (alignment, baselining, decomposition, localisation under
noise, resets, skew and spikes). It can't show that the rules match real
hardware. That requires the fault-injection plan in ``docs/methodology.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.causes import RootCause
from ..io import write_snapshots, write_steps
from ..model import CounterSnapshot
from ..topology import RankBinding, Topology
from ..training.collectives import ring_allreduce_bytes_per_rank
from ..training.events import StepEvent
from .scenarios import SCENARIOS

COUNTERS = (
    "tx_bytes", "rx_bytes", "tx_packets", "rx_packets",
    "rx_pause_frames", "tx_pause_frames", "rx_pause_duration_us", "tx_pause_duration_us",
    "ecn_marked_packets", "cnp_sent", "cnp_handled",
    "out_of_buffer", "rx_discards_phy",
    "packet_seq_err", "out_of_sequence", "local_ack_timeout_err", "adp_retrans",
    "symbol_error", "link_error_recovery", "link_downed", "port_rcv_errors", "rx_crc_errors",
    "fec_corrected_bits",
)  # fmt: skip
CI = {n: i for i, n in enumerate(COUNTERS)}

# Background event rates (per second) in a healthy fabric: rare but non-zero.
BASE_EVENTS = {"out_of_sequence": 0.3, "packet_seq_err": 0.1, "rx_pause_frames": 2.0, "adp_retrans": 0.05, "symbol_error": 0.01}
BASE_FEC_BITS = 2e5  # corrected bits/s: normal on PAM4 links
BASE_ECN_RATIO = 2e-5
BASE_CNP_PER_PKT = 1e-5


@dataclass
class SimConfig:
    hosts: int = 4
    nics_per_host: int = 4
    hosts_per_leaf: int = 2
    link_gbps: float = 400.0
    steps: int = 240
    compute_s: float = 0.30
    allreduce_bytes: float = 2e9
    alpha_s: float = 15e-6
    efficiency: float = 0.85
    jitter: float = 0.01
    sample_interval_s: float = 0.1
    fault_start: int = 120
    fault_end: int = 200
    checkpoint_every: int = 100
    clock_skew_s: float = 0.005
    counter_reset: bool = True
    steps_rank0_only: bool = False
    mtu: int = 4096
    lossless_priority: int = 3
    severity: float = 1.0
    seed: int = 0


@dataclass
class FaultPlan:
    n: int
    bw_factor: np.ndarray = field(init=False)
    compute_factor: np.ndarray = field(init=False)
    byte_share: np.ndarray = field(init=False)
    ecn_ratio: np.ndarray = field(init=False)
    cnp_per_pkt: np.ndarray = field(init=False)
    extra: dict[int, dict[str, float]] = field(default_factory=dict)
    speed_gbps: dict[int, float] = field(default_factory=dict)
    vf_cap_bps: dict[int, float] = field(default_factory=dict)
    culprits: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bw_factor = np.ones(self.n)
        self.compute_factor = np.ones(self.n)
        self.byte_share = np.ones(self.n)
        self.ecn_ratio = np.full(self.n, BASE_ECN_RATIO)
        self.cnp_per_pkt = np.full(self.n, BASE_CNP_PER_PKT)

    def add(self, rank: int, **rates: float) -> None:
        d = self.extra.setdefault(rank, {})
        for k, v in rates.items():
            d[k] = d.get(k, 0.0) + v


@dataclass
class SimResult:
    snapshots: list[CounterSnapshot]
    steps: list[StepEvent]
    topology: Topology
    ground_truth: dict[str, Any]

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "counters": out / "counters.jsonl",
            "steps": out / "steps.jsonl",
            "topology": out / "topology.yaml",
            "ground_truth": out / "ground_truth.json",
        }
        write_snapshots(paths["counters"], self.snapshots)
        write_steps(paths["steps"], self.steps)
        self.topology.dump(paths["topology"])
        paths["ground_truth"].write_text(json.dumps(self.ground_truth, indent=2))
        return paths


def build_topology(cfg: SimConfig) -> Topology:
    ranks = []
    for h in range(cfg.hosts):
        for k in range(cfg.nics_per_host):
            ranks.append(RankBinding(h * cfg.nics_per_host + k, f"host-{h}", f"mlx5_{k}", 1, rail=k, leaf=f"leaf-{h // cfg.hosts_per_leaf}"))
    return Topology(ranks, link_gbps=cfg.link_gbps, lossless_priority=cfg.lossless_priority)


def plan_fault(scenario: str, cfg: SimConfig, topo: Topology, rng: np.random.Generator) -> FaultPlan:
    """Build the fault's effect. ``cfg.severity`` scales it continuously from 0 (none) to 1 (full)."""
    n = len(topo.ranks)
    plan = FaultPlan(n)
    sev = cfg.severity
    c = int(rng.integers(n))

    def slow(full: float) -> float:  # bandwidth factor interpolated toward 1.0
        return 1.0 - sev * (1.0 - full)

    def host_ranks(h: int) -> list[int]:
        return topo.ranks_on_host(f"host-{h}")

    if scenario == "pfc_storm":
        plan.bw_factor[c] = slow(0.6)
        plan.add(c, rx_pause_duration_us=0.4e6 * sev, rx_pause_frames=4800 * sev)
        plan.ecn_ratio[c] *= 1 + 0.5 * sev
        plan.culprits = [c]
    elif scenario == "fabric_congestion":
        leaves = sorted({b.leaf for b in topo.ranks if b.leaf})
        leaf = leaves[int(rng.integers(len(leaves)))]
        members = [b.rank for b in topo.ranks if b.leaf == leaf]
        for r in members:
            plan.bw_factor[r] = slow(0.7)
            plan.ecn_ratio[r] = BASE_ECN_RATIO + sev * (0.01 - BASE_ECN_RATIO)
            plan.cnp_per_pkt[r] = BASE_CNP_PER_PKT + sev * (5e-3 - BASE_CNP_PER_PKT)
            plan.add(r, rx_pause_duration_us=0.02e6 * sev, rx_pause_frames=250 * sev)
        plan.culprits = members
    elif scenario == "host_rx_backpressure":
        victim = (c - 1) % n
        plan.bw_factor[c] = slow(0.6)
        plan.add(c, out_of_buffer=3000 * sev, rx_discards_phy=400 * sev, tx_pause_frames=3500 * sev, tx_pause_duration_us=0.3e6 * sev)
        plan.add(victim, rx_pause_duration_us=0.25e6 * sev, rx_pause_frames=3000 * sev)
        plan.culprits = [c]
    elif scenario == "lossy_retransmit":
        plan.bw_factor[c] = slow(0.55)
        plan.add(c, packet_seq_err=800 * sev, out_of_sequence=1500 * sev, local_ack_timeout_err=3 * sev, adp_retrans=200 * sev)
        plan.culprits = [c]
    elif scenario == "link_degradation":
        plan.bw_factor[c] = slow(0.5)
        if sev >= 0.5:
            plan.speed_gbps[c] = cfg.link_gbps / 2
        plan.add(
            c,
            symbol_error=30 * sev,
            rx_crc_errors=20 * sev,
            link_error_recovery=0.05 * sev,
            fec_corrected_bits=BASE_FEC_BITS * 30 * sev,
            out_of_sequence=40 * sev,
        )
        plan.culprits = [c]
    elif scenario == "sriov_rate_cap":
        h = int(rng.integers(cfg.hosts))
        cap_frac = 1.0 - sev * (1.0 - 0.45)
        for r in host_ranks(h):
            plan.bw_factor[r] = min(1.0, cap_frac / cfg.efficiency)
            plan.vf_cap_bps[r] = cap_frac * cfg.link_gbps * 1e9
        plan.culprits = host_ranks(h)
    elif scenario == "rail_imbalance":
        h = int(rng.integers(cfg.hosts))
        a, b = (int(x) for x in rng.choice(host_ranks(h), size=2, replace=False))
        # A fraction ``sev`` of rank a's traffic is carried by rank b's NIC.
        plan.byte_share[a], plan.byte_share[b] = 1.0 - sev, 1.0 + sev
        plan.bw_factor[a] = plan.bw_factor[b] = 1.0 / (1.0 + sev)
        plan.culprits = [b]
    elif scenario == "compute_straggler":
        plan.compute_factor[c] = 1.0 + 0.3 * sev
        plan.culprits = []
    return plan


def simulate(scenario: str, cfg: SimConfig | None = None) -> SimResult:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    topo = build_topology(cfg)
    n = len(topo.ranks)
    hosts = topo.hosts()
    offsets = {h: float(rng.uniform(-cfg.clock_skew_s, cfg.clock_skew_s)) for h in hosts}
    host_of = [b.host for b in topo.ranks]

    healthy = FaultPlan(n)
    fault = plan_fault(scenario, cfg, topo, rng) if scenario != "healthy" else healthy
    link_Bps = cfg.link_gbps * 1e9 / 8
    vol = ring_allreduce_bytes_per_rank(cfg.allreduce_bytes, n)
    nc = len(COUNTERS)

    bounds: list[list[float]] = [[0.0] for _ in range(n)]
    seg_rates: list[list[np.ndarray]] = [[] for _ in range(n)]
    step_starts: list[float] = []
    step_gauges: list[list[dict[str, float]]] = [[] for _ in range(n)]
    events: list[StepEvent] = []

    t = 0.0
    for s in range(cfg.steps):
        in_fault = cfg.fault_start <= s < cfg.fault_end
        plan = fault if in_fault else healthy
        comp = cfg.compute_s * (1 + cfg.jitter * rng.standard_normal(n)) * plan.compute_factor
        if cfg.checkpoint_every and s > 0 and s % cfg.checkpoint_every == 0:
            comp = comp * 1.6
        starts = t + rng.uniform(0, 2e-4, n)
        comm_start = float((starts + comp).max())
        bw = float((link_Bps * cfg.efficiency * plan.bw_factor).min())
        comm = (2 * (n - 1) * cfg.alpha_s + vol / bw) * (1 + 0.5 * cfg.jitter * rng.standard_normal())
        end = comm_start + comm
        step_dur = end - t
        step_starts.append(t)

        for r in range(n):
            o = offsets[host_of[r]]
            events.append(StepEvent(r, s, float(starts[r] + o), float(end + rng.uniform(0, 2e-4) + o), host_of[r], int(cfg.allreduce_bytes)))

            idle = np.zeros(nc)
            idle[CI["tx_bytes"]] = idle[CI["rx_bytes"]] = 2e6
            idle[CI["tx_packets"]] = idle[CI["rx_packets"]] = 2e6 / 1024
            idle[CI["fec_corrected_bits"]] = BASE_FEC_BITS * rng.lognormal(0, 0.1)

            busy = idle.copy()
            nbytes = vol * plan.byte_share[r] / comm
            pkts = nbytes / cfg.mtu
            busy[CI["tx_bytes"]] = busy[CI["rx_bytes"]] = nbytes + 2e6
            busy[CI["tx_packets"]] = busy[CI["rx_packets"]] = pkts + 2e6 / 1024
            ecn = pkts * plan.ecn_ratio[r] * rng.lognormal(0, 0.3)
            busy[CI["ecn_marked_packets"]] = ecn
            busy[CI["cnp_sent"]] = 0.8 * ecn
            busy[CI["cnp_handled"]] = pkts * plan.cnp_per_pkt[r] * rng.lognormal(0, 0.3)
            for name, rate in plan.extra.get(r, {}).items():
                busy[CI[name]] += rate * rng.lognormal(0, 0.2)
            # Rare background events, as integer counts spread over the comm phase.
            for name, rate in BASE_EVENTS.items():
                busy[CI[name]] += rng.poisson(rate * step_dur) / comm

            bounds[r] += [comm_start, end]
            seg_rates[r] += [idle, busy]
            gauges = {"link_speed_bps": plan.speed_gbps.get(r, cfg.link_gbps) * 1e9}
            if r in fault.vf_cap_bps:
                gauges["vf_max_tx_rate_bps"] = plan.vf_cap_bps.get(r, 0.0)
            step_gauges[r].append(gauges)
        t = end

    # Integrate rates into cumulative counters and sample on each host's clock.
    t_end = t
    starts_arr = np.array(step_starts)
    reset_rank = int(rng.choice([r for r in range(n) if r not in fault.culprits])) if cfg.counter_reset else -1
    reset_at = float(rng.uniform(0.3, 0.5) * t_end)
    phase = {h: float(rng.uniform(0, cfg.sample_interval_s)) for h in hosts}
    snapshots: list[CounterSnapshot] = []
    for r in range(n):
        b = np.array(bounds[r])
        rates = np.vstack(seg_rates[r])
        cum = np.vstack([np.zeros(nc), np.cumsum(rates * np.diff(b)[:, None], axis=0)])
        host = host_of[r]
        grid = np.arange(phase[host], t_end, cfg.sample_interval_s)
        ts = np.clip(grid + rng.normal(0, 3e-4, len(grid)), 0, t_end)
        seg = np.clip(np.searchsorted(b, ts, side="right") - 1, 0, len(rates) - 1)
        values = cum[seg] + rates[seg] * (ts - b[seg])[:, None]
        if r == reset_rank:
            k = min(int(np.searchsorted(b, reset_at, side="right")) - 1, len(rates) - 1)
            at_reset = cum[k] + rates[k] * (reset_at - b[k])
            values = np.where((ts >= reset_at)[:, None], values - at_reset, values)
        values = np.floor(values).astype(np.int64)
        step_idx = np.clip(np.searchsorted(starts_arr, ts, side="right") - 1, 0, cfg.steps - 1)
        dev = topo.ranks[r].device
        o = offsets[host]
        for i in range(len(ts)):
            counters = dict(zip(COUNTERS, values[i].tolist(), strict=True))
            snapshots.append(CounterSnapshot(host, dev, 1, float(ts[i] + o), counters, step_gauges[r][step_idx[i]], "sim"))

    if cfg.steps_rank0_only:
        events = [e for e in events if e.rank == 0]
        topo.clock_offsets = {h: -o for h, o in offsets.items()}

    sc = SCENARIOS[scenario]
    truth = {
        "scenario": scenario,
        "root_cause": sc.root_cause.value if sc.root_cause else None,
        "culprit_ranks": sorted(fault.culprits),
        "fault_steps": [cfg.fault_start, cfg.fault_end] if scenario != "healthy" else None,
        "description": sc.description,
        "seed": cfg.seed,
        "counter_reset_rank": reset_rank,
        "clock_offsets_s": offsets,
    }
    return SimResult(snapshots, events, topo, truth)


__all__ = ["RootCause", "SimConfig", "SimResult", "simulate"]
