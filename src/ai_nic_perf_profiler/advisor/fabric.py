"""Fabric what-if model: which topology or tuning change would buy the most step time?

Model a two-tier leaf/spine fabric running NCCL-style rings (one ring per
rail). A ring hop between hosts under different leaves crosses that leaf's
uplinks. Uplink load depends on:

* **placement**: rank order packed by leaf means only a handful of hops cross
  leaves; scattered placement makes most hops cross.
* **ECMP hashing**: each connection's QPs are hashed onto uplinks. With few,
  fat flows, collisions are likely (balls-into-bins): the most-loaded uplink
  carries well above the average, and the ring runs at *that* link's pace.
  More QPs per connection (``NCCL_IB_QPS_PER_CONNECTION``) raise entropy.
* **oversubscription**: uplinks per leaf versus NICs under it.
* **rail-optimised**: same-rail NICs of up to ``rail_group_hosts`` hosts share
  one rail switch, so intra-group ring hops never reach the spine.

The ring's bottleneck rate is estimated by Monte Carlo over placement and
hashing, then calibrated against the measured step time. Predictions are
therefore relative to what this job actually achieves, not a datasheet.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..training.collectives import ring_allreduce_seconds


@dataclass(frozen=True)
class FabricModel:
    hosts: int
    nics_per_host: int = 8
    hosts_per_leaf: int = 4
    uplinks_per_leaf: int = 32
    link_gbps: float = 400.0
    efficiency: float = 0.85
    alpha_s: float = 15e-6
    qps_per_connection: int = 1
    placement: str = "scattered"  # or "packed"
    rail_optimized: bool = False
    rail_group_hosts: int = 32

    @property
    def oversubscription(self) -> float:
        return self.hosts_per_leaf * self.nics_per_host / self.uplinks_per_leaf


def ring_rate(model: FabricModel, trials: int = 400, seed: int = 0) -> float:
    """Expected ring bottleneck rate as a fraction of NIC line rate (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    h = model.hosts
    group = model.rail_group_hosts if model.rail_optimized else model.hosts_per_leaf
    group_of = np.arange(h) // group
    q = max(1, model.qps_per_connection)
    # Rail-optimised: each rail has its own switch per group, so a group's
    # uplink pool carries one rail's crossing flows. Otherwise all rails of a
    # host share the leaf's uplinks.
    flows_per_hop = 1 if model.rail_optimized else model.nics_per_host
    downlinks = group * flows_per_hop
    uplinks = max(1, round(downlinks / model.oversubscription))
    rates = np.empty(trials)
    for t in range(trials):
        order = np.arange(h) if model.placement == "packed" else rng.permutation(h)
        src, dst = order, np.roll(order, -1)
        crossing = group_of[src] != group_of[dst]
        rate = 1.0
        for g in np.unique(group_of[src[crossing]]):
            flows = int(np.sum(group_of[src[crossing]] == g)) * flows_per_hop
            load = np.bincount(rng.integers(uplinks, size=flows * q), minlength=uplinks).max()
            rate = min(rate, q / load)
        rates[t] = rate
    return float(rates.mean())


@dataclass
class Prediction:
    name: str
    ring_rate: float
    comm_s: float
    step_s: float
    speedup_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def what_if(
    current: FabricModel,
    candidates: dict[str, FabricModel],
    allreduce_bytes: float,
    measured_step_s: float,
    compute_s: float,
    trials: int = 400,
) -> list[Prediction]:
    """Predict step time under each candidate, calibrated so that ``current`` reproduces the measurement."""
    n = current.hosts * current.nics_per_host

    def comm(model: FabricModel) -> tuple[float, float]:
        r = ring_rate(model, trials)
        bw = model.link_gbps * 1e9 / 8 * model.efficiency * r
        return r, ring_allreduce_seconds(allreduce_bytes, n, bw, model.alpha_s)

    measured_comm = max(measured_step_s - compute_s, 1e-9)
    r0, model_comm = comm(current)
    calibration = measured_comm / model_comm
    out = [Prediction("current", r0, measured_comm, measured_step_s, 0.0)]
    for name, model in candidates.items():
        r, c = comm(model)
        c *= calibration
        step = compute_s + c
        out.append(Prediction(name, r, c, step, 100 * (measured_step_s / step - 1)))
    return out


def default_candidates(current: FabricModel) -> dict[str, FabricModel]:
    return {
        "packed placement (topology-aware rank order)": replace(current, placement="packed"),
        "NCCL_IB_QPS_PER_CONNECTION=4": replace(current, qps_per_connection=4),
        "non-blocking leaves (1:1)": replace(current, uplinks_per_leaf=current.hosts_per_leaf * current.nics_per_host),
        "rail-optimized fabric": replace(current, rail_optimized=True),
        "packed + QPs=4": replace(current, placement="packed", qps_per_connection=4),
    }
