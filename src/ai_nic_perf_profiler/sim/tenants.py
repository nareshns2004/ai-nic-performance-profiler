"""Synthetic multi-tenant VF traffic on one PF, for the SR-IOV sizing advisor."""

from __future__ import annotations

import numpy as np

from ..model import CounterSnapshot


def simulate_vf_tenants(
    n_vfs: int = 4,
    pf: str = "ens1f0",
    host: str = "host-0",
    burst_gbps: float = 150.0,
    period_s: float = 0.4,
    duty: float = 0.25,
    synchronized: bool = True,
    cap_gbps: float | None = None,
    duration_s: float = 60.0,
    interval_s: float = 0.02,
    seed: int = 0,
) -> list[CounterSnapshot]:
    """Each VF alternates compute (idle) and all-reduce bursts of ``burst_gbps``.

    With ``synchronized`` the VFs burst together (ranks of one job); otherwise
    each has a random phase (independent jobs).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, interval_s)
    snaps: list[CounterSnapshot] = []
    for vf in range(n_vfs):
        phase = 0.0 if synchronized else rng.uniform(0, period_s)
        in_burst = ((t + phase) % period_s) >= period_s * (1 - duty)
        rate_bps = np.where(in_burst, burst_gbps * 1e9 * rng.lognormal(0, 0.05, len(t)), 1e8)
        if cap_gbps:
            rate_bps = np.minimum(rate_bps, cap_gbps * 1e9)
        cum = np.concatenate([[0.0], np.cumsum(rate_bps[:-1] / 8 * interval_s)]).astype(np.int64)
        gauges = {"vf_max_tx_rate_bps": cap_gbps * 1e9} if cap_gbps else {}
        for i, ts in enumerate(t):
            snaps.append(CounterSnapshot(host, f"{pf}/vf{vf}", 1, float(ts), {"tx_bytes": int(cum[i])}, gauges, "sim"))
    return snaps
