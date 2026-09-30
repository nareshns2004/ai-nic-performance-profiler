"""Align counter rates onto training step windows.

Counters are sampled on a fixed wall-clock grid; training progresses in steps
of varying length. Rather than resample both onto a common time grid, the
step itself is the unit of analysis: for each rank and step we integrate the
rank's NIC rates over exactly ``[step.start, step.end)``, weighting each rate
sample by its overlap with the window.

A rank's step events and its NIC counters come from the same host clock, so
this per-rank alignment is immune to cross-host clock skew. Across ranks we
compare by step number, which acts as a logical clock. Wall-clock offsets are
only needed when a rank's steps were logged on a different host (for example,
rank-0-only logging).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..model import RateSample
from ..topology import Topology
from ..training.events import StepEvent


class RateIndex:
    """Interval index over one endpoint's rate samples."""

    def __init__(self, samples: list[RateSample], offset: float = 0.0) -> None:
        samples = sorted(samples, key=lambda s: s.t0)
        self.t0 = np.array([s.t0 for s in samples], dtype=float) + offset
        self.t1 = np.array([s.t1 for s in samples], dtype=float) + offset
        rate_names = sorted({n for s in samples for n in s.rates})
        gauge_names = sorted({n for s in samples for n in s.gauges})
        self.rate_names, self.gauge_names = rate_names, gauge_names
        self.rates = np.full((len(samples), len(rate_names)), np.nan)
        self.gauges = np.full((len(samples), len(gauge_names)), np.nan)
        rcol = {n: i for i, n in enumerate(rate_names)}
        gcol = {n: i for i, n in enumerate(gauge_names)}
        for i, s in enumerate(samples):
            for n, v in s.rates.items():
                self.rates[i, rcol[n]] = v
            for n, v in s.gauges.items():
                self.gauges[i, gcol[n]] = v

    def window(self, start: float, end: float) -> tuple[dict[str, float], dict[str, float], float]:
        """Overlap-weighted mean rates and gauges over ``[start, end)`` plus coverage in [0, 1]."""
        if end <= start or len(self.t0) == 0:
            return {}, {}, 0.0
        lo = int(np.searchsorted(self.t1, start, side="right"))
        hi = int(np.searchsorted(self.t0, end, side="left"))
        if hi <= lo:
            return {}, {}, 0.0
        overlap = np.clip(np.minimum(self.t1[lo:hi], end) - np.maximum(self.t0[lo:hi], start), 0.0, None)
        coverage = float(overlap.sum() / (end - start))
        return self._weighted(self.rates[lo:hi], overlap, self.rate_names), self._weighted(self.gauges[lo:hi], overlap, self.gauge_names), min(coverage, 1.0)

    @staticmethod
    def _weighted(values: np.ndarray, weights: np.ndarray, names: list[str]) -> dict[str, float]:
        if values.shape[1] == 0:
            return {}
        present = ~np.isnan(values)
        w = weights[:, None] * present
        den = w.sum(axis=0)
        num = np.where(present, values, 0.0).T @ weights
        out: dict[str, float] = {}
        for i, name in enumerate(names):
            if den[i] > 0:
                out[name] = float(num[i] / den[i])
        return out


@dataclass
class StepTable:
    """Per-(step, rank) windows on each rank's own host clock."""

    steps: np.ndarray  # (S,) step numbers, sorted
    ranks: list[int]  # (R,) rank ids in column order
    start: np.ndarray  # (S, R)
    end: np.ndarray  # (S, R)
    inferred: np.ndarray  # (S, R) bool: window borrowed from another rank

    @property
    def durations(self) -> np.ndarray:
        return self.end - self.start

    def step_time(self) -> np.ndarray:
        """Global step time: in synchronous training the slowest rank sets the pace."""
        with np.errstate(all="ignore"):
            return np.nanmax(self.durations, axis=1)


def build_step_table(events: list[StepEvent], topology: Topology) -> StepTable:
    """Arrange step events into a (step x rank) grid.

    Ranks without their own events inherit the window of a rank that has one,
    shifted by the difference in host clock offsets.
    """
    ranks = [b.rank for b in topology.ranks]
    col = {r: i for i, r in enumerate(ranks)}
    by_step: dict[int, dict[int, StepEvent]] = defaultdict(dict)
    for ev in events:
        if ev.rank in col:
            by_step[ev.step][ev.rank] = ev
    steps = np.array(sorted(by_step), dtype=int)
    start = np.full((len(steps), len(ranks)), np.nan)
    end = np.full_like(start, np.nan)
    inferred = np.zeros_like(start, dtype=bool)
    offsets = topology.clock_offsets
    for i, s in enumerate(steps):
        row = by_step[int(s)]
        ref = next(iter(row.values()))
        ref_host = ref.host or topology.binding(ref.rank).host
        for r in ranks:
            own = row.get(r)
            if own is not None:
                start[i, col[r]], end[i, col[r]] = own.start, own.end
                continue
            host = topology.binding(r).host
            # Put the reference window on the common clock, then onto this host's clock.
            shift = offsets.get(ref_host, 0.0) - offsets.get(host, 0.0)
            start[i, col[r]], end[i, col[r]] = ref.start + shift, ref.end + shift
            inferred[i, col[r]] = True
    return StepTable(steps, ranks, start, end, inferred)
