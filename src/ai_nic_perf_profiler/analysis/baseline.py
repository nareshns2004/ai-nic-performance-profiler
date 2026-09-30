"""Robust baselines and degradation episode detection.

Step times are heavy-tailed (checkpoint steps, eval steps, GC pauses), so we
use median/MAD rather than mean/stddev everywhere. A degradation *episode* is
a run of steps that are both statistically (``k`` robust sigmas) and
practically (``min_relative`` slower) worse than baseline.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

MAD_TO_SIGMA = 1.4826


def robust_center_scale(x: np.ndarray, axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    # All-NaN columns (a counter a NIC doesn't expose) are expected: yield NaN quietly.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(x, axis=axis)
        mad = np.nanmedian(np.abs(x - (np.expand_dims(med, axis) if axis is not None else med)), axis=axis)
    return med, mad * MAD_TO_SIGMA


def choose_baseline(steps: np.ndarray, warmup: int, fraction: float, explicit: tuple[int, int] | None = None, min_steps: int = 10) -> np.ndarray:
    """Boolean mask of baseline steps.

    With ``explicit=(a, b)`` the baseline is steps ``a <= step < b``. Otherwise
    it's the first ``fraction`` of post-warmup steps, which assumes the job
    started healthy; pass an explicit window from a known-good run when it didn't.
    """
    if explicit is not None:
        return (steps >= explicit[0]) & (steps < explicit[1])
    post = steps >= steps.min() + warmup if len(steps) else np.zeros(0, dtype=bool)
    idx = np.flatnonzero(post)
    n = max(min_steps, int(len(idx) * fraction))
    mask = np.zeros(len(steps), dtype=bool)
    mask[idx[:n]] = True
    return mask


def degraded_mask(step_time: np.ndarray, baseline: np.ndarray, k: float = 4.0, min_relative: float = 0.03) -> np.ndarray:
    med, sigma = robust_center_scale(step_time[baseline])
    threshold = max(med + k * max(float(sigma), 1e-9), med * (1 + min_relative))
    return np.nan_to_num(step_time, nan=0.0) > threshold


@dataclass(frozen=True, slots=True)
class Episode:
    start_idx: int
    end_idx: int  # exclusive

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


def find_episodes(mask: np.ndarray, merge_gap: int = 2, min_length: int = 3) -> list[Episode]:
    """Contiguous runs of degraded steps, bridging gaps of up to ``merge_gap`` healthy steps.

    Short isolated spikes (below ``min_length``) are dropped: they are usually
    checkpoints or evaluation passes, not fabric problems.
    """
    runs: list[list[int]] = []
    for i in np.flatnonzero(mask):
        if runs and i - runs[-1][1] <= merge_gap + 1:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    return [Episode(a, b + 1) for a, b in runs if b + 1 - a >= min_length]


def cusum(x: np.ndarray, target: float, sigma: float, k: float = 0.5, h: float = 5.0) -> list[int]:
    """One-sided (upward) CUSUM change points, for detecting sustained drifts
    too small for the per-step threshold. ``k`` and ``h`` are in units of sigma."""
    s, alarms = 0.0, []
    sigma = max(sigma, 1e-12)
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        s = max(0.0, s + (v - target) / sigma - k)
        if s > h:
            alarms.append(i)
            s = 0.0
    return alarms
