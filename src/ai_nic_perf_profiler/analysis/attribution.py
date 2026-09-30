"""End-to-end attribution: counters + step events -> explained step-time regressions.

Pipeline::

    snapshots --rates--> RateIndex per endpoint
    step events --------> StepTable (step x rank windows)
                               |
             window integration + feature derivation -> F[step, rank, feature]
                               |
           baseline (median/MAD per rank & feature) -> Z[step, rank, feature] >= 0
                               |
      step time vs worst-rank log1p(Z): non-negative ridge -> seconds per feature
                               |
       per episode: rank localisation + evidence rules -> Diagnosis list

The worst rank per feature enters the regression because synchronous
collectives run at the pace of the slowest link. Localisation then looks
across ranks to find which link that was.
"""

from __future__ import annotations

import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import AnalysisConfig
from ..counters.rates import rate_series
from ..model import CounterSnapshot
from ..topology import Topology
from ..training.events import StepEvent
from .align import RateIndex, build_step_table
from .baseline import choose_baseline, degraded_mask, find_episodes, robust_center_scale
from .diagnosis import Diagnosis, EpisodeEvidence, diagnose
from .features import FEATURE_INDEX, FEATURE_NAMES, SPEC, derive_features
from .regression import nnls_ridge, r_squared

MODEL_FEATURES = [n for n in FEATURE_NAMES if SPEC[n].in_model]
MODEL_COLS = [FEATURE_INDEX[n] for n in MODEL_FEATURES]


@dataclass
class EpisodeReport:
    start_step: int
    end_step: int
    n_steps: int
    kind: str
    baseline_step_s: float
    episode_step_s: float
    slowdown_pct: float
    excess_s: float
    explained_s: float
    unexplained_s: float
    feature_contributions: dict[str, float]
    rank_anomaly: list[tuple[int, float]]
    diagnoses: list[Diagnosis]

    @property
    def primary(self) -> Diagnosis | None:
        return self.diagnoses[0] if self.diagnoses else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "n_steps": self.n_steps,
            "kind": self.kind,
            "baseline_step_s": round(self.baseline_step_s, 5),
            "episode_step_s": round(self.episode_step_s, 5),
            "slowdown_pct": round(self.slowdown_pct, 2),
            "excess_s": round(self.excess_s, 4),
            "explained_s": round(self.explained_s, 4),
            "unexplained_s": round(self.unexplained_s, 4),
            "feature_contributions_s": {k: round(v, 4) for k, v in sorted(self.feature_contributions.items(), key=lambda kv: -kv[1]) if v > 0},
            "top_anomalous_ranks": [{"rank": r, "max_z": round(z, 1)} for r, z in self.rank_anomaly],
            "diagnoses": [d.to_dict() for d in self.diagnoses],
        }


@dataclass
class DataQuality:
    endpoints_expected: int
    endpoints_found: int
    missing_endpoints: list[str]
    windows_total: int
    windows_low_coverage: int
    windows_inferred: int
    counter_flags: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class AttributionReport:
    n_steps: int
    n_ranks: int
    baseline_steps: tuple[int, int]
    baseline_step_s: float
    baseline_sigma_s: float
    model_weights: dict[str, float]
    model_r2: float
    episodes: list[EpisodeReport]
    data_quality: DataQuality
    step_series: dict[str, list[Any]] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self, include_series: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "generated_at": self.generated_at,
            "n_steps": self.n_steps,
            "n_ranks": self.n_ranks,
            "baseline": {"steps": list(self.baseline_steps), "median_step_s": round(self.baseline_step_s, 5), "sigma_s": round(self.baseline_sigma_s, 5)},
            "model": {"r2": round(self.model_r2, 3), "weights_s_per_unit": {k: round(v, 5) for k, v in self.model_weights.items() if v > 0}},
            "episodes": [e.to_dict() for e in self.episodes],
            "data_quality": self.data_quality.to_dict(),
        }
        if include_series:
            d["step_series"] = self.step_series
        return d


def analyze(
    snapshots: list[CounterSnapshot],
    steps: list[StepEvent],
    topology: Topology,
    config: AnalysisConfig | None = None,
) -> AttributionReport:
    cfg = config or AnalysisConfig()

    series = rate_series(snapshots)
    flags = Counter(flag.split(":")[0] for samples in series.values() for s in samples for flag in s.flags)
    indexes = {key: RateIndex(samples) for key, samples in series.items()}
    table = build_step_table(steps, topology)
    if len(table.steps) == 0:
        raise ValueError("no step events match the ranks in the topology")

    n_steps, n_ranks, n_feat = len(table.steps), len(table.ranks), len(FEATURE_NAMES)
    col = {r: j for j, r in enumerate(table.ranks)}

    # 1. Feature tensor F[step, rank, feature].
    F = np.full((n_steps, n_ranks, n_feat), np.nan)
    missing: list[str] = []
    windows = low_cov = 0
    for j, rank in enumerate(table.ranks):
        key = topology.binding(rank).endpoint.key
        index = indexes.get(key)
        if index is None:
            missing.append(key)
            continue
        for i in range(n_steps):
            start, end = table.start[i, j], table.end[i, j]
            if np.isnan(start):
                continue
            windows += 1
            rates, gauges, coverage = index.window(start, end)
            if coverage < cfg.min_coverage:
                low_cov += 1
                continue
            feats = derive_features(rates, gauges, topology.link_bps)
            F[i, j, :] = [feats[n] for n in FEATURE_NAMES]

    # Rail skew is a host-level property: this rail vs. the host's median rail.
    u, rs = FEATURE_INDEX["util"], FEATURE_INDEX["rail_skew"]
    for host in topology.hosts():
        cols = [col[r] for r in topology.ranks_on_host(host)]
        if len(cols) < 2:
            continue
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(F[:, cols, u], axis=1, keepdims=True)
            F[:, cols, rs] = np.where(med > 0, F[:, cols, u] / med, np.nan)

    # 2. Step time, baseline and degradation episodes.
    y = table.step_time()
    valid = ~np.isnan(y)
    post_warmup = table.steps >= table.steps.min() + cfg.warmup_steps
    base = choose_baseline(table.steps, cfg.warmup_steps, cfg.baseline_fraction, cfg.baseline_steps) & valid
    if base.sum() < 3:
        raise ValueError("baseline window has fewer than 3 valid steps; adjust warmup/baseline settings")
    y_med, y_sigma = (float(v) for v in robust_center_scale(y[base]))
    deg = degraded_mask(y, base, cfg.degradation_sigma, cfg.min_relative_slowdown) & ~base & post_warmup
    episodes = find_episodes(deg, cfg.merge_gap_steps, cfg.min_episode_steps)

    # 3. Robust z-scores against each rank's own baseline. Only excursions in
    # the "worse" direction count. A feature absent from the baseline (e.g. no
    # VF cap configured yet) is treated as baseline zero.
    with np.errstate(all="ignore"):
        f_med, f_sigma = robust_center_scale(F[base], axis=0)
    floors = np.array([SPEC[n].floor for n in FEATURE_NAMES])
    scale = np.fmax(f_sigma, floors)
    f_med0 = np.nan_to_num(f_med, nan=0.0)
    Z = np.clip(np.nan_to_num((F - f_med0) / scale, nan=0.0), 0.0, None)

    # 4. Additive decomposition of excess step time.
    X = np.log1p(Z[:, :, MODEL_COLS]).max(axis=1)
    X = np.clip(X - np.median(X[base], axis=0), 0.0, None)
    yc = np.nan_to_num(y - y_med, nan=0.0)
    fit = valid & post_warmup
    w = nnls_ridge(X[fit], yc[fit], cfg.ridge_lambda)
    r2 = r_squared(X[fit], yc[fit], w)
    C = X * w

    # 5. Per-episode localisation and diagnosis.
    reports: list[EpisodeReport] = []
    for ep in episodes:
        sl = slice(ep.start_idx, ep.end_idx)
        excess = float(yc[sl].sum())
        contrib = dict(zip(MODEL_FEATURES, C[sl].sum(axis=0).tolist(), strict=True))
        explained = sum(contrib.values())
        if explained > excess > 0:
            contrib = {k: v * excess / explained for k, v in contrib.items()}
            explained = excess
        unexplained = max(0.0, excess - explained)
        z_mean = Z[sl].mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value_mean = np.nanmean(F[sl], axis=0)
        evidence = EpisodeEvidence(table.ranks, z_mean, value_mean, f_med, contrib, excess, unexplained)
        rank_z = z_mean[:, MODEL_COLS].max(axis=1)
        order = np.argsort(-rank_z)[:5]
        ep_med = float(np.nanmedian(y[sl]))
        reports.append(
            EpisodeReport(
                start_step=int(table.steps[ep.start_idx]),
                end_step=int(table.steps[ep.end_idx - 1]),
                n_steps=ep.length,
                kind="sustained" if ep.length >= 10 else "transient",
                baseline_step_s=y_med,
                episode_step_s=ep_med,
                slowdown_pct=100 * (ep_med / y_med - 1),
                excess_s=excess,
                explained_s=explained,
                unexplained_s=unexplained,
                feature_contributions=contrib,
                rank_anomaly=[(table.ranks[k], float(rank_z[k])) for k in order],
                diagnoses=diagnose(evidence, topology, cfg.culprit_z),
            )
        )

    base_steps = table.steps[base]
    return AttributionReport(
        n_steps=n_steps,
        n_ranks=n_ranks,
        baseline_steps=(int(base_steps.min()), int(base_steps.max()) + 1),
        baseline_step_s=y_med,
        baseline_sigma_s=y_sigma,
        model_weights=dict(zip(MODEL_FEATURES, w.tolist(), strict=True)),
        model_r2=r2,
        episodes=reports,
        data_quality=DataQuality(
            endpoints_expected=n_ranks,
            endpoints_found=n_ranks - len(missing),
            missing_endpoints=missing,
            windows_total=windows,
            windows_low_coverage=low_cov,
            windows_inferred=int(table.inferred.sum()),
            counter_flags=dict(flags),
        ),
        step_series={
            "step": table.steps.tolist(),
            "step_time_s": [None if np.isnan(v) else round(float(v), 5) for v in y],
            "degraded": deg.tolist(),
        },
    )
