"""Measure attribution quality against simulator ground truth.

Metrics per scenario, aggregated over seeds:

* **detect**: a degradation episode overlaps the fault window. For
  ``healthy``, this is the false-positive rate instead.
* **top-1**: the primary diagnosis of the matching episode is the injected root cause.
* **culprit P/R**: precision and recall of the primary diagnosis' culprit ranks
  against the ranks where the fault was injected (not defined for
  ``non_network``, which has no network culprit).
* **explained**: fraction of excess step time the decomposition attributes to
  network features. It should be high for network faults and low for
  ``compute_straggler``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .analysis.attribution import AttributionReport, EpisodeReport, analyze
from .config import AnalysisConfig
from .sim import SCENARIOS, SimConfig, SimResult, simulate


@dataclass
class Trial:
    scenario: str
    seed: int
    detected: bool
    top1: bool
    precision: float | None
    recall: float | None
    explained: float | None
    predicted: str | None
    seconds: float
    true_slowdown_pct: float = 0.0


@dataclass
class ScenarioSummary:
    scenario: str
    trials: list[Trial] = field(default_factory=list)

    def _mean(self, values: list[float | None]) -> float | None:
        vals = [v for v in values if v is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def detect_rate(self) -> float:
        return float(np.mean([t.detected for t in self.trials]))

    @property
    def top1(self) -> float | None:
        if self.scenario == "healthy":
            return None
        return float(np.mean([t.top1 for t in self.trials]))

    def row(self) -> dict[str, float | str | None]:
        return {
            "scenario": self.scenario,
            "trials": len(self.trials),
            "detect": self.detect_rate,
            "top1": self.top1,
            "culprit_precision": self._mean([t.precision for t in self.trials]),
            "culprit_recall": self._mean([t.recall for t in self.trials]),
            "explained": self._mean([t.explained for t in self.trials]),
            "analyze_s": self._mean([t.seconds for t in self.trials]),
            "true_slowdown_pct": self._mean([t.true_slowdown_pct for t in self.trials]),
        }


def _matching_episode(report: AttributionReport, window: list[int] | None) -> EpisodeReport | None:
    if window is None:
        return report.episodes[0] if report.episodes else None
    lo, hi = window
    overlapping = [e for e in report.episodes if e.start_step < hi and e.end_step >= lo]
    return max(overlapping, key=lambda e: e.excess_s, default=None)


def _true_slowdown(result: SimResult, warmup: int) -> float:
    """Median step-time increase inside the fault window, straight from the simulator."""
    window = result.ground_truth["fault_steps"]
    if window is None:
        return 0.0
    per_step: dict[int, float] = {}
    for ev in result.steps:
        per_step[ev.step] = max(per_step.get(ev.step, 0.0), ev.duration)
    before = [v for k, v in per_step.items() if warmup <= k < window[0]]
    during = [v for k, v in per_step.items() if window[0] <= k < window[1]]
    return 100 * (float(np.median(during)) / float(np.median(before)) - 1)


def run_trial(scenario: str, seed: int, sim: SimConfig, analysis: AnalysisConfig) -> Trial:
    result = simulate(scenario, replace(sim, seed=seed))
    t0 = time.perf_counter()
    report = analyze(result.snapshots, result.steps, result.topology, analysis)
    elapsed = time.perf_counter() - t0
    truth = result.ground_truth
    slowdown = _true_slowdown(result, analysis.warmup_steps)
    ep = _matching_episode(report, truth["fault_steps"])
    if scenario == "healthy":
        return Trial(scenario, seed, ep is not None, ep is None, None, None, None, ep.primary.cause.value if ep and ep.primary else None, elapsed)
    if ep is None or ep.primary is None:
        return Trial(scenario, seed, False, False, None, None, None, None, elapsed, slowdown)
    primary = ep.primary
    precision = recall = None
    if truth["culprit_ranks"]:
        predicted, actual = set(primary.culprit_ranks), set(truth["culprit_ranks"])
        precision = len(predicted & actual) / len(predicted) if predicted else 0.0
        recall = len(predicted & actual) / len(actual)
    explained = ep.explained_s / ep.excess_s if ep.excess_s > 0 else None
    return Trial(scenario, seed, True, primary.cause.value == truth["root_cause"], precision, recall, explained, primary.cause.value, elapsed, slowdown)


def evaluate(
    scenarios: list[str] | None = None,
    seeds: int = 5,
    sim: SimConfig | None = None,
    analysis: AnalysisConfig | None = None,
) -> list[ScenarioSummary]:
    sim = sim or SimConfig()
    analysis = analysis or AnalysisConfig()
    out = []
    for name in scenarios or list(SCENARIOS):
        summary = ScenarioSummary(name)
        for seed in range(seeds):
            summary.trials.append(run_trial(name, seed, sim, analysis))
        out.append(summary)
    return out


def to_markdown(summaries: list[ScenarioSummary]) -> str:
    def f(v: object, pct: bool = True) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.0%}" if pct else f"{v:.2f}"
        return str(v)

    lines = [
        "| scenario | trials | detected | top-1 cause | culprit precision | culprit recall | excess explained | analyze (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        r = s.row()
        detect = f"{r['detect']:.0%} (false positive)" if s.scenario == "healthy" else f(r["detect"])
        lines.append(
            f"| {r['scenario']} | {r['trials']} | {detect} | {f(r['top1'])} | {f(r['culprit_precision'])} | "
            f"{f(r['culprit_recall'])} | {f(r['explained'])} | {f(r['analyze_s'], pct=False)} |"
        )
    faults = [s for s in summaries if s.scenario != "healthy"]
    if faults:
        overall = np.mean([t.top1 for s in faults for t in s.trials])
        lines.append("")
        lines.append(f"Overall top-1 root-cause accuracy over {sum(len(s.trials) for s in faults)} fault trials: **{overall:.1%}**")
    return "\n".join(lines)


def sweep(
    param: str,
    values: list[float],
    scenarios: list[str] | None = None,
    seeds: int = 3,
    sim: SimConfig | None = None,
    analysis: AnalysisConfig | None = None,
) -> list[tuple[float, list[ScenarioSummary]]]:
    """Re-run the evaluation while varying one :class:`SimConfig` field
    (e.g. ``severity`` or ``sample_interval_s``)."""
    base = sim or SimConfig()
    faults = scenarios or [s for s in SCENARIOS if s != "healthy"]
    out = []
    for v in values:
        cfg = replace(base)
        setattr(cfg, param, v)
        out.append((v, evaluate(faults, seeds, cfg, analysis)))
    return out


def sweep_markdown(param: str, results: list[tuple[float, list[ScenarioSummary]]]) -> str:
    lines = [
        f"| {param} | median true slowdown | detected | top-1 cause | culprit recall |",
        "|---:|---:|---:|---:|---:|",
    ]
    for value, summaries in results:
        trials = [t for s in summaries for t in s.trials]
        slow = float(np.median([t.true_slowdown_pct for t in trials]))
        recalls = [t.recall for t in trials if t.recall is not None]
        lines.append(
            f"| {value:g} | {slow:.1f}% | {np.mean([t.detected for t in trials]):.0%} | "
            f"{np.mean([t.top1 for t in trials]):.0%} | {np.mean(recalls) if recalls else 0:.0%} |"
        )
    return "\n".join(lines)
