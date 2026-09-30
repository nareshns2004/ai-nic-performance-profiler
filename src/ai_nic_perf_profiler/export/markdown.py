"""Human-readable incident-style report."""

from __future__ import annotations

from ..analysis.attribution import AttributionReport
from ..analysis.causes import DECISION_DOMAIN, SUMMARY, TITLE

BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float | None], width: int = 72) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    step = max(1, len(values) // width)
    binned = []
    for i in range(0, len(values), step):
        chunk = [v for v in values[i : i + step] if v is not None]
        binned.append(max(chunk) if chunk else None)
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    return "".join(" " if v is None else BLOCKS[min(len(BLOCKS) - 1, int((v - lo) / span * len(BLOCKS)))] for v in binned)


def render(report: AttributionReport) -> str:
    ms = lambda s: f"{s * 1e3:.1f} ms"  # noqa: E731
    out = [
        "# NIC attribution report",
        "",
        f"**{report.n_steps} steps x {report.n_ranks} ranks** · baseline steps {report.baseline_steps[0]}-{report.baseline_steps[1] - 1}: "
        f"median {ms(report.baseline_step_s)} (σ {ms(report.baseline_sigma_s)}) · decomposition R² {report.model_r2:.2f}",
        "",
    ]
    series = report.step_series.get("step_time_s") or []
    if series:
        out += ["```", f"step time  {sparkline(series)}", "```", ""]

    if not report.episodes:
        out += ["No degradation episodes detected: step time stayed within the baseline envelope.", ""]

    for n, ep in enumerate(report.episodes, 1):
        explained = ep.explained_s / ep.excess_s if ep.excess_s > 0 else 0.0
        out += [
            f"## Episode {n}: steps {ep.start_step}-{ep.end_step} ({ep.kind}, {ep.n_steps} steps)",
            "",
            f"Step time {ms(ep.baseline_step_s)} → {ms(ep.episode_step_s)} (**{ep.slowdown_pct:+.1f}%**). "
            f"Excess {ep.excess_s:.2f} s over the episode; **{explained:.0%} explained** by NIC counter evidence.",
            "",
        ]
        if not ep.diagnoses:
            out += ["No cause cleared the evidence threshold.", ""]
            continue
        primary = ep.diagnoses[0]
        out += [
            f"### Primary: {TITLE[primary.cause]} ({primary.share:.0%} of excess, evidence {primary.evidence_score:.2f})",
            "",
            f"{SUMMARY[primary.cause]}.",
            "",
            f"- **Decision domain:** {DECISION_DOMAIN[primary.cause]}",
            f"- **Scope:** {primary.scope} · **Suspect:** {primary.suspect}",
        ]
        if primary.evidence:
            out.append("- **Evidence:**")
            out += [f"  - {e}" for e in primary.evidence]
        out += ["", "**Recommended actions**", ""]
        for i, r in enumerate(primary.recommendations, 1):
            out.append(f"{i}. {r.action}  \n   _Verify:_ {r.verify}")
        out.append("")
        if len(ep.diagnoses) > 1:
            out += ["### Also contributing", "", "| cause | share of excess | evidence | culprit ranks | suspect |", "|---|---:|---:|---|---|"]
            for d in ep.diagnoses[1:]:
                out.append(f"| {TITLE[d.cause]} | {d.share:.0%} | {d.evidence_score:.2f} | {', '.join(map(str, d.culprit_ranks[:8])) or '-'} | {d.suspect} |")
            out.append("")
        contrib = sorted(((k, v) for k, v in ep.feature_contributions.items() if v > 1e-6), key=lambda kv: -kv[1])
        if contrib:
            out += ["<details><summary>Feature-level decomposition</summary>", "", "| feature | seconds | share of excess |", "|---|---:|---:|"]
            out += [f"| {k} | {v:.3f} | {v / ep.excess_s:.0%} |" for k, v in contrib]
            out += ["", "</details>", ""]

    dq = report.data_quality
    out += [
        "## Data quality",
        "",
        f"- Endpoints with counters: {dq.endpoints_found}/{dq.endpoints_expected}"
        + (f" (missing: {', '.join(dq.missing_endpoints)})" if dq.missing_endpoints else ""),
        f"- Step windows: {dq.windows_total}, low counter coverage: {dq.windows_low_coverage}, inferred from another rank's clock: {dq.windows_inferred}",
        f"- Counter anomalies handled: {', '.join(f'{k}={v}' for k, v in sorted(dq.counter_flags.items())) or 'none'}",
        "",
    ]
    return "\n".join(out)
