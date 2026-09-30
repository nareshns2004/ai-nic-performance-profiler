"""Library API in ~20 lines: simulate a fault, attribute it, inspect the result.

PYTHONPATH=src python examples/quickstart.py
"""

from ai_nic_perf_profiler import analyze
from ai_nic_perf_profiler.export import markdown
from ai_nic_perf_profiler.sim import SimConfig, simulate

result = simulate("fabric_congestion", SimConfig(seed=4))
report = analyze(result.snapshots, result.steps, result.topology)

for ep in report.episodes:
    d = ep.primary
    print(f"steps {ep.start_step}-{ep.end_step}: {ep.slowdown_pct:+.1f}% step time")
    if d is not None:
        print(f"  -> {d.cause.value} ({d.share:.0%} of excess), scope={d.scope}, suspect: {d.suspect}")
        print(f"     first action: {d.recommendations[0].action}")

print(f"\nground truth: {result.ground_truth['root_cause']} on ranks {result.ground_truth['culprit_ranks']}\n")
print(markdown.render(report)[:1200], "...")
