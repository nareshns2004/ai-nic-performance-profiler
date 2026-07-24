from ai_nic_perf_profiler.metrics import Sample
from ai_nic_perf_profiler.profiler import PerfProfiler


def test_profiler_summary() -> None:
    profiler = PerfProfiler()
    profiler.record_sample(Sample(timestamp=0.0, throughput_gbps=10.0, latency_ms=1.0))
    profiler.record_sample(Sample(timestamp=0.5, throughput_gbps=20.0, latency_ms=0.5))

    summary = profiler.summarize()

    assert summary["sample_count"] == 2
    assert summary["avg_throughput_gbps"] == 15.0
    assert summary["diagnostics_enabled"] is True
