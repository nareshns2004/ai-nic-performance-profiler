from ai_nic_perf_profiler.config import ProfilerConfig
from ai_nic_perf_profiler.metrics import Sample
from ai_nic_perf_profiler.profiler import PerfProfiler


if __name__ == "__main__":
    config = ProfilerConfig(report_path="example_report.json")
    profiler = PerfProfiler(config=config)
    profiler.record_sample(Sample(timestamp=0.0, throughput_gbps=11.2, latency_ms=0.9))
    print(profiler.summarize())
