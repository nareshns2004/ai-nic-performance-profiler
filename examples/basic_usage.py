from ai_nic_perf_profiler.config import ProfilerConfig
from ai_nic_perf_profiler.metrics import Sample
from ai_nic_perf_profiler.profiler import PerfProfiler


if __name__ == "__main__":
    config = ProfilerConfig(report_path="example_report.json", metrics_path="example_metrics.prom")
    profiler = PerfProfiler(config=config)
    profiler.record_sample(Sample(timestamp=0.0, rx_packets=1000, tx_packets=980, throughput_gbps=11.2, latency_ms=0.9))
    profiler.record_sample(Sample(timestamp=0.5, rx_packets=1200, tx_packets=1150, throughput_gbps=12.8, latency_ms=0.8))
    profiler.export()
    print(profiler.summarize())
