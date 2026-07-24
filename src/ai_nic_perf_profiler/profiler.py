"""Profiler orchestration."""

from __future__ import annotations

from .config import ProfilerConfig
from .metrics import MetricCollector, Sample


class PerfProfiler:
    """Minimal execution wrapper for a NIC performance profile run."""

    def __init__(self, config: ProfilerConfig | None = None) -> None:
        self.config = config or ProfilerConfig()
        self.collector = MetricCollector()

    def record_sample(self, sample: Sample) -> None:
        self.collector.add_sample(sample)

    def summarize(self) -> dict[str, float | bool]:
        return {
            "avg_throughput_gbps": self.collector.average_throughput(),
            "sample_count": len(self.collector.samples),
            "diagnostics_enabled": self.config.enable_diagnostics,
        }
