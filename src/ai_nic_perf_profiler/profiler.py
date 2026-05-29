"""Profiler orchestration."""

from __future__ import annotations

from pathlib import Path

from .config import ProfilerConfig
from .metrics import MetricCollector, Sample


class PerfProfiler:
    """Minimal execution wrapper for a NIC performance profile run."""

    def __init__(self, config: ProfilerConfig | None = None) -> None:
        self.config = config or ProfilerConfig()
        self.collector = MetricCollector()

    def record_sample(self, sample: Sample) -> None:
        self.collector.add_sample(sample)

    def summarize(self) -> dict[str, float | int | bool]:
        return {
            "avg_throughput_gbps": self.collector.average_throughput(),
            "peak_throughput_gbps": self.collector.peak_throughput(),
            "latest_throughput_gbps": self.collector.latest_throughput(),
            "avg_latency_ms": self.collector.average_latency(),
            "rx_packets_total": self.collector.total_rx_packets(),
            "tx_packets_total": self.collector.total_tx_packets(),
            "sample_count": len(self.collector.samples),
            "diagnostics_enabled": self.config.enable_diagnostics,
        }

    def export(self) -> None:
        self.collector.export_json(self.config.report_path)
        if self.config.metrics_path:
            self.collector.export_prometheus(self.config.metrics_path)

    def export_to_file(self, report_path: str | Path | None = None, metrics_path: str | Path | None = None) -> None:
        report_target = Path(report_path or self.config.report_path)
        metrics_target = Path(metrics_path or self.config.metrics_path or "") if metrics_path or self.config.metrics_path else None
        self.collector.export_json(report_target)
        if metrics_target is not None:
            self.collector.export_prometheus(metrics_target)
