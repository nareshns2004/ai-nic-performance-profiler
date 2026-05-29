"""Command-line entry point for the profiler."""

from __future__ import annotations

import click

from .config import ProfilerConfig
from .metrics import Sample
from .profiler import PerfProfiler


@click.command()
@click.option("--sample-interval", default=0.5, type=float, show_default=True)
@click.option("--report-path", default="report.json", show_default=True)
@click.option("--metrics-path", default=None, type=str, help="Optional path for Prometheus-compatible metrics export")
def main(sample_interval: float, report_path: str, metrics_path: str | None) -> None:
    """Run a minimal AI NIC profiling session."""
    config = ProfilerConfig(sample_interval=sample_interval, report_path=report_path, metrics_path=metrics_path)
    profiler = PerfProfiler(config=config)
    profiler.record_sample(
        Sample(timestamp=0.0, rx_packets=1200, tx_packets=1180, throughput_gbps=12.5, latency_ms=1.2)
    )
    profiler.record_sample(
        Sample(timestamp=0.5, rx_packets=1400, tx_packets=1350, throughput_gbps=13.1, latency_ms=1.1)
    )
    summary = profiler.summarize()
    profiler.export()
    click.echo(f"Profiler summary: {summary}")
    if metrics_path:
        click.echo(f"Prometheus metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
