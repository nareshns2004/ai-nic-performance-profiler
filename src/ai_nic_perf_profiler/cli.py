"""Command-line entry point for the profiler."""

from __future__ import annotations

import click

from .config import ProfilerConfig
from .metrics import Sample
from .profiler import PerfProfiler


@click.command()
@click.option("--sample-interval", default=0.5, type=float, show_default=True)
@click.option("--report-path", default="report.json", show_default=True)
def main(sample_interval: float, report_path: str) -> None:
    """Run a minimal AI NIC profiling session."""
    config = ProfilerConfig(sample_interval=sample_interval, report_path=report_path)
    profiler = PerfProfiler(config=config)
    profiler.record_sample(Sample(timestamp=0.0, throughput_gbps=12.5, latency_ms=1.2))
    profiler.record_sample(Sample(timestamp=0.5, throughput_gbps=13.1, latency_ms=1.1))
    summary = profiler.summarize()
    click.echo(f"Profiler summary: {summary}")


if __name__ == "__main__":
    main()
