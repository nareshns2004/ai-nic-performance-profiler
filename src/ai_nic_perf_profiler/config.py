"""Configuration primitives for profiler runs."""

from dataclasses import dataclass


@dataclass
class ProfilerConfig:
    """Basic settings for an NIC profiling run."""

    sample_interval: float = 0.5
    report_path: str = "report.json"
    metrics_path: str | None = None
    enable_diagnostics: bool = True
