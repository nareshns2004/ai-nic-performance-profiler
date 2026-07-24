"""Metric collection utilities."""

from dataclasses import dataclass


@dataclass
class Sample:
    """A single observed metric sample."""

    timestamp: float
    rx_packets: int = 0
    tx_packets: int = 0
    throughput_gbps: float = 0.0
    latency_ms: float = 0.0


class MetricCollector:
    """Collects and normalizes metrics for analysis."""

    def __init__(self) -> None:
        self.samples: list[Sample] = []

    def add_sample(self, sample: Sample) -> None:
        self.samples.append(sample)

    def average_throughput(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.throughput_gbps for sample in self.samples) / len(self.samples)
