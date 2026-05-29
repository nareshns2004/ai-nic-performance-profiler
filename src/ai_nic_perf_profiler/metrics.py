"""Metric collection utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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

    def average_latency(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.latency_ms for sample in self.samples) / len(self.samples)

    def peak_throughput(self) -> float:
        if not self.samples:
            return 0.0
        return max(sample.throughput_gbps for sample in self.samples)

    def total_rx_packets(self) -> int:
        return sum(sample.rx_packets for sample in self.samples)

    def total_tx_packets(self) -> int:
        return sum(sample.tx_packets for sample in self.samples)

    def latest_throughput(self) -> float:
        if not self.samples:
            return 0.0
        return self.samples[-1].throughput_gbps

    def to_serializable(self) -> list[dict[str, float | int]]:
        return [
            {
                "timestamp": sample.timestamp,
                "rx_packets": sample.rx_packets,
                "tx_packets": sample.tx_packets,
                "throughput_gbps": sample.throughput_gbps,
                "latency_ms": sample.latency_ms,
            }
            for sample in self.samples
        ]

    def export_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.write_text(json.dumps(self.to_serializable(), indent=2), encoding="utf-8")

    def export_prometheus(self, path: str | Path) -> None:
        output_path = Path(path)
        lines = [
            "# HELP ai_nic_perf_throughput_gbps Observed NIC throughput in Gbps",
            "# TYPE ai_nic_perf_throughput_gbps gauge",
        ]
        for index, sample in enumerate(self.samples):
            lines.append(
                f"ai_nic_perf_throughput_gbps{{sample_id=\"{index}\"}} {sample.throughput_gbps}"
            )
        lines.extend(
            [
                "# HELP ai_nic_perf_latency_ms Observed NIC latency in milliseconds",
                "# TYPE ai_nic_perf_latency_ms gauge",
            ]
        )
        for index, sample in enumerate(self.samples):
            lines.append(f"ai_nic_perf_latency_ms{{sample_id=\"{index}\"}} {sample.latency_ms}")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
