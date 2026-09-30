"""Configuration for the agent and the analysis pipeline (YAML-loadable)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AgentConfig:
    interval_s: float = 1.0
    sources: list[str] = field(default_factory=lambda: ["rdma", "netdev"])
    rdma_devices: list[str] | None = None
    netdev_interfaces: list[str] | None = None
    # ethtool forks a process per interface per tick (~ms each); keep
    # interval_s >= 1 when enabling it.
    ethtool_interfaces: list[str] = field(default_factory=list)
    sriov_pfs: list[str] = field(default_factory=list)
    lossless_priority: int = 3
    output: str = "counters.jsonl"
    prometheus_textfile: str | None = None
    ring_buffer: int = 600


@dataclass
class AnalysisConfig:
    warmup_steps: int = 10
    baseline_steps: tuple[int, int] | None = None
    baseline_fraction: float = 0.25
    degradation_sigma: float = 4.0
    min_relative_slowdown: float = 0.03
    min_episode_steps: int = 3
    merge_gap_steps: int = 2
    culprit_z: float = 6.0
    min_coverage: float = 0.5
    ridge_lambda: float = 1.0


@dataclass
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        doc = yaml.safe_load(Path(path).read_text()) or {}
        return cls(agent=_build(AgentConfig, doc.get("agent", {})), analysis=_build(AnalysisConfig, doc.get("analysis", {})))


def _build(kind: Any, values: dict[str, Any]) -> Any:
    known = {f.name for f in fields(kind)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {kind.__name__} keys: {sorted(unknown)}")
    if values.get("baseline_steps") is not None:
        values = {**values, "baseline_steps": tuple(values["baseline_steps"])}
    return kind(**values)
