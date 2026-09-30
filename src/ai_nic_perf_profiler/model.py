"""Core data types shared by collectors, the agent and the analysis pipeline.

Everything in this module is stdlib-only so the per-host agent stays light:
numpy is imported only by the analysis side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One NIC port (or SR-IOV VF) on one host."""

    host: str
    device: str
    port: int = 1

    @property
    def key(self) -> str:
        return f"{self.host}/{self.device}/{self.port}"


@dataclass(frozen=True, slots=True)
class CounterSnapshot:
    """Cumulative counter values read from one endpoint at one instant.

    ``counters`` uses canonical names (see :mod:`counters.catalog`) and holds the
    cumulative value in canonical units (bytes, packets, events, microseconds).
    ``gauges`` holds instantaneous values such as the negotiated link speed.
    ``timestamp`` is the host's wall clock (seconds since epoch) taken at the
    midpoint of the read.
    """

    host: str
    device: str
    port: int
    timestamp: float
    counters: dict[str, int]
    gauges: dict[str, float] = field(default_factory=dict)
    source: str = ""

    @property
    def endpoint(self) -> Endpoint:
        return Endpoint(self.host, self.device, self.port)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "host": self.host,
            "device": self.device,
            "port": self.port,
            "ts": self.timestamp,
            "c": self.counters,
        }
        if self.gauges:
            out["g"] = self.gauges
        if self.source:
            out["src"] = self.source
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CounterSnapshot:
        return cls(
            host=d["host"],
            device=d["device"],
            port=int(d.get("port", 1)),
            timestamp=float(d["ts"]),
            counters={k: int(v) for k, v in d["c"].items()},
            gauges={k: float(v) for k, v in d.get("g", {}).items()},
            source=d.get("src", ""),
        )


@dataclass(frozen=True, slots=True)
class RateSample:
    """Per-second rates for one endpoint over the interval ``[t0, t1)``.

    A counter is absent from ``rates`` when its delta could not be trusted
    (saturated counter, ambiguous reset); ``flags`` records why.
    """

    host: str
    device: str
    port: int
    t0: float
    t1: float
    rates: dict[str, float]
    gauges: dict[str, float]
    flags: tuple[str, ...] = ()

    @property
    def endpoint(self) -> Endpoint:
        return Endpoint(self.host, self.device, self.port)

    @property
    def dt(self) -> float:
        return self.t1 - self.t0
