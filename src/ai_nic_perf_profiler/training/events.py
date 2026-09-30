"""Training step events: the throughput side of the attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StepEvent:
    """One optimizer step on one rank, timed on that rank's host clock."""

    rank: int
    step: int
    start: float
    end: float
    host: str = ""
    comm_bytes: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"rank": self.rank, "step": self.step, "start": self.start, "end": self.end}
        if self.host:
            d["host"] = self.host
        if self.comm_bytes:
            d["comm_bytes"] = self.comm_bytes
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepEvent:
        return cls(
            rank=int(d["rank"]),
            step=int(d["step"]),
            start=float(d["start"]),
            end=float(d["end"]),
            host=d.get("host", ""),
            comm_bytes=int(d.get("comm_bytes", 0)),
        )
