"""Training step events: the throughput side of the attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StepEvent:
    """One optimizer step on one rank, timed on that rank's host clock.

    ``start``/``end`` are wall-clock seconds; ``mono_start_ns``/``mono_end_ns``
    are the same instants on ``CLOCK_MONOTONIC`` (absent in older files).
    """

    rank: int
    step: int
    start: float
    end: float
    host: str = ""
    comm_bytes: int = 0
    mono_start_ns: int | None = None
    mono_end_ns: int | None = None

    @property
    def has_mono(self) -> bool:
        return self.mono_start_ns is not None and self.mono_end_ns is not None

    @property
    def duration(self) -> float:
        if self.mono_start_ns is not None and self.mono_end_ns is not None:
            return (self.mono_end_ns - self.mono_start_ns) / 1e9
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"rank": self.rank, "step": self.step, "start": self.start, "end": self.end}
        if self.host:
            d["host"] = self.host
        if self.comm_bytes:
            d["comm_bytes"] = self.comm_bytes
        if self.has_mono:
            d["mono_start_ns"] = self.mono_start_ns
            d["mono_end_ns"] = self.mono_end_ns
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
            mono_start_ns=int(d["mono_start_ns"]) if d.get("mono_start_ns") is not None else None,
            mono_end_ns=int(d["mono_end_ns"]) if d.get("mono_end_ns") is not None else None,
        )
