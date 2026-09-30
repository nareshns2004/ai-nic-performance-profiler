"""JSONL persistence for counter snapshots and step events (``.gz`` supported)."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO

from .model import CounterSnapshot
from .training.events import StepEvent


def _open(path: str | Path, mode: str) -> IO[str]:
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode + "t", encoding="utf-8")  # type: ignore[return-value]
    return p.open(mode, encoding="utf-8")


def _iter_json(path: str | Path) -> Iterator[dict]:
    with _open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                # A partially written last line is normal when the agent is killed.
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc


def read_snapshots(path: str | Path) -> list[CounterSnapshot]:
    return [CounterSnapshot.from_dict(d) for d in _iter_json(path)]


def write_snapshots(path: str | Path, snapshots: Iterable[CounterSnapshot], append: bool = False) -> None:
    with _open(path, "a" if append else "w") as fh:
        for snap in snapshots:
            fh.write(json.dumps(snap.to_dict(), separators=(",", ":")) + "\n")


def read_steps(paths: Iterable[str | Path]) -> list[StepEvent]:
    events: list[StepEvent] = []
    for path in paths:
        events.extend(StepEvent.from_dict(d) for d in _iter_json(path))
    return events


def write_steps(path: str | Path, events: Iterable[StepEvent]) -> None:
    with _open(path, "w") as fh:
        for ev in events:
            fh.write(json.dumps(ev.to_dict(), separators=(",", ":")) + "\n")
