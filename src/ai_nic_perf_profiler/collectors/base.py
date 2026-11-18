"""Collector interface and shared helpers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path

from ..model import CounterSnapshot

Clock = Callable[[], float]
MonoClock = Callable[[], int]
Stamp = tuple[float, int]


def midpoint(a: Stamp, b: Stamp) -> Stamp:
    """Midpoint of two (wall seconds, monotonic ns) stamps."""
    return (a[0] + b[0]) / 2, (a[1] + b[1]) // 2


class Collector(ABC):
    """Reads cumulative counters from one kind of source on the local host."""

    name: str = "collector"

    def __init__(self, host: str, clock: Clock = time.time, mono_clock: MonoClock = time.monotonic_ns) -> None:
        self.host = host
        self.clock = clock
        self.mono_clock = mono_clock

    def stamp(self) -> Stamp:
        """Read both clocks: wall time for cross-host display, monotonic for per-host alignment."""
        return self.clock(), self.mono_clock()

    @abstractmethod
    def collect(self) -> list[CounterSnapshot]:
        """Read every endpoint this collector owns. Must not raise on a single bad file."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


def read_int(path: Path) -> int | None:
    """Read an integer sysfs attribute. Returns None when the file is absent or unreadable.

    Some attributes return EINVAL while a link is down (``speed``) and some
    hw_counters are write-only on older kernels, so errors are expected.
    """
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return None


def read_counter_dir(directory: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not directory.is_dir():
        return out
    for entry in directory.iterdir():
        value = read_int(entry)
        if value is not None:
            out[entry.name] = value
    return out


def merge_snapshots(snapshots: Iterable[CounterSnapshot], aliases: dict[str, str] | None = None) -> list[CounterSnapshot]:
    """Merge snapshots of the same endpoint from different sources.

    On RoCE the RDMA device (``mlx5_0``) and its netdev (``ens1f0np0``) are the
    same port. ``aliases`` maps netdev names onto RDMA device names so ethtool
    PFC counters and RDMA hw_counters land on one endpoint. Earlier snapshots
    win on conflicting counter names, so list the most accurate source first.
    """
    aliases = aliases or {}
    merged: dict[tuple[str, str, int], CounterSnapshot] = {}
    for snap in snapshots:
        device = aliases.get(snap.device, snap.device)
        key = (snap.host, device, snap.port)
        existing = merged.get(key)
        if existing is None:
            merged[key] = CounterSnapshot(snap.host, device, snap.port, snap.timestamp, dict(snap.counters), dict(snap.gauges), snap.source, snap.mono_ns)
            continue
        counters = {**snap.counters, **existing.counters}
        gauges = {**snap.gauges, **existing.gauges}
        source = "+".join(filter(None, [existing.source, snap.source]))
        ts = (existing.timestamp + snap.timestamp) / 2
        mono = (existing.mono_ns + snap.mono_ns) // 2 if existing.mono_ns is not None and snap.mono_ns is not None else None
        merged[key] = CounterSnapshot(snap.host, device, snap.port, ts, counters, gauges, source, mono)
    return list(merged.values())
