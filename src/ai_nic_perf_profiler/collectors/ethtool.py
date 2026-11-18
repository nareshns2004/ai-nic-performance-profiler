"""PHY and PFC counters from ``ethtool -S`` (mlx5 naming).

This is the only place per-priority PFC pause counters and pause *durations*
are exposed on RoCE NICs, so it's needed even alongside RDMA sysfs.

Two read paths (``backend=``):

* ``"ioctl"``: the same SIOCETHTOOL calls ``ethtool -S`` makes, done
  in-process (:mod:`.ethtool_ioctl`). No fork; the default via ``"auto"``.
* ``"exec"``: run ``ethtool -S`` and parse its output. One fork+exec per
  interface per tick (~ms each).

``"auto"`` tries the ioctl and falls back to ``exec`` per interface if the
ioctl fails (unusual kernels, non-Linux test hosts).
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from typing import Literal

from ..counters.catalog import ETHTOOL_COUNTERS, canonicalize
from ..model import CounterSnapshot
from .base import Clock, Collector, midpoint
from .ethtool_ioctl import EthtoolIoctl

Runner = Callable[[list[str]], str]
Backend = Literal["auto", "ioctl", "exec"]


def parse_ethtool_stats(text: str) -> dict[str, int]:
    """Parse ``ethtool -S`` output (``     name: value`` lines)."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        name, sep, value = line.strip().partition(":")
        if not sep:
            continue
        try:
            out[name.strip()] = int(value.strip())
        except ValueError:
            continue
    return out


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5).stdout


class EthtoolCollector(Collector):
    name = "ethtool"

    def __init__(
        self,
        host: str,
        interfaces: list[str],
        lossless_priority: int = 3,
        runner: Runner = _run,
        clock: Clock = time.time,
        backend: Backend = "auto",
        reader: EthtoolIoctl | None = None,
    ) -> None:
        super().__init__(host, clock)
        if backend not in ("auto", "ioctl", "exec"):
            raise ValueError(f"unknown ethtool backend {backend!r}")
        self.interfaces = interfaces
        self.lossless_priority = lossless_priority
        self.runner = runner
        self.backend: Backend = backend
        self.reader = reader or (EthtoolIoctl() if backend != "exec" else None)
        # Interfaces where the ioctl failed under "auto"; they use exec from then on.
        self.exec_fallback: set[str] = set()

    def read_raw(self, iface: str) -> dict[str, int] | None:
        """Raw ``ethtool -S`` name -> value for one interface, or None if it can't be read."""
        if self.reader is not None and iface not in self.exec_fallback:
            try:
                return self.reader.read(iface)
            except OSError:
                if self.backend == "ioctl":
                    return None
                self.exec_fallback.add(iface)
        try:
            return parse_ethtool_stats(self.runner(["ethtool", "-S", iface]))
        except (OSError, subprocess.SubprocessError):
            return None

    def collect(self) -> list[CounterSnapshot]:
        out: list[CounterSnapshot] = []
        for iface in self.interfaces:
            t_start = self.stamp()
            raw = self.read_raw(iface)
            if raw is None:
                continue
            ts, mono = midpoint(t_start, self.stamp())
            counters = canonicalize(raw, ETHTOOL_COUNTERS, self.lossless_priority)
            out.append(CounterSnapshot(self.host, iface, 1, ts, counters, {}, self.name, mono))
        return out

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()
