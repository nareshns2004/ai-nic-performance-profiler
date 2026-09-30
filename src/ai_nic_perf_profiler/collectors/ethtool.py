"""PHY and PFC counters from ``ethtool -S`` (mlx5 naming).

This is the only place per-priority PFC pause counters and pause *durations*
are exposed on RoCE NICs, so it's needed even alongside RDMA sysfs. Each read
forks a process (~ms), so run it at a coarser interval than sysfs, or replace
it with the ETHTOOL_MSG_STATS_GET netlink call.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

from ..counters.catalog import ETHTOOL_COUNTERS, canonicalize
from ..model import CounterSnapshot
from .base import Clock, Collector

Runner = Callable[[list[str]], str]


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
    ) -> None:
        super().__init__(host, clock)
        self.interfaces = interfaces
        self.lossless_priority = lossless_priority
        self.runner = runner

    def collect(self) -> list[CounterSnapshot]:
        out: list[CounterSnapshot] = []
        for iface in self.interfaces:
            t_start = self.clock()
            try:
                text = self.runner(["ethtool", "-S", iface])
            except (OSError, subprocess.SubprocessError):
                continue
            t_end = self.clock()
            counters = canonicalize(parse_ethtool_stats(text), ETHTOOL_COUNTERS, self.lossless_priority)
            out.append(CounterSnapshot(self.host, iface, 1, (t_start + t_end) / 2, counters, {}, self.name))
        return out
