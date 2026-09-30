"""Kernel netdev statistics from ``/sys/class/net/<if>/statistics``.

Works on any Linux host, which makes it the fallback for development machines
and TCP-transport clusters. On RoCE it only sees traffic that goes through the
kernel stack: RDMA traffic bypasses it, so use :mod:`rdma_sysfs` or
:mod:`ethtool` for the real wire counters.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..counters.catalog import NETDEV_COUNTERS, canonicalize
from ..model import CounterSnapshot
from .base import Clock, Collector, read_counter_dir, read_int


class NetdevSysfsCollector(Collector):
    name = "netdev"

    def __init__(
        self,
        host: str,
        root: str | Path = "/sys/class/net",
        interfaces: list[str] | None = None,
        include_virtual: bool = False,
        clock: Clock = time.time,
    ) -> None:
        super().__init__(host, clock)
        self.root = Path(root)
        self.interfaces = interfaces
        self.include_virtual = include_virtual

    def discover(self) -> list[str]:
        if self.interfaces:
            return list(self.interfaces)
        if not self.root.is_dir():
            return []
        names = []
        for entry in sorted(self.root.iterdir()):
            # Physical and SR-IOV interfaces have a backing ``device`` link;
            # bridges, veths and loopback don't.
            if not self.include_virtual and not (entry / "device").exists():
                continue
            names.append(entry.name)
        return names

    def collect(self) -> list[CounterSnapshot]:
        out: list[CounterSnapshot] = []
        for iface in self.discover():
            base = self.root / iface
            t_start = self.clock()
            raw = read_counter_dir(base / "statistics")
            t_end = self.clock()
            if not raw:
                continue
            gauges: dict[str, float] = {}
            speed_mbps = read_int(base / "speed")
            if speed_mbps is not None and speed_mbps > 0:
                gauges["link_speed_bps"] = speed_mbps * 1e6
            out.append(CounterSnapshot(self.host, iface, 1, (t_start + t_end) / 2, canonicalize(raw, NETDEV_COUNTERS), gauges, self.name))
        return out
