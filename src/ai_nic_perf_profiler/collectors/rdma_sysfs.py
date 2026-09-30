"""RDMA port counters from ``/sys/class/infiniband``.

Layout (per the kernel ``ib_core`` sysfs ABI)::

    /sys/class/infiniband/mlx5_0/
        device/net/ens1f0np0            -> RoCE netdev for this HCA
        ports/1/rate                    "400 Gb/sec (4X NDR)"
        ports/1/link_layer              "InfiniBand" | "Ethernet"
        ports/1/counters/port_xmit_data  (4-octet units)
        ports/1/hw_counters/out_of_buffer

Reading sysfs is a handful of ``open/read`` syscalls per counter and needs no
fork, so it is cheap enough to run at sub-second intervals, unlike shelling out
to ``ethtool``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from ..counters.catalog import RDMA_HW_COUNTERS, RDMA_SYSFS_COUNTERS, canonicalize
from ..model import CounterSnapshot
from .base import Clock, Collector, read_counter_dir

_RATE = re.compile(r"([\d.]+)\s*Gb/sec")


def parse_rate(text: str) -> float | None:
    """Parse the ``ports/<p>/rate`` attribute into bits per second."""
    m = _RATE.search(text)
    return float(m.group(1)) * 1e9 if m else None


@dataclass(frozen=True, slots=True)
class RdmaPort:
    device: str
    port: int
    link_layer: str
    rate_bps: float | None
    netdev: str | None
    path: Path


class RdmaSysfsCollector(Collector):
    name = "rdma_sysfs"

    def __init__(
        self,
        host: str,
        root: str | Path = "/sys/class/infiniband",
        devices: list[str] | None = None,
        include_hw_counters: bool = True,
        clock: Clock = time.time,
    ) -> None:
        super().__init__(host, clock)
        self.root = Path(root)
        self.devices = devices
        self.include_hw_counters = include_hw_counters
        self._ports: list[RdmaPort] | None = None

    def discover(self) -> list[RdmaPort]:
        ports: list[RdmaPort] = []
        if not self.root.is_dir():
            return ports
        for dev_dir in sorted(self.root.iterdir()):
            if self.devices and dev_dir.name not in self.devices:
                continue
            net_dir = dev_dir / "device" / "net"
            netdevs = sorted(p.name for p in net_dir.iterdir()) if net_dir.is_dir() else []
            ports_dir = dev_dir / "ports"
            if not ports_dir.is_dir():
                continue
            for port_dir in sorted(ports_dir.iterdir(), key=lambda p: p.name):
                if not port_dir.name.isdigit():
                    continue
                rate = _read_text(port_dir / "rate")
                ports.append(
                    RdmaPort(
                        device=dev_dir.name,
                        port=int(port_dir.name),
                        link_layer=_read_text(port_dir / "link_layer") or "unknown",
                        rate_bps=parse_rate(rate) if rate else None,
                        netdev=netdevs[0] if netdevs else None,
                        path=port_dir,
                    )
                )
        return ports

    def netdev_aliases(self) -> dict[str, str]:
        """Map RoCE netdev names to RDMA device names, for :func:`merge_snapshots`."""
        return {p.netdev: p.device for p in self._discover_cached() if p.netdev}

    def _discover_cached(self) -> list[RdmaPort]:
        if self._ports is None:
            self._ports = self.discover()
        return self._ports

    def collect(self) -> list[CounterSnapshot]:
        out: list[CounterSnapshot] = []
        for port in self._discover_cached():
            t_start = self.clock()
            raw = read_counter_dir(port.path / "counters")
            counters = canonicalize(raw, RDMA_SYSFS_COUNTERS)
            if self.include_hw_counters:
                counters.update(canonicalize(read_counter_dir(port.path / "hw_counters"), RDMA_HW_COUNTERS))
            t_end = self.clock()
            # Speed can change under us (auto-negotiation downshift), so re-read it.
            rate = _read_text(port.path / "rate")
            rate_bps = parse_rate(rate) if rate else None
            gauges = {"link_speed_bps": rate_bps} if rate_bps else {}
            out.append(CounterSnapshot(self.host, port.device, port.port, (t_start + t_end) / 2, counters, gauges, self.name))
        return out


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None
