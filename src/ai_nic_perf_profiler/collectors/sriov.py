"""SR-IOV virtual function counters and rate caps.

Per-VF statistics and rate limits are reported by the PF driver through
``ip -s -j link show dev <pf>`` (``vfinfo_list``). VF capacity comes from
``/sys/class/net/<pf>/device/sriov_{num,total}vfs``.

Each VF is reported as its own endpoint named ``<pf>/vf<N>``, with the rate cap
as a gauge, which is what the partition-sizing advisor consumes.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..model import CounterSnapshot
from .base import Clock, Collector, read_int
from .ethtool import Runner, _run


@dataclass(frozen=True, slots=True)
class SriovCapacity:
    pf: str
    num_vfs: int
    total_vfs: int


def read_capacity(pf: str, root: str | Path = "/sys/class/net") -> SriovCapacity | None:
    dev = Path(root) / pf / "device"
    num, total = read_int(dev / "sriov_numvfs"), read_int(dev / "sriov_totalvfs")
    if num is None or total is None:
        return None
    return SriovCapacity(pf, num, total)


def parse_ip_link_json(doc: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-VF stats from ``ip -s -j link show``.

    Returns dicts with ``vf``, ``tx_bytes``, ``rx_bytes``, ``tx_packets``,
    ``rx_packets``, ``max_tx_rate_mbps``, ``min_tx_rate_mbps``. Fields missing in
    the driver's output are omitted; older iproute2 uses ``max_tx_rate``
    instead of ``rate.max_tx``.
    """
    links = doc if isinstance(doc, list) else [doc]
    vfs: list[dict[str, Any]] = []
    for link in links:
        for vf in link.get("vfinfo_list", []):
            entry: dict[str, Any] = {"vf": int(vf["vf"])}
            rate = vf.get("rate", {})
            max_rate = rate.get("max_tx", vf.get("max_tx_rate"))
            min_rate = rate.get("min_tx", vf.get("min_tx_rate"))
            if max_rate is not None:
                entry["max_tx_rate_mbps"] = int(max_rate)
            if min_rate is not None:
                entry["min_tx_rate_mbps"] = int(min_rate)
            stats = vf.get("stats", {})
            for direction in ("rx", "tx"):
                d = stats.get(direction, {})
                if "bytes" in d:
                    entry[f"{direction}_bytes"] = int(d["bytes"])
                if "packets" in d:
                    entry[f"{direction}_packets"] = int(d["packets"])
            vfs.append(entry)
    return vfs


class SriovCollector(Collector):
    name = "sriov"

    def __init__(self, host: str, pfs: list[str], runner: Runner = _run, clock: Clock = time.time) -> None:
        super().__init__(host, clock)
        self.pfs = pfs
        self.runner = runner

    def collect(self) -> list[CounterSnapshot]:
        out: list[CounterSnapshot] = []
        for pf in self.pfs:
            t_start = self.clock()
            try:
                doc = json.loads(self.runner(["ip", "-s", "-j", "link", "show", "dev", pf]))
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
                continue
            ts = (t_start + self.clock()) / 2
            for vf in parse_ip_link_json(doc):
                counters: dict[str, int] = {k: vf[k] for k in ("tx_bytes", "rx_bytes", "tx_packets", "rx_packets") if k in vf}
                gauges: dict[str, float] = {}
                if "max_tx_rate_mbps" in vf:
                    gauges["vf_max_tx_rate_bps"] = vf["max_tx_rate_mbps"] * 1e6
                if "min_tx_rate_mbps" in vf:
                    gauges["vf_min_tx_rate_bps"] = vf["min_tx_rate_mbps"] * 1e6
                out.append(CounterSnapshot(self.host, f"{pf}/vf{vf['vf']}", 1, ts, counters, gauges, self.name))
        return out
