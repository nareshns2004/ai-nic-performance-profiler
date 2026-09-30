"""Per-host sampling agent.

Design points:

* **Drift-free schedule**: deadlines advance by a fixed interval from a
  monotonic clock, so a slow read doesn't shift every later sample. If a read
  overruns a whole interval, the missed ticks are skipped and counted rather
  than bursting to catch up.
* **Wall-clock timestamps** on samples (not monotonic) so that step events
  from the training process on the same host line up. Cross-host comparison
  goes through step numbers; see :mod:`analysis.align`.
* **Bounded memory**: recent rate samples live in a ring buffer. The full
  cumulative history goes to an append-only JSONL file that is cheap to ship
  off-host when an episode needs analysis.
* **Self-observability**: collection latency and overruns are exported
  alongside the NIC metrics, because a monitoring agent that silently falls
  behind is worse than none.
"""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .collectors import (
    Collector,
    EthtoolCollector,
    NetdevSysfsCollector,
    RdmaSysfsCollector,
    SriovCollector,
    merge_snapshots,
)
from .config import AgentConfig
from .counters.rates import to_rate
from .export.prometheus import Metric, rate_metrics, render, write_textfile
from .io import write_snapshots
from .model import CounterSnapshot, RateSample


@dataclass
class AgentStats:
    ticks: int = 0
    snapshots: int = 0
    overruns: int = 0
    errors: int = 0
    last_collect_s: float = 0.0
    total_collect_s: float = 0.0

    @property
    def mean_collect_s(self) -> float:
        return self.total_collect_s / self.ticks if self.ticks else 0.0


def build_collectors(cfg: AgentConfig, host: str | None = None) -> tuple[list[Collector], dict[str, str]]:
    """Instantiate collectors from config. Returns (collectors, netdev->rdma aliases)."""
    host = host or socket.gethostname()
    collectors: list[Collector] = []
    aliases: dict[str, str] = {}
    if "rdma" in cfg.sources:
        rdma = RdmaSysfsCollector(host, devices=cfg.rdma_devices)
        aliases = rdma.netdev_aliases()
        collectors.append(rdma)
    if cfg.ethtool_interfaces:
        collectors.append(EthtoolCollector(host, cfg.ethtool_interfaces, cfg.lossless_priority))
    if "netdev" in cfg.sources:
        # On RoCE hosts the RDMA collector already covers these ports with wire-level counters.
        skip = set(aliases)
        netdev = NetdevSysfsCollector(host, interfaces=cfg.netdev_interfaces)
        if skip and not cfg.netdev_interfaces:
            netdev.interfaces = [i for i in netdev.discover() if i not in skip]
        collectors.append(netdev)
    if cfg.sriov_pfs:
        collectors.append(SriovCollector(host, cfg.sriov_pfs))
    return collectors, aliases


class Agent:
    def __init__(
        self,
        collectors: list[Collector],
        interval_s: float = 1.0,
        output: str | Path | None = None,
        prometheus_textfile: str | Path | None = None,
        aliases: dict[str, str] | None = None,
        ring_size: int = 600,
        on_sample: Callable[[list[RateSample]], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.collectors = collectors
        self.interval_s = interval_s
        self.output = Path(output) if output else None
        self.prometheus_textfile = prometheus_textfile
        self.aliases = aliases or {}
        self.ring: deque[RateSample] = deque(maxlen=ring_size)
        self.on_sample = on_sample
        self.stats = AgentStats()
        self._last: dict[str, CounterSnapshot] = {}
        self._monotonic = monotonic
        self.stop_event = threading.Event()

    def tick(self) -> list[RateSample]:
        t0 = self._monotonic()
        raw: list[CounterSnapshot] = []
        for c in self.collectors:
            try:
                raw.extend(c.collect())
            except Exception:  # a broken collector must not kill the agent
                self.stats.errors += 1
        snaps = merge_snapshots(raw, self.aliases)
        if self.output:
            write_snapshots(self.output, snaps, append=True)
        rates = []
        for snap in snaps:
            key = snap.endpoint.key
            prev = self._last.get(key)
            self._last[key] = snap
            if prev is not None and (r := to_rate(prev, snap)) is not None:
                rates.append(r)
        self.ring.extend(rates)
        elapsed = self._monotonic() - t0
        self.stats.ticks += 1
        self.stats.snapshots += len(snaps)
        self.stats.last_collect_s = elapsed
        self.stats.total_collect_s += elapsed
        if self.prometheus_textfile and rates:
            write_textfile(self.prometheus_textfile, render([*rate_metrics(rates), *self._self_metrics()]))
        if self.on_sample and rates:
            self.on_sample(rates)
        return rates

    def _self_metrics(self) -> list[Metric]:
        dur = Metric("nicprof_agent_collect_seconds", "Wall time of the last collection pass")
        dur.add(self.stats.last_collect_s)
        over = Metric("nicprof_agent_overruns_total", "Ticks skipped because collection overran the interval", "counter")
        over.add(self.stats.overruns)
        errs = Metric("nicprof_agent_collector_errors_total", "Collector exceptions", "counter")
        errs.add(self.stats.errors)
        return [dur, over, errs]

    def run(self, duration_s: float | None = None, max_ticks: int | None = None) -> AgentStats:
        start = self._monotonic()
        deadline = start
        while not self.stop_event.is_set():
            self.tick()
            if max_ticks is not None and self.stats.ticks >= max_ticks:
                break
            deadline += self.interval_s
            now = self._monotonic()
            if duration_s is not None and now - start >= duration_s:
                break
            if now > deadline:
                missed = int((now - deadline) // self.interval_s) + 1
                self.stats.overruns += missed
                deadline += missed * self.interval_s
            # Event.wait doubles as an interruptible sleep for SIGTERM handling.
            self.stop_event.wait(max(0.0, deadline - self._monotonic()))
        for c in self.collectors:
            c.close()
        return self.stats
