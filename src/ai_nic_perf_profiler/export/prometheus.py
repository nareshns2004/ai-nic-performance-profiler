"""Prometheus text exposition (for node_exporter's textfile collector).

Cardinality matters at fleet scale: 10k hosts x 8 NICs x ~25 counters is 2M
series if every raw counter is exported. The agent therefore exports only
*rates* for a curated set of counters, and the attribution output is a handful
of series per job. The raw cumulative counters stay in the local JSONL ring,
to be pulled on demand when an episode needs analysis.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..analysis.attribution import AttributionReport
from ..model import RateSample

EXPORTED_RATES = (
    "tx_bytes", "rx_bytes", "rx_pause_duration_us", "tx_pause_frames", "ecn_marked_packets", "cnp_handled",
    "out_of_buffer", "rx_discards_phy", "packet_seq_err", "local_ack_timeout_err", "symbol_error", "link_downed",
)  # fmt: skip


@dataclass
class Metric:
    name: str
    help: str
    type: str = "gauge"
    samples: list[tuple[dict[str, str], float]] = field(default_factory=list)

    def add(self, value: float, **labels: object) -> None:
        self.samples.append(({k: str(v) for k, v in labels.items()}, value))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render(metrics: Iterable[Metric]) -> str:
    lines: list[str] = []
    for m in metrics:
        if not m.samples:
            continue
        lines.append(f"# HELP {m.name} {m.help}")
        lines.append(f"# TYPE {m.name} {m.type}")
        for labels, value in m.samples:
            label_str = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
            lines.append(f"{m.name}{{{label_str}}} {value:.6g}" if label_str else f"{m.name} {value:.6g}")
    return "\n".join(lines) + "\n"


def write_textfile(path: str | Path, text: str) -> None:
    """Write atomically: the textfile collector must never scrape a half-written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def rate_metrics(samples: Iterable[RateSample]) -> list[Metric]:
    rate = Metric("nicprof_counter_rate", "Per-second rate of a NIC counter over the last interval")
    speed = Metric("nicprof_link_speed_bps", "Negotiated link speed")
    cap = Metric("nicprof_vf_max_tx_rate_bps", "SR-IOV VF transmit cap (0 = unlimited)")
    for s in samples:
        ep = {"host": s.host, "device": s.device, "port": s.port}
        for name in EXPORTED_RATES:
            if name in s.rates:
                rate.add(s.rates[name], counter=name, **ep)
        if "link_speed_bps" in s.gauges:
            speed.add(s.gauges["link_speed_bps"], **ep)
        if "vf_max_tx_rate_bps" in s.gauges:
            cap.add(s.gauges["vf_max_tx_rate_bps"], **ep)
    return [rate, speed, cap]


def report_metrics(report: AttributionReport, job: str = "default") -> list[Metric]:
    base = Metric("nicprof_baseline_step_seconds", "Median step time in the baseline window")
    base.add(report.baseline_step_s, job=job)
    slow = Metric("nicprof_episode_slowdown_ratio", "Median step time in episode / baseline")
    attributed = Metric("nicprof_attributed_seconds", "Excess step time attributed to a root cause over the episode")
    unexplained = Metric("nicprof_unexplained_seconds", "Excess step time without network evidence")
    for ep in report.episodes:
        eid = f"{ep.start_step}-{ep.end_step}"
        slow.add(1 + ep.slowdown_pct / 100, job=job, episode=eid)
        unexplained.add(ep.unexplained_s, job=job, episode=eid)
        for d in ep.diagnoses:
            attributed.add(d.attributed_s, job=job, episode=eid, cause=d.cause.value, scope=d.scope)
    return [base, slow, attributed, unexplained]
