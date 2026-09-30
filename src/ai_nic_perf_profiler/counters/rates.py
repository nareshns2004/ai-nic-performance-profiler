"""Turn cumulative counter snapshots into trustworthy per-second rates.

Three things go wrong with raw hardware counters and all of them show up in
production fleets:

* **Wrap** - 32-bit mlx5 Q counters wrap at 2**32. A decrease that is small
  relative to the counter range is treated as a wrap.
* **Saturation** - IBTA PortCounters error fields (``symbol_error`` is 16-bit,
  ``link_downed`` 8-bit) stick at their maximum until someone runs
  ``perfquery -R``. Once saturated, the delta is unknowable; we drop the
  counter and flag it rather than report a fake zero.
* **Reset** - driver reload, firmware reset or a manual clear. A decrease that
  is not a plausible wrap is a reset; the post-reset value is a lower bound on
  the true delta, which we use.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise

from ..model import CounterSnapshot, RateSample
from .catalog import spec_for


def counter_delta(name: str, prev: int, cur: int) -> tuple[int | None, str | None]:
    """Return ``(delta, flag)`` for one counter between two reads."""
    spec = spec_for(name)
    if spec.overflow == "saturate" and prev >= spec.max_value:
        return None, f"saturated:{name}"
    if cur >= prev:
        return cur - prev, None
    if spec.overflow == "wrap" and spec.bits <= 32:
        wrapped = cur + (1 << spec.bits) - prev
        # A counter cannot plausibly advance more than half its range between
        # two samples; anything larger is a reset that happened to land low.
        if wrapped <= (1 << spec.bits) // 2:
            return wrapped, f"wrap:{name}"
    return cur, f"reset:{name}"


def to_rate(prev: CounterSnapshot, cur: CounterSnapshot) -> RateSample | None:
    dt = cur.timestamp - prev.timestamp
    if dt <= 0:
        return None
    rates: dict[str, float] = {}
    flags: list[str] = []
    for name, value in cur.counters.items():
        before = prev.counters.get(name)
        if before is None:
            continue
        delta, flag = counter_delta(name, before, value)
        if flag:
            flags.append(flag)
        if delta is not None:
            rates[name] = delta / dt
    return RateSample(
        host=cur.host,
        device=cur.device,
        port=cur.port,
        t0=prev.timestamp,
        t1=cur.timestamp,
        rates=rates,
        gauges=dict(cur.gauges),
        flags=tuple(flags),
    )


def rate_series(snapshots: Iterable[CounterSnapshot]) -> dict[str, list[RateSample]]:
    """Group snapshots by endpoint and convert consecutive pairs into rates."""
    by_endpoint: dict[str, list[CounterSnapshot]] = defaultdict(list)
    for snap in snapshots:
        by_endpoint[snap.endpoint.key].append(snap)
    out: dict[str, list[RateSample]] = {}
    for key, snaps in by_endpoint.items():
        snaps.sort(key=lambda s: s.timestamp)
        series = [r for a, b in pairwise(snaps) if (r := to_rate(a, b)) is not None]
        out[key] = series
    return out
