"""Framework-agnostic step recorder to drop into a training loop.

Usage with PyTorch::

    rec = StepRecorder("steps.rank3.jsonl", rank=3)   # CUDA events when torch+CUDA are present
    for batch in loader:
        with rec.step():
            loss = model(batch).sum()
            loss.backward()
            opt.step()
    rec.close()

CUDA work is asynchronous, so a host-side ``time.time()`` around the step body
measures when the CPU *enqueued* the work, not when the GPU ran it, and the
step window drifts away from the NIC traffic it caused. There are three ways
to time a step (``timing=``):

``"cuda_events"`` (default when torch and a GPU are available)
    Record a CUDA event at each step edge and resolve it later, once the GPU
    has passed it. Event times are mapped onto the host's monotonic clock via
    a calibration point (one ``synchronize`` at start-up and every
    ``recalibrate_every`` steps). Steady-state cost is two ``cudaEventRecord``
    calls per step and no CPU/GPU serialisation, so the measurement does not
    change the step time it is measuring.

``"sync"`` (pass ``sync=torch.cuda.synchronize``)
    Synchronise at both step edges. Exact, but it drains the GPU queue twice
    per step, removing CPU/GPU overlap at the step boundary: the measured step
    can be slower than the unmeasured one. Kept for frameworks without events.

``"host"``
    Host clock only, no synchronisation. Fine for CPU-bound loops and tests.

Every event carries both wall-clock time (for cross-host display) and
``CLOCK_MONOTONIC`` nanoseconds (for per-host alignment that survives an NTP
step mid-job). See ``docs/methodology.md`` §1.
"""

from __future__ import annotations

import json
import socket
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, Protocol

from .events import StepEvent

Timing = Literal["auto", "cuda_events", "sync", "host"]


class EventBackend(Protocol):
    """The four operations the recorder needs from a GPU event API."""

    def record(self) -> Any: ...
    def query(self, event: Any) -> bool: ...
    def wait(self, event: Any) -> None: ...
    def elapsed_ms(self, start: Any, end: Any) -> float: ...
    def synchronize(self) -> None: ...


class TorchCudaEvents:
    """:class:`EventBackend` over ``torch.cuda.Event`` on the current stream."""

    def __init__(self) -> None:
        import torch  # imported lazily: torch is never a hard dependency

        self._torch = torch

    def record(self) -> Any:
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def query(self, event: Any) -> bool:
        return bool(event.query())

    def wait(self, event: Any) -> None:
        event.synchronize()

    def elapsed_ms(self, start: Any, end: Any) -> float:
        return float(start.elapsed_time(end))

    def synchronize(self) -> None:
        self._torch.cuda.synchronize()


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass
class _Pending:
    step: int
    comm_bytes: int
    start_ev: Any
    end_ev: Any
    anchor: _Anchor


@dataclass(frozen=True)
class _Anchor:
    """A GPU event whose completion time is known on both host clocks."""

    event: Any
    mono_ns: int
    wall: float


class StepRecorder:
    def __init__(
        self,
        path: str | Path,
        rank: int,
        host: str | None = None,
        sync: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.time,
        flush_every: int = 50,
        timing: Timing = "auto",
        mono_clock: Callable[[], int] = time.monotonic_ns,
        events: EventBackend | None = None,
        recalibrate_every: int = 1000,
    ) -> None:
        self.rank = rank
        self.host = host or socket.gethostname()
        self.sync = sync
        self.clock = clock
        self.mono_clock = mono_clock
        self.flush_every = flush_every
        self.recalibrate_every = recalibrate_every
        self.timing: Timing = self._resolve_timing(timing, sync, events)
        self._events: EventBackend | None = None
        if self.timing == "cuda_events":
            self._events = events or TorchCudaEvents()
        self._anchor: _Anchor | None = None
        self._pending: deque[_Pending] = deque()
        self._step = 0
        self._written = 0
        # Long-lived handle for the life of the training job; closed via close()/__exit__.
        self._fh: IO[str] = Path(path).open("a", encoding="utf-8")  # noqa: SIM115

    @staticmethod
    def _resolve_timing(timing: Timing, sync: Callable[[], None] | None, events: EventBackend | None) -> Timing:
        if timing != "auto":
            if timing == "sync" and sync is None:
                raise ValueError("timing='sync' needs a sync= callable (e.g. torch.cuda.synchronize)")
            return timing
        if sync is not None:
            return "sync"  # explicit sync= keeps the 0.2 behaviour
        if events is not None or _cuda_available():
            return "cuda_events"
        return "host"

    def _now(self) -> tuple[float, int]:
        return self.clock(), self.mono_clock()

    @contextmanager
    def step(self, comm_bytes: int = 0) -> Iterator[None]:
        if self.timing == "cuda_events":
            assert self._events is not None
            if self._anchor is None or self._step % self.recalibrate_every == 0:
                self._calibrate()
            assert self._anchor is not None
            start_ev = self._events.record()
            yield
            end_ev = self._events.record()
            self._pending.append(_Pending(self._step, comm_bytes, start_ev, end_ev, self._anchor))
            self._step += 1
            self._drain(block=False)
            return
        if self.timing == "sync" and self.sync:
            self.sync()
        wall0, mono0 = self._now()
        yield
        if self.timing == "sync" and self.sync:
            self.sync()
        wall1, mono1 = self._now()
        self._write(StepEvent(self.rank, self._step, wall0, wall1, self.host, comm_bytes, mono0, mono1))
        self._step += 1

    def _calibrate(self) -> None:
        """Pin a GPU event to the host clocks with one synchronize.

        After ``synchronize`` returns, the anchor event has completed; reading
        the host clocks immediately afterwards bounds its time to within the
        wake-up latency (tens of microseconds). Re-anchoring periodically keeps
        GPU/host clock drift (ppm-level) from accumulating over long jobs.
        """
        assert self._events is not None
        self._events.synchronize()
        ev = self._events.record()
        self._events.synchronize()
        wall, mono = self._now()
        self._anchor = _Anchor(ev, mono, wall)

    def _resolve(self, p: _Pending) -> StepEvent:
        assert self._events is not None
        a = p.anchor
        start_s = self._events.elapsed_ms(a.event, p.start_ev) / 1e3
        end_s = self._events.elapsed_ms(a.event, p.end_ev) / 1e3
        mono0, mono1 = a.mono_ns + round(start_s * 1e9), a.mono_ns + round(end_s * 1e9)
        return StepEvent(self.rank, p.step, a.wall + start_s, a.wall + end_s, self.host, p.comm_bytes, mono0, mono1)

    def _drain(self, block: bool) -> None:
        assert self._events is not None
        while self._pending:
            head = self._pending[0]
            if not self._events.query(head.end_ev):
                if not block:
                    return
                self._events.wait(head.end_ev)
            self._write(self._resolve(self._pending.popleft()))

    def _write(self, event: StepEvent) -> None:
        self._fh.write(json.dumps(event.to_dict()) + "\n")
        self._written += 1
        if self._written % self.flush_every == 0:
            self._fh.flush()

    @property
    def pending(self) -> int:
        """Steps recorded on the GPU but not yet resolved and written."""
        return len(self._pending)

    def close(self) -> None:
        if self._fh.closed:
            return
        if self._events is not None:
            self._drain(block=True)
        self._fh.close()

    def __enter__(self) -> StepRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
