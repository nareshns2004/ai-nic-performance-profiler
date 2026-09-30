"""Framework-agnostic step recorder to drop into a training loop.

Usage with PyTorch::

    rec = StepRecorder("steps.rank3.jsonl", rank=3, sync=torch.cuda.synchronize)
    for batch in loader:
        with rec.step():
            loss = model(batch).sum()
            loss.backward()
            opt.step()

CUDA work is asynchronous, so without ``sync`` the host-side step boundary
drifts from the GPU-side one and the step window no longer lines up with the
NIC traffic it caused. Synchronising once per step costs little relative to a
multi-hundred-millisecond step; for tighter loops, pass a function that
records CUDA events instead.
"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from .events import StepEvent


class StepRecorder:
    def __init__(
        self,
        path: str | Path,
        rank: int,
        host: str | None = None,
        sync: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.time,
        flush_every: int = 50,
    ) -> None:
        self.rank = rank
        self.host = host or socket.gethostname()
        self.sync = sync
        self.clock = clock
        self.flush_every = flush_every
        self._step = 0
        # Long-lived handle for the life of the training job; closed via close()/__exit__.
        self._fh: IO[str] = Path(path).open("a", encoding="utf-8")  # noqa: SIM115

    @contextmanager
    def step(self, comm_bytes: int = 0) -> Iterator[None]:
        if self.sync:
            self.sync()
        start = self.clock()
        yield
        if self.sync:
            self.sync()
        end = self.clock()
        event = StepEvent(self.rank, self._step, start, end, self.host, comm_bytes)
        self._fh.write(json.dumps(event.to_dict()) + "\n")
        self._step += 1
        if self._step % self.flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> StepRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
