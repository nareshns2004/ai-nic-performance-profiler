"""B7: StepRecorder CUDA-event mode, tested with a fake event backend (no torch needed)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_nic_perf_profiler.io import read_steps
from ai_nic_perf_profiler.training import StepRecorder


@dataclass
class FakeEvent:
    gpu_t: float  # seconds on the fake GPU timeline
    done: bool = False


class FakeGpu:
    """A GPU that runs `lag` seconds behind the host and only completes work when told to."""

    def __init__(self) -> None:
        self.gpu_now = 50.0
        self.events: list[FakeEvent] = []
        self.syncs = 0

    def record(self) -> FakeEvent:
        ev = FakeEvent(self.gpu_now)
        self.events.append(ev)
        return ev

    def query(self, event: FakeEvent) -> bool:
        return event.done

    def wait(self, event: FakeEvent) -> None:
        # One stream: completing an event completes everything recorded before it.
        for ev in self.events[: self.events.index(event) + 1]:
            ev.done = True

    def elapsed_ms(self, start: FakeEvent, end: FakeEvent) -> float:
        assert start.done and end.done, "elapsed_time on an incomplete event raises in CUDA"
        return (end.gpu_t - start.gpu_t) * 1e3

    def synchronize(self) -> None:
        self.syncs += 1
        for ev in self.events:
            ev.done = True

    def run(self, seconds: float) -> None:
        self.gpu_now += seconds

    def complete_all(self) -> None:
        for ev in self.events:
            ev.done = True


class HostClock:
    def __init__(self) -> None:
        self.wall, self.mono_ns = 1_700_000_000.0, 3_000_000_000

    def now_wall(self) -> float:
        return self.wall

    def now_mono(self) -> int:
        return self.mono_ns


def test_cuda_event_mode_resolves_lazily_without_per_step_sync(tmp_path) -> None:
    gpu, host = FakeGpu(), HostClock()
    rec = StepRecorder(tmp_path / "s.jsonl", rank=1, host="h", clock=host.now_wall, mono_clock=host.now_mono, events=gpu, flush_every=1)
    assert rec.timing == "cuda_events"
    for dur in (0.30, 0.45, 0.40):
        with rec.step():
            gpu.run(dur)
    # One calibration (two synchronizes) at start-up; nothing per step.
    assert gpu.syncs == 2
    assert rec.pending == 3  # GPU hasn't finished: nothing written yet
    gpu.complete_all()
    with rec.step():
        gpu.run(0.5)
    assert rec.pending == 1  # earlier steps drained once their end events completed
    rec.close()
    events = read_steps([tmp_path / "s.jsonl"])
    assert [e.step for e in events] == [0, 1, 2, 3]
    assert [round(e.duration, 6) for e in events] == [0.30, 0.45, 0.40, 0.50]
    # Windows are on the host clocks, anchored at the calibration point.
    assert events[0].mono_start_ns == host.mono_ns and events[0].start == host.wall
    assert events[1].mono_start_ns == events[0].mono_end_ns


def test_cuda_event_mode_recalibrates_periodically(tmp_path) -> None:
    gpu, host = FakeGpu(), HostClock()
    rec = StepRecorder(tmp_path / "s.jsonl", rank=0, host="h", clock=host.now_wall, mono_clock=host.now_mono, events=gpu, recalibrate_every=2)
    for _ in range(5):
        with rec.step():
            gpu.run(0.1)
    rec.close()
    assert gpu.syncs == 2 * 3  # steps 0, 2, 4
    assert len(read_steps([tmp_path / "s.jsonl"])) == 5


def test_host_mode_without_torch_or_sync(tmp_path, monkeypatch) -> None:
    import ai_nic_perf_profiler.training.recorder as recorder

    monkeypatch.setattr(recorder, "_cuda_available", lambda: False)
    rec = StepRecorder(tmp_path / "s.jsonl", rank=0, host="h")
    assert rec.timing == "host"
    with rec, rec.step():
        pass
    [ev] = read_steps([tmp_path / "s.jsonl"])
    assert ev.mono_start_ns is not None and ev.duration >= 0


def test_sync_timing_requires_callable(tmp_path) -> None:
    with pytest.raises(ValueError):
        StepRecorder(tmp_path / "s.jsonl", rank=0, timing="sync")
