"""B4: per-host alignment must survive a wall-clock step (NTP/chrony makestep) mid-job."""

from __future__ import annotations

from dataclasses import replace

from ai_nic_perf_profiler.analysis import analyze
from ai_nic_perf_profiler.analysis.align import rebase_on_monotonic
from ai_nic_perf_profiler.analysis.attribution import AttributionReport
from ai_nic_perf_profiler.counters import to_rate
from ai_nic_perf_profiler.io import read_snapshots, read_steps, write_snapshots, write_steps
from ai_nic_perf_profiler.model import CounterSnapshot
from ai_nic_perf_profiler.sim import SimConfig, SimResult, simulate
from ai_nic_perf_profiler.training.events import StepEvent

FAST = SimConfig(steps=140, fault_start=70, fault_end=120, checkpoint_every=50, seed=3)
JUMP_S = 2.0


def _with_mono(result: SimResult) -> tuple[list[CounterSnapshot], list[StepEvent]]:
    """Stamp every record with a per-host monotonic clock that agrees with wall time."""
    boot = {h: 1_000.0 + 37.0 * i for i, h in enumerate(result.topology.hosts())}

    def mono(host: str, t: float) -> int:
        return round((t - min(s.timestamp for s in result.snapshots) + boot[host]) * 1e9)

    snaps = [replace(s, mono_ns=mono(s.host, s.timestamp)) for s in result.snapshots]
    steps = [replace(e, mono_start_ns=mono(e.host, e.start), mono_end_ns=mono(e.host, e.end)) for e in result.steps]
    return snaps, steps


def _step_wall_clock(snaps: list[CounterSnapshot], steps: list[StepEvent], host: str, at: float) -> tuple[list[CounterSnapshot], list[StepEvent]]:
    """Jump ``host``'s wall clock forward by JUMP_S at time ``at``; monotonic stamps are untouched."""

    def bump(t: float) -> float:
        return t + JUMP_S if t >= at else t

    snaps = [replace(s, timestamp=bump(s.timestamp)) if s.host == host else s for s in snaps]
    steps = [replace(e, start=bump(e.start), end=bump(e.end)) if e.host == host else e for e in steps]
    return snaps, steps


def _summary(report: AttributionReport) -> list[tuple[int, int, str, list[int], float]]:
    out = []
    for ep in report.episodes:
        p = ep.primary
        assert p is not None
        out.append((ep.start_step, ep.end_step, p.cause.value, sorted(p.culprit_ranks), round(ep.explained_s / ep.excess_s, 2)))
    return out


def _jump_point(result: SimResult, host: str) -> float:
    # Mid-step on the host, halfway through the job (before the fault window).
    ev = next(e for e in result.steps if e.host == host and e.step == 40)
    return (ev.start + ev.end) / 2


def test_wall_clock_step_does_not_change_attribution_with_monotonic() -> None:
    result = simulate("pfc_storm", FAST)
    snaps, steps = _with_mono(result)
    clean = analyze(snaps, steps, result.topology)

    host = result.topology.hosts()[0]
    jumped_snaps, jumped_steps = _step_wall_clock(snaps, steps, host, _jump_point(result, host))
    jumped = analyze(jumped_snaps, jumped_steps, result.topology)

    assert _summary(jumped) == _summary(clean)
    assert jumped.data_quality.hosts_on_monotonic == len(result.topology.hosts())
    assert jumped.step_series["step_time_s"] == clean.step_series["step_time_s"]


def test_wall_clock_step_corrupts_wall_only_alignment() -> None:
    # The failure mode B4 describes, kept as a regression guard: without
    # monotonic stamps the straddling step absorbs the jump.
    result = simulate("pfc_storm", FAST)
    host = result.topology.hosts()[0]
    clean = analyze(result.snapshots, result.steps, result.topology)
    snaps, steps = _step_wall_clock(result.snapshots, result.steps, host, _jump_point(result, host))
    jumped = analyze(snaps, steps, result.topology)
    assert jumped.data_quality.hosts_on_monotonic == 0
    assert max(jumped.step_series["step_time_s"]) > max(clean.step_series["step_time_s"]) + JUMP_S / 2


def test_mixed_old_and_new_records_fall_back_to_wall_time() -> None:
    result = simulate("healthy", FAST)
    snaps, steps = _with_mono(result)
    steps[0] = replace(steps[0], mono_start_ns=None, mono_end_ns=None)
    _, _, hosts = rebase_on_monotonic(snaps, steps, result.topology)
    assert steps[0].host not in hosts
    assert len(hosts) == len(result.topology.hosts()) - 1


def test_rate_interval_uses_monotonic_when_present() -> None:
    a = CounterSnapshot("h", "mlx5_0", 1, 100.0, {"tx_bytes": 0}, mono_ns=5_000_000_000)
    b = CounterSnapshot("h", "mlx5_0", 1, 103.0, {"tx_bytes": 1000}, mono_ns=6_000_000_000)  # wall jumped +2 s
    r = to_rate(a, b)
    assert r is not None and r.rates == {"tx_bytes": 1000.0}


def test_old_files_without_monotonic_still_load(tmp_path) -> None:
    (tmp_path / "c.jsonl").write_text('{"host":"h","device":"d","port":1,"ts":1.5,"c":{"tx_bytes":1}}\n')
    (tmp_path / "s.jsonl").write_text('{"rank":0,"step":0,"start":1.0,"end":2.0}\n')
    [snap] = read_snapshots(tmp_path / "c.jsonl")
    [ev] = read_steps([tmp_path / "s.jsonl"])
    assert snap.mono_ns is None and ev.mono_start_ns is None and ev.duration == 1.0


def test_monotonic_round_trips_through_jsonl(tmp_path) -> None:
    snap = CounterSnapshot("h", "d", 1, 1.5, {"tx_bytes": 1}, mono_ns=123)
    ev = StepEvent(0, 0, 1.0, 2.0, "h", 0, 10, 2_000_000_010)
    write_snapshots(tmp_path / "c.jsonl", [snap])
    write_steps(tmp_path / "s.jsonl", [ev])
    assert read_snapshots(tmp_path / "c.jsonl") == [snap]
    [back] = read_steps([tmp_path / "s.jsonl"])
    assert back == ev and back.duration == 2.0
