"""End-to-end: simulate each fault, analyse, check cause and culprit against ground truth."""

import pytest

from ai_nic_perf_profiler.analysis import analyze
from ai_nic_perf_profiler.analysis.causes import RootCause
from ai_nic_perf_profiler.io import read_snapshots, read_steps
from ai_nic_perf_profiler.sim import SCENARIOS, SimConfig, simulate
from ai_nic_perf_profiler.topology import Topology

FAST = SimConfig(steps=140, fault_start=70, fault_end=120, checkpoint_every=50, seed=3)
FAULTS = [s for s in SCENARIOS if s != "healthy"]


@pytest.mark.parametrize("scenario", FAULTS)
def test_attributes_injected_fault(scenario: str) -> None:
    result = simulate(scenario, FAST)
    report = analyze(result.snapshots, result.steps, result.topology)
    truth = result.ground_truth
    lo, hi = truth["fault_steps"]
    [episode] = [e for e in report.episodes if e.start_step < hi and e.end_step >= lo]
    assert episode.primary is not None
    assert episode.primary.cause.value == truth["root_cause"]
    if truth["culprit_ranks"]:
        assert set(episode.primary.culprit_ranks) == set(truth["culprit_ranks"])
        assert episode.explained_s / episode.excess_s > 0.8
    else:
        assert episode.primary.cause is RootCause.NON_NETWORK


def test_healthy_run_has_no_episodes_despite_spikes_reset_and_skew() -> None:
    result = simulate("healthy", FAST)
    report = analyze(result.snapshots, result.steps, result.topology)
    assert report.episodes == []
    assert report.data_quality.counter_flags.get("reset", 0) > 0  # the injected reset was handled


def test_backpressure_names_victims_separately() -> None:
    result = simulate("host_rx_backpressure", FAST)
    report = analyze(result.snapshots, result.steps, result.topology)
    d = report.episodes[0].primary
    assert d is not None
    victim = (result.ground_truth["culprit_ranks"][0] - 1) % len(result.topology.ranks)
    assert d.victim_ranks == [victim]


def test_rank0_only_step_logging_uses_clock_offsets() -> None:
    result = simulate("pfc_storm", SimConfig(**{**FAST.__dict__, "steps_rank0_only": True}))
    report = analyze(result.snapshots, result.steps, result.topology)
    assert report.data_quality.windows_inferred > 0
    assert report.episodes[0].primary.culprit_ranks == result.ground_truth["culprit_ranks"]


def test_round_trip_through_files(tmp_path) -> None:
    result = simulate("lossy_retransmit", FAST)
    paths = result.write(tmp_path)
    report = analyze(read_snapshots(paths["counters"]), read_steps([paths["steps"]]), Topology.load(paths["topology"]))
    assert report.episodes[0].primary.cause is RootCause.LOSSY_RETRANSMISSION
    d = report.to_dict()
    assert d["episodes"][0]["diagnoses"][0]["recommendations"]


def test_missing_endpoint_is_reported_not_fatal() -> None:
    result = simulate("pfc_storm", FAST)
    culprit = result.ground_truth["culprit_ranks"][0]
    drop = next(r for r in range(16) if r != culprit)
    key = result.topology.binding(drop).endpoint.key
    snaps = [s for s in result.snapshots if s.endpoint.key != key]
    report = analyze(snaps, result.steps, result.topology)
    assert report.data_quality.missing_endpoints == [key]
    assert report.episodes[0].primary.cause is RootCause.PFC_HOL_BLOCKING
