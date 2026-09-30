import math

import numpy as np
import pytest

from ai_nic_perf_profiler.analysis.align import RateIndex, build_step_table
from ai_nic_perf_profiler.analysis.baseline import choose_baseline, cusum, degraded_mask, find_episodes, robust_center_scale
from ai_nic_perf_profiler.analysis.causes import RootCause
from ai_nic_perf_profiler.analysis.diagnosis import allocate, score_causes
from ai_nic_perf_profiler.analysis.features import derive_features
from ai_nic_perf_profiler.analysis.regression import nnls_ridge
from ai_nic_perf_profiler.model import RateSample
from ai_nic_perf_profiler.topology import RankBinding, Topology
from ai_nic_perf_profiler.training.events import StepEvent


def rs(t0: float, t1: float, **rates: float) -> RateSample:
    return RateSample("h", "d", 1, t0, t1, rates, {"link_speed_bps": 100.0})


def test_window_is_overlap_weighted() -> None:
    idx = RateIndex([rs(0, 1, x=10.0), rs(1, 2, x=20.0), rs(2, 3, x=30.0)])
    rates, gauges, cov = idx.window(0.5, 2.0)  # 0.5 s of 10, 1 s of 20
    assert rates["x"] == pytest.approx((0.5 * 10 + 1.0 * 20) / 1.5)
    assert cov == pytest.approx(1.0)
    assert gauges["link_speed_bps"] == 100.0


def test_window_reports_partial_coverage() -> None:
    idx = RateIndex([rs(0, 1, x=1.0)])
    _, _, cov = idx.window(0.5, 1.5)
    assert cov == pytest.approx(0.5)
    assert idx.window(5, 6) == ({}, {}, 0.0)


def test_window_ignores_missing_counter_in_some_samples() -> None:
    idx = RateIndex([rs(0, 1, x=10.0), rs(1, 2)])
    rates, _, _ = idx.window(0, 2)
    assert rates["x"] == 10.0


def test_step_table_infers_missing_ranks_with_clock_offsets() -> None:
    topo = Topology(
        [RankBinding(0, "a", "mlx5_0"), RankBinding(1, "b", "mlx5_0")],
        clock_offsets={"a": 0.0, "b": -0.5},  # b's clock runs 0.5 s ahead
    )
    table = build_step_table([StepEvent(0, 0, 10.0, 11.0, "a")], topo)
    assert table.start[0, 1] == pytest.approx(10.5)
    assert table.inferred[0, 1] and not table.inferred[0, 0]
    assert table.step_time()[0] == pytest.approx(1.0)


def test_robust_scale_ignores_outliers() -> None:
    x = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 50.0])
    med, sigma = robust_center_scale(x)
    assert med == pytest.approx(1.025)
    assert sigma < 0.2


def test_baseline_and_episode_detection() -> None:
    steps = np.arange(100)
    y = np.full(100, 1.0) + 0.001 * np.sin(steps)
    y[60:80] = 1.2
    y[30] = 1.5  # single spike: a checkpoint, not an episode
    base = choose_baseline(steps, warmup=5, fraction=0.25)
    assert base[5] and not base[4] and base.sum() == 23
    eps = find_episodes(degraded_mask(y, base), merge_gap=2, min_length=3)
    assert [(e.start_idx, e.end_idx) for e in eps] == [(60, 80)]


def test_explicit_baseline() -> None:
    mask = choose_baseline(np.arange(10), 0, 0.5, explicit=(2, 5))
    assert mask.tolist() == [False, False, True, True, True] + [False] * 5


def test_cusum_finds_small_sustained_shift() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.0, 1, 200)])
    alarms = np.array(cusum(x, target=0.0, sigma=1.0))
    # In-control average run length at k=0.5, h=5 is ~465 samples, so an
    # occasional false alarm is expected; after the shift, alarms are dense.
    assert (alarms < 200).sum() <= 1
    assert (alarms >= 200).sum() >= 5


def test_nnls_ridge_recovers_nonnegative_weights() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (200, 3))
    y = X @ np.array([2.0, 0.0, 0.5])
    w = nnls_ridge(X, y, lam=1e-6)
    assert w == pytest.approx([2.0, 0.0, 0.5], abs=1e-3)
    # A truly negative relation is clamped to zero, never negative.
    w2 = nnls_ridge(X, -X[:, 0], lam=1e-6)
    assert (w2 >= 0).all() and w2[0] == 0


def test_features_are_normalised() -> None:
    f = derive_features(
        {"tx_bytes": 25e9, "rx_bytes": 10e9, "rx_packets": 1e6, "tx_packets": 2e6, "ecn_marked_packets": 1e3, "cnp_handled": 40, "rx_pause_duration_us": 2e5},
        {"link_speed_bps": 200e9, "vf_max_tx_rate_bps": 250e9},
        expected_link_bps=400e9,
    )
    assert f["util"] == pytest.approx(0.5)  # vs expected speed, not the downshifted one
    assert f["link_speed_deficit"] == pytest.approx(0.5)
    assert f["ecn_mark_ratio"] == pytest.approx(1e-3)
    assert f["cnp_per_kpkt"] == pytest.approx(0.02)
    assert f["tx_paused_frac"] == pytest.approx(0.2)
    assert f["vf_cap_util"] == pytest.approx(0.8)
    assert math.isnan(f["drop_rate"])  # absent counters stay NaN, never 0


def test_pause_origin_outranks_pause_victim() -> None:
    scores = score_causes({"tx_paused_frac": 30.0, "pause_sent_rate": 60.0})
    assert scores[RootCause.HOST_RX_BACKPRESSURE] > scores[RootCause.PFC_HOL_BLOCKING]
    alone = score_causes({"tx_paused_frac": 30.0})
    assert alone[RootCause.PFC_HOL_BLOCKING] > 0.9


def test_physical_errors_explain_retransmissions() -> None:
    scores = score_causes({"retrans_rate": 40.0, "phy_err_rate": 40.0})
    assert scores[RootCause.LINK_DEGRADATION] > scores[RootCause.LOSSY_RETRANSMISSION]


def test_allocate_splits_by_evidence_and_orphans_weak_features() -> None:
    seconds, orphaned = allocate({"tx_paused_frac": 1.0, "ecn_mark_ratio": 0.5}, {RootCause.HOST_RX_BACKPRESSURE: 0.9, RootCause.PFC_HOL_BLOCKING: 0.1})
    assert seconds[RootCause.HOST_RX_BACKPRESSURE] == pytest.approx(0.9)
    assert seconds[RootCause.PFC_HOL_BLOCKING] == pytest.approx(0.1)
    assert orphaned == pytest.approx(0.5)
