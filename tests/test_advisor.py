import pytest

from ai_nic_perf_profiler.advisor.fabric import FabricModel, default_candidates, ring_rate, what_if
from ai_nic_perf_profiler.advisor.sriov import plan_partitions
from ai_nic_perf_profiler.sim.tenants import simulate_vf_tenants
from ai_nic_perf_profiler.training.collectives import busbw, effective_bandwidth, ring_allreduce_seconds


def test_busbw_matches_nccl_tests_convention() -> None:
    # 8 ranks, 1 GB in 10 ms: algbw 100 GB/s, busbw = 100 * 2*7/8
    assert busbw("allreduce", 1e9, 0.01, 8) == pytest.approx(175e9)
    assert busbw("allgather", 1e9, 0.01, 8) == pytest.approx(87.5e9)


def test_ring_allreduce_time_round_trips() -> None:
    t = ring_allreduce_seconds(2e9, 16, 40e9, alpha_s=10e-6)
    assert effective_bandwidth(2e9, 16, t, alpha_s=10e-6) == pytest.approx(40e9)
    assert ring_allreduce_seconds(1e9, 1, 1e9) == 0.0


def test_synchronised_bursts_have_no_multiplexing_gain() -> None:
    plan = plan_partitions(simulate_vf_tenants(synchronized=True, duration_s=20), 400e9, "ens1f0/vf")
    assert plan.synchrony_index > 0.9
    assert plan.policy.startswith("oversubscribed")
    assert plan.max_vfs_per_pf < 4


def test_independent_bursts_can_share_a_pf() -> None:
    plan = plan_partitions(simulate_vf_tenants(synchronized=False, duration_s=20), 400e9, "ens1f0/vf")
    assert plan.synchrony_index < 0.7
    assert plan.policy.startswith("statistical multiplexing")
    assert plan.recommended_max_tx_bps is None


def test_capped_vfs_are_flagged_as_censored() -> None:
    plan = plan_partitions(simulate_vf_tenants(cap_gbps=90, duration_s=20), 400e9, "ens1f0/vf")
    assert all(p.censored for p in plan.profiles)
    assert "lower bounds" in plan.notes[0]


def test_plan_requires_matching_endpoints() -> None:
    with pytest.raises(ValueError):
        plan_partitions(simulate_vf_tenants(duration_s=5), 400e9, "nope")


def test_packed_placement_avoids_uplinks() -> None:
    packed = FabricModel(hosts=16, hosts_per_leaf=4, uplinks_per_leaf=16, placement="packed")
    scattered = FabricModel(hosts=16, hosts_per_leaf=4, uplinks_per_leaf=16, placement="scattered")
    assert ring_rate(packed, trials=200) > ring_rate(scattered, trials=200)


def test_more_qps_reduce_ecmp_collisions() -> None:
    one = FabricModel(hosts=16, placement="scattered", qps_per_connection=1)
    four = FabricModel(hosts=16, placement="scattered", qps_per_connection=4)
    assert ring_rate(four, trials=200) > ring_rate(one, trials=200)


def test_single_leaf_job_runs_at_line_rate() -> None:
    assert ring_rate(FabricModel(hosts=4, hosts_per_leaf=4), trials=10) == 1.0


def test_what_if_is_calibrated_to_measurement() -> None:
    current = FabricModel(hosts=32, hosts_per_leaf=4, uplinks_per_leaf=16)
    preds = what_if(current, default_candidates(current), 4e9, measured_step_s=0.6, compute_s=0.4, trials=100)
    assert preds[0].step_s == 0.6 and preds[0].speedup_pct == 0.0
    assert all(p.step_s >= 0.4 for p in preds)
    assert max(p.speedup_pct for p in preds[1:]) > 0
