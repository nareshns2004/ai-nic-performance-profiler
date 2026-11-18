"""B2: adversarial simulator mode adds nuisances the diagnosis rules do not encode."""

from __future__ import annotations

import pytest

from ai_nic_perf_profiler.sim import SimConfig, simulate
from ai_nic_perf_profiler.sim.cluster import NUISANCES

FAST = SimConfig(steps=140, fault_start=70, fault_end=120, checkpoint_every=50, seed=3)


def test_default_runs_are_unchanged_by_the_adversary_stream() -> None:
    # The adversary uses its own RNG, so turning it off reproduces the published numbers exactly.
    a = simulate("pfc_storm", FAST)
    b = simulate("pfc_storm", SimConfig(**{**FAST.__dict__, "adversarial": False, "nuisances": ()}))
    assert [s.counters for s in a.snapshots] == [s.counters for s in b.snapshots]


def test_adversarial_changes_counters_but_not_step_times() -> None:
    clean = simulate("lossy_retransmit", FAST)
    adv = simulate("lossy_retransmit", SimConfig(**{**FAST.__dict__, "adversarial": True}))
    assert [e.duration for e in adv.steps] == [e.duration for e in clean.steps]  # nuisances are non-causal
    assert [s.counters for s in adv.snapshots] != [s.counters for s in clean.snapshots]
    assert adv.ground_truth["adversarial"] is True
    assert adv.ground_truth["root_cause"] == clean.ground_truth["root_cause"]


@pytest.mark.parametrize("nuisance", NUISANCES)
def test_each_nuisance_can_be_enabled_alone(nuisance: str) -> None:
    clean = simulate("healthy", FAST)
    adv = simulate("healthy", SimConfig(**{**FAST.__dict__, "adversarial": True, "nuisances": (nuisance,)}))
    assert [s.counters for s in adv.snapshots] != [s.counters for s in clean.snapshots]


def test_unknown_nuisance_is_rejected() -> None:
    with pytest.raises(ValueError):
        simulate("healthy", SimConfig(**{**FAST.__dict__, "adversarial": True, "nuisances": ("bogus",)}))
