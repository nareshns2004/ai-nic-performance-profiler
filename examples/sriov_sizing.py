"""Why "sum of peaks" and "sum of means" are both wrong for SR-IOV sizing.

Four VFs, each bursting to 150 Gb/s for 25% of the time, share a 400G PF.
Whether they fit depends on whether their bursts line up.

    PYTHONPATH=src python examples/sriov_sizing.py
"""

from ai_nic_perf_profiler.advisor.sriov import plan_partitions
from ai_nic_perf_profiler.sim.tenants import simulate_vf_tenants

for label, synchronized in (("ranks of ONE data-parallel job (bursts aligned)", True), ("four independent jobs (random phase)", False)):
    plan = plan_partitions(simulate_vf_tenants(synchronized=synchronized, duration_s=30), 400e9, "ens1f0/vf")
    print(f"\n{label}")
    print(f"  sum of per-VF p99: {plan.sum_p99_bps / 1e9:6.0f} Gb/s")
    print(f"  p99 of aggregate:  {plan.p99_of_sum_bps / 1e9:6.0f} Gb/s   synchrony index {plan.synchrony_index:.2f}")
    print(f"  policy: {plan.policy}; max VFs/PF at this profile: {plan.max_vfs_per_pf}")
    for note in plan.notes:
        print(f"  - {note}")
