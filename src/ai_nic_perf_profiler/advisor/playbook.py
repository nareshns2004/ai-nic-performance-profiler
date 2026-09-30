"""Remediation playbook: what to change, and how to check that it worked.

Each entry names a concrete knob. Knob names are real (mlx5, NCCL, UCX,
iproute2), but defaults and availability vary by driver, firmware and library
version, so check them against your stack before changing anything in production.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis.causes import RootCause as RC


@dataclass(frozen=True, slots=True)
class Recommendation:
    action: str
    verify: str


PLAYBOOK: dict[RC, tuple[Recommendation, ...]] = {
    RC.FABRIC_CONGESTION: (
        Recommendation(
            "Increase flow entropy for ECMP: NCCL_IB_QPS_PER_CONNECTION=4 spreads each peer connection over more 5-tuples",
            "Run `nicprof advise fabric` to estimate the collision penalty; re-measure cross-leaf all_reduce_perf busbw",
        ),
        Recommendation(
            "Enable adaptive routing on the fabric if the switches support it, or use topology-aware rank placement so ring neighbours share a leaf",
            "ECN mark ratio during collectives returns to baseline",
        ),
        Recommendation(
            "If congestion is persistent at the uplinks, reduce leaf oversubscription for this job class (fabric topology decision)",
            "Compare the what-if step time from `nicprof advise fabric` with the cost of the change",
        ),
    ),
    RC.PFC_HOL_BLOCKING: (
        Recommendation(
            "Set switch ECN marking thresholds below PFC XOFF on the RDMA priority so DCQCN reacts before PFC fires",
            "tx_paused_frac < 1% during collectives while ECN marking stays active",
        ),
        Recommendation(
            "Enable PFC storm protection: switch PFC watchdog, and on mlx5 `ethtool --set-tunable <if> pfc-prevention-tout <ms>`",
            "No sustained pause storms in the switch PFC watchdog logs",
        ),
        Recommendation(
            "Inspect the switch egress queue toward the paused host for a stuck or slow receiver further downstream",
            "Pause source identified; the port is no longer paused after the fix",
        ),
    ),
    RC.HOST_RX_BACKPRESSURE: (
        Recommendation(
            "Check the NIC's PCIe link: `lspci -vv -s <bdf> | grep -E 'LnkCap|LnkSta'`. A x16 link trained at x8, or at a lower Gen, halves the drain rate",
            "LnkSta matches LnkCap; pause_sent_rate returns to baseline",
        ),
        Recommendation(
            "Verify that GPUDirect RDMA is active (nvidia-peermem loaded, NCCL_NET_GDR_LEVEL) and that NIC and GPU share a PCIe switch (`nvidia-smi topo -m`)",
            "NCCL_DEBUG=INFO shows GDRDMA enabled on every rank",
        ),
        Recommendation(
            "If out_of_buffer is rising, receive WQEs are being exhausted: raise the posted receive depth (for UCX, UCX_RC_RX_QUEUE_LEN) or reduce outstanding requests per QP",
            "out_of_buffer rate returns to zero",
        ),
    ),
    RC.LOSSY_RETRANSMISSION: (
        Recommendation(
            "Confirm that RDMA traffic lands on the lossless priority: DSCP trust and priority mapping on the NIC (`mlnx_qos -i <if>`) and on the switches",
            "No drops on the RDMA queue in switch counters; packet_seq_err back to baseline",
        ),
        Recommendation(
            "Find the dropping hop with switch per-queue drop counters along the leaf-spine path",
            "Drop source identified",
        ),
        Recommendation(
            "Mitigation only: tune NCCL_IB_TIMEOUT / NCCL_IB_RETRY_CNT. Longer timeouts hide loss but lengthen each stall",
            "ack_timeout_rate is zero",
        ),
    ),
    RC.LINK_DEGRADATION: (
        Recommendation(
            "Drain the node from the scheduler, then reseat or replace the cable/transceiver; read BER and eye margins with `mlxlink -d <dev> -m -c`",
            "phy_err_rate zero and pre-FEC BER within spec after the swap",
        ),
        Recommendation(
            "If the link downshifted, compare `ports/<p>/rate` with the expected speed and check auto-negotiation on both ends",
            "link_speed_deficit is 0",
        ),
        Recommendation(
            "Alert on the trend of corrected FEC bits; a rising pre-FEC BER comes before uncorrectable errors and flaps",
            "Alert in place on fec_corrected_rate slope",
        ),
    ),
    RC.SRIOV_RATE_CAP: (
        Recommendation(
            "Resize the partition: raise the VF cap (`ip link set <pf> vf <n> max_tx_rate <Mbps>`) or place fewer VFs of this profile per PF",
            "`nicprof advise sriov` shows time-at-cap < 5% for the VF",
        ),
        Recommendation(
            "Prefer min_tx_rate guarantees over hard caps when tenants' bursts are not synchronised (burst synchrony index well below 1)",
            "Aggregate PF p99 demand stays below line rate",
        ),
    ),
    RC.RAIL_IMBALANCE: (
        Recommendation(
            "Check NCCL_IB_HCA and GPU-to-NIC affinity: each GPU should use its PCIe-local NIC (`nvidia-smi topo -m`, NCCL_DEBUG=INFO ring/graph dump)",
            "Per-rail tx bytes within 10% of each other during collectives",
        ),
        Recommendation(
            "Look for a NIC that is down or missing, which makes NCCL collapse onto fewer rails. On rail-optimized fabrics keep PXN enabled",
            "All rails carry traffic",
        ),
    ),
    RC.NON_NETWORK: (
        Recommendation(
            "Time per-rank compute (CUDA events around forward/backward) to find the straggler: GPU clock throttling (`nvidia-smi -q -d PERFORMANCE`), XID/ECC errors, data loader stalls",
            "Per-rank compute time back within baseline spread",
        ),
        Recommendation(
            "Network counters show no anomaly beyond baseline, so hand the investigation to the compute/input-pipeline owners",
            "n/a",
        ),
    ),
}
