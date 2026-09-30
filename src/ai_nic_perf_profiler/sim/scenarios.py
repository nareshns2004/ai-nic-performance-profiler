"""Fault scenarios with ground truth for evaluating attribution."""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis.causes import RootCause


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    root_cause: RootCause | None
    description: str


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario("healthy", None, "No fault. Checkpoint spikes, a counter reset and clock skew only"),
        Scenario("pfc_storm", RootCause.PFC_HOL_BLOCKING, "Switch keeps one rank's transmitter paused ~40% of comm time; ECN barely reacts"),
        Scenario("fabric_congestion", RootCause.FABRIC_CONGESTION, "Uplinks of one leaf congested: ECN marking and CNP throttling on every rank under it"),
        Scenario(
            "host_rx_backpressure",
            RootCause.HOST_RX_BACKPRESSURE,
            "One host can't drain its NIC (RQ starvation); it sends PFC pauses that stall its ring predecessor",
        ),
        Scenario("lossy_retransmit", RootCause.LOSSY_RETRANSMISSION, "Drops on one path trigger sequence-error NAKs and ACK timeouts"),
        Scenario("link_degradation", RootCause.LINK_DEGRADATION, "Dirty optic: symbol/CRC errors, FEC corrections up 30x, link downshifts 400G -> 200G"),
        Scenario("sriov_rate_cap", RootCause.SRIOV_RATE_CAP, "Partition resized mid-run: one host's VFs capped at 45% of line rate"),
        Scenario("rail_imbalance", RootCause.RAIL_IMBALANCE, "A rank's traffic lands on a neighbour's NIC (wrong NCCL_IB_HCA mapping)"),
        Scenario("compute_straggler", RootCause.NON_NETWORK, "One GPU runs 30% slower (thermal throttling); the network is healthy"),
    ]
}
