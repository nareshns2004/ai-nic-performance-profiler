"""Fault scenarios with ground truth for evaluating attribution."""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis.causes import RootCause


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    root_cause: RootCause | None
    description: str
    # Composite scenarios: two single-fault scenarios injected at once on
    # disjoint hosts. The first is the slower bottleneck (the primary cause);
    # the second is masked by it and should surface as a secondary suspect.
    components: tuple[str, ...] = ()
    secondary_cause: RootCause | None = None

    @property
    def is_composite(self) -> bool:
        return bool(self.components)


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
        Scenario(
            "multi_link_backpressure",
            RootCause.LINK_DEGRADATION,
            "Two faults at once: a degraded link (bottleneck) and host RX backpressure on another host (masked)",
            components=("link_degradation", "host_rx_backpressure"),
            secondary_cause=RootCause.HOST_RX_BACKPRESSURE,
        ),
        Scenario(
            "multi_lossy_congestion",
            RootCause.LOSSY_RETRANSMISSION,
            "Two faults at once: a lossy path (bottleneck) and congested uplinks on the other leaf (masked)",
            components=("lossy_retransmit", "fabric_congestion"),
            secondary_cause=RootCause.FABRIC_CONGESTION,
        ),
    ]
}

SINGLE_FAULTS: tuple[str, ...] = tuple(n for n, s in SCENARIOS.items() if s.root_cause is not None and not s.is_composite)
MULTI_FAULTS: tuple[str, ...] = tuple(n for n, s in SCENARIOS.items() if s.is_composite)
