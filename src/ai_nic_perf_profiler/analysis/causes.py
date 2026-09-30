"""Root-cause taxonomy.

Every cause maps to the decision it informs: fabric topology, RDMA tuning,
SR-IOV partition sizing, physical-layer maintenance, or "not the network".
The last one matters as much as the others: exonerating the fabric quickly
sends the investigation to the right team.
"""

from __future__ import annotations

from enum import Enum


class RootCause(str, Enum):
    FABRIC_CONGESTION = "fabric_congestion"
    PFC_HOL_BLOCKING = "pfc_hol_blocking"
    HOST_RX_BACKPRESSURE = "host_rx_backpressure"
    LOSSY_RETRANSMISSION = "lossy_retransmission"
    LINK_DEGRADATION = "link_degradation"
    SRIOV_RATE_CAP = "sriov_rate_cap"
    RAIL_IMBALANCE = "rail_imbalance"
    NON_NETWORK = "non_network"


DECISION_DOMAIN = {
    RootCause.FABRIC_CONGESTION: "fabric topology",
    RootCause.PFC_HOL_BLOCKING: "fabric QoS configuration",
    RootCause.HOST_RX_BACKPRESSURE: "RDMA / host tuning",
    RootCause.LOSSY_RETRANSMISSION: "RDMA transport tuning",
    RootCause.LINK_DEGRADATION: "physical layer maintenance",
    RootCause.SRIOV_RATE_CAP: "SR-IOV partition sizing",
    RootCause.RAIL_IMBALANCE: "fabric topology / NIC mapping",
    RootCause.NON_NETWORK: "outside the network (compute, input pipeline, stragglers)",
}

SUMMARY = {
    RootCause.FABRIC_CONGESTION: "ECN marking and CNP rate throttling indicate in-fabric congestion (oversubscription or ECMP collisions)",
    RootCause.PFC_HOL_BLOCKING: "Transmitters are held by PFC pause frames without matching ECN reaction: pause storm / head-of-line blocking",
    RootCause.HOST_RX_BACKPRESSURE: "A receiver cannot drain its NIC (RQ starvation, PCIe or memory pressure) and pushes pause frames upstream",
    RootCause.LOSSY_RETRANSMISSION: "Packet loss is triggering RDMA transport retransmission and ACK timeouts",
    RootCause.LINK_DEGRADATION: "Physical-layer errors, link retrains or a speed downshift on the port",
    RootCause.SRIOV_RATE_CAP: "The VF is pinned at its transmit rate cap while the job needs more bandwidth",
    RootCause.RAIL_IMBALANCE: "Traffic is concentrated on a subset of rails on a host (NIC-to-GPU mapping mismatch)",
    RootCause.NON_NETWORK: "Step time regressed without a matching network signal",
}

TITLE = {
    RootCause.FABRIC_CONGESTION: "Fabric congestion",
    RootCause.PFC_HOL_BLOCKING: "PFC head-of-line blocking",
    RootCause.HOST_RX_BACKPRESSURE: "Host RX backpressure",
    RootCause.LOSSY_RETRANSMISSION: "Lossy fabric / RDMA retransmission",
    RootCause.LINK_DEGRADATION: "Link degradation",
    RootCause.SRIOV_RATE_CAP: "SR-IOV rate cap",
    RootCause.RAIL_IMBALANCE: "Rail imbalance",
    RootCause.NON_NETWORK: "Not the network",
}
