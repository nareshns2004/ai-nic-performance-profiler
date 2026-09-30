"""Counter catalog: one canonical vocabulary across RDMA sysfs, ethtool and netdev.

The same physical quantity has a different name (and sometimes a different unit
and width) depending on where it is read from. For example, bytes transmitted are

* ``port_xmit_data`` in ``/sys/class/infiniband/<dev>/ports/<p>/counters``,
  counted in 4-octet words (IBTA PortCounters semantics),
* ``tx_bytes_phy`` in ``ethtool -S`` on mlx5, counted in octets,
* ``tx_bytes`` in ``/sys/class/net/<if>/statistics``, counted in octets but
  *above* the RDMA stack, so RoCE traffic that bypasses the kernel is invisible.

Collectors translate raw names into canonical names and canonical units, and
the rate engine uses :data:`CANONICAL` to handle counter width and overflow
behaviour (legacy IB error counters saturate rather than wrap).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Overflow = Literal["wrap", "saturate"]


@dataclass(frozen=True, slots=True)
class CounterSpec:
    name: str
    unit: str
    category: str
    description: str
    bits: int = 64
    overflow: Overflow = "wrap"

    @property
    def max_value(self) -> int:
        return (1 << self.bits) - 1


def _spec(name: str, unit: str, category: str, description: str, bits: int = 64, overflow: Overflow = "wrap") -> CounterSpec:
    return CounterSpec(name, unit, category, description, bits, overflow)


# Canonical counters. Widths follow the narrowest source we map from: IBTA
# PortCounters error fields are 8/16/32-bit and saturate; mlx5 Q counters
# (out_of_buffer, out_of_sequence, ...) are 32-bit and wrap.
CANONICAL: dict[str, CounterSpec] = {
    s.name: s
    for s in [
        # Throughput
        _spec("tx_bytes", "bytes", "throughput", "Bytes transmitted on the wire"),
        _spec("rx_bytes", "bytes", "throughput", "Bytes received on the wire"),
        _spec("tx_packets", "packets", "throughput", "Packets transmitted"),
        _spec("rx_packets", "packets", "throughput", "Packets received"),
        # Lossless Ethernet / IB flow control. Direction matters: a pause frame
        # *received* stops our transmitter (downstream congestion), a pause
        # frame *sent* means our receive buffers are filling (host can't drain).
        _spec("rx_pause_frames", "frames", "flow_control", "PFC pause frames received on the lossless priority (our TX was paused)"),
        _spec("tx_pause_frames", "frames", "flow_control", "PFC pause frames sent on the lossless priority (our RX was backing up)"),
        _spec("rx_pause_duration_us", "us", "flow_control", "Time our transmitter spent paused by the peer"),
        _spec("tx_pause_duration_us", "us", "flow_control", "Time we held the peer paused"),
        _spec("xmit_wait", "ticks", "flow_control", "IB PortXmitWait: ticks with data queued but no credits to send", bits=32, overflow="saturate"),
        # Congestion control (DCQCN): NP = notification point (receiver),
        # RP = reaction point (sender).
        _spec("ecn_marked_packets", "packets", "congestion", "RoCE packets received with ECN CE mark (NP)"),
        _spec("cnp_sent", "packets", "congestion", "Congestion Notification Packets sent (NP)"),
        _spec("cnp_handled", "packets", "congestion", "CNPs received and acted on by rate limiter (RP)"),
        # Receive-side resource exhaustion
        _spec("out_of_buffer", "packets", "drops", "Packets dropped: no posted receive WQE (RQ starvation)", bits=32),
        _spec("rx_discards_phy", "packets", "drops", "Packets discarded by the port for lack of NIC buffer"),
        _spec("rx_dropped", "packets", "drops", "Kernel netdev receive drops"),
        # RDMA transport recovery
        _spec("packet_seq_err", "events", "transport", "NAK sequence errors received (loss detected by responder)", bits=32),
        _spec("out_of_sequence", "packets", "transport", "Out-of-sequence packets received", bits=32),
        _spec("local_ack_timeout_err", "events", "transport", "QP ACK timeouts on the requester (costly go-back-N)", bits=32),
        _spec("rnr_nak_retry_err", "events", "transport", "Receiver-not-ready NAK retries exceeded", bits=32),
        _spec("implied_nak_seq_err", "events", "transport", "Implied NAK sequence errors", bits=32),
        _spec("adp_retrans", "events", "transport", "Adaptive retransmissions (RoCE)", bits=32),
        # Physical layer
        _spec("symbol_error", "events", "physical", "Minor link errors (IB SymbolErrorCounter)", bits=16, overflow="saturate"),
        _spec("link_error_recovery", "events", "physical", "Successful link retrains", bits=8, overflow="saturate"),
        _spec("link_downed", "events", "physical", "Link went down", bits=8, overflow="saturate"),
        _spec("port_rcv_errors", "packets", "physical", "Packets received with errors", bits=16, overflow="saturate"),
        _spec("rx_crc_errors", "packets", "physical", "Frames received with bad FCS"),
        _spec("fec_corrected_bits", "bits", "physical", "Bits corrected by FEC (non-zero is normal on PAM4 links; trend matters)"),
    ]
}

# Gauges (instantaneous values).
GAUGES = {
    "link_speed_bps": "Negotiated link speed",
    "vf_max_tx_rate_bps": "SR-IOV VF transmit rate cap (0 = unlimited)",
    "vf_min_tx_rate_bps": "SR-IOV VF guaranteed transmit rate",
}


@dataclass(frozen=True, slots=True)
class RawMapping:
    canonical: str
    scale: int = 1


# /sys/class/infiniband/<dev>/ports/<p>/counters/*
RDMA_SYSFS_COUNTERS: dict[str, RawMapping] = {
    "port_xmit_data": RawMapping("tx_bytes", 4),
    "port_rcv_data": RawMapping("rx_bytes", 4),
    "port_xmit_packets": RawMapping("tx_packets"),
    "port_rcv_packets": RawMapping("rx_packets"),
    "port_xmit_wait": RawMapping("xmit_wait"),
    "symbol_error": RawMapping("symbol_error"),
    "link_error_recovery": RawMapping("link_error_recovery"),
    "link_downed": RawMapping("link_downed"),
    "port_rcv_errors": RawMapping("port_rcv_errors"),
}

# /sys/class/infiniband/<dev>/ports/<p>/hw_counters/* (mlx5)
RDMA_HW_COUNTERS: dict[str, RawMapping] = {
    "out_of_buffer": RawMapping("out_of_buffer"),
    "out_of_sequence": RawMapping("out_of_sequence"),
    "packet_seq_err": RawMapping("packet_seq_err"),
    "local_ack_timeout_err": RawMapping("local_ack_timeout_err"),
    "rnr_nak_retry_err": RawMapping("rnr_nak_retry_err"),
    "implied_nak_seq_err": RawMapping("implied_nak_seq_err"),
    "roce_adp_retrans": RawMapping("adp_retrans"),
    "np_ecn_marked_roce_packets": RawMapping("ecn_marked_packets"),
    "np_cnp_sent": RawMapping("cnp_sent"),
    "rp_cnp_handled": RawMapping("cnp_handled"),
}

# ethtool -S <iface> (mlx5 naming). Per-priority names are handled separately.
ETHTOOL_COUNTERS: dict[str, RawMapping] = {
    "tx_bytes_phy": RawMapping("tx_bytes"),
    "rx_bytes_phy": RawMapping("rx_bytes"),
    "tx_packets_phy": RawMapping("tx_packets"),
    "rx_packets_phy": RawMapping("rx_packets"),
    "rx_discards_phy": RawMapping("rx_discards_phy"),
    "rx_crc_errors_phy": RawMapping("rx_crc_errors"),
    "rx_corrected_bits_phy": RawMapping("fec_corrected_bits"),
    "link_down_events_phy": RawMapping("link_downed"),
}

# /sys/class/net/<if>/statistics/*
NETDEV_COUNTERS: dict[str, RawMapping] = {
    "tx_bytes": RawMapping("tx_bytes"),
    "rx_bytes": RawMapping("rx_bytes"),
    "tx_packets": RawMapping("tx_packets"),
    "rx_packets": RawMapping("rx_packets"),
    "rx_dropped": RawMapping("rx_dropped"),
    "rx_crc_errors": RawMapping("rx_crc_errors"),
}

_PRIO_PAUSE = re.compile(r"^(rx|tx)_prio(\d)_pause(_duration)?$")


def canonicalize(raw: dict[str, int], table: dict[str, RawMapping], lossless_priority: int | None = None) -> dict[str, int]:
    """Translate raw counter names/units into canonical ones.

    Per-priority PFC counters (``rx_prio3_pause``) are mapped only for the
    configured lossless priority; pause activity on other priorities is not
    RDMA-relevant.
    """
    out: dict[str, int] = {}
    for name, value in raw.items():
        mapping = table.get(name)
        if mapping is not None:
            out[mapping.canonical] = value * mapping.scale
            continue
        if lossless_priority is None:
            continue
        m = _PRIO_PAUSE.match(name)
        if m and int(m.group(2)) == lossless_priority:
            direction, is_duration = m.group(1), bool(m.group(3))
            key = f"{direction}_pause_duration_us" if is_duration else f"{direction}_pause_frames"
            out[key] = value
    return out


def spec_for(name: str) -> CounterSpec:
    """Spec for a canonical counter; unknown counters are treated as 64-bit wrapping."""
    spec = CANONICAL.get(name)
    if spec is None:
        return CounterSpec(name, "count", "other", "unclassified counter")
    return spec
