"""Derive dimensionless, comparable features from windowed counter rates.

Raw rates are not comparable across ports (a 200G and a 400G port) or across
load levels (more packets means more ECN marks). Features normalise by link
speed or packet count where that is physically meaningful, and every feature
is oriented so that *higher is worse*.

``floor`` is the smallest scale used when z-scoring against the baseline.
Error counters are usually exactly zero in the baseline (MAD = 0), so without
a floor a single stray event would produce an infinite z-score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .causes import RootCause as RC

NAN = float("nan")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    causes: tuple[RC, ...]
    floor: float
    description: str
    in_model: bool = True


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "tx_paused_frac", (RC.PFC_HOL_BLOCKING, RC.HOST_RX_BACKPRESSURE, RC.FABRIC_CONGESTION), 0.004, "Fraction of time our transmitter was paused by PFC"
    ),
    FeatureSpec("xmit_wait_rate", (RC.PFC_HOL_BLOCKING, RC.FABRIC_CONGESTION, RC.HOST_RX_BACKPRESSURE), 1e4, "IB credit-starvation ticks per second"),
    FeatureSpec("pause_sent_rate", (RC.HOST_RX_BACKPRESSURE,), 20.0, "PFC pause frames we sent per second"),
    FeatureSpec("drop_rate", (RC.HOST_RX_BACKPRESSURE,), 5.0, "Receive drops (RQ starvation, NIC buffer) per second"),
    FeatureSpec("ecn_mark_ratio", (RC.FABRIC_CONGESTION,), 5e-4, "ECN-marked fraction of received packets"),
    FeatureSpec("cnp_per_kpkt", (RC.FABRIC_CONGESTION,), 0.2, "CNPs handled per thousand transmitted packets"),
    FeatureSpec("retrans_rate", (RC.LOSSY_RETRANSMISSION, RC.LINK_DEGRADATION), 5.0, "Sequence errors / out-of-order / adaptive retransmits per second"),
    FeatureSpec("ack_timeout_rate", (RC.LOSSY_RETRANSMISSION, RC.LINK_DEGRADATION), 0.5, "RDMA ACK timeouts per second (each stalls a QP for ms)"),
    FeatureSpec("phy_err_rate", (RC.LINK_DEGRADATION,), 0.5, "Symbol, CRC and receive errors and link retrains per second"),
    FeatureSpec("link_flap_rate", (RC.LINK_DEGRADATION,), 0.01, "Link-down events per second"),
    FeatureSpec("fec_corrected_rate", (RC.LINK_DEGRADATION,), 1e4, "FEC-corrected bits per second (non-zero is normal; the trend matters)"),
    FeatureSpec("link_speed_deficit", (RC.LINK_DEGRADATION,), 0.02, "Shortfall of negotiated speed vs expected speed"),
    FeatureSpec("vf_cap_util", (RC.SRIOV_RATE_CAP,), 0.03, "Transmit rate as a fraction of the VF rate cap"),
    FeatureSpec("rail_skew", (RC.RAIL_IMBALANCE,), 0.05, "This rail's utilisation relative to the host's median rail"),
    FeatureSpec("util", (), 0.02, "Link utilisation (context only; not a slowdown cause by itself)", in_model=False),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURE_INDEX: dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
SPEC: dict[str, FeatureSpec] = {f.name: f for f in FEATURES}


def _sum(rates: dict[str, float], *names: str) -> float:
    vals = [rates[n] for n in names if n in rates]
    return sum(vals) if vals else NAN


def derive_features(rates: dict[str, float], gauges: dict[str, float], expected_link_bps: float) -> dict[str, float]:
    """Compute per-endpoint features for one window. ``rail_skew`` is filled in later at host level."""
    # Utilisation is relative to the *expected* speed so that a downshifted link
    # carrying the same bytes doesn't look like a hot rail.
    link_bps = expected_link_bps
    tx_b, rx_b = rates.get("tx_bytes", NAN), rates.get("rx_bytes", NAN)
    tx_p, rx_p = rates.get("tx_packets", NAN), rates.get("rx_packets", NAN)
    directions = [v for v in (tx_b, rx_b) if not math.isnan(v)]
    util = max(directions) * 8 / link_bps if directions else NAN

    speed = gauges.get("link_speed_bps")
    cap = gauges.get("vf_max_tx_rate_bps", 0.0)
    ecn = rates.get("ecn_marked_packets", NAN)
    cnp = rates.get("cnp_handled", NAN)
    pause_us = rates.get("rx_pause_duration_us", NAN)

    return {
        "tx_paused_frac": min(pause_us / 1e6, 1.0) if not math.isnan(pause_us) else NAN,
        "xmit_wait_rate": rates.get("xmit_wait", NAN),
        "pause_sent_rate": rates.get("tx_pause_frames", NAN),
        "drop_rate": _sum(rates, "out_of_buffer", "rx_discards_phy", "rx_dropped"),
        "ecn_mark_ratio": ecn / max(rx_p, 1.0) if not math.isnan(ecn) and not math.isnan(rx_p) else NAN,
        "cnp_per_kpkt": cnp / max(tx_p / 1000, 1.0) if not math.isnan(cnp) and not math.isnan(tx_p) else NAN,
        "retrans_rate": _sum(rates, "packet_seq_err", "out_of_sequence", "implied_nak_seq_err", "adp_retrans"),
        "ack_timeout_rate": _sum(rates, "local_ack_timeout_err", "rnr_nak_retry_err"),
        "phy_err_rate": _sum(rates, "symbol_error", "link_error_recovery", "port_rcv_errors", "rx_crc_errors"),
        "link_flap_rate": rates.get("link_downed", NAN),
        "fec_corrected_rate": rates.get("fec_corrected_bits", NAN),
        "link_speed_deficit": max(0.0, 1 - speed / expected_link_bps) if speed else NAN,
        "vf_cap_util": tx_b * 8 / cap if cap and cap > 0 and not math.isnan(tx_b) else NAN,
        "rail_skew": NAN,
        "util": util,
    }
