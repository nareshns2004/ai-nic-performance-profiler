# Counter reference

Canonical names used throughout `nicprof`, where they come from, and what a rise means.
Widths and semantics are from the IBTA PortCounters spec and mlx5 driver
documentation; verify them against your firmware.

## Throughput

| Canonical | RDMA sysfs (`counters/`) | ethtool (mlx5) | netdev | Notes |
|---|---|---|---|---|
| `tx_bytes` | `port_xmit_data` ×4 | `tx_bytes_phy` | `tx_bytes` | IB data counters are in **4-octet words**. netdev doesn't see kernel-bypass RDMA traffic |
| `rx_bytes` | `port_rcv_data` ×4 | `rx_bytes_phy` | `rx_bytes` | |
| `tx_packets` / `rx_packets` | `port_xmit_packets` / `port_rcv_packets` | `*_packets_phy` | `*_packets` | |

## Lossless flow control (PFC / IB credits)

| Canonical | Source | Meaning when it rises |
|---|---|---|
| `rx_pause_frames` | ethtool `rx_prio<N>_pause` | The **switch paused us**: congestion downstream of our port |
| `rx_pause_duration_us` | ethtool `rx_prio<N>_pause_duration` | Time our TX spent paused. The best single "we were blocked" signal |
| `tx_pause_frames` | ethtool `tx_prio<N>_pause` | **We paused the switch**: our RX buffers were filling, and the host can't drain the NIC |
| `tx_pause_duration_us` | ethtool `tx_prio<N>_pause_duration` | Time we held the switch paused |
| `xmit_wait` | sysfs `port_xmit_wait` (32-bit, **saturates**) | IB: ticks with data queued but no credits |

`<N>` is the lossless priority (`lossless_priority`, default 3). Counters for other priorities are ignored.

## Congestion control (DCQCN)

| Canonical | hw_counters | Role |
|---|---|---|
| `ecn_marked_packets` | `np_ecn_marked_roce_packets` | Notification point (receiver): packets arrived with ECN CE |
| `cnp_sent` | `np_cnp_sent` | NP sent a CNP back to the sender |
| `cnp_handled` | `rp_cnp_handled` | Reaction point (sender) received a CNP and **cut its rate** |

## Receive-side drops

| Canonical | Source | Meaning |
|---|---|---|
| `out_of_buffer` | hw_counters `out_of_buffer` (32-bit) | No receive WQE posted: RQ starvation. NCCL's RDMA-write-with-immediate consumes receive WQEs |
| `rx_discards_phy` | ethtool | NIC dropped for lack of internal buffer (often PCIe back-pressure) |
| `rx_dropped` | netdev | Kernel-path drops |

## RDMA transport recovery

| Canonical | hw_counters | Meaning |
|---|---|---|
| `packet_seq_err` | `packet_seq_err` | Responder saw a gap and NAKed: loss |
| `out_of_sequence` | `out_of_sequence` | Out-of-order arrivals (loss or multipath reordering) |
| `local_ack_timeout_err` | `local_ack_timeout_err` | Requester timed out waiting for an ACK. **Expensive**: go-back-N after ms |
| `rnr_nak_retry_err` | `rnr_nak_retry_err` | Receiver-not-ready retries exceeded |
| `implied_nak_seq_err` | `implied_nak_seq_err` | Implied NAK |
| `adp_retrans` | `roce_adp_retrans` | Adaptive retransmissions |

## Physical layer

| Canonical | Source | Width | Meaning |
|---|---|---|---|
| `symbol_error` | sysfs `symbol_error` | 16-bit, **saturates** | Minor link errors |
| `link_error_recovery` | sysfs | 8-bit, **saturates** | Successful retrains |
| `link_downed` | sysfs / ethtool `link_down_events_phy` | 8-bit, **saturates** | Link flaps |
| `port_rcv_errors` | sysfs | 16-bit, **saturates** | Bad packets received |
| `rx_crc_errors` | ethtool `rx_crc_errors_phy` / netdev | 64 | FCS errors |
| `fec_corrected_bits` | ethtool `rx_corrected_bits_phy` | 64 | **Non-zero is normal** on PAM4 (50G/lane+) links; alert on the *trend* |

Gauges: `link_speed_bps` (sysfs `rate`, netdev `speed`), `vf_max_tx_rate_bps`, `vf_min_tx_rate_bps` (from `ip -j link`).

## Counter hygiene in `counters/rates.py`

| Situation | Detection | Handling |
|---|---|---|
| 32-bit wrap | `cur < prev` and wrapped delta ≤ half the range | Wrapped delta, flag `wrap` |
| Reset (driver reload, `perfquery -R`) | `cur < prev` and not a plausible wrap, or any 64-bit decrease | `delta = cur` (lower bound), flag `reset` |
| Saturated IB error counter | `prev == 2^bits − 1` | Drop the counter for that interval, flag `saturated` |

All flags are counted into the report's **data quality** section.
