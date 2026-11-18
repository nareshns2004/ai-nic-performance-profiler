# Interview guide

Questions this project invites, with what a strong answer contains. Levels: **L1** any backend engineer,
**L2** senior infra/networking, **L3** staff / GPU-fabric specialist. Open with the pause-propagation story
from `nicprof demo`. It leads into the three areas this project covers: PFC/ECN, the regression, and validation.

## A. Problem framing and architecture

| Question | Level | A strong answer contains |
|---|---|---|
| Why is this hard? Isn't it just correlation? | L1 | Skew across hosts, variable step lengths, victims vs origins, collinear counters; sizing the effect, not just naming it |
| Why the step as the unit, not time? | L2 | Step time is the dependent variable; the per-host clock is shared with that host's NICs; step number is a Lamport-style logical clock across hosts |
| Walk me through `analyze()`. | L2 | monotonic rebase → rates → windows → F → Z → worst-rank X → NNLS-ridge → C → rules → culprits / victims / scope / secondary suspects |
| What if a host's clock jumps mid-job? | L2 | Both sides record `CLOCK_MONOTONIC` too; each host is rebased onto it (`mono + median(wall − mono)`); rates use Δmono; the clock-step test proves attribution is unchanged |
| How does it scale to 16k GPUs? | L3 | Per-rank feature extraction at the edge with a local StepRecorder; ship steps × features; the central regression is tiny; attribution as a few series per job |
| What if the topology file is wrong? | L2 | Localisation degrades; validate it from LLDP / `ibnetdiscover` / NCCL topology dumps |

## B. RDMA / RoCE / lossless Ethernet

| Question | Level | A strong answer contains |
|---|---|---|
| Explain PFC and why it propagates. | L2 | Per-priority XOFF/XON, hop-by-hop, buffer headroom, upstream spread, HOL blocking |
| `rx_prio3_pause` vs `tx_prio3_pause`: which is the culprit? | L2 | tx = we paused the switch (our RX can't drain: origin); rx = we were paused (victim / downstream) |
| Walk through DCQCN. | L2 | Switch ECN-marks above Kmin → NP sends CNP → RP cuts rate, recovers; Kmin below PFC XOFF so ECN acts first |
| Why is a single loss so expensive in RoCE RC? | L2 | Go-back-N retransmission, ACK timeouts in ms, QP stalls; that's why the fabric is made lossless |
| What does `out_of_buffer` mean, and why does NCCL care? | L3 | No posted receive WQE; write-with-immediate consumes RQ entries; fix RQ depth / outstanding requests |
| How do you tell a bad optic from congestion? | L2 | Symbol/CRC errors, FEC trend, speed downshift, flaps vs ECN/CNP; physical errors explain retransmits |
| Why multiply `port_xmit_data` by 4? | L2 | IBTA counts 4-octet words; plus 32-bit wrap vs 64-bit extended counters (audit B6) |
| What happens when an IB error counter saturates? | L2 | 8/16-bit PortCounters stick at max until cleared; already-saturated → drop + flag; reaching max inside an interval → keep the lower-bound delta + flag (audit B5) |
| PCIe link trained at x8: what do you see? | L3 | Drain rate halves, NIC buffers fill, it sends pauses, ring neighbours get paused; LnkSta vs LnkCap |

## C. Statistics and the model

| Question | Level | A strong answer contains |
|---|---|---|
| Why not XGBoost + SHAP? | L2 | No labelled incidents; useful from day one; exact Shapley for an additive model; operator trust; non-negativity as a physical prior |
| Why non-negative? Why ridge? Why log1p? | L2 | Anomalies only add time; ridge splits collinear credit (ECN/CNP); log1p tames 10-1000× z ranges |
| Prove that w·Δx is the Shapley value. | L3 | Linear model: the marginal contribution of f is w_f·Δx_f in every coalition, so the average is the same |
| Why median/MAD and a floor? | L2 | Heavy-tailed steps; zero-inflated error counters make MAD = 0 → z = ∞ without a floor |
| R² is 0.75 but you say 98% explained. Explain. | L3 | Fit over all steps vs share of one episode's excess; rescaling when explained > observed |
| What happens with two faults at once? | L3 | With the same onset their features rise together, so the regression cannot tell which one is the bottleneck: in the simulator top-1 is near chance, but both are listed (primary + secondary suspects) every time. Fix both, then re-check |
| How small a regression can you detect? | L2 | ~4% in the simulator; detection, not attribution, is the limit; CUSUM for sustained drift |

## D. Validation and honesty

| Question | Level | A strong answer contains |
|---|---|---|
| Your accuracy is 100%. Why should I believe it? | L2 | You shouldn't, yet: the simulator shares the rules' domain model, so it is a pipeline-consistency number. Adversarial mode drops it to 45%; the decoy-port ablation shows why (correlated non-causal counters, and a cross-rank veto in the lossy rule). Then the hardware plan H1-H10 |
| How would you inject each fault on real hardware? | L3 | The H-table: `ib_write_bw` cross-leaf, ECN Kmin > XOFF, PCIe downgrade, lossy DSCP, `max_tx_rate`, wrong `NCCL_IB_HCA`, `nvidia-smi -lgc` (`scripts/hw_campaign/`) |
| What does the agent cost the training job? | L2 | Sysfs ~1 ms/port; ethtool via in-process SIOCETHTOOL ioctl (54 µs vs 656 µs fork+exec on a non-mlx5 test NIC); StepRecorder uses CUDA events, not per-step syncs; overhead on GPUs is H9 |
| Why the ioctl and not netlink `ETHTOOL_MSG_STATS_GET`? | L3 | Netlink stats cover only standard IEEE/RMON groups; per-priority PFC and mlx5 `*_phy` counters are driver-private `ETH_SS_STATS` strings, which only the ioctl exposes |

## E. Systems design extensions

| Question | Level | A strong answer contains |
|---|---|---|
| Design fleet-wide "GPU-hours lost to the network". | L3 | Per-job attribution series → Prometheus / warehouse; sum attributed seconds × GPUs; cause and scope labels; alert on top offenders |
| Add switch telemetry. What changes? | L3 | Origin confirmation via egress queue depth, per-port ECN/PFC; join on LLDP topology; resolves fabric-wide scope |
| Attribute per collective instead of per step? | L3 | NCCL profiler plugin events per op; windows per collective; the same F → Z → X pipeline at a finer grain |
| SR-IOV: 4 VFs bursting to 150G on a 400G PF. Cap them? | L2 | Depends on synchrony: ≈ 0.96 → fits 2; ≈ 0.48 → fits 4, use `min_tx_rate` guarantees, not hard caps; censored-demand caveat |
| Is 1:1 leaves worth buying? | L3 | Fabric what-if: placement + `NCCL_IB_QPS_PER_CONNECTION` first (simulator: +35% vs +10%); ECMP collision math |

## F. Behavioural hooks

- **Disagreed with a common practice:** sorting dashboards by pause counters blames victims.
- **Chose simple over clever:** NNLS-ridge + rules over a black-box model, and why.
- **Found a flaw in my own work:** B4 (a wall-clock step corrupted alignment), B5 (silent saturation), or the
  adversarial decoy that showed the lossy-retransmission rule is vetoed by physical errors on *another* rank.

## Trap questions

- "Is FEC-corrected bits > 0 a problem?" Not on PAM4; watch the trend.
- "Does high ECN marking mean the switch is broken?" No, it's working; look for the CNP reaction.
- "Can PFC alone make a fabric lossless and fast?" Lossless yes, fast no: HOL blocking and storms.
- "Does 98% explained mean 98% accurate?" No.
