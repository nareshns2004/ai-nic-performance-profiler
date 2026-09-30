# Attribution methodology

## 1. Alignment: the step is the unit of analysis

Counters are sampled on a wall-clock grid (e.g. every 100 ms); steps have
variable length. For rank *r* and step *s* with window `[a, b)` on *r*'s host
clock, each rate sample `[t0, t1)` contributes with weight equal to its overlap:

```
rate(r, s) = Σ_i rate_i · |[t0_i, t1_i) ∩ [a, b)|  /  Σ_i |[t0_i, t1_i) ∩ [a, b)|
coverage   = Σ_i |overlap_i| / (b − a)            (windows with coverage < 0.5 are dropped)
```

**Clock skew.** A rank's steps and its NIC's counters are timestamped by the
same host clock, so per-rank alignment is exact regardless of NTP/PTP quality.
Across ranks, comparison is by **step number**, which works as a logical clock. Wall-clock
offsets matter only when a rank's step events are missing and borrowed from
another host (rank-0-only logging). Then `topology.clock_offsets` shifts the window.

**Sampling resolution.** Window averages blur when the sampling interval is
comparable to the step time: a 1 s sample straddling three 0.4 s steps smears
one step's anomaly into its neighbours. `benchmarks/RESULTS.md` §4 shows this
doesn't hurt sustained episodes (100% top-1 even at 1.6 s sampling vs 0.39 s steps);
it matters for transient episodes only a few steps long.

## 2. Features

Each feature is dimensionless where physics allows and oriented so that
*higher is worse*:

| Feature | Definition | Why normalised this way |
|---|---|---|
| `tx_paused_frac` | Δ`rx_prioN_pause_duration` µs / 1e6 | Fraction of time our transmitter was held |
| `pause_sent_rate` | Δ`tx_prioN_pause` / s | We are asking the switch to stop: our RX is backing up |
| `drop_rate` | Δ(`out_of_buffer` + `rx_discards_phy` + `rx_dropped`) / s | Receive-side starvation |
| `ecn_mark_ratio` | ECN-marked / received packets | Per-packet, so it is independent of load |
| `cnp_per_kpkt` | CNPs handled / 1k transmitted packets | DCQCN throttling intensity at the sender |
| `retrans_rate`, `ack_timeout_rate` | Transport recovery events / s | Timeouts separate: each stalls a QP for ms |
| `phy_err_rate`, `link_flap_rate`, `fec_corrected_rate`, `link_speed_deficit` | Physical layer | Speed deficit vs *expected* speed |
| `vf_cap_util` | tx bps / VF `max_tx_rate` | Pressure against an SR-IOV cap |
| `rail_skew` | util(rail) / median util(host's rails) | Host-relative, so it is load-independent |

Utilisation is computed against the *expected* link speed, not the negotiated
one, so a downshifted link carrying the same bytes doesn't look like a hot rail.

## 3. Baselines and episodes

* Baseline: an explicit known-good step range, or else the first 25% of
  post-warmup steps. Step time uses the median and σ = 1.4826 · MAD.
* A step is degraded if `y > max(median + kσ, median · (1 + min_relative))`,
  with k = 4 and min_relative = 3% by default.
* Episodes are degraded runs of ≥ 3 steps, with gaps of ≤ 2 steps bridged.
  Single-step spikes (checkpoints, eval) are ignored.
* Per rank and feature, z-scores use that rank's own baseline with a
  per-feature floor on σ. Error counters are usually exactly 0 in the baseline, so
  without the floor, one stray event would be z = ∞.

## 4. Decomposition: how many milliseconds does each signal explain?

```
y_s  = step_time_s − median(baseline)                      (excess, seconds)
X_sf = log1p( max_r Z_srf ) − median_baseline(·)            (worst rank per feature)
w    = argmin_w ‖X w − y‖² + λ‖w‖²   s.t. w ≥ 0
C_sf = w_f · X_sf                                           (seconds from feature f at step s)
```

Design choices:

* **Worst rank per feature**: a synchronous collective runs at the speed of its
  slowest link, so the fastest rank's counters carry no information about step time.
* **Non-negative weights**: an anomaly can only add time. Without the
  constraint, correlated features (ECN and CNP rise together) get large
  opposite-sign weights that cancel out and mean nothing.
* **Ridge**: splits credit between collinear features instead of choosing
  arbitrarily. Credit is then aggregated by cause, which makes the split
  inside a cause irrelevant.
* **log1p**: compresses z-scores spanning 10–1000× so that one extreme counter
  doesn't dominate. The model stays linear in the transformed features.
* **Shapley-exact**: for an additive model with the baseline as the reference
  point, `w_f · (x_f − x_f_baseline)` *is* feature f's Shapley value. The
  decomposition is order-independent and sums exactly to the prediction, without
  sampling. If the explained total exceeds the observed excess, contributions
  are scaled down proportionally, and what remains unexplained is reported as such.

## 5. Diagnosis: from "which feature" to "which cause"

Evidence strength per cause is `1 − exp(−z/8)` of the strongest relevant
per-rank episode-mean z-score, with interactions:

| Cause | Evidence | Suppressed by |
|---|---|---|
| Link degradation | physical errors, flaps, speed deficit, ½·FEC trend | — |
| Lossy retransmission | retransmits, ACK timeouts | physical errors (they explain the loss) |
| Host RX backpressure | pauses *sent*, receive drops | — |
| PFC HOL blocking | time *paused*, IB xmit_wait | a pause origin inside the job; strong ECN/CNP |
| Fabric congestion | ECN ratio, CNP rate | — |
| SR-IOV rate cap | tx / VF cap | — |
| Rail imbalance | rail skew | — |

Seconds attributed to a feature are split across its candidate causes in
proportion to evidence (e.g. `tx_paused_frac` → PFC / backpressure /
congestion). Seconds whose candidate causes all lack evidence are moved to
**unexplained**. If ≥ 20% of the excess is unexplained, a **Not the network** diagnosis is
emitted. Exonerating the fabric is a useful output in its own right.

**Pause propagation.** PFC is hop-by-hop. A receiver that can't drain (PCIe
downgrade, RQ starvation) pauses its switch port, the switch buffer fills, and
the switch pauses the *senders*. The loudest pause counters therefore belong
to victims. The rule engine treats "some rank is sending pauses" as the origin,
and reports paused ranks separately as `victim_ranks`.

**Scope.** Culprits on one rank → `rank`; one host → `host`; several hosts
under one leaf → `leaf` (suspect: that leaf's uplinks); several leaves →
`fabric` (suspect: spine / ECMP).

## 6. Evaluation and what it does and doesn't prove

`nicprof evaluate` runs the simulator's fault scenarios over many seeds and
reports detection, top-1 root cause, culprit precision/recall and explained
fraction. Sweeps over fault severity and sampling interval show where accuracy
breaks down. Results: `benchmarks/RESULTS.md`.

**What the simulator validates:** the pipeline mechanics under realistic nuisance
conditions: counter resets, clock skew, sampling phase jitter, checkpoint
spikes, missing endpoints, rank-0-only logging, low-severity faults near the
noise floor, and the separation between network faults and compute stragglers.

**What it does not validate:** that real hardware produces these signatures.
The simulator and the rules encode the same domain model, so 100% at full
severity is a consistency check, not evidence of real-world accuracy.

### Hardware validation plan

Inject each fault on a real cluster with a known ground truth, and measure the same metrics:

| Scenario | Injection method |
|---|---|
| Fabric congestion | Background `ib_write_bw` flows pinned to cross one leaf's uplinks |
| PFC HOL blocking | Raise the switch ECN Kmin above PFC XOFF on one port |
| Host RX backpressure | Force the NIC's PCIe link to a lower width/Gen (BIOS or `setpci`), or throttle the RQ depth |
| Lossy retransmission | Move RDMA traffic to a lossy priority via DSCP mapping; add a switch ACL drop rate |
| Link degradation | Known-marginal optic; or force a speed downshift with `ethtool -s` / `mlxlink` |
| SR-IOV rate cap | `ip link set <pf> vf <n> max_tx_rate` mid-job |
| Rail imbalance | Misconfigure `NCCL_IB_HCA` on one host |
| Compute straggler | `nvidia-smi -lgc` to lower one GPU's clocks |

## 7. Known limitations

* **Bottleneck masking.** With a min-gated collective, a secondary fault on a
  faster link contributes 0 ms until the primary is fixed. The additive model
  correctly attributes 0 ms to it, but fixing the primary can then reveal the
  secondary. Per-rank evidence (`top_anomalous_ranks`) still lists it.
* **Linear-in-log model.** Large, saturating effects (a link flap that stalls
  everything for 2 s) are only approximately additive.
* **Detection floor.** Slowdowns below the `min_relative_slowdown` gate and the
  step-jitter floor aren't detected (≈3–4% in the default simulator). CUSUM on
  step time (`baseline.cusum`) can find smaller sustained drifts and is the
  natural next detector.
* **Overlapped communication.** With heavy compute/comm overlap, a slower network can
  cost less step time than the counters suggest. The regression learns that
  (smaller weights), but the per-rank evidence can look alarming for a small real effect.
* **Topology input.** Localisation is only as good as the rank→NIC→leaf map.
