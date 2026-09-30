# Design decisions

Short ADR-style records: the decision, the alternatives, and why.

## ADR-1: The training step is the unit of analysis, not wall-clock time

**Decision.** Integrate counter rates over each rank's own step windows, and compare across ranks by step number.

**Alternatives.** Resample everything onto a common 1 s grid, then correlate time series.

**Why.** Step time is the quantity the business cares about, so it should be the dependent variable, not one more
time series. Per-rank windows share a clock with that rank's NIC, so cross-host clock skew drops out.
Step number is a logical clock (Lamport-style) that orders events across hosts without synchronised clocks.
A fixed grid would mix fast and slow steps inside one bucket and blur the effect being measured.

## ADR-2: Additive, constrained, interpretable model over a black-box learner

**Decision.** Non-negative ridge regression of excess step time on baseline-relative anomaly features, with
rules layered on top for causal interpretation.

**Alternatives.** Gradient-boosted trees plus SHAP; causal discovery; pure rules.

**Why.** (a) The output is "N ms of the regression is due to X". For a linear model with a baseline reference,
coefficient × delta is the exact Shapley value, so there's no approximation or sampling to defend. (b) Non-negativity
encodes a true physical prior: network anomalies don't speed training up. (c) There is very little labelled incident data,
and a model that has to be trained per cluster doesn't work on day one. (d) Operators have to trust it: every millisecond
traces back to a counter they can read with `ethtool -S`.
Pure rules can name a cause but can't size it. The regression can size a feature's effect but doesn't know that
"paused" means "victim". Hence the hybrid.

## ADR-3: Worst rank per feature enters the regression

**Decision.** `X[s, f] = max over ranks of log1p(Z[s, r, f])`.

**Why.** Synchronous collectives (ring/tree all-reduce) complete at the pace of the slowest participant. Averaging
across ranks would dilute a single bad link 1/N and make it invisible at 1000+ ranks. Localisation then goes back to
the per-rank tensor to find *which* rank.

## ADR-4: Robust statistics everywhere (median/MAD), with per-feature floors

**Why.** Step times are heavy-tailed (checkpoints, eval, GC) and error counters are zero-inflated. A mean/σ baseline
is pulled around by the very outliers we're trying to find. MAD of an all-zero error counter is 0, so the floor
turns "3 CRC errors" into a moderate z-score instead of infinity.

## ADR-5: Read sysfs directly; ethtool only where necessary

**Decision.** The primary collector reads `/sys/class/infiniband/*/ports/*/{counters,hw_counters}`. `ethtool -S` is
opt-in for PFC pause durations and PHY counters, which aren't exposed in RDMA sysfs.

**Why.** Sysfs reads are a few syscalls per counter with no fork (~0.9 ms per netdev port measured in
`benchmarks/RESULTS.md`), cheap enough for sub-second sampling. `ethtool` forks a process per interface per tick. The planned replacement is
the `ETHTOOL_MSG_STATS_GET` netlink API, or rdma-core's `rdma statistic`, which avoid the fork.

## ADR-6: Edge rate computation; export curated rates, keep raw counters local

**Why.** At 10k hosts × 8 NICs × ~25 counters, exporting raw counters to Prometheus is 2M series of mostly zeros.
The agent exports ~12 rates per port and keeps the full cumulative history in a local append-only file, which is
pulled only when an episode needs analysis. Attribution outputs are a few series per *job*. This is the usual
observability trade: aggregate at the edge, keep detail on the host, fetch it on demand.

## ADR-7: Handle counter pathologies explicitly and surface them

**Why.** Real fleets have 32-bit counters that wrap, IB error counters that saturate and stay stuck, and driver reloads
that reset everything. Silently turning those into zero or negative rates produces confident wrong answers. Every
such event is flagged and counted in the report's data-quality section. An observability tool should report its own
blind spots.

## ADR-8: Ship a fault-injection simulator and measure accuracy

**Why.** "How do you know it's right?" is the first question anyone should ask of an attribution system. The
simulator provides ground truth for regression tests and sensitivity curves (fault severity, sampling interval)
in CI. It's explicitly *not* claimed as proof of real-world accuracy. `docs/methodology.md` has a hardware
fault-injection plan for that.

## ADR-9: Advisors turn attribution into decisions

**Why.** The tagline promises data-driven decisions on fabric topology, RDMA tuning and SR-IOV sizing. Attribution
says what hurt. The advisors estimate what a change would buy, calibrated to measured step time:
* SR-IOV: size by the p99 of *aggregate* VF demand, not the sum of per-VF peaks. The synchrony index says whether
  statistical multiplexing is available. Demand on capped VFs is censored and flagged.
* Fabric: a Monte Carlo over rank placement and ECMP hashing, run on an alpha-beta ring model. This shows, for example,
  when topology-aware placement plus more QPs per connection beats buying non-blocking leaves.
