# Changelog

## Unreleased: hardening and honest evaluation

Works through audit items B2-B5, B7, B9, B10 and B12 from the project booklet.

### Fixed
- **B5: silent saturation.** A saturating IB error counter that *reaches* its maximum inside an interval
  (`prev < max`, `cur == max`) is now flagged `saturated`. Its delta is kept as a lower bound; before, it was
  returned with no flag.
- **B4: wall-clock steps.** Counter snapshots (`mono_ns`) and step events (`mono_start_ns` / `mono_end_ns`) now carry
  `CLOCK_MONOTONIC` alongside wall time. `rebase_on_monotonic` aligns each host on its monotonic clock when every
  record from it has one, and rate intervals use Δmono. A test steps one host's wall clock by +2 s mid-run and checks
  that attribution is unchanged. Files from 0.2 still load and fall back to wall time.
- **B10:** repository URL in `pyproject.toml` (the GitHub repo is `ai-nic-performance-profiler`).

### Added
- **B7: CUDA-event step timing.** `StepRecorder` defaults to CUDA events when torch and a GPU are available: one
  `synchronize` to calibrate at start-up (and every `recalibrate_every` steps), none per step, resolved lazily.
  `sync=` and a host-clock mode remain. torch is still optional.
- **B3: two-fault scenarios** (`multi_link_backpressure`, `multi_lossy_congestion`), placed on disjoint hosts, plus
  **secondary suspects** in episode reports (JSON and Markdown). `nicprof evaluate --multi`.
- **B2: adversarial simulator mode** (`--adversarial`, `SimConfig(adversarial=True, nuisances=...)`): noisy ECN
  baseline, background tenants and a decoy port. Uses its own RNG stream, so default runs are unchanged. It is
  reported separately in RESULTS.md, with a per-nuisance ablation.
- **B12: fork-free ethtool stats** via the SIOCETHTOOL ioctl (`collectors/ethtool_ioctl.py`), the same calls
  `ethtool -S` makes. `--ethtool-backend auto|ioctl|exec`; `auto` falls back to forking per interface. Netlink
  `ETHTOOL_MSG_STATS_GET` was not used because it doesn't expose driver-private counters such as `rx_prio3_pause`.
- **B9:** `docs/interview-guide.md`.
- Data quality reports how many hosts were aligned on the monotonic clock.

### Changed
- `evaluate()` and sweeps default to the single-fault scenarios; composites are evaluated separately.

## 0.2.0: attribution engine

Rebuilt from the 0.1 scaffold into an end-to-end attribution system.

### Added
- **Collectors**: RDMA sysfs (`counters/` + `hw_counters/`), netdev sysfs, `ethtool -S` (per-priority PFC and pause
  durations, PHY), SR-IOV per-VF stats and rate caps via `ip -s -j link`. RoCE netdev ↔ RDMA device merge.
- **Counter catalog** with canonical names across sources, 4-octet scaling for IB data counters, and explicit
  wrap / saturation / reset handling with data-quality flags.
- **Agent** with drift-free scheduling, ring buffer, JSONL sink, atomic Prometheus textfile output and self-metrics.
- **StepRecorder** training hook (framework-agnostic, CUDA-sync aware) and a collective cost model (busbw, alpha-beta ring).
- **Attribution engine**: step-window alignment, robust baselines, episode detection, per-rank z-scores,
  non-negative ridge decomposition (Shapley-exact), evidence rules with PFC pause-propagation handling, culprit /
  victim / scope localisation, remediation playbook.
- **Advisors**: SR-IOV partition sizing (synchrony index, censored-demand detection) and a fabric what-if model
  (placement, ECMP entropy, oversubscription, rail-optimised).
- **Fault-injection simulator** (9 scenarios, continuous severity) and an evaluation harness with accuracy
  tables and sensitivity sweeps; `benchmarks/run_benchmarks.py` regenerates `benchmarks/RESULTS.md`.
- `nicprof` CLI: `discover`, `collect`, `simulate`, `analyze`, `demo`, `evaluate`, `advise sriov|fabric`, `scenarios`.
- Docs: architecture, methodology, counter reference, design decisions, interview guide.
- Deployment: Dockerfile, Kubernetes DaemonSet, systemd unit, Prometheus alert rules; GitHub Actions CI.

### Changed
- License metadata aligned with the Apache-2.0 LICENSE file (pyproject previously said MIT).
- Fixed the `Makefile` `test` target (it was indented under `install`).

### Removed
- Virtualenv, `__pycache__` and `egg-info` directories from version control.
- The 0.1 `PerfProfiler` / `MetricCollector` placeholder API.

## 0.1.0
- Initial scaffold.
