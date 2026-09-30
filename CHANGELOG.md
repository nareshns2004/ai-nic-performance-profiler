# Changelog

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
