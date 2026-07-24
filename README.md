# ai-nic-perf-profiler

An observability primitive that closes the attribution gap between NIC hardware counters and distributed training throughput degradation, enabling data-driven decisions on fabric topology, RDMA tuning, and SR-IOV partition sizing across GPU cluster deployments.

## Project structure

- `src/ai_nic_perf_profiler/` — Python package containing the profiler implementation.
- `tests/` — smoke tests for the profiling package.
- `examples/` — minimal usage examples.
- `configs/` — default runtime configuration.
- `scripts/` — operational scripts for running the profiler.

## Quick start

```bash
PYTHONPATH=src python3 -m ai_nic_perf_profiler
```

## Verification

The scaffold was verified by compiling the package and running a minimal in-memory profiler example successfully.

