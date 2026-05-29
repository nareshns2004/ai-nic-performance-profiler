# Architecture Overview

This repository provides a minimal Python scaffold for an AI NIC performance profiler.

## Components

- `ai_nic_perf_profiler/cli.py`: command-line entry point.
- `ai_nic_perf_profiler/config.py`: configuration helpers.
- `ai_nic_perf_profiler/metrics.py`: NIC and training metrics aggregation.
- `ai_nic_perf_profiler/profiler.py`: orchestration logic for runs and reports.

## Intended workflow

1. Collect NIC hardware counters.
2. Pair them with distributed training throughput and latency samples.
3. Produce a compact attribution report that indicates likely bottlenecks.
