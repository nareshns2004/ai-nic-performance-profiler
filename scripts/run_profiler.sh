#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m ai_nic_perf_profiler --sample-interval 0.5 --report-path report.json
