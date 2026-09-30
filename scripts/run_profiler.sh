#!/usr/bin/env bash
# End-to-end walkthrough on simulated data: simulate a fault, analyse it, export all formats.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

SCENARIO="${1:-fabric_congestion}"
OUT="run/${SCENARIO}"

python3 -m ai_nic_perf_profiler simulate "$SCENARIO" --out-dir "$OUT" --seed 11
python3 -m ai_nic_perf_profiler analyze \
  --counters "$OUT/counters.jsonl" \
  --steps "$OUT/steps.jsonl" \
  --topology "$OUT/topology.yaml" \
  --json "$OUT/report.json" \
  --markdown "$OUT/report.md" \
  --prom "$OUT/nicprof.prom"
echo
echo "Artifacts in $OUT/: report.md, report.json, nicprof.prom, ground_truth.json"
