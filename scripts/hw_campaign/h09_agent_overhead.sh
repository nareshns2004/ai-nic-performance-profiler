#!/usr/bin/env bash
# H9 agent overhead: identical training runs without the agent and with it at 1, 0.25 and
# 0.1 s sampling. Also run once with StepRecorder sync= mode to measure what per-step
# CUDA synchronisation costs vs CUDA events (audit B7).
# Pass: step-time delta not significant at 0.25 s. Report every row, significant or not.
# No hardware state changes; each training launch is still confirmed.
source "$(dirname "$0")/lib.sh"
require_env HOSTS TRAIN_CMD
: "${TRAIN_CMD_SYNC_MODE:=}"   # optional: same job with StepRecorder(sync=torch.cuda.synchronize)
campaign_init H9 agent_overhead "" "n/a (overhead measurement)"
discover_hosts

run_variant() {
  local name="$1" interval="$2" cmd="$3"
  log "variant $name"
  if [[ "$interval" != none ]]; then
    INTERVAL="$interval"
    start_collection
  fi
  TRAIN_CMD="$cmd" start_training
  wait "${BG_PIDS[@]}" 2>/dev/null || true
  BG_PIDS=()
  mkdir -p "$RUN_DIR/$name"
  for h in $HOSTS; do scp -q "$h:$REMOTE_DIR/steps.rank*.jsonl" "$RUN_DIR/$name/" 2>/dev/null || true; done
  on_host "${HOSTS%% *}" "rm -f $REMOTE_DIR/steps.rank*.jsonl" || true
}

run_variant no_agent none "$TRAIN_CMD"
run_variant agent_1s 1 "$TRAIN_CMD"
run_variant agent_0.25s 0.25 "$TRAIN_CMD"
run_variant agent_0.1s 0.1 "$TRAIN_CMD"
[[ -n "$TRAIN_CMD_SYNC_MODE" ]] && run_variant sync_mode none "$TRAIN_CMD_SYNC_MODE"

python3 - "$RUN_DIR" <<'PY'
import statistics, sys
from pathlib import Path
from ai_nic_perf_profiler.io import read_steps
root = Path(sys.argv[1])
print("| variant | steps | median step (ms) | p90 (ms) |\n|---|---:|---:|---:|")
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    ev = read_steps(sorted(d.glob("steps.rank*.jsonl")))
    per = {}
    for e in ev:
        per[e.step] = max(per.get(e.step, 0.0), e.duration)
    vals = sorted(list(per.values())[10:])  # skip warmup
    if vals:
        print(f"| {d.name} | {len(vals)} | {statistics.median(vals)*1e3:.2f} | {vals[int(0.9*len(vals))]*1e3:.2f} |")
PY
