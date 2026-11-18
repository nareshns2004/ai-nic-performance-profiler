# shellcheck shell=bash
# Shared helpers for the hardware validation campaign (docs/methodology.md, H1-H10).
#
# Safety model
#   * Read-only commands (sysfs, ethtool -S, ip -j link, lspci, nvidia-smi topo -m,
#     nicprof collect) run without asking.
#   * EVERY state-changing command (setpci, ethtool -s, ip link set, nvidia-smi -lgc,
#     switch config, traffic generators, launching training) goes through `gated`:
#     it is printed in full and runs only after you type "yes".
#   * Each inject registers its revert. On exit or Ctrl-C, pending reverts are
#     offered one by one (still gated), so a failed run doesn't leave a host degraded.
#   * NICPROF_DRY_RUN=1 prints every gated command and runs none of them.
#
# Never run this campaign on a shared or production cluster.
set -euo pipefail

CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${NICPROF_DRY_RUN:=0}"
: "${RESULTS_ROOT:=$PWD/hw_results}"
: "${INTERVAL:=0.25}"
: "${BASELINE_S:=120}"   # healthy period before injecting
: "${FAULT_S:=120}"      # fault period
: "${RECOVERY_S:=60}"    # healthy period after reverting
: "${REMOTE_DIR:=/tmp/nicprof_campaign}"
: "${SSH:=ssh -o BatchMode=yes}"

PENDING_REVERTS=()
BG_PIDS=()

log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*" >&2; }

require_env() {
  local missing=()
  for v in "$@"; do [[ -n "${!v:-}" ]] || missing+=("$v"); done
  if ((${#missing[@]})); then
    echo "missing required settings: ${missing[*]} (see campaign.env.example)" >&2
    exit 2
  fi
}

# Run a read-only command (no confirmation).
ro() { echo "+ $*" >&2; "$@"; }

# Print a state-changing command and ask for confirmation (foreground only).
# Returns 0 to run, 1 to skip, 3 in dry-run mode.
confirm() {
  local why="$1"; shift
  printf '\n\033[33m>>> STATE CHANGE: %s\033[0m\n    %s\n' "$why" "$(printf '%q ' "$@")" >&2
  if [[ "$NICPROF_DRY_RUN" == 1 ]]; then
    echo "    (dry run: not executed)" >&2
    return 3
  fi
  local answer
  read -r -p "    Type 'yes' to run, anything else to skip: " answer </dev/tty
  [[ "$answer" == yes ]] && return 0
  echo "    skipped" >&2
  return 1
}

# Run a state-changing command only after explicit confirmation.
#   gated "why" cmd args...
gated() {
  local rc=0
  confirm "$@" || rc=$?
  ((rc == 3)) && return 0
  ((rc == 0)) || return 1
  shift
  "$@"
}

# Same, but run in the background after confirming in the foreground. Sets BG_PID.
gated_bg() {
  local rc=0
  BG_PID=""
  confirm "$@" || rc=$?
  ((rc == 3)) && return 0
  ((rc == 0)) || return 1
  shift
  "$@" &
  BG_PID=$!
  BG_PIDS+=("$BG_PID")
}

# Register the command that undoes an injection (run gated on exit if still pending).
on_revert() { PENDING_REVERTS+=("$(printf '%q ' "$@")"); }

revert_all() {
  local i
  for ((i = ${#PENDING_REVERTS[@]} - 1; i >= 0; i--)); do
    # eval, not `bash -c`: reverts may call shell functions such as on_host.
    gated "revert" eval "${PENDING_REVERTS[$i]}" || echo "    !!! revert NOT applied; do it by hand: ${PENDING_REVERTS[$i]}" >&2
  done
  PENDING_REVERTS=()
}

cleanup() {
  local rc=$?
  if ((${#PENDING_REVERTS[@]})); then
    log "exiting with ${#PENDING_REVERTS[@]} injection(s) still applied"
    revert_all
  fi
  for pid in "${BG_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  exit "$rc"
}
trap cleanup EXIT INT TERM

on_host() { local h="$1"; shift; $SSH "$h" "$@"; }

# campaign_init <H-id> <scenario> <expected_cause> <expected culprit description>
campaign_init() {
  RUN_ID="$1-$2-$(date +%Y%m%d-%H%M%S)"
  RUN_DIR="$RESULTS_ROOT/$RUN_ID"
  mkdir -p "$RUN_DIR"
  python3 - "$RUN_DIR/ground_truth.json" "$1" "$2" "$3" "$4" <<'PY'
import json, sys, time
path, hid, scenario, cause, culprit = sys.argv[1:]
json.dump({"experiment": hid, "scenario": scenario, "root_cause": cause or None, "expected_culprit": culprit,
           "label": "hardware", "started_at": time.time()}, open(path, "w"), indent=2)
PY
  exec > >(tee -a "$RUN_DIR/campaign.log") 2>&1
  log "run $RUN_ID -> $RUN_DIR"
}

# Mark the fault window in the run log (wall + monotonic, per host clocks are in the data).
mark() { echo "{\"event\": \"$1\", \"wall\": $(date +%s.%N)}" >> "$RUN_DIR/events.jsonl"; log "mark: $1"; }

# Read-only host inventory: what hardware state was the run on?
discover_hosts() {
  for h in $HOSTS; do
    log "discover $h (read-only)"
    {
      echo "### $h"
      on_host "$h" "nicprof discover" || true
      on_host "$h" "for d in /sys/class/infiniband/*; do echo \$d; cat \$d/ports/*/rate 2>/dev/null; done" || true
      on_host "$h" "nvidia-smi topo -m" || true
      [[ -n "${NIC_BDFS:-}" ]] && on_host "$h" "for b in $NIC_BDFS; do lspci -vv -s \$b | grep -E 'LnkCap:|LnkSta:'; done" || true
    } >> "$RUN_DIR/hosts.txt" 2>&1
  done
}

# Start the counter agent on every host (read-only collection).
start_collection() {
  local total=$((BASELINE_S + FAULT_S + RECOVERY_S + 60))
  local eth_args=""
  for ifc in ${ETHTOOL_IFS:-}; do eth_args+=" --ethtool-if $ifc"; done
  for h in $HOSTS; do
    log "start nicprof collect on $h (${total}s, every ${INTERVAL}s)"
    on_host "$h" "mkdir -p $REMOTE_DIR && nicprof collect --interval $INTERVAL --source rdma$eth_args --duration $total --out $REMOTE_DIR/counters.$h.jsonl" &
    BG_PIDS+=($!)
  done
}

# Launch the training job (gated: it consumes GPUs). TRAIN_CMD must write
# StepRecorder files to $REMOTE_DIR/steps.rank*.jsonl on each host.
start_training() {
  local extra_env="${1:-}"
  gated_bg "launch training job${extra_env:+ with $extra_env}" bash -c "$extra_env $TRAIN_CMD" || log "training not launched"
}

wait_s() { log "waiting $1 s ($2)"; sleep "$1"; }

# Copy counters and step files back, then analyse locally (read-only).
collect_and_analyze() {
  log "waiting for training and collectors to finish"
  wait "${BG_PIDS[@]}" 2>/dev/null || true
  BG_PIDS=()
  for h in $HOSTS; do
    scp -q "$h:$REMOTE_DIR/counters.$h.jsonl" "$RUN_DIR/" || echo "no counters from $h" >&2
    scp -q "$h:$REMOTE_DIR/steps.rank*.jsonl" "$RUN_DIR/" 2>/dev/null || true
  done
  if ! compgen -G "$RUN_DIR/counters.*.jsonl" > /dev/null; then
    log "no counter files collected; nothing to analyse"
    return 0
  fi
  cat "$RUN_DIR"/counters.*.jsonl > "$RUN_DIR/counters.jsonl"
  cp "$TOPOLOGY" "$RUN_DIR/topology.yaml"
  ro nicprof analyze --counters "$RUN_DIR/counters.jsonl" --steps "$RUN_DIR"/steps.rank*.jsonl \
    --topology "$RUN_DIR/topology.yaml" --json "$RUN_DIR/report.json" --markdown "$RUN_DIR/report.md" || true
  log "done: compare $RUN_DIR/report.md with $RUN_DIR/ground_truth.json; results are labelled 'hardware'"
}

# The standard experiment shape: baseline -> inject -> fault -> revert -> recovery.
#   run_fault_experiment inject_fn revert_fn
run_fault_experiment() {
  local inject_fn="$1" revert_fn="$2"
  discover_hosts
  start_collection
  start_training
  wait_s "$BASELINE_S" "healthy baseline"
  mark inject
  "$inject_fn" || log "injection skipped; the run continues as a healthy control"
  wait_s "$FAULT_S" "fault window"
  mark revert
  "$revert_fn"
  wait_s "$RECOVERY_S" "recovery"
  collect_and_analyze
}
