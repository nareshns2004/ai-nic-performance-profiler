#!/usr/bin/env bash
# H2 PFC HOL blocking: raise the switch ECN Kmin above the PFC XOFF threshold on one port,
# so PFC fires before DCQCN reacts.
# Pass: primary = pfc_hol_blocking on the rank behind that port; paused senders flagged as
# victims; no false origin.
# Switch syntax is vendor-specific: put the exact commands in SWITCH_INJECT_CMD / SWITCH_REVERT_CMD.
source "$(dirname "$0")/lib.sh"
require_env HOSTS TOPOLOGY TRAIN_CMD SWITCH_INJECT_CMD SWITCH_REVERT_CMD
campaign_init H2 pfc_storm pfc_hol_blocking "rank behind the reconfigured switch port"

inject() {
  gated "switch: ECN Kmin above PFC XOFF" bash -c "$SWITCH_INJECT_CMD" || return 1
  on_revert bash -c "$SWITCH_REVERT_CMD"
}
run_fault_experiment inject revert_all
