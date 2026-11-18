#!/usr/bin/env bash
# H1 fabric congestion: background ib_write_bw flows pinned across one leaf's uplinks.
# Pass: primary = fabric_congestion, scope = leaf (the leaf whose uplinks carry the flows).
source "$(dirname "$0")/lib.sh"
require_env HOSTS TOPOLOGY TRAIN_CMD BW_SERVER BW_CLIENT BW_DEVICE
: "${BW_QPS:=8}"
campaign_init H1 fabric_congestion fabric_congestion "ranks under the leaf whose uplinks carry $BW_CLIENT -> $BW_SERVER"

inject() {
  # The sink and source sit under different leaves, so the flows cross the uplinks.
  gated_bg "start ib_write_bw sink on $BW_SERVER" on_host "$BW_SERVER" "ib_write_bw -d $BW_DEVICE -q $BW_QPS -D $FAULT_S --report_gbits" || return 1
  sleep 2
  gated_bg "start ib_write_bw source on $BW_CLIENT" on_host "$BW_CLIENT" "ib_write_bw -d $BW_DEVICE -q $BW_QPS -D $FAULT_S --report_gbits $BW_SERVER" || return 1
  on_revert on_host "$BW_CLIENT" "pkill -x ib_write_bw || true"
  on_revert on_host "$BW_SERVER" "pkill -x ib_write_bw || true"
}
run_fault_experiment inject revert_all
