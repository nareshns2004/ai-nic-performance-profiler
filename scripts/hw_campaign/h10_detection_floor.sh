#!/usr/bin/env bash
# H10 detection floor on real jitter: repeat H1 at decreasing background load.
# Pass: report the measured detection curve (slowdown vs detected / correct), whatever it is.
# Rate limiting uses perftest's --rate_limit / --rate_units; check `ib_write_bw --help` on
# your perftest version before running.
source "$(dirname "$0")/lib.sh"
require_env HOSTS TOPOLOGY TRAIN_CMD BW_SERVER BW_CLIENT BW_DEVICE
: "${LOADS_GBPS:=200 100 50 25 12}"
H10_ROOT="$RESULTS_ROOT/H10-$(date +%Y%m%d-%H%M%S)"

for load in $LOADS_GBPS; do
  RESULTS_ROOT="$H10_ROOT" campaign_init "H10-${load}g" fabric_congestion fabric_congestion "leaf uplinks at ${load} Gb/s background"
  inject() {
    gated_bg "ib_write_bw sink on $BW_SERVER" on_host "$BW_SERVER" "ib_write_bw -d $BW_DEVICE -q 8 -D $FAULT_S --report_gbits" || return 1
    sleep 2
    gated_bg "ib_write_bw source on $BW_CLIENT at $load Gb/s" \
      on_host "$BW_CLIENT" "ib_write_bw -d $BW_DEVICE -q 8 -D $FAULT_S --report_gbits --rate_limit $load --rate_units g $BW_SERVER" || return 1
    on_revert on_host "$BW_CLIENT" "pkill -x ib_write_bw || true"
    on_revert on_host "$BW_SERVER" "pkill -x ib_write_bw || true"
  }
  run_fault_experiment inject revert_all
done
log "per-load reports under $H10_ROOT; tabulate slowdown vs detection by hand or with nicprof's report.json"
