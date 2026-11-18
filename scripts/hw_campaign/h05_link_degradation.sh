#!/usr/bin/env bash
# H5 link degradation: force a speed downshift on one port mid-job.
# Pass: primary = link_degradation with link_speed_deficit evidence on that rank.
# A forced renegotiation flaps the link once; expect link_downed too.
source "$(dirname "$0")/lib.sh"
require_env HOSTS TOPOLOGY TRAIN_CMD TARGET_HOST TARGET_IF TARGET_SPEED_MBPS
campaign_init H5 link_degradation link_degradation "rank on $TARGET_HOST:$TARGET_IF"

inject() {
  on_host "$TARGET_HOST" "ethtool $TARGET_IF" | tee -a "$RUN_DIR/link_before.txt" || true   # read-only
  gated "force $TARGET_IF to $TARGET_SPEED_MBPS Mb/s" \
    on_host "$TARGET_HOST" "sudo ethtool -s $TARGET_IF speed $TARGET_SPEED_MBPS duplex full autoneg off" || return 1
  on_revert on_host "$TARGET_HOST" "sudo ethtool -s $TARGET_IF autoneg on"
}

revert() {
  revert_all
  sleep 5
  on_host "$TARGET_HOST" "ethtool $TARGET_IF" | tee -a "$RUN_DIR/link_after.txt" || true
  log "check link_after.txt matches link_before.txt (speed, advertised modes)"
}
run_fault_experiment inject revert
