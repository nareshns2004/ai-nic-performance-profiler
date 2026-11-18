#!/usr/bin/env bash
# H6 SR-IOV cap: lower one VF's max_tx_rate mid-job (the job must run on that VF).
# Pass: primary = sriov_rate_cap on that host's VF(s).
source "$(dirname "$0")/lib.sh"
require_env HOSTS TOPOLOGY TRAIN_CMD TARGET_HOST TARGET_PF TARGET_VF TARGET_VF_RATE_MBPS
campaign_init H6 sriov_rate_cap sriov_rate_cap "VF $TARGET_VF of $TARGET_HOST:$TARGET_PF"

inject() {
  local orig
  orig=$(on_host "$TARGET_HOST" "ip -j link show dev $TARGET_PF" | python3 -c '
import json, sys
from ai_nic_perf_profiler.collectors import parse_ip_link_json
vf = int(sys.argv[1])
print(next((v.get("max_tx_rate_mbps", 0) for v in parse_ip_link_json(json.load(sys.stdin)) if v["vf"] == vf), 0))' "$TARGET_VF")
  log "VF $TARGET_VF max_tx_rate before: $orig Mb/s (0 = unlimited; saved)"
  gated "cap VF $TARGET_VF on $TARGET_PF at $TARGET_VF_RATE_MBPS Mb/s" \
    on_host "$TARGET_HOST" "sudo ip link set dev $TARGET_PF vf $TARGET_VF max_tx_rate $TARGET_VF_RATE_MBPS" || return 1
  on_revert on_host "$TARGET_HOST" "sudo ip link set dev $TARGET_PF vf $TARGET_VF max_tx_rate $orig"
}
run_fault_experiment inject revert_all
