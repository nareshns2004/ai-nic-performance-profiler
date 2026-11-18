#!/usr/bin/env bash
# H3 host RX backpressure: retrain one NIC's PCIe link at a lower speed, so it can't
# drain the wire and starts sending PFC pauses.
# Pass: culprit = that rank; its ring neighbours reported as victims, not causes.
#
# Mechanism (PCIe base spec): set Target Link Speed (Link Control 2 [3:0], CAP_EXP+0x30)
# on the downstream port ABOVE the NIC, then set Retrain Link (Link Control bit 5,
# CAP_EXP+0x10) on that port. The original Link Control 2 value is saved and restored.
# Lane width can't be forced this way; use BIOS settings for a width downgrade.
source "$(dirname "$0")/lib.sh"
require_env HOSTS TOPOLOGY TRAIN_CMD TARGET_HOST TARGET_BRIDGE_BDF TARGET_GEN NIC_BDFS
campaign_init H3 host_rx_backpressure host_rx_backpressure "rank on $TARGET_HOST whose NIC sits under $TARGET_BRIDGE_BDF"

link_state() { on_host "$TARGET_HOST" "for b in $NIC_BDFS; do sudo lspci -vv -s \$b | grep -E 'LnkCap:|LnkSta:'; done" || true; }

inject() {
  local orig
  orig=$(on_host "$TARGET_HOST" "sudo setpci -s $TARGET_BRIDGE_BDF CAP_EXP+0x30.w")   # read-only
  log "Link Control 2 on $TARGET_BRIDGE_BDF = 0x$orig (saved); link before:"
  link_state
  gated "PCIe target link speed -> Gen$TARGET_GEN on $TARGET_BRIDGE_BDF" \
    on_host "$TARGET_HOST" "sudo setpci -s $TARGET_BRIDGE_BDF CAP_EXP+0x30.w=$(printf '%04x' "$TARGET_GEN"):000f" || return 1
  on_revert on_host "$TARGET_HOST" "sudo setpci -s $TARGET_BRIDGE_BDF CAP_EXP+0x30.w=$orig && sudo setpci -s $TARGET_BRIDGE_BDF CAP_EXP+0x10.w=0020:0020"
  gated "retrain PCIe link on $TARGET_BRIDGE_BDF" on_host "$TARGET_HOST" "sudo setpci -s $TARGET_BRIDGE_BDF CAP_EXP+0x10.w=0020:0020" || return 1
  sleep 1
  log "link after:"
  link_state
}

revert() {
  revert_all
  sleep 1
  log "link after revert (must match the 'before' LnkSta):"
  link_state
}
run_fault_experiment inject revert
