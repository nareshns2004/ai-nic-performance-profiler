#!/usr/bin/env bash
# H8 compute straggler: lock one GPU's clocks low mid-job. The network stays healthy.
# Pass: "not the network" (most of the excess unexplained by NIC counters).
source "$(dirname "$0")/lib.sh"
require_env HOSTS TOPOLOGY TRAIN_CMD TARGET_HOST TARGET_GPU TARGET_GPU_CLOCKS
campaign_init H8 compute_straggler non_network "none (GPU $TARGET_GPU on $TARGET_HOST is slow, network healthy)"

inject() {
  gated "lock GPU $TARGET_GPU clocks on $TARGET_HOST to $TARGET_GPU_CLOCKS MHz" \
    on_host "$TARGET_HOST" "sudo nvidia-smi -i $TARGET_GPU -lgc $TARGET_GPU_CLOCKS" || return 1
  on_revert on_host "$TARGET_HOST" "sudo nvidia-smi -i $TARGET_GPU -rgc"
}
run_fault_experiment inject revert_all
