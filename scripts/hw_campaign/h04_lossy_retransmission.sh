#!/usr/bin/env bash
# H4 lossy retransmission: run the job on a lossy (non-PFC) traffic class, then add a
# switch drop (ACL / policer) on that class mid-job.
# Pass: primary = lossy_retransmission, NOT link_degradation (no PHY errors anywhere).
# LOSSY_NCCL_IB_TC is the NCCL_IB_TC value whose DSCP maps to a priority WITHOUT PFC on
# your fabric; check your QoS map, do not guess it. Switch syntax is vendor-specific.
source "$(dirname "$0")/lib.sh"
require_env HOSTS TOPOLOGY TRAIN_CMD LOSSY_NCCL_IB_TC SWITCH_INJECT_CMD SWITCH_REVERT_CMD
campaign_init H4 lossy_retransmit lossy_retransmission "rank(s) whose path crosses the dropping switch port"

discover_hosts
start_collection
start_training "NCCL_IB_TC=$LOSSY_NCCL_IB_TC"
wait_s "$BASELINE_S" "healthy baseline (lossy class, no drops yet)"
mark inject
gated "switch: drop a fraction of the lossy class" bash -c "$SWITCH_INJECT_CMD" && on_revert bash -c "$SWITCH_REVERT_CMD"
wait_s "$FAULT_S" "fault window"
mark revert
revert_all
wait_s "$RECOVERY_S" "recovery"
collect_and_analyze
