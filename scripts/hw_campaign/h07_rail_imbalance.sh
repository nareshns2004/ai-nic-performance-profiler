#!/usr/bin/env bash
# H7 rail imbalance: run the job with a wrong NCCL_IB_HCA mapping on one host, so two
# ranks share a NIC. No host state changes; it is a job-launch setting, so this
# experiment is a pair of runs (correct mapping as control, then wrong mapping)
# instead of a mid-job injection.
# Pass: control run has no episode; the wrong-mapping run is diagnosed as rail_imbalance
# on that host's over-subscribed NIC (compare its step time with the control's).
# TRAIN_CMD_WRONG_HCA must be your launcher with NCCL_IB_HCA overridden on TARGET_HOST only.
source "$(dirname "$0")/lib.sh"
require_env HOSTS TOPOLOGY TRAIN_CMD TRAIN_CMD_WRONG_HCA TARGET_HOST
H7_ROOT="$RESULTS_ROOT/H7-$(date +%Y%m%d-%H%M%S)"

RESULTS_ROOT="$H7_ROOT" campaign_init H7-control healthy "" "none (correct NCCL_IB_HCA)"
discover_hosts
start_collection
start_training
collect_and_analyze

RESULTS_ROOT="$H7_ROOT" campaign_init H7-wrong-hca rail_imbalance rail_imbalance "NIC on $TARGET_HOST that receives the extra rank's traffic"
TRAIN_CMD="$TRAIN_CMD_WRONG_HCA"
start_collection
start_training
collect_and_analyze
