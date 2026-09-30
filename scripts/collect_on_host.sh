#!/usr/bin/env bash
# Collect real counters on a GPU host for the duration of a training job.
#   scripts/collect_on_host.sh /data/nicprof ens1f0np0 ens1f1np1
set -euo pipefail
OUT_DIR="${1:?output dir}"; shift
mkdir -p "$OUT_DIR"
ETHTOOL_ARGS=()
for ifc in "$@"; do ETHTOOL_ARGS+=(--ethtool-if "$ifc"); done
exec nicprof collect --interval 1 --source rdma "${ETHTOOL_ARGS[@]}" \
  --out "$OUT_DIR/counters.$(hostname).jsonl" \
  --prom-textfile /var/lib/node_exporter/textfile/nicprof.prom
