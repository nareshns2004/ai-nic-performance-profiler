# Hardware validation campaign (H1-H10)

Scripts for the hardware plan in [`docs/methodology.md`](../../docs/methodology.md#hardware-validation-plan).
Each one runs: read-only discovery → start `nicprof collect` on every host → launch training →
healthy baseline → **inject** → fault window → **revert** → recovery → copy files back → `nicprof analyze`.
Results land in `hw_results/<run-id>/` with a `ground_truth.json` labelled `"hardware"`.

**These scripts have not been run yet.** Review each one against your hardware before using it.

## Safety

- Every state-changing command (`setpci`, `ethtool -s`, `ip link set`, `nvidia-smi -lgc`, switch
  commands, traffic generators, training launches) is printed in full and runs only after you type `yes`.
- Each injection registers its revert. On normal exit, error or Ctrl-C, pending reverts are offered again.
- `NICPROF_DRY_RUN=1` prints every gated command and runs none of them. Start every script this way.
- Never run on a shared or production cluster. Use one fault at a time.

## Setup

```bash
cp scripts/hw_campaign/campaign.env.example campaign.env   # fill in; nothing is guessed for you
set -a; source campaign.env; set +a
NICPROF_DRY_RUN=1 scripts/hw_campaign/h05_link_degradation.sh   # read the plan first
scripts/hw_campaign/h05_link_degradation.sh                     # then for real
```

`TRAIN_CMD` must run a job that writes `StepRecorder` files to `$REMOTE_DIR/steps.rank*.jsonl` on each
host (see `examples/pytorch_training_hook.py`), and `nicprof` must be installed on every host.

## Experiments

| Script | Fault | State change (gated) | Pass criterion |
|---|---|---|---|
| `h01_fabric_congestion.sh` | Fabric congestion | `ib_write_bw` flows across one leaf's uplinks | Primary = fabric congestion, scope = leaf |
| `h02_pfc_hol_blocking.sh` | PFC HOL blocking | Switch: ECN Kmin above PFC XOFF (your `SWITCH_*_CMD`) | PFC HOL; victims flagged; no false origin |
| `h03_host_rx_backpressure.sh` | Host RX backpressure | `setpci` target link speed + retrain on the NIC's upstream port | Culprit = that rank; senders are victims |
| `h04_lossy_retransmission.sh` | Lossy retransmission | Job on a lossy class (`NCCL_IB_TC`) + switch drop | Lossy retransmission, not link degradation |
| `h05_link_degradation.sh` | Link degradation | `ethtool -s <if> speed … autoneg off` | Link degradation with speed-deficit evidence |
| `h06_sriov_rate_cap.sh` | SR-IOV cap | `ip link set dev <pf> vf <n> max_tx_rate …` | SR-IOV cap on that host's VFs |
| `h07_rail_imbalance.sh` | Rail imbalance | Second run with a wrong `NCCL_IB_HCA` on one host | Rail imbalance on that host |
| `h08_compute_straggler.sh` | Compute straggler | `nvidia-smi -i <gpu> -lgc …` | Not the network |
| `h09_agent_overhead.sh` | Agent overhead | Training runs only | Step-time delta not significant at 0.25 s |
| `h10_detection_floor.sh` | Detection floor | H1 at decreasing `ib_write_bw` rate | Report the measured curve |

Minimum hardware: 2 hosts × 2 RDMA NICs on one switch covers H3-H9; H1-H2 need two leaves or a
switch you can configure.

## Known gaps (fill in before running)

- **Switch commands (H2, H4)** are vendor-specific and deliberately not written here.
- **H3** changes the PCIe *speed* only. A width downgrade needs BIOS settings. `TARGET_BRIDGE_BDF` must be
  the downstream port above the NIC (`lspci -t`), not the NIC itself.
- **H4** needs the `NCCL_IB_TC` value that maps to a priority *without* PFC on your fabric. Read it from
  your QoS map; don't guess it.
- **H10** uses perftest's `--rate_limit` / `--rate_units`; check `ib_write_bw --help` for your version.
