# nicprof: guidance for Claude Code

## What this is
Attributes distributed-training step-time regressions to NIC counter evidence
(RDMA sysfs, ethtool PFC/PHY, SR-IOV). Pipeline: monotonic rebase -> rates ->
step-window alignment -> median/MAD z-scores -> worst-rank NNLS-ridge ->
evidence rules -> playbook. Read docs/methodology.md and docs/design-decisions.md
before changing analysis/.

## Commands
- make dev                    # install with dev deps
- make check                  # ruff + mypy + pytest; must pass before every commit
- nicprof demo                # end-to-end smoke test (< 1 s)
- nicprof evaluate --seeds 2  # quick accuracy table (~6 s); --multi, --adversarial for the separate tables
- make bench                  # full benchmarks (~15 min); only when asked

## Invariants (do not break)
- Features are "higher is worse"; regression weights stay non-negative.
- Worst rank per feature enters the regression (ADR-3).
- Counter pathologies (wrap, saturate, reset) are flagged and counted, never silent.
- Pause semantics: tx_pause = origin, rx_pause / paused time = victim.
- "Not the network" must stay reachable; never force a network cause.
- Per-host alignment uses CLOCK_MONOTONIC when every record from the host has it;
  files without monotonic fields must keep loading (wall-time fallback).
- Default (non-adversarial) simulator runs must stay bit-for-bit reproducible;
  new randomness goes on its own RNG stream.
- torch is never a hard dependency (StepRecorder imports it lazily).

## Honesty rules
- Never edit numbers in benchmarks/RESULTS.md or README by hand; regenerate them
  with benchmarks/run_benchmarks.py. Label every number "simulator" or "hardware".
- Never invent hardware results, counter names or driver behaviour. If unsure,
  leave a TODO and say so in the PR description.
- Don't tune diagnosis rules against the adversarial simulator to raise its score;
  fix a rule only for a reason that holds on real hardware, and say which.

## How to work
- One audit item (B1-B12) or feature per branch and PR; reference it in the title.
- Write or update a failing test first, then the fix.
- Update docs/ and CHANGELOG.md in the same PR when behaviour changes.
- Keep dependencies to numpy, click, PyYAML unless asked.

## Hardware safety (when running on real hosts)
- Read-only by default: sysfs, ethtool -S, ip -j link, lspci, nvidia-smi topo -m.
- Ask before ANY command that changes state: setpci, ethtool -s, ip link set,
  mlxconfig, mlxlink, nvidia-smi -lgc, sysctl -w, switch configuration, reboots.
- Never run fault injection on a shared or production cluster.
- scripts/hw_campaign/ is run by a human, one fault at a time; never run it yourself.
