"""``nicprof`` command-line interface."""

from __future__ import annotations

import json
import signal
import socket
import tempfile
from dataclasses import replace
from pathlib import Path

import click

from . import __version__
from .config import AgentConfig, AnalysisConfig, Config


def _parse_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    a, _, b = value.partition(":")
    return int(a), int(b)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="nicprof")
def main() -> None:
    """Attribute distributed-training slowdowns to NIC hardware counter evidence."""


# --------------------------------------------------------------------------- discover / collect


@main.command()
@click.option("--rdma-root", default="/sys/class/infiniband", show_default=True)
@click.option("--net-root", default="/sys/class/net", show_default=True)
def discover(rdma_root: str, net_root: str) -> None:
    """List NIC endpoints visible on this host."""
    from .collectors import NetdevSysfsCollector, RdmaSysfsCollector

    host = socket.gethostname()
    ports = RdmaSysfsCollector(host, rdma_root).discover()
    click.echo(f"RDMA ports ({rdma_root}): {len(ports)}")
    for p in ports:
        rate = f"{p.rate_bps / 1e9:.0f} Gb/s" if p.rate_bps else "unknown rate"
        click.echo(f"  {p.device} port {p.port}  {p.link_layer:<10} {rate:<14} netdev={p.netdev or '-'}")
    netdev = NetdevSysfsCollector(host, net_root)
    ifaces = netdev.discover()
    click.echo(f"Physical netdevs ({net_root}): {len(ifaces)}")
    for snap in netdev.collect():
        speed = snap.gauges.get("link_speed_bps")
        click.echo(f"  {snap.device:<16} {f'{speed / 1e9:.0f} Gb/s' if speed else 'link down/unknown'}")


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config (agent section)")
@click.option("--interval", type=float, help="Sampling interval in seconds")
@click.option("--duration", type=float, help="Stop after this many seconds (default: run until SIGTERM)")
@click.option("--out", help="Append counter snapshots to this JSONL file")
@click.option("--source", "sources", multiple=True, type=click.Choice(["rdma", "netdev"]), help="Sysfs sources to read")
@click.option("--ethtool-if", "ethtool_ifs", multiple=True, help="Interface to read PFC/PHY counters from via ethtool -S")
@click.option("--sriov-pf", "sriov_pfs", multiple=True, help="PF whose VF stats to collect")
@click.option("--prom-textfile", help="Write Prometheus metrics here (node_exporter textfile collector)")
@click.option("--host", help="Host label (default: hostname)")
def collect(config_path, interval, duration, out, sources, ethtool_ifs, sriov_pfs, prom_textfile, host) -> None:  # type: ignore[no-untyped-def]
    """Run the per-host counter sampling agent."""
    from .agent import Agent, build_collectors

    cfg = Config.load(config_path).agent if config_path else AgentConfig()
    overrides = {
        "interval_s": interval,
        "output": out,
        "sources": list(sources) or None,
        "ethtool_interfaces": list(ethtool_ifs) or None,
        "sriov_pfs": list(sriov_pfs) or None,
        "prometheus_textfile": prom_textfile,
    }
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    collectors, aliases = build_collectors(cfg, host)
    agent = Agent(collectors, cfg.interval_s, cfg.output, cfg.prometheus_textfile, aliases, cfg.ring_buffer)
    signal.signal(signal.SIGTERM, lambda *_: agent.stop_event.set())
    click.echo(f"collecting every {cfg.interval_s}s from {', '.join(c.name for c in collectors)} -> {cfg.output}", err=True)
    try:
        stats = agent.run(duration_s=duration)
    except KeyboardInterrupt:
        stats = agent.stats
    click.echo(
        f"ticks={stats.ticks} snapshots={stats.snapshots} overruns={stats.overruns} errors={stats.errors} mean_collect={stats.mean_collect_s * 1e3:.2f}ms",
        err=True,
    )


# --------------------------------------------------------------------------- simulate / analyze


@main.command()
def scenarios() -> None:
    """List simulator fault scenarios."""
    from .sim import SCENARIOS

    for s in SCENARIOS.values():
        click.echo(f"{s.name:<22} {(s.root_cause.value if s.root_cause else '-'):<22} {s.description}")


@main.command()
@click.argument("scenario")
@click.option("--out-dir", default="run", show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--severity", default=1.0, show_default=True, help="Fault magnitude, 0..1")
@click.option("--steps", default=240, show_default=True)
@click.option("--hosts", default=4, show_default=True)
@click.option("--nics-per-host", default=4, show_default=True)
@click.option("--sample-interval", default=0.1, show_default=True)
@click.option("--rank0-only", is_flag=True, help="Log step events on rank 0 only (exercises clock-offset alignment)")
def simulate(scenario, out_dir, seed, severity, steps, hosts, nics_per_host, sample_interval, rank0_only) -> None:  # type: ignore[no-untyped-def]
    """Generate counters + step events for a fault SCENARIO with ground truth."""
    from .sim import SimConfig
    from .sim import simulate as run_sim

    cfg = SimConfig(
        hosts=hosts,
        nics_per_host=nics_per_host,
        steps=steps,
        seed=seed,
        severity=severity,
        sample_interval_s=sample_interval,
        steps_rank0_only=rank0_only,
        fault_start=steps // 2,
        fault_end=steps * 5 // 6,
    )
    result = run_sim(scenario, cfg)
    paths = result.write(out_dir)
    click.echo(f"wrote {len(result.snapshots)} snapshots, {len(result.steps)} step events to {out_dir}/")
    truth = {k: result.ground_truth[k] for k in ("root_cause", "culprit_ranks", "fault_steps")}
    click.echo(f"ground truth: {json.dumps(truth)}")
    click.echo(f"next: nicprof analyze --counters {paths['counters']} --steps {paths['steps']} --topology {paths['topology']}")


@main.command()
@click.option("--counters", required=True, type=click.Path(exists=True), help="Counter snapshots JSONL (.gz ok)")
@click.option("--steps", "steps_paths", required=True, multiple=True, type=click.Path(exists=True), help="Step events JSONL; repeat per rank file")
@click.option("--topology", required=True, type=click.Path(exists=True), help="Rank-to-NIC binding YAML")
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config (analysis section)")
@click.option("--baseline", help="Baseline step range a:b (default: first 25%% after warmup)")
@click.option("--json", "json_out", help="Write the full report as JSON")
@click.option("--markdown", "md_out", help="Write the Markdown report here (default: print it)")
@click.option("--prom", "prom_out", help="Write attribution metrics in Prometheus text format")
@click.option("--job", default="default", show_default=True, help="Job label for Prometheus output")
def analyze(counters, steps_paths, topology, config_path, baseline, json_out, md_out, prom_out, job) -> None:  # type: ignore[no-untyped-def]
    """Attribute step-time regressions to NIC counter evidence."""
    from .analysis import analyze as run_analysis
    from .export import markdown, prometheus
    from .io import read_snapshots, read_steps
    from .topology import Topology

    cfg = Config.load(config_path).analysis if config_path else AnalysisConfig()
    if baseline:
        cfg = replace(cfg, baseline_steps=_parse_range(baseline))
    report = run_analysis(read_snapshots(counters), read_steps(steps_paths), Topology.load(topology), cfg)
    if json_out:
        Path(json_out).write_text(json.dumps(report.to_dict(), indent=2))
    if prom_out:
        prometheus.write_textfile(prom_out, prometheus.render(prometheus.report_metrics(report, job)))
    text = markdown.render(report)
    if md_out:
        Path(md_out).write_text(text)
        click.echo(f"report written to {md_out}")
    else:
        click.echo(text)


@main.command()
@click.option("--scenario", default="host_rx_backpressure", show_default=True)
@click.option("--seed", default=7, show_default=True)
@click.option("--keep", type=click.Path(), help="Keep generated files in this directory")
def demo(scenario: str, seed: int, keep: str | None) -> None:
    """Simulate a fault, run the full pipeline from files on disk, print the report."""
    from .analysis import analyze as run_analysis
    from .export import markdown
    from .io import read_snapshots, read_steps
    from .sim import SimConfig
    from .sim import simulate as run_sim
    from .topology import Topology

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(keep or tmp)
        result = run_sim(scenario, SimConfig(seed=seed))
        paths = result.write(out)
        report = run_analysis(read_snapshots(paths["counters"]), read_steps([paths["steps"]]), Topology.load(paths["topology"]))
        click.echo(markdown.render(report))
        truth = result.ground_truth
        click.echo(f"---\nground truth: {truth['root_cause']} on ranks {truth['culprit_ranks']} (steps {truth['fault_steps']})")


@main.command()
@click.option("--seeds", default=5, show_default=True)
@click.option("--scenario", "scenario_names", multiple=True, help="Limit to these scenarios")
@click.option("--sweep", "sweep_param", type=click.Choice(["severity", "sample_interval_s"]), help="Sweep a simulator parameter")
@click.option("--values", default="", help="Comma-separated sweep values")
def evaluate(seeds: int, scenario_names: tuple[str, ...], sweep_param: str | None, values: str) -> None:
    """Measure attribution accuracy against simulator ground truth."""
    from .evaluate import evaluate as run_eval
    from .evaluate import sweep, sweep_markdown, to_markdown

    names = list(scenario_names) or None
    if sweep_param:
        vals = [float(v) for v in values.split(",") if v] or ([0.1, 0.2, 0.35, 0.5, 1.0] if sweep_param == "severity" else [0.05, 0.1, 0.2, 0.4, 0.8])
        click.echo(sweep_markdown(sweep_param, sweep(sweep_param, vals, names, seeds)))
    else:
        click.echo(to_markdown(run_eval(names, seeds)))


# --------------------------------------------------------------------------- advisors


@main.group()
def advise() -> None:
    """Decision support: SR-IOV sizing and fabric what-if."""


@advise.command("sriov")
@click.option("--counters", required=True, type=click.Path(exists=True), help="Snapshots containing per-VF counters")
@click.option("--select", default="/vf", show_default=True, help="Substring selecting the VFs of one PF (host/device/port key)")
@click.option("--line-gbps", default=400.0, show_default=True)
@click.option("--headroom", default=0.1, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def advise_sriov(counters: str, select: str, line_gbps: float, headroom: float, as_json: bool) -> None:
    """Size VF rate policy and VFs-per-PF from measured burst demand."""
    from .advisor.sriov import plan_partitions
    from .io import read_snapshots

    plan = plan_partitions(read_snapshots(counters), line_gbps * 1e9, select, headroom)
    d = plan.to_dict()
    if as_json:
        click.echo(json.dumps(d, indent=2))
        return
    click.echo(f"VFs analysed: {len(plan.profiles)} on a {line_gbps:.0f} Gb/s PF (headroom {headroom:.0%})")
    for v in d["vfs"]:
        cap = f"cap {v['cap_gbps']} Gb/s, at cap {v['time_at_cap']:.0%}" if v["cap_gbps"] else "uncapped"
        click.echo(f"  {v['endpoint']:<28} p50 {v['p50_gbps']:>7} p99 {v['p99_gbps']:>7} Gb/s  active {v['active_frac']:.0%}  {cap}")
    click.echo(f"sum of per-VF p99:   {d['sum_p99_gbps']} Gb/s")
    click.echo(f"p99 of aggregate:    {d['p99_of_sum_gbps']} Gb/s   (synchrony index {d['synchrony_index']})")
    click.echo(f"policy:              {d['policy']}")
    click.echo(f"max VFs per PF:      {d['max_vfs_per_pf']}")
    click.echo(f"min_tx_rate:         {d['recommended_min_tx_gbps']} Gb/s   max_tx_rate: {d['recommended_max_tx_gbps'] or 'none (work-conserving)'}")
    for note in d["notes"]:
        click.echo(f"- {note}")


@advise.command("fabric")
@click.option("--hosts", default=64, show_default=True)
@click.option("--nics-per-host", default=8, show_default=True)
@click.option("--hosts-per-leaf", default=4, show_default=True)
@click.option("--uplinks-per-leaf", default=16, show_default=True)
@click.option("--link-gbps", default=400.0, show_default=True)
@click.option("--qps", default=1, show_default=True, help="NCCL_IB_QPS_PER_CONNECTION today")
@click.option("--placement", type=click.Choice(["scattered", "packed"]), default="scattered", show_default=True)
@click.option("--allreduce-gb", default=4.0, show_default=True, help="Gradient bytes all-reduced per step (GB)")
@click.option("--step-s", type=float, help="Measured step time (or use --report)")
@click.option("--compute-s", type=float, required=True, help="Compute (non-exposed-comm) time per step")
@click.option("--report", "report_path", type=click.Path(exists=True), help="Take step time from an analyze --json report baseline")
def advise_fabric(hosts, nics_per_host, hosts_per_leaf, uplinks_per_leaf, link_gbps, qps, placement, allreduce_gb, step_s, compute_s, report_path) -> None:  # type: ignore[no-untyped-def]
    """Predict step time under topology / placement / ECMP-entropy changes."""
    from .advisor.fabric import FabricModel, default_candidates, what_if

    if report_path:
        step_s = json.loads(Path(report_path).read_text())["baseline"]["median_step_s"]
    if step_s is None:
        raise click.UsageError("pass --step-s or --report")
    current = FabricModel(hosts, nics_per_host, hosts_per_leaf, uplinks_per_leaf, link_gbps, qps_per_connection=qps, placement=placement)
    click.echo(f"current: {hosts} hosts x {nics_per_host} NICs, {current.oversubscription:.1f}:1 leaf oversubscription, {placement} placement, QPs/conn={qps}")
    click.echo(f"{'scenario':<46} {'ring rate':>9} {'comm':>9} {'step':>9} {'speedup':>8}")
    for p in what_if(current, default_candidates(current), allreduce_gb * 1e9, step_s, compute_s):
        click.echo(f"{p.name:<46} {p.ring_rate:>9.2f} {p.comm_s * 1e3:>7.1f}ms {p.step_s * 1e3:>7.1f}ms {p.speedup_pct:>+7.1f}%")


if __name__ == "__main__":
    main()
