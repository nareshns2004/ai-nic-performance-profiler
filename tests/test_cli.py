import json

from click.testing import CliRunner

from ai_nic_perf_profiler.cli import main


def test_simulate_then_analyze(tmp_path) -> None:
    runner = CliRunner()
    r = runner.invoke(main, ["simulate", "link_degradation", "--out-dir", str(tmp_path), "--steps", "120", "--seed", "2"])
    assert r.exit_code == 0, r.output
    report = tmp_path / "report.json"
    r = runner.invoke(
        main,
        ["analyze", "--counters", str(tmp_path / "counters.jsonl"), "--steps", str(tmp_path / "steps.jsonl"), "--topology", str(tmp_path / "topology.yaml"),
         "--json", str(report), "--prom", str(tmp_path / "m.prom")],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "Primary: Link degradation" in r.output
    assert json.loads(report.read_text())["episodes"][0]["diagnoses"][0]["cause"] == "link_degradation"
    assert (tmp_path / "m.prom").exists()


def test_demo_and_scenarios() -> None:
    runner = CliRunner()
    r = runner.invoke(main, ["demo", "--scenario", "rail_imbalance"])
    assert r.exit_code == 0, r.output
    assert "Primary: Rail imbalance" in r.output
    assert "pfc_storm" in runner.invoke(main, ["scenarios"]).output


def test_advise_fabric() -> None:
    r = CliRunner().invoke(main, ["advise", "fabric", "--hosts", "16", "--step-s", "0.5", "--compute-s", "0.35"])
    assert r.exit_code == 0, r.output
    assert "packed placement" in r.output


def test_advise_sriov(tmp_path) -> None:
    from ai_nic_perf_profiler.io import write_snapshots
    from ai_nic_perf_profiler.sim.tenants import simulate_vf_tenants

    path = tmp_path / "vf.jsonl"
    write_snapshots(path, simulate_vf_tenants(synchronized=False, duration_s=10))
    r = CliRunner().invoke(main, ["advise", "sriov", "--counters", str(path), "--select", "ens1f0/vf"])
    assert r.exit_code == 0, r.output
    assert "synchrony index" in r.output


def test_collect_runs_briefly(tmp_path) -> None:
    out = tmp_path / "c.jsonl"
    r = CliRunner().invoke(main, ["collect", "--source", "netdev", "--interval", "0.05", "--duration", "0.2", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert "ticks=" in r.output
