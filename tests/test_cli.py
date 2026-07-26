from click.testing import CliRunner

from ai_nic_perf_profiler.cli import main


def test_cli_smoke(tmp_path) -> None:
    runner = CliRunner()
    report_path = tmp_path / "demo.json"
    metrics_path = tmp_path / "metrics.prom"
    result = runner.invoke(
        main,
        ["--sample-interval", "0.1", "--report-path", str(report_path), "--metrics-path", str(metrics_path)],
    )

    assert result.exit_code == 0
    assert "Profiler summary" in result.output
    assert report_path.exists()
    assert metrics_path.exists()
    assert "ai_nic_perf_throughput_gbps" in metrics_path.read_text()
