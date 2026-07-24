from click.testing import CliRunner

from ai_nic_perf_profiler.cli import main


def test_cli_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--sample-interval", "0.1", "--report-path", "demo.json"])

    assert result.exit_code == 0
    assert "Profiler summary" in result.output
