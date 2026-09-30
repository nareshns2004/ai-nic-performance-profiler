import json
import os

from ai_nic_perf_profiler.agent import Agent
from ai_nic_perf_profiler.analysis import analyze
from ai_nic_perf_profiler.collectors import Collector
from ai_nic_perf_profiler.config import Config
from ai_nic_perf_profiler.export import markdown, prometheus
from ai_nic_perf_profiler.io import read_snapshots, read_steps, write_snapshots, write_steps
from ai_nic_perf_profiler.model import CounterSnapshot, RateSample
from ai_nic_perf_profiler.sim import SimConfig, simulate
from ai_nic_perf_profiler.training import StepEvent, StepRecorder


def test_prometheus_escaping_and_format() -> None:
    m = prometheus.Metric("x_total", "help", "counter")
    m.add(1.5, label='a"b\\c\nd')
    text = prometheus.render([m, prometheus.Metric("empty", "skipped")])
    assert text == '# HELP x_total help\n# TYPE x_total counter\nx_total{label="a\\"b\\\\c\\nd"} 1.5\n'


def test_textfile_write_is_atomic(tmp_path) -> None:
    target = tmp_path / "sub" / "nicprof.prom"
    prometheus.write_textfile(target, "a 1\n")
    prometheus.write_textfile(target, "a 2\n")
    assert target.read_text() == "a 2\n"
    assert os.listdir(target.parent) == ["nicprof.prom"]  # no temp files left behind


def test_rate_metrics_only_export_curated_counters() -> None:
    r = RateSample("h", "mlx5_0", 1, 0, 1, {"tx_bytes": 5.0, "some_random_counter": 1.0}, {"link_speed_bps": 4e11})
    text = prometheus.render(prometheus.rate_metrics([r]))
    assert 'counter="tx_bytes"' in text and "some_random_counter" not in text
    assert "nicprof_link_speed_bps" in text


def test_report_renders_to_markdown_and_metrics() -> None:
    result = simulate("fabric_congestion", SimConfig(steps=140, fault_start=70, fault_end=120, seed=1))
    report = analyze(result.snapshots, result.steps, result.topology)
    md = markdown.render(report)
    assert "Primary: Fabric congestion" in md
    assert "uplinks of leaf-" in md
    prom = prometheus.render(prometheus.report_metrics(report, job="llm"))
    assert 'nicprof_attributed_seconds{cause="fabric_congestion"' in prom
    json.dumps(report.to_dict())  # serialisable


def test_sparkline() -> None:
    assert markdown.sparkline([1, 1, 2, None]) == "▁▁█ "
    assert markdown.sparkline([]) == ""


class FakeCollector(Collector):
    name = "fake"

    def __init__(self) -> None:
        super().__init__("h")
        self.n = 0

    def collect(self) -> list[CounterSnapshot]:
        self.n += 1
        return [CounterSnapshot("h", "mlx5_0", 1, float(self.n), {"tx_bytes": 1000 * self.n})]


class BrokenCollector(Collector):
    name = "broken"

    def collect(self) -> list[CounterSnapshot]:
        raise RuntimeError("boom")


def test_agent_ticks_rates_ring_and_outputs(tmp_path) -> None:
    out, prom = tmp_path / "c.jsonl", tmp_path / "m.prom"
    agent = Agent([FakeCollector(), BrokenCollector("h")], interval_s=0.001, output=out, prometheus_textfile=prom, ring_size=2)
    stats = agent.run(max_ticks=4)
    assert stats.ticks == 4 and stats.errors == 4
    assert len(read_snapshots(out)) == 4
    assert len(agent.ring) == 2 and agent.ring[-1].rates["tx_bytes"] == 1000.0
    text = prom.read_text()
    assert "nicprof_counter_rate" in text and "nicprof_agent_collector_errors_total 4" in text


def test_jsonl_gzip_round_trip(tmp_path) -> None:
    snaps = [CounterSnapshot("h", "d", 1, 1.0, {"tx_bytes": 5}, {"link_speed_bps": 1.0}, "x")]
    write_snapshots(tmp_path / "s.jsonl.gz", snaps)
    assert read_snapshots(tmp_path / "s.jsonl.gz") == snaps
    events = [StepEvent(0, 1, 2.0, 3.0, "h", 10)]
    write_steps(tmp_path / "e.jsonl", events)
    assert read_steps([tmp_path / "e.jsonl"]) == events


def test_step_recorder(tmp_path) -> None:
    ticks = iter([1.0, 1.5, 2.0, 2.75])
    syncs = []
    with StepRecorder(tmp_path / "s.jsonl", rank=3, host="h", sync=lambda: syncs.append(1), clock=lambda: next(ticks)) as rec:
        for _ in range(2):
            with rec.step():
                pass
    events = read_steps([tmp_path / "s.jsonl"])
    assert [(e.step, e.duration) for e in events] == [(0, 0.5), (1, 0.75)]
    assert len(syncs) == 4


def test_config_loads_and_rejects_unknown_keys(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("agent:\n  interval_s: 0.5\nanalysis:\n  baseline_steps: [10, 50]\n")
    cfg = Config.load(p)
    assert cfg.agent.interval_s == 0.5 and cfg.analysis.baseline_steps == (10, 50)
    p.write_text("agent:\n  bogus: 1\n")
    try:
        Config.load(p)
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValueError")
