from ai_nic_perf_profiler.counters import canonicalize, counter_delta, rate_series, to_rate
from ai_nic_perf_profiler.counters.catalog import ETHTOOL_COUNTERS, RDMA_SYSFS_COUNTERS
from ai_nic_perf_profiler.model import CounterSnapshot


def snap(ts: float, **counters: int) -> CounterSnapshot:
    return CounterSnapshot("h", "mlx5_0", 1, ts, counters)


def test_ib_data_counters_are_scaled_from_4_octet_words() -> None:
    assert canonicalize({"port_xmit_data": 10, "port_rcv_data": 1}, RDMA_SYSFS_COUNTERS) == {"tx_bytes": 40, "rx_bytes": 4}


def test_only_lossless_priority_pause_counters_are_mapped() -> None:
    raw = {"rx_prio3_pause": 7, "rx_prio0_pause": 99, "rx_prio3_pause_duration": 1500, "tx_prio3_pause": 2, "tx_bytes_phy": 10}
    out = canonicalize(raw, ETHTOOL_COUNTERS, lossless_priority=3)
    assert out == {"rx_pause_frames": 7, "rx_pause_duration_us": 1500, "tx_pause_frames": 2, "tx_bytes": 10}


def test_32bit_counter_wrap() -> None:
    delta, flag = counter_delta("out_of_buffer", 2**32 - 10, 5)
    assert delta == 15 and flag == "wrap:out_of_buffer"


def test_64bit_decrease_is_reset_not_wrap() -> None:
    delta, flag = counter_delta("tx_bytes", 10_000, 300)
    assert delta == 300 and flag == "reset:tx_bytes"


def test_32bit_large_decrease_is_reset() -> None:
    # Wrapping would imply > half the counter range between samples: implausible.
    delta, flag = counter_delta("out_of_buffer", 1_000_000_000, 10)
    assert delta == 10 and flag == "reset:out_of_buffer"


def test_saturated_counter_is_dropped() -> None:
    delta, flag = counter_delta("symbol_error", 65535, 65535)
    assert delta is None and flag == "saturated:symbol_error"


def test_to_rate_divides_by_interval_and_skips_new_counters() -> None:
    r = to_rate(snap(10.0, tx_bytes=0), snap(12.0, tx_bytes=1000, rx_bytes=5))
    assert r is not None
    assert r.rates == {"tx_bytes": 500.0}
    assert r.dt == 2.0


def test_to_rate_rejects_non_increasing_time() -> None:
    assert to_rate(snap(10.0, tx_bytes=0), snap(10.0, tx_bytes=5)) is None


def test_rate_series_groups_and_sorts() -> None:
    other = CounterSnapshot("h", "mlx5_1", 1, 0.0, {"tx_bytes": 0})
    series = rate_series([snap(2.0, tx_bytes=20), other, snap(0.0, tx_bytes=0), snap(1.0, tx_bytes=10)])
    assert [r.rates["tx_bytes"] for r in series["h/mlx5_0/1"]] == [10.0, 10.0]
    assert series["h/mlx5_1/1"] == []
