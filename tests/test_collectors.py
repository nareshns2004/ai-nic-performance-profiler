import json

from ai_nic_perf_profiler.collectors import (
    EthtoolCollector,
    NetdevSysfsCollector,
    RdmaSysfsCollector,
    SriovCollector,
    merge_snapshots,
    parse_ethtool_stats,
    parse_ip_link_json,
    parse_rate,
    read_capacity,
)


def test_parse_rate() -> None:
    assert parse_rate("400 Gb/sec (4X NDR)") == 400e9
    assert parse_rate("26.5625 Gb/sec (1X HDR)") == 26.5625e9
    assert parse_rate("garbage") is None


def test_rdma_collector_reads_counters_and_hw_counters(fake_rdma_sysfs) -> None:
    root, _ = fake_rdma_sysfs
    c = RdmaSysfsCollector("h", root, clock=iter([1.0, 1.1]).__next__)
    [ports] = c.discover()
    assert (ports.device, ports.port, ports.link_layer, ports.netdev) == ("mlx5_0", 1, "Ethernet", "ens1f0np0")
    [snap] = c.collect()
    assert snap.timestamp == 1.05
    assert snap.counters["tx_bytes"] == 4000
    assert snap.counters["out_of_buffer"] == 5
    assert snap.counters["cnp_sent"] == 1
    assert snap.gauges["link_speed_bps"] == 400e9
    assert c.netdev_aliases() == {"ens1f0np0": "mlx5_0"}


def test_rdma_collector_tolerates_missing_root(tmp_path) -> None:
    assert RdmaSysfsCollector("h", tmp_path / "nope").collect() == []


def test_netdev_skips_virtual_interfaces(fake_net_sysfs) -> None:
    c = NetdevSysfsCollector("h", fake_net_sysfs)
    assert c.discover() == ["eth0"]
    [snap] = c.collect()
    assert snap.counters == {"rx_bytes": 100, "tx_bytes": 200, "rx_dropped": 0}
    assert snap.gauges["link_speed_bps"] == 25e9


ETHTOOL_OUT = """NIC statistics:
     rx_bytes_phy: 123456
     tx_bytes_phy: 654321
     rx_prio3_pause: 17
     rx_prio3_pause_duration: 3400
     rx_prio0_pause: 5
     rx_discards_phy: 2
     not_a_number: abc
"""


def test_parse_ethtool_stats() -> None:
    stats = parse_ethtool_stats(ETHTOOL_OUT)
    assert stats["rx_bytes_phy"] == 123456
    assert "not_a_number" not in stats


def test_ethtool_collector_uses_runner_and_maps_pfc() -> None:
    calls = []

    def runner(cmd: list[str]) -> str:
        calls.append(cmd)
        return ETHTOOL_OUT

    [snap] = EthtoolCollector("h", ["ens1f0np0"], lossless_priority=3, runner=runner).collect()
    assert calls == [["ethtool", "-S", "ens1f0np0"]]
    assert snap.counters["rx_pause_frames"] == 17
    assert snap.counters["rx_pause_duration_us"] == 3400
    assert snap.counters["rx_discards_phy"] == 2


def test_merge_joins_roce_netdev_onto_rdma_device() -> None:
    from ai_nic_perf_profiler.model import CounterSnapshot

    rdma = CounterSnapshot("h", "mlx5_0", 1, 10.0, {"tx_bytes": 4000, "out_of_buffer": 1}, {"link_speed_bps": 4e11}, "rdma_sysfs")
    eth = CounterSnapshot("h", "ens1f0np0", 1, 10.2, {"tx_bytes": 9999, "rx_pause_frames": 3}, {}, "ethtool")
    [m] = merge_snapshots([rdma, eth], {"ens1f0np0": "mlx5_0"})
    assert m.device == "mlx5_0"
    assert m.counters == {"tx_bytes": 4000, "out_of_buffer": 1, "rx_pause_frames": 3}  # first source wins
    assert m.source == "rdma_sysfs+ethtool"
    assert abs(m.timestamp - 10.1) < 1e-9


IP_LINK = [
    {
        "ifname": "ens1f0",
        "vfinfo_list": [
            {"vf": 0, "rate": {"max_tx": 100000, "min_tx": 0}, "stats": {"rx": {"bytes": 10, "packets": 1}, "tx": {"bytes": 20, "packets": 2}}},
            {"vf": 1, "max_tx_rate": 0, "stats": {"tx": {"bytes": 5}}},
        ],
    }
]


def test_parse_ip_link_json() -> None:
    vfs = parse_ip_link_json(IP_LINK)
    assert vfs[0] == {"vf": 0, "max_tx_rate_mbps": 100000, "min_tx_rate_mbps": 0, "rx_bytes": 10, "rx_packets": 1, "tx_bytes": 20, "tx_packets": 2}
    assert vfs[1] == {"vf": 1, "max_tx_rate_mbps": 0, "tx_bytes": 5}


def test_sriov_collector_emits_one_endpoint_per_vf() -> None:
    snaps = SriovCollector("h", ["ens1f0"], runner=lambda cmd: json.dumps(IP_LINK)).collect()
    assert [s.device for s in snaps] == ["ens1f0/vf0", "ens1f0/vf1"]
    assert snaps[0].gauges["vf_max_tx_rate_bps"] == 100e9


def test_read_capacity(tmp_path) -> None:
    dev = tmp_path / "ens1f0" / "device"
    dev.mkdir(parents=True)
    (dev / "sriov_numvfs").write_text("4\n")
    (dev / "sriov_totalvfs").write_text("16\n")
    cap = read_capacity("ens1f0", tmp_path)
    assert cap is not None and (cap.num_vfs, cap.total_vfs) == (4, 16)
    assert read_capacity("missing", tmp_path) is None
