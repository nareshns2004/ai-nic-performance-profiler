"""B12: fork-free ethtool statistics via SIOCETHTOOL, against a fake kernel."""

from __future__ import annotations

import ctypes
import socket
import struct

import pytest

from ai_nic_perf_profiler.collectors import EthtoolCollector, parse_ethtool_stats
from ai_nic_perf_profiler.collectors.ethtool_ioctl import (
    ETH_GSTRING_LEN,
    ETHTOOL_GSSET_INFO,
    ETHTOOL_GSTATS,
    ETHTOOL_GSTRINGS,
    SIOCETHTOOL,
    EthtoolIoctl,
)

MLX5_LIKE = {"rx_bytes_phy": 123456, "tx_bytes_phy": 654321, "rx_prio3_pause": 17, "rx_prio3_pause_duration": 3400, "rx_discards_phy": 2}


class FakeKernel:
    """Implements the three SIOCETHTOOL commands the way net/ethtool/ioctl.c does."""

    def __init__(self, stats: dict[str, dict[str, int]]) -> None:
        self.stats = stats
        self.calls: list[int] = []

    def __call__(self, fd: int, request: int, ifreq: bytearray) -> None:
        assert request == SIOCETHTOOL
        name = bytes(ifreq[:16]).split(b"\0", 1)[0].decode()
        (addr,) = struct.unpack_from("P", ifreq, 16)
        if name not in self.stats:
            raise OSError(19, "No such device")
        stats = self.stats[name]
        (cmd,) = struct.unpack("=I", ctypes.string_at(addr, 4))
        self.calls.append(cmd)
        n = len(stats)
        if cmd == ETHTOOL_GSSET_INFO:
            ctypes.memmove(addr, struct.pack("=IIQI", cmd, 0, 1 << 1, n), 20)
        elif cmd == ETHTOOL_GSTRINGS:
            data = b"".join(k.encode().ljust(ETH_GSTRING_LEN, b"\0") for k in stats)
            ctypes.memmove(addr, struct.pack("=III", cmd, 1, n) + data, 12 + len(data))
        elif cmd == ETHTOOL_GSTATS:
            data = struct.pack(f"={n}Q", *stats.values())
            ctypes.memmove(addr, struct.pack("=II", cmd, n) + data, 8 + len(data))
        else:
            raise OSError(95, "Operation not supported")


def test_ioctl_reader_matches_ethtool_dash_s() -> None:
    kernel = FakeKernel({"ens1f0np0": MLX5_LIKE})
    reader = EthtoolIoctl(kernel)
    assert reader.read("ens1f0np0") == MLX5_LIKE
    assert kernel.calls == [ETHTOOL_GSSET_INFO, ETHTOOL_GSTRINGS, ETHTOOL_GSTATS]
    # Names are cached: later reads are two ioctls, no string fetch, no fork.
    kernel.calls.clear()
    kernel.stats["ens1f0np0"] = {**MLX5_LIKE, "rx_prio3_pause": 18}
    assert reader.read("ens1f0np0")["rx_prio3_pause"] == 18
    assert kernel.calls == [ETHTOOL_GSSET_INFO, ETHTOOL_GSTATS]
    reader.close()


def test_ioctl_reader_refetches_names_when_count_changes() -> None:
    kernel = FakeKernel({"eth0": {"a": 1}})
    reader = EthtoolIoctl(kernel)
    assert reader.read("eth0") == {"a": 1}
    kernel.stats["eth0"] = {"a": 2, "b": 3}  # driver reload added counters
    assert reader.read("eth0") == {"a": 2, "b": 3}


def test_collector_uses_ioctl_without_forking() -> None:
    def runner(cmd: list[str]) -> str:
        raise AssertionError("must not fork")

    c = EthtoolCollector("h", ["ens1f0np0"], 3, runner=runner, reader=EthtoolIoctl(FakeKernel({"ens1f0np0": MLX5_LIKE})))
    [snap] = c.collect()
    assert snap.counters["rx_pause_frames"] == 17 and snap.counters["rx_pause_duration_us"] == 3400
    assert snap.mono_ns is not None


def test_auto_falls_back_to_exec_per_interface() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> str:
        calls.append(cmd)
        return "NIC statistics:\n     rx_prio3_pause: 5\n"

    reader = EthtoolIoctl(FakeKernel({"good0": MLX5_LIKE}))
    c = EthtoolCollector("h", ["good0", "odd0"], 3, runner=runner, reader=reader)
    snaps = {s.device: s for s in c.collect()}
    assert snaps["good0"].counters["rx_pause_frames"] == 17
    assert snaps["odd0"].counters["rx_pause_frames"] == 5
    c.collect()
    assert calls == [["ethtool", "-S", "odd0"]] * 2
    assert c.exec_fallback == {"odd0"}


def test_forced_ioctl_backend_does_not_fork() -> None:
    def runner(cmd: list[str]) -> str:
        raise AssertionError("must not fork")

    c = EthtoolCollector("h", ["missing0"], 3, runner=runner, backend="ioctl", reader=EthtoolIoctl(FakeKernel({})))
    assert c.collect() == []


def test_bad_backend_rejected() -> None:
    with pytest.raises(ValueError):
        EthtoolCollector("h", [], backend="netlink")  # type: ignore[arg-type]


def _local_ifaces() -> list[str]:
    return [name for _, name in socket.if_nameindex() if name != "lo"]


def test_ioctl_matches_ethtool_binary_on_this_host() -> None:
    """Read-only parity check against the real `ethtool -S` where both work."""
    import shutil
    import subprocess

    if not shutil.which("ethtool"):
        pytest.skip("ethtool not installed")
    reader = EthtoolIoctl()
    compared = 0
    for iface in _local_ifaces():
        try:
            via_ioctl = reader.read(iface)
            text = subprocess.run(["ethtool", "-S", iface], capture_output=True, text=True, check=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        via_exec = parse_ethtool_stats(text)
        # Counters can tick between the two reads; names must match exactly.
        assert set(via_ioctl) == set(via_exec), iface
        compared += 1
    reader.close()
    if not compared:
        pytest.skip("no interface exposes ethtool stats here")
