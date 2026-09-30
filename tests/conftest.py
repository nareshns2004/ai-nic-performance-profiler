from __future__ import annotations

from pathlib import Path

import pytest


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n")


@pytest.fixture
def fake_rdma_sysfs(tmp_path: Path):
    """A minimal /sys/class/infiniband tree with one RoCE port. Returns (root, setter)."""
    root = tmp_path / "infiniband"
    port = root / "mlx5_0" / "ports" / "1"
    (root / "mlx5_0" / "device" / "net" / "ens1f0np0").mkdir(parents=True)
    _write(port / "rate", "400 Gb/sec (4X NDR)")
    _write(port / "link_layer", "Ethernet")

    def set_counters(counters: dict[str, int] | None = None, hw: dict[str, int] | None = None) -> None:
        for k, v in (counters or {}).items():
            _write(port / "counters" / k, v)
        for k, v in (hw or {}).items():
            _write(port / "hw_counters" / k, v)

    set_counters({"port_xmit_data": 1000, "port_rcv_data": 2000, "symbol_error": 0}, {"out_of_buffer": 5, "np_cnp_sent": 1})
    return root, set_counters


@pytest.fixture
def fake_net_sysfs(tmp_path: Path) -> Path:
    root = tmp_path / "net"
    for iface, has_device in (("eth0", True), ("docker0", False)):
        base = root / iface
        base.mkdir(parents=True)
        if has_device:
            (base / "device").mkdir()
        _write(base / "speed", 25000)
        for k, v in {"rx_bytes": 100, "tx_bytes": 200, "rx_dropped": 0}.items():
            _write(base / "statistics" / k, v)
    return root
