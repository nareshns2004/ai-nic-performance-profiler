"""Fork-free ``ethtool -S``: read driver statistics with the SIOCETHTOOL ioctl.

``ethtool -S <if>`` is three ioctls on an ordinary socket:

1. ``ETHTOOL_GSSET_INFO`` for the number of strings in the ``ETH_SS_STATS`` set,
2. ``ETHTOOL_GSTRINGS`` for their names (32 bytes each),
3. ``ETHTOOL_GSTATS`` for the values (one ``u64`` each, same order).

Doing them in-process removes the fork+exec per interface per tick
(audit B12). Names are cached per interface and re-read whenever the count
changes (driver reload, channel reconfiguration).

Why not netlink ``ETHTOOL_MSG_STATS_GET``? It returns only the standardised
IEEE 802.3 / RMON groups. The counters this tool needs, mlx5's per-priority
``rx_prio3_pause*`` and the ``*_phy`` counters, are driver-private strings that
are only exposed through the ``ETH_SS_STATS`` string set, i.e. through this ioctl.
No privileges are needed: ``ETHTOOL_GSTATS`` is a read-only command.
"""

from __future__ import annotations

import ctypes
import fcntl
import socket
import struct
from collections.abc import Callable

SIOCETHTOOL = 0x8946
ETHTOOL_GSTRINGS = 0x0000001B
ETHTOOL_GSTATS = 0x0000001D
ETHTOOL_GSSET_INFO = 0x00000037
ETH_SS_STATS = 1
ETH_GSTRING_LEN = 32
IFNAMSIZ = 16
_IFREQ_SIZE = 40  # sizeof(struct ifreq) on LP64
# The kernel writes as many entries as the driver has at call time, whatever
# count we pass. Slack keeps a count that grew since GSSET_INFO inside our buffer.
_SLACK = 256

# ioctl(fd, request, buffer address): replaceable for tests.
Ioctl = Callable[[int, int, bytearray], None]


def _kernel_ioctl(fd: int, request: int, ifreq: bytearray) -> None:
    fcntl.ioctl(fd, request, ifreq, True)


class EthtoolIoctl:
    """Reads ``ETH_SS_STATS`` for network interfaces over one long-lived socket."""

    def __init__(self, ioctl: Ioctl = _kernel_ioctl) -> None:
        self._ioctl = ioctl
        self._sock: socket.socket | None = None
        self._names: dict[str, list[str]] = {}

    def _fd(self) -> int:
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self._sock.fileno()

    def _call(self, iface: str, buf: ctypes.Array[ctypes.c_char]) -> None:
        name = iface.encode()
        if len(name) >= IFNAMSIZ:
            raise OSError(f"interface name too long: {iface!r}")
        ifreq = bytearray(struct.pack(f"{IFNAMSIZ}sP", name, ctypes.addressof(buf)).ljust(_IFREQ_SIZE, b"\0"))
        self._ioctl(self._fd(), SIOCETHTOOL, ifreq)

    def count(self, iface: str) -> int:
        # struct ethtool_sset_info { u32 cmd; u32 reserved; u64 sset_mask; u32 data[1]; }
        buf = ctypes.create_string_buffer(struct.pack("=IIQI", ETHTOOL_GSSET_INFO, 0, 1 << ETH_SS_STATS, 0))
        self._call(iface, buf)
        _, _, mask, n = struct.unpack("=IIQI", buf.raw[:20])
        if not mask & (1 << ETH_SS_STATS):
            return 0
        return int(n)

    def names(self, iface: str, n: int) -> list[str]:
        # struct ethtool_gstrings { u32 cmd; u32 string_set; u32 len; u8 data[]; }
        buf = ctypes.create_string_buffer(struct.pack("=III", ETHTOOL_GSTRINGS, ETH_SS_STATS, n) + bytes((n + _SLACK) * ETH_GSTRING_LEN))
        self._call(iface, buf)
        got = struct.unpack("=III", buf.raw[:12])[2]
        if got != n:
            raise OSError(f"{iface}: string count changed during read ({n} -> {got})")
        raw = buf.raw[12 : 12 + n * ETH_GSTRING_LEN]
        return [raw[i : i + ETH_GSTRING_LEN].split(b"\0", 1)[0].decode(errors="replace") for i in range(0, len(raw), ETH_GSTRING_LEN)]

    def values(self, iface: str, n: int) -> list[int]:
        # struct ethtool_stats { u32 cmd; u32 n_stats; u64 data[]; }
        buf = ctypes.create_string_buffer(struct.pack("=II", ETHTOOL_GSTATS, n) + bytes((n + _SLACK) * 8))
        self._call(iface, buf)
        got = struct.unpack("=II", buf.raw[:8])[1]
        if got != n:
            raise OSError(f"{iface}: stats count changed during read ({n} -> {got})")
        return list(struct.unpack(f"={n}Q", buf.raw[8 : 8 + n * 8]))

    def read(self, iface: str) -> dict[str, int]:
        """Same name -> value mapping as parsing ``ethtool -S <iface>``. Raises OSError on failure."""
        # Ask for the count every time: GSTATS writes as many values as the
        # driver has *now*, so a stale count could under-size the buffer.
        n = self.count(iface)
        names = self._names.get(iface)
        if names is None or len(names) != n:
            names = self._names[iface] = self.names(iface, n)
        return dict(zip(names, self.values(iface, n), strict=True))

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
