"""Rank-to-NIC bindings and fabric metadata.

The binding is what turns "rank 11 is slow" into "mlx5_3 on host-2 under
leaf-1 is the problem". It usually comes from the job launcher
(``NCCL_IB_HCA`` / the NCCL topology dump) plus the cluster inventory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .model import Endpoint


@dataclass(frozen=True, slots=True)
class RankBinding:
    rank: int
    host: str
    device: str
    port: int = 1
    rail: int | None = None
    leaf: str | None = None

    @property
    def endpoint(self) -> Endpoint:
        return Endpoint(self.host, self.device, self.port)


@dataclass
class Topology:
    ranks: list[RankBinding]
    link_gbps: float = 400.0
    lossless_priority: int = 3
    # Seconds to add to each host's timestamps to put them on a common clock.
    # Only needed when step events for a rank were recorded on another host.
    clock_offsets: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_rank = {b.rank: b for b in self.ranks}

    @property
    def link_bps(self) -> float:
        return self.link_gbps * 1e9

    def binding(self, rank: int) -> RankBinding:
        return self._by_rank[rank]

    def hosts(self) -> list[str]:
        return sorted({b.host for b in self.ranks})

    def ranks_on_host(self, host: str) -> list[int]:
        return [b.rank for b in self.ranks if b.host == host]

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_gbps": self.link_gbps,
            "lossless_priority": self.lossless_priority,
            "clock_offsets": self.clock_offsets,
            "ranks": [{k: v for k, v in asdict(b).items() if v is not None} for b in self.ranks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Topology:
        return cls(
            ranks=[RankBinding(**r) for r in d["ranks"]],
            link_gbps=float(d.get("link_gbps", 400.0)),
            lossless_priority=int(d.get("lossless_priority", 3)),
            clock_offsets={k: float(v) for k, v in (d.get("clock_offsets") or {}).items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> Topology:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
