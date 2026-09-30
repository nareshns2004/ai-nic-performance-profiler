"""Collective communication cost model.

Bus bandwidth conventions follow nccl-tests (``PERFORMANCE.md``): algorithm
bandwidth is ``bytes / time`` and bus bandwidth normalises it by the fraction
of data each rank must actually move, so it's comparable to link speed
regardless of rank count.

Timing follows the alpha-beta (Hockney) model: ``T = steps * alpha + volume / bandwidth``.
For ring all-reduce on ``n`` ranks, each rank sends ``2(n-1)/n * S`` bytes
over ``2(n-1)`` sequential steps, and the slowest link in the ring gates
everyone, which is why a single degraded NIC slows the whole job.
"""

from __future__ import annotations

BUSBW_FACTOR = {
    "allreduce": lambda n: 2 * (n - 1) / n,
    "allgather": lambda n: (n - 1) / n,
    "reducescatter": lambda n: (n - 1) / n,
    "alltoall": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
}


def algbw(size_bytes: float, seconds: float) -> float:
    return size_bytes / seconds


def busbw(op: str, size_bytes: float, seconds: float, n_ranks: int) -> float:
    return algbw(size_bytes, seconds) * BUSBW_FACTOR[op](n_ranks)


def ring_allreduce_bytes_per_rank(size_bytes: float, n_ranks: int) -> float:
    return 2 * (n_ranks - 1) / n_ranks * size_bytes


def ring_allreduce_seconds(size_bytes: float, n_ranks: int, bw_bytes_per_s: float, alpha_s: float = 10e-6) -> float:
    """Ring all-reduce time with the bottleneck link bandwidth ``bw_bytes_per_s``."""
    if n_ranks <= 1:
        return 0.0
    return 2 * (n_ranks - 1) * alpha_s + ring_allreduce_bytes_per_rank(size_bytes, n_ranks) / bw_bytes_per_s


def effective_bandwidth(size_bytes: float, n_ranks: int, seconds: float, alpha_s: float = 10e-6) -> float:
    """Invert :func:`ring_allreduce_seconds` to get the bottleneck bandwidth a measurement implies."""
    transfer = seconds - 2 * (n_ranks - 1) * alpha_s
    if transfer <= 0:
        raise ValueError("measured time is below the latency floor; alpha is too large")
    return ring_allreduce_bytes_per_rank(size_bytes, n_ranks) / transfer
