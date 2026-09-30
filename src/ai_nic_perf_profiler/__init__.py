"""ai-nic-perf-profiler: attribute distributed-training slowdowns to NIC counter evidence."""

from .analysis import AttributionReport, RootCause, analyze
from .model import CounterSnapshot, Endpoint, RateSample
from .topology import RankBinding, Topology
from .training import StepEvent, StepRecorder

__version__ = "0.2.0"

__all__ = [
    "AttributionReport",
    "CounterSnapshot",
    "Endpoint",
    "RankBinding",
    "RateSample",
    "RootCause",
    "StepEvent",
    "StepRecorder",
    "Topology",
    "__version__",
    "analyze",
]
