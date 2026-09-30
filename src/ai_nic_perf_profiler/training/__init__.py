"""Training-side signals: step timing and collective cost model."""

from .events import StepEvent
from .recorder import StepRecorder

__all__ = ["StepEvent", "StepRecorder"]
