"""Fault-injection simulator for end-to-end testing and evaluation."""

from .cluster import SimConfig, SimResult, simulate
from .scenarios import MULTI_FAULTS, SCENARIOS, SINGLE_FAULTS, Scenario

__all__ = ["MULTI_FAULTS", "SCENARIOS", "SINGLE_FAULTS", "Scenario", "SimConfig", "SimResult", "simulate"]
