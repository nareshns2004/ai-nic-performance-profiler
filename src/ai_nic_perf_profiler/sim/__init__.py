"""Fault-injection simulator for end-to-end testing and evaluation."""

from .cluster import SimConfig, SimResult, simulate
from .scenarios import SCENARIOS, Scenario

__all__ = ["SCENARIOS", "Scenario", "SimConfig", "SimResult", "simulate"]
