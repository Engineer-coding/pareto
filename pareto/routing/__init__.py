"""Adaptive query routing — signals + rule-based router."""

from pareto.routing.router import QueryRouter, RouteDecision
from pareto.routing.signals import QuerySignals, detect_language, extract_signals

__all__ = [
    "QuerySignals",
    "extract_signals",
    "detect_language",
    "QueryRouter",
    "RouteDecision",
]