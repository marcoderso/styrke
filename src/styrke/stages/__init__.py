"""Teststufen. Schritt 1 enthält die Stufen 0 bis 2."""

from . import s0_feasibility, s1_single, s2_concurrency

STAGES = {
    0: ("Machbarkeit", s0_feasibility.run),
    1: ("Einzelstrom", s1_single.run),
    2: ("Parallelität", s2_concurrency.run),
}

__all__ = ["STAGES"]
