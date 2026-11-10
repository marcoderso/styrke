"""Teststufen 0 bis 8."""

from . import (
    s0_feasibility,
    s1_single,
    s2_concurrency,
    s3_kvfill,
    s4_prefix,
    s5_interference,
    s6_scenario,
    s7_endurance,
    s8_quality,
)

STAGES = {
    0: ("Machbarkeit", s0_feasibility.run),
    1: ("Einzelstrom", s1_single.run),
    2: ("Parallelität", s2_concurrency.run),
    3: ("KV-Füllstand", s3_kvfill.run),
    4: ("Präfix-Cache", s4_prefix.run),
    5: ("Störwirkung", s5_interference.run),
    6: ("Szenario", s6_scenario.run),
    7: ("Dauerlauf", s7_endurance.run),
    8: ("Qualität", s8_quality.run),
}

__all__ = ["STAGES"]
