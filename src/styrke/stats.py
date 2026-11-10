"""Kleine Statistik-Helfer ohne Abhängigkeiten."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    """Perzentil mit linearer Interpolation; ``q`` zwischen 0 und 100."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(data[lo])
    return float(data[lo] + (data[hi] - data[lo]) * (pos - lo))


def median(values: Sequence[float]) -> float | None:
    return percentile(values, 50)


def mean(values: Iterable[float]) -> float | None:
    data = [v for v in values if v is not None]
    return sum(data) / len(data) if data else None


def describe(values: Sequence[float], digits: int = 3) -> dict[str, float | int | None]:
    """Anzahl, Minimum, Median, 95. und 99. Perzentil, Maximum und Mittelwert."""
    data = [v for v in values if v is not None]

    def r(x: float | None) -> float | None:
        return None if x is None else round(x, digits)

    return {
        "n": len(data),
        "min": r(min(data)) if data else None,
        "p50": r(percentile(data, 50)),
        "p95": r(percentile(data, 95)),
        "p99": r(percentile(data, 99)),
        "max": r(max(data)) if data else None,
        "mean": r(mean(data)),
    }
