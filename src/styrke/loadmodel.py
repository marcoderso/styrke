"""Lastmodell: von Berechtigten zu gleichzeitigen Anfragen.

    lambda = N * a * t * k / 3600      Ankunftsrate in Aufrufen pro Sekunde
    A      = lambda * S                mittlere Zahl gleichzeitiger Aufrufe
    C_q    = A + z_q * sqrt(A)         Gleichzeitigkeit, die in q Prozent der Zeit reicht

N = Berechtigte, a = Anteil aktiv in der Spitzenstunde, t = Nutzeraktionen je
aktivem Nutzer in dieser Stunde, k = Modellaufrufe je Aktion, S = mittlere
Bearbeitungszeit eines Aufrufs in Sekunden. Die Formel unterstellt voneinander
unabhängige Nutzer.
"""

from __future__ import annotations

import math
from typing import Any

Z = {95: 1.645, 99: 2.326}


def arrival_rate(n: float, a: float, t: float, k: float) -> float:
    """Aufrufe pro Sekunde in der Spitzenstunde."""
    return n * a * t * k / 3600.0


def concurrency(n: float, a: float, t: float, k: float, s: float, quantile: int = 99) -> dict[str, float]:
    lam = arrival_rate(n, a, t, k)
    mean_active = lam * s
    return {
        "lambda": lam,
        "mean": mean_active,
        "peak": mean_active + Z[quantile] * math.sqrt(mean_active),
    }


def supported_users(rate_per_s: float, a: float, t: float, k: float) -> float:
    """Wie viele Berechtigte eine tragbare Ankunftsrate bei gegebenem Profil bedient."""
    per_user = a * t * k / 3600.0
    return rate_per_s / per_user if per_user > 0 else float("inf")


def table(load_model: dict[str, Any], quantile: int = 99) -> list[dict[str, Any]]:
    """Eine Zeile je Profil aus dem Abschnitt ``load_model`` der Testmatrix."""
    n = load_model.get("entitled", 0)
    rows = []
    for name, p in (load_model.get("profiles") or {}).items():
        result = concurrency(n, p["a"], p["t"], p["k"], p["s"], quantile)
        rows.append(
            {
                "profile": name,
                **p,
                "lambda": round(result["lambda"], 3),
                "mean": round(result["mean"], 1),
                "peak": math.ceil(result["peak"]),
            }
        )
    return rows
