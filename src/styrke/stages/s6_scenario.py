"""Stufe 6, Szenario: Tragfähigkeit im gemischten Betrieb.

Die Ankunftsrate an Modellaufrufen steigt von Abschnitt zu Abschnitt. Je
Abschnitt treffen Gespräche im Sitzungsmix des Lastmodells ein (siehe
``scenario.py``). Ein Abschnitt gilt als getragen, solange kein Rückstau
wächst, die Aufrufe bis zum Ende fertig werden und kaum Fehler auftreten.
Nach dem ersten nicht getragenen Abschnitt folgt noch einer zur Bestätigung,
dann endet die Stufe.

Zielwerte für die Antwortzeit setzt diese Stufe nicht. Sie hält je Abschnitt
die Antwortzeiten fest; ab welcher Rate sie zu lang sind, entscheidet die
Auswertung. Das Ergebnis hier ist die höchste Rate ohne Überlast und daraus
die Zahl der Berechtigten, die sie je Nutzungsprofil bedient.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .common import StageContext, ensure_calibrated, fmt
from .scenario import (
    Scenario,
    calls_per_session,
    overload_reasons,
    session_shapes,
    shapes_as_dicts,
    summarize,
    users_supported,
)

HEADER = (
    f"  {'Aufrufe/s':>10} {'erreicht':>9} {'Aufrufe':>8} {'laufend':>8} {'TTFT p50':>9} {'TTFT p95':>9} "
    f"{'Dauer p50':>10} {'tok/s':>7} {'Fehler':>7}  Urteil"
)


def format_row(row: dict[str, Any]) -> str:
    verdict = "getragen" if not row.get("overload") else "Überlast: " + "; ".join(row["overload"])
    return (
        f"  {row['rate']:>10.2f} {row['achieved_rate']:>9.2f} {row['calls']:>8} {fmt(row['inflight_mean'], 1):>8} "
        f"{fmt(row['ttft_p50_s'], 2, ' s'):>9} {fmt(row['ttft_p95_s'], 2, ' s'):>9} "
        f"{fmt(row['total_p50_s'], 1, ' s'):>10} {row['throughput_tps']:>7.0f} {row['errors']:>7}  {verdict}"
    )


async def run(ctx: StageContext) -> dict[str, Any]:
    params = ctx.params["s6"]
    await ensure_calibrated(ctx)
    shapes, dropped = session_shapes(ctx, params)
    if not shapes:
        summary = {"skipped": "keine Sitzungsart passt in den Kontext", "sessions_dropped": dropped}
        ctx.store.write_json("summary_s6.json", summary)
        ctx.log(f"  ausgelassen: {summary['skipped']}")
        return summary
    for drop in dropped:
        ctx.log(f"  Sitzungsart {drop['kind']} entfällt: {drop['reason']}")
    ctx.log(
        "  Sitzungsmix: "
        + ", ".join(f"{s.kind} {s.share:.0%}" for s in shapes)
        + f"; im Mittel {calls_per_session(shapes):.1f} Aufrufe je Gespräch"
    )

    rows: list[dict[str, Any]] = []
    overloaded_at: float | None = None
    extra_left = 0
    ctx.log(HEADER)
    for rate in params["rates"]:
        ctx.store.event("level_start", stage=6, level=rate)
        scenario = Scenario(ctx, stage=6, rate=rate, shapes=shapes, thinking=params["thinking"])
        await scenario.run(params["duration_s"], params["drain_s"])
        row = summarize(scenario, params["warmup_s"], params["duration_s"], ctx.load_model)
        row["overload"] = overload_reasons(row, params)
        rows.append(row)
        ctx.store.event("level_end", stage=6, level=rate, overload=row["overload"], throughput_tps=row["throughput_tps"])
        ctx.log(format_row(row))

        if row["overload"] and overloaded_at is None:
            overloaded_at = rate
            extra_left = params["extra_levels_after_overload"]
        elif overloaded_at is not None:
            extra_left -= 1
        if overloaded_at is not None and extra_left <= 0:
            break
        await asyncio.sleep(params["pause_s"])

    carried: dict[str, Any] | None = None
    for row in rows:
        if row["overload"]:
            break
        carried = row
    summary = {
        "thinking": params["thinking"],
        "sessions": shapes_as_dicts(shapes),
        "sessions_dropped": dropped,
        "calls_per_session": round(calls_per_session(shapes), 3),
        "rows": rows,
        "overloaded_at": overloaded_at,
        # Höchste vorgegebene Rate ohne Überlast, und was dabei tatsächlich an Aufrufen ankam.
        "max_sustained_rate": carried["rate"] if carried else None,
        "max_sustained_achieved_rate": carried["achieved_rate"] if carried else None,
        "users_supported": users_supported(carried["achieved_rate"], ctx.load_model) if carried else None,
    }
    ctx.store.write_json("summary_s6.json", summary)

    if carried is None:
        ctx.log("  Ergebnis: schon die kleinste Rate wird nicht getragen")
    else:
        users = ", ".join(f"{name} {n}" for name, n in (summary["users_supported"] or {}).items())
        end = "bis zur höchsten gefahrenen Rate" if overloaded_at is None else f"Überlast ab {overloaded_at:g}"
        ctx.log(f"  Ergebnis: {carried['rate']:g} Aufrufe/s getragen ({end}). Berechtigte je Profil: {users}")
    return summary
