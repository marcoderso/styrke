"""Stufe 2, Parallelität: Verhalten unter steigender Dauerlast.

Je Laststufe schicken N Arbeiter ununterbrochen Anfragen (geschlossener
Kreislauf). Nach einer Anlaufzeit folgen mehrere gleich lange Messfenster.
Der Durchsatz eines Fensters ist die Zahl der Tokens, deren Zeitstempel in das
Fenster fallen. Abgebrochen wird bei Sättigung, nicht bei Speichermangel:
wächst der Durchsatz kaum noch, folgen zwei weitere Stufen, dann ist Schluss.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..stats import mean, percentile
from .common import StageContext, ensure_calibrated, fmt
from .load import ClosedLoop


def window_throughput(
    records: list[dict[str, Any]], w_start: float, window_s: float, windows: int
) -> list[float]:
    """Tokens pro Sekunde je Messfenster, aus den Zeitstempeln der Antwortstücke.

    Die Tokenzahl einer Anfrage wird gleichmäßig auf ihre Stücke verteilt;
    Engines, die mehrere Tokens je Stück senden, werden so richtig gezählt.
    """
    tokens = [0.0] * windows
    for rec in records:
        chunks = rec.get("chunk_ms") or []
        if not chunks:
            continue
        per_chunk = (rec.get("completion_tokens") or len(chunks)) / len(chunks)
        base = rec["ts_submit"]
        for ms in chunks:
            idx = int((base + ms / 1000.0 - w_start) // window_s)
            if 0 <= idx < windows:
                tokens[idx] += per_chunk
    return [t / window_s for t in tokens]


def summarize_level(
    n: int, records: list[dict[str, Any]], w_start: float, window_s: float, windows: int
) -> dict[str, Any]:
    per_window = window_throughput(records, w_start, window_s, windows)
    w_end = w_start + window_s * windows
    cancelled = [r for r in records if r["error"] == "abgebrochen"]
    errors = [r for r in records if not r["ok"] and r["error"] != "abgebrochen"]
    in_window = [r for r in records if r["ok"] and w_start <= r["ts_submit"] < w_end]

    def pct(key: str, q: float) -> float | None:
        value = percentile([r[key] for r in in_window], q)
        return None if value is None else round(value, 4)

    return {
        "level": n,
        "throughput_tps": round(mean(per_window) or 0.0, 2),
        "throughput_min_tps": round(min(per_window), 2),
        "throughput_max_tps": round(max(per_window), 2),
        "windows_tps": [round(v, 2) for v in per_window],
        "requests": len(records),
        "requests_measured": len(in_window),
        "errors": len(errors),
        "cancelled": len(cancelled),
        "error_rate": round(len(errors) / len(records), 4) if records else 0.0,
        "decode_tps_p50": pct("decode_tps", 50),
        "ttft_p50_s": pct("ttft_s", 50),
        "ttft_p95_s": pct("ttft_s", 95),
        "total_p50_s": pct("total_s", 50),
        "total_p95_s": pct("total_s", 95),
        "error_examples": sorted({r["error"] for r in errors if r["error"]})[:3],
    }


async def run_level(
    ctx: StageContext, n: int, params: dict[str, Any], stage: int = 2, tags: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Fährt eine Laststufe: Anlauf, Messfenster, Auslauf. Auch Stufe 3 nutzt das."""
    loop = ClosedLoop(
        ctx, n, stage=stage, prompt_tokens=params["prompt_tokens"], max_tokens=params["max_tokens"],
        thinking=params["thinking"], tags=tags,
    )
    loop.start()
    try:
        await asyncio.sleep(params["warmup_s"])
        w_start = time.time()
        await asyncio.sleep(params["window_s"] * params["windows"])
        records = await loop.stop(params["drain_s"])
    finally:
        await loop.abort()
    return summarize_level(n, records, w_start, params["window_s"], params["windows"])


async def run(ctx: StageContext) -> dict[str, Any]:
    params = ctx.params["s2"]
    await ensure_calibrated(ctx)
    rows: list[dict[str, Any]] = []
    best = 0.0
    saturated_at: int | None = None
    extra_left = 0
    stop_reason = "alle Laststufen gefahren"

    ctx.log(f"  {'parallel':>8} {'tok/s gesamt':>13} {'tok/s je Anfrage':>17} {'TTFT p50':>9} {'TTFT p95':>9} {'Fehler':>7}")
    for i, n in enumerate(params["levels"]):
        ctx.store.event("level_start", stage=2, level=n)
        row = await run_level(ctx, n, params)
        rows.append(row)
        ctx.store.event("level_end", stage=2, level=n, throughput_tps=row["throughput_tps"])
        ctx.log(
            f"  {n:>8} {row['throughput_tps']:>13.1f} {fmt(row['decode_tps_p50'], 1):>17} "
            f"{fmt(row['ttft_p50_s'], 2, ' s'):>9} {fmt(row['ttft_p95_s'], 2, ' s'):>9} {row['errors']:>7}"
        )

        if row["error_rate"] > params["max_error_rate"]:
            stop_reason = f"Fehleranteil {row['error_rate']:.0%} bei {n} parallelen Anfragen"
            break
        thr = row["throughput_tps"]
        if saturated_at is None and i > 0 and best > 0 and thr < best * (1 + params["saturation_gain"]):
            saturated_at = n
            extra_left = params["extra_levels_after_saturation"]
        elif saturated_at is not None:
            extra_left -= 1
        best = max(best, thr)
        if saturated_at is not None and extra_left <= 0:
            stop_reason = f"Sättigung ab {saturated_at} parallelen Anfragen"
            break
        await asyncio.sleep(params["pause_s"])

    if saturated_at is not None and stop_reason == "alle Laststufen gefahren":
        stop_reason = f"Sättigung ab {saturated_at} parallelen Anfragen; alle Laststufen gefahren"
    top = max(rows, key=lambda r: r["throughput_tps"]) if rows else None
    summary = {
        "rows": rows,
        "saturated_at": saturated_at,
        "stop_reason": stop_reason,
        "max_throughput_tps": top["throughput_tps"] if top else None,
        "max_throughput_level": top["level"] if top else None,
    }
    ctx.store.write_json("summary_s2.json", summary)
    ctx.log(f"  Ende der Stufe: {stop_reason}")
    return summary
