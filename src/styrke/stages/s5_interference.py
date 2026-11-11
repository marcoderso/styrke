"""Stufe 5, Störwirkung: Was eine große Anfrage den anderen antut.

Im Hintergrund laufen ununterbrochen kurze Chats. Nach einer Ruhephase kommt
eine einzelne große, kalte Anfrage dazu. Verglichen wird, wie die Chats in vier
Abschnitten laufen:

- ``baseline``: vor der großen Anfrage;
- ``prefill``: vom Absenden der großen Anfrage bis zu ihrem ersten Token, also
  während die Engine den langen Prompt einliest;
- ``decode``: während die große Anfrage ihre Antwort schreibt;
- ``recovery``: nach ihrem Ende.

Je Abschnitt: Durchsatz der Chats, Abstände zwischen zwei Tokens derselben
Antwort (Ruckeln), die längste Zeit ganz ohne Token und die Zeit bis zum
ersten Token für Chats, die in dem Abschnitt begonnen haben.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..client import RequestSpec
from ..stats import percentile
from .common import StageContext, ensure_calibrated, fmt
from .load import ClosedLoop, gaps_between, longest_silence, throughput_between


def phase_stats(background: list[dict[str, Any]], t0: float, t1: float) -> dict[str, Any] | None:
    """Wie die Hintergrund-Chats im Abschnitt ``[t0, t1)`` liefen."""
    if t1 <= t0:
        return None
    gaps = gaps_between(background, t0, t1)
    started = [r["ttft_s"] for r in background if t0 <= r["ts_submit"] < t1 and r["ok"] and r["ttft_s"] is not None]

    def ms(q: float) -> float | None:
        value = percentile(gaps, q)
        return None if value is None else round(1000 * value, 1)

    return {
        "seconds": round(t1 - t0, 2),
        "tps": round(throughput_between(background, t0, t1) or 0.0, 1),
        "gap_p50_ms": ms(50),
        "gap_p99_ms": ms(99),
        "gap_max_ms": ms(100),
        "silence_max_s": round(longest_silence(background, t0, t1) or 0.0, 3),
        "requests_started": len(started),
        "ttft_p50_s": None if not started else round(percentile(started, 50) or 0.0, 4),
        "ttft_max_s": max(started) if started else None,
    }


def _slowdown(base: dict[str, Any] | None, phase: dict[str, Any] | None) -> float | None:
    """Einbruch des Durchsatzes gegenüber der Ruhephase in Prozent."""
    if not base or not phase or not base["tps"]:
        return None
    return round(100 * (1 - phase["tps"] / base["tps"]), 1)


def analyze(background: list[dict[str, Any]], injection: dict[str, Any]) -> dict[str, Any]:
    """Wertet eine Einspielung aus; ``injection`` trägt die Zeitmarken und den Datensatz der großen Anfrage."""
    big = injection["record"]
    t_inj, t_end = injection["t_inject"], injection["t_end"]
    t_first = t_inj + big["ttft_s"] if big.get("ttft_s") is not None else t_end
    base = phase_stats(background, injection["t_baseline"], t_inj)
    prefill = phase_stats(background, t_inj, t_first)
    decode = phase_stats(background, t_first, t_end)
    return {
        "big_target": injection["target"],
        "rep": injection["rep"],
        "big_ok": big["ok"],
        "big_error": big["error"],
        "big_prompt_tokens": big.get("prompt_tokens"),
        "big_ttft_s": big.get("ttft_s"),
        "big_total_s": big.get("total_s"),
        "big_decode_tps": big.get("decode_tps"),
        "baseline": base,
        "prefill": prefill,
        "decode": decode,
        "recovery": phase_stats(background, t_end, injection["t_recovered"]),
        "slowdown_prefill_pct": _slowdown(base, prefill),
        "slowdown_decode_pct": _slowdown(base, decode),
    }


async def run(ctx: StageContext) -> dict[str, Any]:
    params = ctx.params["s5"]
    await ensure_calibrated(ctx)
    skipped: list[dict[str, Any]] = []
    targets = []
    for target in params["big_lengths"]:
        if ctx.fits(target, params["big_max_tokens"] + 256):
            targets.append(target)
        else:
            skipped.append({"big_target": target, "reason": "passt nicht in den Kontext"})

    if not targets:
        # Keine der vorgesehenen Längen passt: ersatzweise die größte, die laut Stufe 0 läuft.
        fallback = min(ctx.prompt_limit(), ctx.model.max_context - params["big_max_tokens"] - 256)
        if fallback >= params["fallback_min_tokens"]:
            targets = [fallback]
            ctx.log(f"  keine der vorgesehenen großen Anfragen passt; ersatzweise {fallback} Tokens")

    loop = ClosedLoop(
        ctx, params["chats"], stage=5, prompt_tokens=params["prompt_tokens"], max_tokens=params["max_tokens"],
        thinking=params["thinking"], tags={"what": "background"},
    )
    injections: list[dict[str, Any]] = []
    loop.start()
    try:
        await asyncio.sleep(params["warmup_s"])
        for target in targets:
            for rep in range(params["reps"]):
                # Der große Prompt entsteht vor der Ruhephase; sein Bau soll sie nicht stören.
                prompt = await ctx.build(target)
                t_baseline = time.time()
                await asyncio.sleep(params["baseline_s"])
                t_inject = time.time()
                ctx.store.event("injection_start", stage=5, big_target=target, rep=rep)
                rec = await ctx.request(
                    RequestSpec(
                        messages=prompt.messages,
                        max_tokens=params["big_max_tokens"],
                        body=ctx.body_for(params["thinking"]),
                        tags={
                            "stage": 5, "what": "big", "prompt_target": target, "rep": rep,
                            "prompt_local": prompt.local_tokens, "thinking": params["thinking"],
                        },
                    )
                )
                t_end = time.time()
                ctx.store.event("injection_end", stage=5, big_target=target, rep=rep, ok=rec["ok"])
                await asyncio.sleep(params["recovery_s"])
                injections.append(
                    {
                        "target": target, "rep": rep, "record": rec, "t_baseline": t_baseline,
                        "t_inject": t_inject, "t_end": t_end, "t_recovered": time.time(),
                    }
                )
        background = await loop.stop(params["drain_s"])
    finally:
        await loop.abort()

    rows = [analyze(background, inj) for inj in injections]
    summary = {
        "chats": params["chats"],
        "rows": rows,
        "skipped": skipped,
        "background_requests": len(background),
        "background_errors": sum(1 for r in background if not r["ok"] and r["error"] != "abgebrochen"),
    }
    ctx.store.write_json("summary_s5.json", summary)

    ctx.log(
        f"  {'große Anfrage':>14} {'ihr TTFT':>9} {'Chats tok/s: vorher':>20} {'beim Einlesen':>14} "
        f"{'Einbruch':>9} {'längste Pause':>14}"
    )
    for row in rows:
        if not row["big_ok"]:
            ctx.log(f"  {row['big_target']:>14} FEHLER: {row['big_error']}")
            continue
        base, prefill = row["baseline"] or {}, row["prefill"] or {}
        ctx.log(
            f"  {row['big_target']:>14} {fmt(row['big_ttft_s'], 1, ' s'):>9} {fmt(base.get('tps'), 1):>20} "
            f"{fmt(prefill.get('tps'), 1):>14} {fmt(row['slowdown_prefill_pct'], 0, ' %'):>9} "
            f"{fmt(prefill.get('silence_max_s'), 2, ' s'):>14}"
        )
    for skip in skipped:
        ctx.log(f"  ausgelassen: {skip['big_target']} Tokens, {skip['reason']}")
    return summary
