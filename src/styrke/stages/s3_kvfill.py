"""Stufe 3, KV-Füllstand: Einfluss eines vollen Cache.

Der KV-Pool der Engine wird schrittweise mit gehaltenen Gesprächen belegt:
Jedes Gespräch beginnt mit einem langen Prompt, dessen Zwischenstand die
Engine im Cache behält. Je Füllstand läuft danach eine Kurzform der Stufe 2
mit frischen, kurzen Anfragen. Zum Schluss bekommt ein Teil der gehaltenen
Gespräche eine Folgefrage; an ihrer Antwortzeit zeigt sich, ob der Cache das
Gespräch unter Last behalten oder verdrängt hat.

Die Füllstände bauen aufeinander auf, die Engine startet dazwischen nicht neu.
Der Pool enthält deshalb außer den gehaltenen Gesprächen auch die Reste der
kurzen Anfragen; wie voll er laut Engine wirklich ist, steht in den
mitgeschriebenen Kennzahlen (``engine_metrics``).
"""

from __future__ import annotations

from typing import Any

from ..stats import median
from .common import StageContext, ensure_calibrated, fmt
from .load import Chat, cached_share, looks_cached
from .s2_concurrency import run_level

MIN_SESSIONS_IN_POOL = 20  # ein gehaltenes Gespräch belegt höchstens ein Zwanzigstel des Pools
RESERVE_TOKENS = 1024  # Platz für Folgefragen und kurze Antworten eines gehaltenen Gesprächs


def held_tokens(held: list[Chat]) -> int:
    return sum(chat.last_prompt_tokens or chat.first_target for chat in held)


async def fill_to(ctx: StageContext, held: list[Chat], target_tokens: int, fill: float, params: dict[str, Any]) -> int:
    """Eröffnet nacheinander Gespräche, bis zusammen ``target_tokens`` gehalten sind. Liefert die Fehlerzahl."""
    errors = 0
    # Nur so viele Gespräche, wie unter den Füllstand passen: Ein weiteres würde ihn überschreiten.
    while held_tokens(held) + params["session_tokens"] <= target_tokens * 1.02 and errors < 3:
        chat = Chat(ctx, stage=3, thinking=params["thinking"], tags={"fill": fill})
        await chat.open(params["session_tokens"])
        rec = await chat.send(params["probe_max_tokens"], what="fill")
        if not rec["ok"]:
            errors += 1
            ctx.log(f"  Füllen fehlgeschlagen: {rec['error']}")
            continue
        held.append(chat)
    return errors


async def probe(ctx: StageContext, held: list[Chat], fill: float, params: dict[str, Any]) -> dict[str, Any]:
    """Folgefrage an eine Auswahl gehaltener Gespräche: Sind sie noch im Cache?"""
    count = min(len(held), params["probe_sessions"])
    if count == 0:
        return {"probe_n": 0}
    step = len(held) / count
    sample = [held[int(i * step)] for i in range(count)]  # gleichmäßig von alt bis neu
    ttfts: list[float] = []
    shares: list[float] = []
    verdicts: list[bool] = []
    for chat in sample:
        cold = chat.cold_ttft_s  # vor dem Senden lesen: gilt dem ersten Zug des Gesprächs
        await chat.add_followup()
        rec = await chat.send(params["probe_max_tokens"], what="probe", fill=fill)
        if not rec["ok"]:
            continue
        ttfts.append(rec["ttft_s"])
        share = cached_share(rec)
        if share is not None:
            shares.append(share)
        verdict = looks_cached(rec["ttft_s"], cold)
        if verdict is not None:
            verdicts.append(verdict)
    return {
        "probe_n": len(ttfts),
        "probe_ttft_p50_s": median(ttfts),
        # Behalten nach Antwortzeit: Die Folgefrage kam in höchstens der halben Zeit des kalten Prompts.
        # Leer, wenn schon der kalte Prompt zu schnell war, um das zu unterscheiden.
        "retained_by_ttft": round(sum(verdicts) / len(verdicts), 3) if verdicts else None,
        # Behalten laut Engine: mindestens der halbe Prompt kam aus dem Cache. Leer, wenn sie nichts meldet.
        "retained_by_cache_report": round(sum(1 for s in shares if s >= 0.5) / len(shares), 3) if shares else None,
        "cached_share_p50": median(shares),
    }


async def run(ctx: StageContext) -> dict[str, Any]:
    params = ctx.params["s3"]
    await ensure_calibrated(ctx)
    pool = params.get("pool_tokens") or ctx.kv_pool_tokens
    if not pool:
        summary = {
            "skipped": "Die Größe des KV-Pools ist nicht bekannt. In der Testmatrix unter stages.s3.pool_tokens angeben."
        }
        ctx.store.write_json("summary_s3.json", summary)
        ctx.log(f"  ausgelassen: {summary['skipped']}")
        return summary

    session_tokens = params["session_tokens"]
    if not ctx.fits(session_tokens, RESERVE_TOKENS):
        session_tokens = min(ctx.prompt_limit(), ctx.model.max_context - RESERVE_TOKENS)
        ctx.log(f"  Gespräche auf {session_tokens} Tokens gekürzt, damit sie in den Kontext passen")
    if session_tokens > pool // MIN_SESSIONS_IN_POOL:
        # Bei kleinem Pool wären wenige große Gespräche zu grob: Der Füllstand ließe sich nicht treffen.
        session_tokens = max(1000, pool // MIN_SESSIONS_IN_POOL)
        ctx.log(f"  Gespräche auf {session_tokens} Tokens verkleinert, damit sich der Füllstand des Pools treffen lässt")
    params = {**params, "session_tokens": session_tokens}

    held: list[Chat] = []
    fills: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    ctx.log(f"  KV-Pool: {pool} Tokens, gehaltene Gespräche zu je {session_tokens} Tokens")
    for fill in params["fills"]:
        ctx.store.event("fill_start", stage=3, fill=fill)
        errors = await fill_to(ctx, held, int(fill * pool), fill, params)
        info: dict[str, Any] = {
            "fill": fill,
            "held_sessions": len(held),
            "held_tokens": held_tokens(held),
            "fill_errors": errors,
            "cold_ttft_p50_s": median([c.cold_ttft_s for c in held]),
            "engine_metrics_after_fill": await ctx.adapter.metrics(ctx.session),
        }
        ctx.store.event("fill_done", stage=3, fill=fill, held_sessions=len(held), held_tokens=info["held_tokens"])
        ctx.log(
            f"  Füllstand {fill:.0%}: {len(held)} Gespräche mit zusammen {info['held_tokens']} Tokens gehalten"
            + (f" ({errors} Fehler beim Füllen)" if errors else "")
        )

        ctx.log(f"  {'parallel':>10} {'tok/s gesamt':>13} {'tok/s je Anfrage':>17} {'TTFT p50':>9} {'TTFT p95':>9} {'Fehler':>7}")
        for n in params["levels"]:
            ctx.store.event("level_start", stage=3, level=n, fill=fill)
            row = {"fill": fill, **await run_level(ctx, n, params, stage=3, tags={"fill": fill})}
            rows.append(row)
            ctx.store.event("level_end", stage=3, level=n, fill=fill, throughput_tps=row["throughput_tps"])
            ctx.log(
                f"  {n:>10} {row['throughput_tps']:>13.1f} {fmt(row['decode_tps_p50'], 1):>17} "
                f"{fmt(row['ttft_p50_s'], 2, ' s'):>9} {fmt(row['ttft_p95_s'], 2, ' s'):>9} {row['errors']:>7}"
            )

        info["engine_metrics_after_load"] = await ctx.adapter.metrics(ctx.session)
        info.update(await probe(ctx, held, fill, params))
        if info["probe_n"]:
            kept = info["retained_by_ttft"]
            if kept is None:
                kept = info["retained_by_cache_report"]
            ctx.log(
                f"  Folgefrage an {info['probe_n']} gehaltene Gespräche: erstes Token nach "
                f"{fmt(info['probe_ttft_p50_s'], 2, ' s')} (kalt: {fmt(info['cold_ttft_p50_s'], 2, ' s')}), "
                f"noch im Cache: {fmt(None if kept is None else 100 * kept, 0, ' %')}"
            )
        fills.append(info)

    summary = {"pool_tokens": pool, "session_tokens": session_tokens, "fills": fills, "rows": rows}
    ctx.store.write_json("summary_s3.json", summary)
    return summary
