"""Stufe 4, Präfix-Cache: Nutzen des Cache bei Folgefragen.

Je Promptlänge beginnt ein Gespräch kalt, also mit einem Prompt, den die
Engine noch nie gesehen hat. Darauf folgt sofort eine Folgefrage; ihr Anfang
ist der Engine bekannt, sie muss nur den neuen Teil einlesen. Der Unterschied
der Zeiten bis zum ersten Token ist der Nutzen des Cache.

Zwei Zusatzmessungen zeigen, wie lange der Nutzen hält:

- nach einer Pause: eine weitere Folgefrage nach ``pause_s`` Sekunden Ruhe;
- nach Verdrängung: Zwischen zwei Folgefragen wird mehr fremder Text durch
  die Engine geschoben, als der KV-Pool fasst.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..client import RequestSpec
from ..stats import median
from .common import StageContext, ensure_calibrated, fmt
from .load import Chat, cached_share, looks_cached


def _ratio(cold: float | None, warm: float | None) -> float | None:
    return round(cold / warm, 1) if cold and warm else None


def summarize_length(target: int, samples: list[dict[str, Any]], pause_s: float) -> dict[str, Any]:
    """Mediane einer Promptlänge über alle Wiederholungen."""

    def ttfts(key: str) -> list[float]:
        return [s[key]["ttft_s"] for s in samples if s.get(key) and s[key]["ok"]]

    def shares(key: str) -> list[float]:
        values = [cached_share(s[key]) for s in samples if s.get(key) and s[key]["ok"]]
        return [v for v in values if v is not None]

    cold, warm, paused = median(ttfts("cold")), median(ttfts("followup")), median(ttfts("after_pause"))
    prompt_tokens = [s["cold"]["prompt_tokens"] for s in samples if s.get("cold") and s["cold"].get("prompt_tokens")]
    return {
        "prompt_target": target,
        "prompt_tokens": median(prompt_tokens),
        "n": len(ttfts("followup")),
        "errors": sum(1 for s in samples for rec in s.values() if rec is not None and not rec["ok"]),
        "cold_ttft_s": cold,
        "followup_ttft_s": warm,
        "speedup": _ratio(cold, warm),
        "followup_cached_share": median(shares("followup")),
        "pause_s": pause_s,
        "after_pause_ttft_s": paused,
        "after_pause_cached_share": median(shares("after_pause")),
    }


async def measure_lengths(ctx: StageContext, params: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    max_tokens = params["max_tokens"]
    reserve = 3 * (max_tokens + 128) + 256  # drei Züge mit Antwort und Folgefrage
    for target in params["lengths"]:
        if not ctx.fits(target, reserve):
            skipped.append({"prompt_target": target, "reason": "passt nicht in den Kontext"})
            continue
        samples: list[dict[str, Any]] = []
        for rep in range(params["reps"]):
            chat = Chat(ctx, stage=4, thinking=params["thinking"], tags={"rep": rep})
            await chat.open(target)
            sample: dict[str, Any] = {"cold": await chat.send(max_tokens, what="cold"), "followup": None, "after_pause": None}
            samples.append(sample)
            if not sample["cold"]["ok"]:
                ctx.log(f"  {target:>7} Tokens, Lauf {rep + 1}: FEHLER: {sample['cold']['error']}")
                break
            await chat.add_followup()
            sample["followup"] = await chat.send(max_tokens, what="followup")
            line = (
                f"  {target:>7} Tokens, Lauf {rep + 1}: kalt {fmt(sample['cold']['ttft_s'], 2, ' s')}, "
                f"Folgefrage {fmt(sample['followup']['ttft_s'], 2, ' s')}"
            )
            if rep == 0 and params["pause_s"] > 0 and sample["followup"]["ok"]:
                await asyncio.sleep(params["pause_s"])
                await chat.add_followup()
                sample["after_pause"] = await chat.send(max_tokens, what="after_pause", idle_s=params["pause_s"])
                line += f", nach {params['pause_s']:g} s Pause {fmt(sample['after_pause']['ttft_s'], 2, ' s')}"
            ctx.log(line)
        rows.append(summarize_length(target, samples, params["pause_s"]))
    return rows, skipped


async def measure_eviction(ctx: StageContext, params: dict[str, Any]) -> dict[str, Any]:
    """Folgefrage, nachdem mehr fremder Text durch die Engine ging, als der KV-Pool fasst."""
    pool = params.get("pool_tokens") or ctx.kv_pool_tokens
    lengths = [n for n in params["evict_lengths"] if ctx.fits(n, 3 * (params["max_tokens"] + 128) + 256)]
    if not lengths:
        return {"skipped": "keine Promptlänge für die Verdrängung vorgesehen oder passend"}
    if not pool:
        return {
            "skipped": "Die Größe des KV-Pools ist nicht bekannt. In der Testmatrix unter stages.s4.pool_tokens angeben."
        }
    max_tokens = params["max_tokens"]
    victims: list[tuple[int, Chat, dict[str, Any]]] = []
    for target in lengths:
        chat = Chat(ctx, stage=4, thinking=params["thinking"], tags={"part": "eviction"})
        await chat.open(target)
        cold = await chat.send(max_tokens, what="cold")
        if not cold["ok"]:
            continue
        await chat.add_followup()
        before = await chat.send(max_tokens, what="before_eviction")
        victims.append((target, chat, before))

    chunk = min(params["evict_chunk_tokens"], ctx.prompt_limit())
    goal = int(params["evict_factor"] * pool)
    ctx.log(f"  Verdrängung: {goal} Tokens fremder Text in Stücken zu {chunk} (KV-Pool: {pool} Tokens)")
    ctx.store.event("flood_start", stage=4, tokens=goal)
    sent = 0
    while sent < goal:
        prompt = await ctx.build(chunk)
        rec = await ctx.request(
            RequestSpec(
                messages=prompt.messages, max_tokens=1, body=ctx.body_for("off"),
                tags={"stage": 4, "what": "flood", "prompt_target": chunk, "prompt_local": prompt.local_tokens},
            )
        )
        if not rec["ok"]:
            ctx.log(f"  Verdrängung abgebrochen: {rec['error']}")
            break
        sent += rec.get("prompt_tokens") or chunk
    ctx.store.event("flood_end", stage=4, tokens=sent)

    rows: list[dict[str, Any]] = []
    for target, chat, before in victims:
        await chat.add_followup()
        after = await chat.send(max_tokens, what="after_eviction")
        cold_ttft = chat.cold_ttft_s
        cached_after = looks_cached(after["ttft_s"], cold_ttft) if after["ok"] else None
        row = {
            "prompt_target": target,
            "cold_ttft_s": cold_ttft,
            "before_ttft_s": before["ttft_s"],
            "after_ttft_s": after["ttft_s"],
            "before_cached_share": cached_share(before),
            "after_cached_share": cached_share(after),
            # Verdrängt nach Antwortzeit: Die Folgefrage braucht wieder mehr als die halbe Zeit des kalten
            # Prompts. Leer, wenn schon der kalte Prompt zu schnell war, um das zu unterscheiden.
            "evicted_by_ttft": None if cached_after is None else not cached_after,
        }
        rows.append(row)
        ctx.log(
            f"  {target:>7} Tokens: kalt {fmt(cold_ttft, 2, ' s')}, Folgefrage vorher {fmt(before['ttft_s'], 2, ' s')}, "
            f"nach Verdrängung {fmt(after['ttft_s'], 2, ' s')}"
        )
    return {"pool_tokens": pool, "flood_tokens": sent, "rows": rows}


async def run(ctx: StageContext) -> dict[str, Any]:
    params = ctx.params["s4"]
    await ensure_calibrated(ctx)
    rows, skipped = await measure_lengths(ctx, params)
    eviction = await measure_eviction(ctx, params)
    if eviction.get("skipped"):
        ctx.log(f"  Verdrängung ausgelassen: {eviction['skipped']}")
    summary = {"rows": rows, "skipped": skipped, "eviction": eviction}
    ctx.store.write_json("summary_s4.json", summary)

    ctx.log("")
    ctx.log(f"  {'Prompt':>8} {'kalt s':>8} {'Folgefrage s':>13} {'Faktor':>7} {'aus Cache':>10} {'nach Pause s':>13}")
    for row in rows:
        share = row["followup_cached_share"]
        ctx.log(
            f"  {row['prompt_target']:>8} {fmt(row['cold_ttft_s'], 2):>8} {fmt(row['followup_ttft_s'], 2):>13} "
            f"{fmt(row['speedup'], 1):>7} {fmt(None if share is None else 100 * share, 0, ' %'):>10} "
            f"{fmt(row['after_pause_ttft_s'], 2):>13}"
        )
    return summary
