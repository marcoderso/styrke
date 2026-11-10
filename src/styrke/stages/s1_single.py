"""Stufe 1, Einzelstrom: Grundwerte ohne Konkurrenz, über die Promptlänge."""

from __future__ import annotations

import asyncio
from typing import Any

from ..client import RequestSpec
from ..stats import median
from .common import StageContext, ensure_calibrated, fmt


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fasst die Wiederholungen je Denkmodus und Promptlänge zusammen (Mediane)."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault((rec["thinking"], rec["prompt_target"]), []).append(rec)
    rows = []
    for (thinking, target), recs in sorted(groups.items(), key=lambda kv: (kv[0][0] != "off", kv[0][1])):
        ok = [r for r in recs if r["ok"]]
        prefill = []
        for r in ok:
            fresh = (r["prompt_tokens"] or r.get("prompt_local") or 0) - (r["cached_tokens"] or 0)
            if r["ttft_s"] and fresh > 0:
                prefill.append(fresh / r["ttft_s"])
        rows.append(
            {
                "thinking": thinking,
                "prompt_target": target,
                "n": len(recs),
                "n_ok": len(ok),
                "prompt_tokens": median([r["prompt_tokens"] or r.get("prompt_local") for r in ok]),
                "ttft_s": median([r["ttft_s"] for r in ok]),
                "ttfc_s": median([r["ttfc_s"] for r in ok]),
                "decode_tps": median([r["decode_tps"] for r in ok]),
                "prefill_tps": median(prefill),
                "total_s": median([r["total_s"] for r in ok]),
                "completion_tokens": median([r["completion_tokens"] for r in ok if r["completion_tokens"]]),
                "errors": sorted({r["error"] for r in recs if r["error"]}),
            }
        )
    return rows


async def run(ctx: StageContext) -> dict[str, Any]:
    params = ctx.params["s1"]
    await ensure_calibrated(ctx)
    feasibility = ctx.store.read_json("feasibility.json") or {}
    limit = ctx.model.max_context
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for thinking in params["thinking"]:
        if thinking == "on" and not ctx.model.thinking_mode("on"):
            skipped.append({"thinking": thinking, "reason": "Modell hat keinen Denkmodus in der Matrix"})
            continue
        max_tokens = params["max_tokens"][thinking]
        for target in params["lengths"]:
            if target + max_tokens + 256 > limit:
                skipped.append({"thinking": thinking, "prompt_target": target, "reason": "passt nicht in den Kontext"})
                continue
            failed_probe = next(
                (p for p in feasibility.get("context_probes", []) if p.get("ok") is False and p["target"] <= target),
                None,
            )
            if failed_probe:
                skipped.append(
                    {"thinking": thinking, "prompt_target": target, "reason": "Kontextprobe in Stufe 0 gescheitert"}
                )
                continue
            for rep in range(params["reps"]):
                prompt = ctx.builder.build(target, seed=ctx.next_seed(), system_prefix=ctx.system_prefix_for(thinking))
                rec = await ctx.request(
                    RequestSpec(
                        messages=prompt.messages,
                        max_tokens=max_tokens,
                        body=ctx.body_for(thinking),
                        tags={
                            "stage": 1, "thinking": thinking, "prompt_target": target, "rep": rep,
                            "prompt_local": prompt.local_tokens,
                        },
                    )
                )
                records.append(rec)
                ctx.log(
                    f"  Thinking {thinking:<3} {target:>7} Tokens, Lauf {rep + 1}: "
                    + (
                        f"erstes Token {fmt(rec['ttft_s'], 2, ' s')}, {fmt(rec['decode_tps'], 1)} tok/s"
                        if rec["ok"]
                        else f"FEHLER: {rec['error']}"
                    )
                )
                await asyncio.sleep(params["pause_s"])

    rows = summarize(records)
    summary = {"rows": rows, "skipped": skipped}
    ctx.store.write_json("summary_s1.json", summary)

    ctx.log("")
    ctx.log(f"  {'Thinking':<8} {'Prompt':>8} {'TTFT s':>8} {'Antwort s':>10} {'tok/s':>7} {'Einlesen tok/s':>15} {'ok':>5}")
    for row in rows:
        ctx.log(
            f"  {row['thinking']:<8} {row['prompt_target']:>8} {fmt(row['ttft_s'], 2):>8} {fmt(row['ttfc_s'], 2):>10} "
            f"{fmt(row['decode_tps'], 1):>7} {fmt(row['prefill_tps'], 0):>15} {row['n_ok']}/{row['n']:<3}"
        )
    return summary
