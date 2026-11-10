"""Stufe 0, Machbarkeit: Lädt die Zelle, antwortet sie sinnvoll, welcher Kontext geht?"""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from ..client import RequestSpec
from ..sampler import gpu_snapshot
from .common import StageContext, ensure_calibrated, fmt

SANITY_QUESTION = "Nenne drei europäische Hauptstädte und jeweils den Fluss, an dem sie liegen. Antworte in drei kurzen Sätzen."


async def _get_json(session: aiohttp.ClientSession, url: str) -> Any:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return None
            return await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError):
        return None


async def run(ctx: StageContext) -> dict[str, Any]:
    params = ctx.params["s0"]
    info: dict[str, Any] = {"loaded": True, "load_time_s": ctx.adapter.load_time_s}
    info["models_endpoint"] = await _get_json(ctx.session, ctx.adapter.base_url + "/v1/models")

    # Sinnprobe. Lehnt die Engine das Feld stream_options ab, wird ohne es wiederholt.
    def sanity_spec(thinking: str, what: str) -> RequestSpec:
        prefix = ctx.system_prefix_for(thinking)
        messages = ([{"role": "system", "content": prefix}] if prefix else []) + [
            {"role": "user", "content": SANITY_QUESTION}
        ]
        return RequestSpec(
            messages=messages,
            max_tokens=params["sanity_max_tokens"],
            body=ctx.body_for(thinking),
            tags={"stage": 0, "what": what, "thinking": thinking},
            keep_text=True,
        )

    rec = await ctx.request(sanity_spec("off", "sanity"))
    if not rec["ok"] and rec["http_status"] in (400, 422):
        ctx.usage_in_stream = False
        retry = await ctx.request(sanity_spec("off", "sanity"))
        if retry["ok"]:
            rec = retry
        else:
            ctx.usage_in_stream = True  # lag nicht an stream_options
    info["sanity_ok"] = rec["ok"]
    info["sanity_error"] = rec["error"]
    info["usage_in_stream"] = rec["usage_source"] == "stream"
    if rec["ok"] and not info["usage_in_stream"]:
        ctx.usage_in_stream = False
    ctx.store.write_json("answers/s0_sanity.json", {"question": SANITY_QUESTION, "answer": rec.get("_text")})
    if not rec["ok"]:
        info["loaded"] = False
        ctx.store.write_json("feasibility.json", info)
        ctx.log(f"  Sinnprobe fehlgeschlagen: {rec['error']}")
        return info

    # Lässt sich das Nachdenken schalten?
    rec_on = await ctx.request(sanity_spec("on", "sanity_thinking"))
    info["thinking"] = {
        "off_has_reasoning": rec["chars_reasoning"] > 0,
        "on_has_reasoning": rec_on["chars_reasoning"] > 0,
        "on_ok": rec_on["ok"],
    }

    info["calibration"] = await ensure_calibrated(ctx, force=True)

    # Kontextproben: steigende Promptlängen bis zum ersten Fehlschlag.
    probes: list[dict[str, Any]] = []
    for target in params["context_probes"]:
        if target + 64 > ctx.model.max_context:
            probes.append({"target": target, "skipped": "über max_context des Modells"})
            continue
        prompt = ctx.builder.build(target, seed=ctx.next_seed())
        rec = await ctx.request(
            RequestSpec(
                messages=prompt.messages,
                max_tokens=8,
                body=ctx.body_for("off"),
                tags={"stage": 0, "what": "context_probe", "prompt_target": target, "prompt_local": prompt.local_tokens},
            )
        )
        probes.append(
            {
                "target": target,
                "ok": rec["ok"],
                "prompt_tokens": rec["prompt_tokens"],
                "ttft_s": rec["ttft_s"],
                "error": rec["error"],
            }
        )
        ctx.log(
            f"  Kontextprobe {target:>7} Tokens: "
            + (f"ok, erstes Token nach {fmt(rec['ttft_s'], 1, ' s')}" if rec["ok"] else f"FEHLER: {rec['error']}")
        )
        if not rec["ok"]:
            break
    info["context_probes"] = probes
    ok_lengths = [p["prompt_tokens"] or p["target"] for p in probes if p.get("ok")]  # ohne Zählwert der Engine: Zielgröße
    info["max_context_ok"] = max(ok_lengths) if ok_lengths else None

    info["gpu"] = gpu_snapshot()
    info["engine_metrics"] = await ctx.adapter.metrics(ctx.session)
    ctx.store.write_json("feasibility.json", info)

    cal = info["calibration"]
    ctx.log(
        f"  geladen in {fmt(info['load_time_s'], 0, ' s')}, Zählwerte im Stream: "
        f"{'ja' if info['usage_in_stream'] else 'nein'}, Promptlänge trifft auf "
        f"{fmt(cal.get('check_deviation_pct'), 2, ' %')} genau, größter geprüfter Kontext: {info['max_context_ok']}"
    )
    return info
