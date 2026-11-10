"""Gemeinsames Werkzeug der Stufen: Kontext, Anfragen, Eichung der Promptlänge."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

import aiohttp

from ..client import CancelledWithRecord, RequestSpec, complete, run_request
from ..config import ModelConfig
from ..engines.base import EngineAdapter
from ..prompts import Prompt, PromptBuilder
from ..store import RunStore


@dataclass
class StageContext:
    store: RunStore
    adapter: EngineAdapter
    session: aiohttp.ClientSession
    model: ModelConfig
    params: dict[str, dict[str, Any]]
    builder: PromptBuilder
    log: Callable[[str], None] = print
    usage_in_stream: bool = True
    attempt: int = 1  # der wievielte Anlauf dieser Stufe (nach einem Abbruch größer als 1)
    load_model: dict[str, Any] = field(default_factory=dict)  # Abschnitt load_model der Testmatrix
    kv_pool_tokens: int | None = None  # Größe des KV-Pools der Engine in Tokens, soweit bekannt
    _seed: int = field(default=0, repr=False)

    # -- Bausteine einer Anfrage ----------------------------------------

    def body_for(self, thinking: str) -> dict[str, Any]:
        """Zusätzliche Felder im Anfragekörper, die das Nachdenken an- oder abschalten."""
        return dict(self.model.thinking_mode(thinking).get("body") or {})

    def system_prefix_for(self, thinking: str) -> str | None:
        return self.model.thinking_mode(thinking).get("system_prefix")

    def sampling(self) -> dict[str, Any]:
        common = self.params["common"]
        return {key: common[key] for key in ("temperature", "top_p") if common.get(key) is not None}

    def next_seed(self) -> int:
        self._seed += 1
        return self._seed

    async def build(self, target_tokens: int, **kwargs: Any) -> Prompt:
        """Baut einen Prompt mit neuem Startwert.

        Lange Prompts entstehen in einem Nebenthread: Das Zählen von einigen
        hunderttausend Tokens dauert spürbar, und in dieser Zeit dürfen die
        Zeitstempel gleichzeitig laufender Anfragen nicht stocken.
        """
        seed = self.next_seed()
        if target_tokens < 4000:
            return self.builder.build(target_tokens, seed=seed, **kwargs)
        return await asyncio.to_thread(self.builder.build, target_tokens, seed=seed, **kwargs)

    def prompt_limit(self) -> int:
        """Größte Promptlänge, die nach Stufe 0 als lauffähig gilt."""
        limit = self.model.max_context
        probes = (self.store.read_json("feasibility.json") or {}).get("context_probes") or []
        failed = [p["target"] for p in probes if p.get("ok") is False]
        if failed:
            ok = [p["target"] for p in probes if p.get("ok")]
            limit = min(limit, max(ok) if ok else min(failed) - 1)
        return limit

    def fits(self, prompt_tokens: int, reserve_tokens: int = 0) -> bool:
        """Passt ein Prompt samt ``reserve_tokens`` (Antworten, Folgefragen) in den Kontext?"""
        return prompt_tokens <= self.prompt_limit() and prompt_tokens + reserve_tokens <= self.model.max_context

    def token_counter(self) -> Callable[[str], int] | None:
        """Zählfunktion für Antworten, falls die Engine keine Zählwerte mitliefert."""
        counter = self.builder.counter
        return counter.count if counter.kind == "tokenizer" else None

    # -- Anfrage abschicken und festhalten --------------------------------

    async def request(self, spec: RequestSpec) -> dict[str, Any]:
        """Schickt eine Anfrage, schreibt den Datensatz und gibt ihn zurück."""
        spec.tags.setdefault("attempt", self.attempt)
        try:
            rec = await run_request(
                self.session,
                self.adapter.base_url,
                self.adapter.served_name,
                spec,
                sampling=self.sampling(),
                timeout_s=self.params["common"]["request_timeout_s"],
                include_usage=self.usage_in_stream,
                count_tokens=None if self.usage_in_stream else self.token_counter(),
            )
        except CancelledWithRecord as exc:
            exc.record.pop("_text", None)
            self.store.requests.write(exc.record)
            raise
        text = rec.pop("_text", None)
        self.store.requests.write(rec)
        if text is not None:
            rec["_text"] = text
        return rec


async def calibrate(ctx: StageContext) -> dict[str, Any]:
    """Eicht die Promptlänge an der Engine.

    Mit Tokenizer wird nur der Aufschlag des Chat-Templates bestimmt. Ohne
    Tokenizer wird zusätzlich das Zeichen-pro-Token-Verhältnis gemessen.
    Danach prüft eine zweite Anfrage, wie genau die Zielgröße getroffen wird.
    """
    builder = ctx.builder
    counter = builder.counter
    builder.overhead_tokens = 0
    result: dict[str, Any] = {"counter": counter.kind, "ok": False}

    async def measure(target: int) -> tuple[Any, int | None]:
        prompt = builder.build(target, seed=ctx.next_seed())
        status, data = await complete(
            ctx.session, ctx.adapter.base_url, ctx.adapter.served_name,
            prompt.messages, max_tokens=1, body=ctx.body_for("off"),
        )
        actual = None
        if status == 200 and isinstance(data, dict):
            usage = data.get("usage") or {}
            actual = usage.get("prompt_tokens", usage.get("input_tokens"))
        return prompt, actual

    prompt, actual = await measure(800)
    if not actual:
        result["note"] = "Engine meldet keine Promptlänge; Längen bleiben ungeeicht"
        return result

    if counter.kind == "tokenizer":
        builder.overhead_tokens = max(0, actual - prompt.local_tokens)
    else:
        assumed_overhead = 12
        chars = sum(len(m["content"]) for m in prompt.messages)
        counter.chars_per_token = chars / max(1, actual - assumed_overhead)
        builder.overhead_tokens = assumed_overhead
        result["chars_per_token"] = round(counter.chars_per_token, 4)
    result["overhead_tokens"] = builder.overhead_tokens

    check, check_actual = await measure(2000)
    result["check_target"] = check.target_tokens
    result["check_actual"] = check_actual
    result["ok"] = bool(check_actual)
    if check_actual:
        result["check_deviation_pct"] = round(100 * (check_actual - check.target_tokens) / check.target_tokens, 2)
    return result


async def ensure_calibrated(ctx: StageContext, force: bool = False) -> dict[str, Any]:
    """Eicht einmal je Lauf; spätere Stufen übernehmen die gespeicherten Werte.

    Übernimmt auch, was Stufe 0 über die Zählwerte im Stream herausgefunden
    hat; das zählt, wenn eine spätere Stufe in einem eigenen Aufruf läuft.
    """
    if not force and (ctx.store.read_json("feasibility.json") or {}).get("usage_in_stream") is False:
        ctx.usage_in_stream = False
    saved = None if force else ctx.store.read_json("calibration.json")
    if saved and saved.get("ok") and saved.get("counter") == ctx.builder.counter.kind:
        ctx.builder.overhead_tokens = int(saved.get("overhead_tokens") or 0)
        if saved.get("chars_per_token") and ctx.builder.counter.kind == "ratio":
            ctx.builder.counter.chars_per_token = float(saved["chars_per_token"])
        return saved
    result = await calibrate(ctx)
    ctx.store.write_json("calibration.json", result)
    return result


def fmt(value: float | None, digits: int = 1, unit: str = "") -> str:
    """Zahl für die Konsolenausgabe; ``-`` wenn nichts gemessen wurde."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}{unit}"
