"""Gemeinsames Werkzeug der Stufen: Kontext, Anfragen, Eichung der Promptlänge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import aiohttp

from ..client import CancelledWithRecord, RequestSpec, complete, run_request
from ..config import ModelConfig
from ..engines.base import EngineAdapter
from ..prompts import PromptBuilder
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
    """Eicht einmal je Lauf; spätere Stufen übernehmen die gespeicherten Werte."""
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
