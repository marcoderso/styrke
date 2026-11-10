"""Streaming-Client für OpenAI-kompatible Chat-Schnittstellen.

Eine Anfrage wird abgeschickt, die Antwort gestreamt und jedes eintreffende
Stück mit einem Zeitstempel versehen. Der Antworttext selbst wird nicht
aufbewahrt, außer der Aufrufer verlangt es ausdrücklich.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import aiohttp


@dataclass
class RequestSpec:
    """Was abgeschickt wird."""

    messages: list[dict[str, Any]]
    max_tokens: int
    body: dict[str, Any] = field(default_factory=dict)  # zusätzliche Felder im Anfragekörper
    tags: dict[str, Any] = field(default_factory=dict)  # wandern unverändert in den Datensatz
    keep_text: bool = False


def usage_fields(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Zieht die Zählwerte aus dem usage-Block; Engines benennen sie leicht verschieden."""
    if not isinstance(usage, dict):
        return {}
    out: dict[str, Any] = {}
    out["prompt_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens"))
    out["completion_tokens"] = usage.get("completion_tokens", usage.get("output_tokens"))
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if cached is None:
        cached = usage.get("cached_tokens", usage.get("cache_read_input_tokens"))
    out["cached_tokens"] = cached
    cdetails = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    out["reasoning_tokens"] = cdetails.get("reasoning_tokens") if isinstance(cdetails, dict) else None
    return out


class SseParser:
    """Zerlegt einen Bytestrom in die JSON-Objekte der ``data:``-Zeilen."""

    def __init__(self) -> None:
        self._buf = b""
        self.done = False

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        self._buf += chunk
        events: list[dict[str, Any]] = []
        while True:
            pos = self._buf.find(b"\n")
            if pos < 0:
                break
            line = self._buf[:pos].rstrip(b"\r")
            self._buf = self._buf[pos + 1 :]
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                self.done = True
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
        return events


def delta_texts(event: dict[str, Any]) -> tuple[str, str, bool, str | None]:
    """Liefert (Denk-Text, Antwort-Text, Werkzeugaufruf?, finish_reason) eines Stream-Ereignisses."""
    choices = event.get("choices") or []
    if not choices:
        return "", "", False, None
    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
    content = delta.get("content") or ""
    if not isinstance(reasoning, str):
        reasoning = ""
    if not isinstance(content, str):
        content = ""
    tool = bool(delta.get("tool_calls"))
    return reasoning, content, tool, choice.get("finish_reason")


async def run_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    spec: RequestSpec,
    *,
    sampling: dict[str, Any] | None = None,
    timeout_s: float = 1800,
    include_usage: bool = True,
    count_tokens: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Schickt eine Anfrage und liefert den Messdatensatz.

    Zeiten sind Sekunden ab dem Absenden. ``chunk_ms`` enthält für jedes
    Antwortstück den Zeitpunkt in Millisekunden, ``chunk_kinds`` je Stück ein
    Zeichen: r = Nachdenken, c = Antwort, t = Werkzeugaufruf.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": spec.messages,
        "max_tokens": spec.max_tokens,
        "stream": True,
    }
    body.update(sampling or {})
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    body.update(spec.body)

    rec: dict[str, Any] = {
        "id": uuid.uuid4().hex[:12],
        "ts_submit": round(time.time(), 4),
        **spec.tags,
        "max_tokens": spec.max_tokens,
        "ok": False,
        "http_status": None,
        "error": None,
        "ttft_s": None,
        "ttfr_s": None,
        "ttfc_s": None,
        "total_s": None,
        "n_chunks": 0,
        "chunk_ms": [],
        "chunk_kinds": "",
        "chars_reasoning": 0,
        "chars_content": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "reasoning_tokens": None,
        "usage_source": None,
        "usage": None,
        "finish_reason": None,
        "decode_tps": None,
    }
    chunk_ms: list[int] = []
    kinds: list[str] = []
    texts: list[str] = []
    need_text = spec.keep_text or count_tokens is not None
    usage: dict[str, Any] | None = None
    parser = SseParser()
    t0 = time.perf_counter()

    def finish() -> dict[str, Any]:
        rec["total_s"] = round(time.perf_counter() - t0, 4)
        rec["n_chunks"] = len(chunk_ms)
        rec["chunk_ms"] = chunk_ms
        rec["chunk_kinds"] = "".join(kinds)
        rec["usage"] = usage  # unverändert, falls die Engine mehr meldet als die Felder oben
        fields = usage_fields(usage)
        for key, value in fields.items():
            if value is not None:
                rec[key] = value
        if rec["completion_tokens"] is not None:
            rec["usage_source"] = "stream"
        elif chunk_ms:
            if count_tokens is not None:
                rec["completion_tokens"] = count_tokens("".join(texts))
                rec["usage_source"] = "tokenizer"
            else:
                rec["completion_tokens"] = len(chunk_ms)
                rec["usage_source"] = "chunks"
        ttft, total, ctok = rec["ttft_s"], rec["total_s"], rec["completion_tokens"]
        if ttft is not None and ctok and ctok > 1 and total > ttft:
            rec["decode_tps"] = round((ctok - 1) / (total - ttft), 3)
        if spec.keep_text:
            rec["_text"] = "".join(texts)
        return rec

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=timeout_s)
    url = base_url.rstrip("/") + "/v1/chat/completions"
    try:
        async with session.post(url, json=body, timeout=timeout) as resp:
            rec["http_status"] = resp.status
            if resp.status != 200:
                text = await resp.text()
                rec["error"] = f"HTTP {resp.status}: {text[:500]}"
                return finish()
            async for raw in resp.content.iter_any():
                now = time.perf_counter() - t0
                for event in parser.feed(raw):
                    if event.get("error"):
                        rec["error"] = f"Fehler im Stream: {json.dumps(event['error'], ensure_ascii=False)[:500]}"
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    reasoning, content, tool, reason = delta_texts(event)
                    if reason:
                        rec["finish_reason"] = reason
                    if not (reasoning or content or tool):
                        continue
                    if rec["ttft_s"] is None:
                        rec["ttft_s"] = round(now, 4)
                    if reasoning:
                        if rec["ttfr_s"] is None:
                            rec["ttfr_s"] = round(now, 4)
                        rec["chars_reasoning"] += len(reasoning)
                        kinds.append("r")
                    elif content:
                        if rec["ttfc_s"] is None:
                            rec["ttfc_s"] = round(now, 4)
                        rec["chars_content"] += len(content)
                        kinds.append("c")
                    else:
                        kinds.append("t")
                    chunk_ms.append(int(now * 1000))
                    if need_text:
                        texts.append(reasoning or content)
            if rec["error"] is None:
                rec["ok"] = bool(chunk_ms) or rec["finish_reason"] is not None
                if not rec["ok"]:
                    rec["error"] = "leere Antwort"
    except asyncio.CancelledError:
        rec["error"] = "abgebrochen"
        finish()
        raise CancelledWithRecord(rec) from None
    except asyncio.TimeoutError:
        rec["error"] = f"Zeitüberschreitung nach {timeout_s:.0f} s ohne Daten"
    except aiohttp.ClientError as exc:
        rec["error"] = f"Verbindungsfehler: {type(exc).__name__}: {exc}"
    return finish()


class CancelledWithRecord(asyncio.CancelledError):
    """Abbruch einer laufenden Anfrage; trägt den bis dahin gemessenen Datensatz."""

    def __init__(self, record: dict[str, Any]):
        super().__init__()
        self.record = record


async def complete(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    body: dict[str, Any] | None = None,
    timeout_s: float = 600,
) -> tuple[int, dict[str, Any] | str]:
    """Nicht gestreamte Anfrage; liefert (HTTP-Status, JSON oder Fehlertext)."""
    payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    payload.update(body or {})
    timeout = aiohttp.ClientTimeout(total=timeout_s, sock_connect=15)
    url = base_url.rstrip("/") + "/v1/chat/completions"
    async with session.post(url, json=payload, timeout=timeout) as resp:
        text = await resp.text()
        if resp.status != 200:
            return resp.status, text[:500]
        try:
            return resp.status, json.loads(text)
        except json.JSONDecodeError:
            return resp.status, text[:500]
