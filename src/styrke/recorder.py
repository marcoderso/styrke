"""Mitschreiber: durchreichender Proxy, der nur Längen und Zeiten festhält.

Er sitzt im Testbetrieb zwischen den Nutzern und der Engine, reicht jede
Anfrage unverändert weiter und schreibt je Anfrage eine Zeile mit Kennzahlen:
wie lang war der Prompt, wie lang die Antwort, wie lange hat es gedauert.
Inhalte (Texte von Prompts und Antworten) werden nie gespeichert.

Die Mitschrift liefert echte Werte für das Lastmodell: wie viele Nutzer in der
Spitzenstunde aktiv sind, wie viele Aufrufe sie auslösen und wie lange ein
Aufruf dauert.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from .client import SseParser, delta_texts, usage_fields
from .stats import describe
from .store import JsonlWriter, read_jsonl

# Kopfzeilen, die nur für eine einzelne Verbindung gelten und nicht weitergereicht werden.
REQUEST_DROP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding",
}
RESPONSE_DROP = {"connection", "keep-alive", "transfer-encoding"}
MAX_JSON_BUFFER = 64 * 1024 * 1024
LLM_PATHS = ("/chat/completions", "/completions", "/messages", "/responses")


def _content_chars(content: Any, texts: list[str] | None) -> tuple[int, int]:
    """Zeichenzahl und Bildanzahl eines Nachrichteninhalts (Text oder Liste von Teilen)."""
    if isinstance(content, str):
        if texts is not None:
            texts.append(content)
        return len(content), 0
    chars = images = 0
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                chars += len(part)
                if texts is not None:
                    texts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chars += len(text)
                    if texts is not None:
                        texts.append(text)
                elif part.get("type") in ("image_url", "image", "input_image"):
                    images += 1
    return chars, images


def request_meta(body: bytes, collect_text: bool = False) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """Kennzahlen einer Anfrage. Liefert (Kennzahlen, Texte nur zum Zählen, geparster Körper)."""
    meta: dict[str, Any] = {"req_bytes": len(body)}
    texts: list[str] = []
    try:
        obj = json.loads(body) if body else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return meta, texts, None
    if not isinstance(obj, dict):
        return meta, texts, None
    sink = texts if collect_text else None

    if isinstance(obj.get("model"), str):
        meta["model"] = obj["model"]
    meta["stream"] = bool(obj.get("stream"))
    prompt_chars = system_chars = images = 0
    messages = obj.get("messages")
    if isinstance(messages, list):
        meta["n_messages"] = len(messages)
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            chars, imgs = _content_chars(msg.get("content"), sink)
            if msg.get("tool_calls"):
                chars += len(json.dumps(msg["tool_calls"], ensure_ascii=False))
            images += imgs
            if msg.get("role") in ("system", "developer"):
                system_chars += chars
            prompt_chars += chars
    for key in ("system", "instructions", "prompt", "input"):  # Anthropic-, Responses- und Completions-Format
        chars, imgs = _content_chars(obj.get(key), sink)
        if key in ("system", "instructions"):
            system_chars += chars
        prompt_chars += chars
        images += imgs
    meta["prompt_chars"] = prompt_chars
    meta["system_chars"] = system_chars
    meta["n_images"] = images
    tools = obj.get("tools")
    meta["n_tools"] = len(tools) if isinstance(tools, list) else 0
    if meta["n_tools"]:
        meta["tools_chars"] = len(json.dumps(tools, ensure_ascii=False))
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        if isinstance(obj.get(key), int):
            meta["max_tokens"] = obj[key]
            break
    for key in ("temperature", "top_p", "top_k"):
        if isinstance(obj.get(key), (int, float)) and not isinstance(obj.get(key), bool):
            meta[key] = obj[key]
    kwargs = obj.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and isinstance(kwargs.get("enable_thinking"), bool):
        meta["enable_thinking"] = kwargs["enable_thinking"]
    if isinstance(obj.get("user"), str):
        meta["_user"] = obj["user"]
    return meta, texts, obj


class ResponseObserver:
    """Liest die Antwort mit, ohne sie zu verändern, und zieht Kennzahlen heraus."""

    def __init__(self, content_type: str, keep_text: bool):
        self.sse = "text/event-stream" in content_type
        self.json = "json" in content_type and not self.sse
        self.keep_text = keep_text
        self.parser = SseParser()
        self.buffer = bytearray()
        self.resp_bytes = 0
        self.ttft_s: float | None = None
        self.out_chars = 0
        self.reasoning_chars = 0
        self.tool_call = False
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] = {}
        self.texts: list[str] = []

    def _text(self, reasoning: str, content: str, tool: bool, now: float) -> None:
        if (reasoning or content or tool) and self.ttft_s is None:
            self.ttft_s = round(now, 4)
        self.reasoning_chars += len(reasoning)
        self.out_chars += len(content)
        self.tool_call = self.tool_call or tool
        if self.keep_text and (reasoning or content):
            self.texts.append(reasoning or content)

    def _event(self, event: dict[str, Any], now: float) -> None:
        for holder in (event, event.get("message"), event.get("response")):
            if isinstance(holder, dict) and isinstance(holder.get("usage"), dict):
                self.usage.update(holder["usage"])
        if event.get("choices"):
            reasoning, content, tool, reason = delta_texts(event)
            if reason:
                self.finish_reason = reason
            self._text(reasoning, content, tool, now)
            return
        kind = event.get("type")
        delta = event.get("delta")
        if kind == "content_block_delta" and isinstance(delta, dict):  # Anthropic-Format
            self._text(delta.get("thinking") or "", delta.get("text") or "", "partial_json" in delta, now)
        elif isinstance(kind, str) and kind.endswith(".delta") and isinstance(delta, str):  # Responses-Format
            if "reasoning" in kind:
                self._text(delta, "", False, now)
            elif "output_text" in kind:
                self._text("", delta, False, now)
            else:
                self._text("", "", True, now)
        elif kind == "message_delta" and isinstance(delta, dict) and delta.get("stop_reason"):
            self.finish_reason = delta["stop_reason"]

    def feed(self, chunk: bytes, now: float) -> None:
        self.resp_bytes += len(chunk)
        if self.sse:
            for event in self.parser.feed(chunk):
                self._event(event, now)
        elif self.json and len(self.buffer) < MAX_JSON_BUFFER:
            self.buffer.extend(chunk)

    def finish(self) -> dict[str, Any]:
        if self.json and self.buffer:
            try:
                obj = json.loads(bytes(self.buffer))
            except (json.JSONDecodeError, UnicodeDecodeError):
                obj = None
            if isinstance(obj, dict):
                if isinstance(obj.get("usage"), dict):
                    self.usage.update(obj["usage"])
                choices = obj.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    content = msg.get("content") if isinstance(msg.get("content"), str) else ""
                    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                    self.out_chars += len(content)
                    self.reasoning_chars += len(reasoning) if isinstance(reasoning, str) else 0
                    self.tool_call = bool(msg.get("tool_calls"))
                    self.finish_reason = choices[0].get("finish_reason")
                    if self.keep_text:
                        self.texts.append((reasoning if isinstance(reasoning, str) else "") + content)
                elif isinstance(obj.get("content"), list):  # Anthropic-Format
                    for block in obj["content"]:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            self.out_chars += len(block["text"])
                    self.finish_reason = obj.get("stop_reason")
        out: dict[str, Any] = {
            "resp_bytes": self.resp_bytes,
            "ttft_s": self.ttft_s,
            "out_chars": self.out_chars,
            "reasoning_chars": self.reasoning_chars,
            "tool_call": self.tool_call,
            "finish_reason": self.finish_reason,
        }
        fields = {k: v for k, v in usage_fields(self.usage).items() if v is not None}
        if fields:
            out.update(fields)
            out["tokens_source"] = "usage"
        return out


class Recorder:
    """Zustand des Mitschreibers: Ziel, Ausgabeordner, Zähler."""

    def __init__(self, upstream: str, out_dir: str | Path, counter: Any = None, inject_usage: bool = False):
        self.upstream = upstream.rstrip("/")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.counter = counter
        self.inject_usage = inject_usage
        self.session: aiohttp.ClientSession | None = None
        self._writer: JsonlWriter | None = None
        self._day = ""
        salt_file = self.out_dir / ".salt"
        if not salt_file.is_file():
            salt_file.write_text(secrets.token_hex(16), encoding="ascii")
        self._salt = salt_file.read_text(encoding="ascii").strip()

    def pseudonym(self, value: str | None) -> str | None:
        """Unumkehrbare Kennung statt Adresse oder Nutzername."""
        if not value:
            return None
        return hashlib.sha256((self._salt + value).encode()).hexdigest()[:12]

    def write(self, rec: dict[str, Any]) -> None:
        day = datetime.now().strftime("%Y-%m-%d")
        if day != self._day or self._writer is None:
            if self._writer:
                self._writer.close()
            self._writer = JsonlWriter(self.out_dir / f"usage-{day}.jsonl")
            self._day = day
        self._writer.write(rec)

    async def start(self, app: web.Application) -> None:
        self.session = aiohttp.ClientSession(
            auto_decompress=False,
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=15),
        )

    async def close(self, app: web.Application) -> None:
        if self.session:
            await self.session.close()
        if self._writer:
            self._writer.close()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        assert self.session is not None
        t0 = time.perf_counter()
        body = await request.read()
        is_llm = request.method == "POST" and request.path.endswith(LLM_PATHS)
        meta, texts, obj = request_meta(body, collect_text=self.counter is not None) if is_llm else ({}, [], None)
        user = meta.pop("_user", None)

        if (
            self.inject_usage and obj is not None and obj.get("stream")
            and request.path.endswith("/chat/completions") and "stream_options" not in obj
        ):
            obj["stream_options"] = {"include_usage": True}
            body = json.dumps(obj, ensure_ascii=False).encode()
            meta["usage_injected"] = True

        rec: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "client": self.pseudonym(request.remote),
            "user": self.pseudonym(user),
            "method": request.method,
            "path": request.path,
            **meta,
            "status": None,
            "aborted": False,
            "error": None,
        }
        count_prompt = None
        if self.counter is not None and texts:
            count_prompt = asyncio.create_task(asyncio.to_thread(self.counter.count, "\n".join(texts)))
        texts = []  # nicht länger im Speicher halten als nötig

        headers = {k: v for k, v in request.headers.items() if k.lower() not in REQUEST_DROP}
        observer: ResponseObserver | None = None
        response: web.StreamResponse | None = None
        try:
            async with self.session.request(
                request.method, self.upstream + request.path_qs, headers=headers, data=body or None
            ) as up:
                rec["status"] = up.status
                rec["ttfb_s"] = round(time.perf_counter() - t0, 4)
                observer = ResponseObserver(up.headers.get("Content-Type", ""), keep_text=self.counter is not None)
                response = web.StreamResponse(status=up.status, reason=up.reason)
                for key, value in up.headers.items():
                    if key.lower() not in RESPONSE_DROP:
                        response.headers[key] = value
                await response.prepare(request)
                async for chunk in up.content.iter_any():
                    observer.feed(chunk, time.perf_counter() - t0)
                    await response.write(chunk)
                await response.write_eof()
        except asyncio.CancelledError:
            rec["aborted"] = True
            raise
        except ConnectionResetError:  # der Aufrufer hat die Verbindung geschlossen
            rec["aborted"] = True
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            rec["error"] = type(exc).__name__
            if response is None:
                response = web.Response(status=502, text=f"Engine nicht erreichbar: {type(exc).__name__}")
        finally:
            rec["duration_s"] = round(time.perf_counter() - t0, 4)
            if observer is not None:
                rec.update(observer.finish())
            try:
                if count_prompt is not None:
                    counted = await count_prompt
                    if rec.get("prompt_tokens") is None:
                        rec["prompt_tokens"] = counted
                        rec.setdefault("tokens_source", "tokenizer")
                if self.counter is not None and observer is not None and rec.get("completion_tokens") is None:
                    if observer.texts:
                        rec["completion_tokens"] = await asyncio.to_thread(self.counter.count, "".join(observer.texts))
                        rec.setdefault("tokens_source", "tokenizer")
            except (asyncio.CancelledError, Exception):  # Zählen ist Beiwerk
                pass
            if is_llm or rec["status"] is None or rec["error"]:
                self.write(rec)
        assert response is not None
        return response


def make_app(upstream: str, out_dir: str | Path, counter: Any = None, inject_usage: bool = False) -> web.Application:
    recorder = Recorder(upstream, out_dir, counter=counter, inject_usage=inject_usage)
    app = web.Application(client_max_size=1024**3)
    app.on_startup.append(recorder.start)
    app.on_cleanup.append(recorder.close)
    app.router.add_route("*", "/{tail:.*}", recorder.handle)
    return app


# -- Auswertung ----------------------------------------------------------------


def load_usage(usage_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(usage_dir).glob("usage-*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def peak_concurrency(rows: list[dict[str, Any]]) -> int:
    """Größte Zahl gleichzeitig laufender Anfragen."""
    events: list[tuple[float, int]] = []
    for r in rows:
        events.append((r["ts"], 1))
        events.append((r["ts"] + (r.get("duration_s") or 0.0), -1))
    events.sort()
    active = peak = 0
    for _, step in events:
        active += step
        peak = max(peak, active)
    return peak


def report(usage_dir: str | Path, entitled: int | None = None) -> dict[str, Any]:
    """Fasst die Mitschrift zusammen und schätzt daraus die Werte des Lastmodells."""
    rows = [
        r for r in load_usage(usage_dir)
        if r.get("status") == 200 and r.get("method") == "POST" and str(r.get("path", "")).endswith(LLM_PATHS)
    ]
    if not rows:
        return {"requests": 0}

    def who(r: dict[str, Any]) -> str | None:
        return r.get("user") or r.get("client")

    by_hour: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_hour[datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H")].append(r)
    peak_hour, peak_rows = max(by_hour.items(), key=lambda kv: len(kv[1]))
    active = {who(r) for r in peak_rows if who(r)}
    result: dict[str, Any] = {
        "requests": len(rows),
        "days": len({datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d") for r in rows}),
        "clients": len({r.get("client") for r in rows if r.get("client")}),
        "users": len({r.get("user") for r in rows if r.get("user")}),
        "share_stream": round(sum(1 for r in rows if r.get("stream")) / len(rows), 3),
        "share_with_tools": round(sum(1 for r in rows if r.get("n_tools")) / len(rows), 3),
        "prompt_tokens": describe([r.get("prompt_tokens") for r in rows], 0),
        "prompt_chars": describe([r.get("prompt_chars") for r in rows], 0),
        "completion_tokens": describe([r.get("completion_tokens") for r in rows], 0),
        "duration_s": describe([r.get("duration_s") for r in rows], 2),
        "ttft_s": describe([r.get("ttft_s") for r in rows], 2),
        "peak_concurrency": peak_concurrency(rows),
        "peak_hour": {
            "hour": peak_hour,
            "requests": len(peak_rows),
            "active": len(active),
            "requests_per_active": round(len(peak_rows) / len(active), 1) if active else None,
            "mean_duration_s": round(sum(r.get("duration_s") or 0 for r in peak_rows) / len(peak_rows), 1),
            "concurrency": peak_concurrency(peak_rows),
        },
    }
    if entitled:
        result["peak_hour"]["share_active"] = round(len(active) / entitled, 3)
    return result
