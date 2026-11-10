"""Attrappe einer OpenAI-kompatiblen Engine für Tests.

Verhält sich in den Punkten, die styrke misst, wie eine echte Engine: Der
Prompt wird mit begrenztem Tempo eingelesen, Tokens kommen mit einer Rate, die
bei mehr gleichzeitigen Anfragen sinkt, nur ``max_running`` Anfragen laufen
gleichzeitig, der Rest wartet. Zählwerte kommen im letzten Stück des Streams.

Dazu ein Präfix-Cache: Beginnt ein Prompt wie ein früherer, wird nur der neue
Teil eingelesen. Der Cache fasst ``kv_pool_tokens`` Tokens und verdrängt die am
längsten ungenutzten Einträge. Mit ``prefill_slowdown`` bremst das Einlesen
eines Prompts die gleichzeitig laufenden Antworten.
"""

from __future__ import annotations

import asyncio
import json

from aiohttp import web

WORDS = "Das ist ein Testtext der Attrappe und er hat keinerlei Bedeutung für die Messung ".split()


class MockEngine:
    def __init__(
        self,
        max_running: int = 4,
        tokens_per_s: float = 400.0,
        prefill_tokens_per_s: float = 200_000.0,
        max_context: int = 8192,
        chars_per_token: float = 4.0,
        template_overhead: int = 9,
        send_usage: bool = True,
        accept_stream_options: bool = True,
        kv_pool_tokens: int | None = None,
        report_cached: bool = True,
        prefill_slowdown: float = 0.0,
    ):
        self.max_running = max_running
        self.tokens_per_s = tokens_per_s
        self.prefill_tokens_per_s = prefill_tokens_per_s
        self.max_context = max_context
        self.chars_per_token = chars_per_token
        self.template_overhead = template_overhead
        self.send_usage = send_usage
        self.accept_stream_options = accept_stream_options
        self.sem = asyncio.Semaphore(max_running)
        self.active = 0
        self.peak_active = 0
        self.requests = 0
        self.bodies: list[dict] = []
        self.kv_pool_tokens = kv_pool_tokens
        self.report_cached = report_cached
        self.prefill_slowdown = prefill_slowdown
        self.prefilling = 0
        self.canned: dict[str, str] = {}  # Stichwort in der Frage -> feste Antwort (nur nicht gestreamt)
        self.cache: list[list] = []  # Einträge [Text, Tokens], der zuletzt genutzte am Ende
        self.evictions = 0

    # -- Präfix-Cache -------------------------------------------------------

    @staticmethod
    def canonical(messages: list[dict]) -> str:
        return "\x1e".join(f"{m.get('role')}:{m.get('content') or ''}" for m in messages)

    @staticmethod
    def common_prefix(a: str, b: str) -> int:
        if a[:64] != b[:64]:
            return 0
        lo, hi = 64, min(len(a), len(b))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def cache_lookup(self, text: str) -> int:
        """Tokens des Prompts, die schon im Cache liegen; der Treffer gilt als frisch genutzt."""
        best, best_idx = 0, -1
        for idx, (cached, _) in enumerate(self.cache):
            n = self.common_prefix(text, cached)
            if n > best:
                best, best_idx = n, idx
        if best_idx >= 0:
            self.cache.append(self.cache.pop(best_idx))
        return int(best / self.chars_per_token)

    def cache_store(self, text: str, tokens: int) -> None:
        # Ein Eintrag, den der neue Prompt fortsetzt, geht im neuen auf.
        self.cache = [e for e in self.cache if not text.startswith(e[0])]
        self.cache.append([text, tokens])
        if self.kv_pool_tokens is not None:
            while len(self.cache) > 1 and sum(e[1] for e in self.cache) > self.kv_pool_tokens:
                self.cache.pop(0)
                self.evictions += 1

    async def cache_status(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "kv_pool": {"capacity_tokens": self.kv_pool_tokens, "used_tokens": sum(e[1] for e in self.cache)},
                "entries": len(self.cache),
                "evictions": self.evictions,
            }
        )

    def prompt_tokens(self, messages: list[dict]) -> int:
        chars = sum(len(m.get("content") or "") for m in messages)
        return int(chars / self.chars_per_token) + self.template_overhead

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": "attrappe", "context_length": self.max_context}]})

    async def stats(self, request: web.Request) -> web.Response:
        return web.json_response({"active": self.active, "completed": self.requests})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.bodies.append(body)
        if "stream_options" in body and not self.accept_stream_options:
            return web.json_response({"error": {"message": "unknown field stream_options"}}, status=400)
        messages = body.get("messages") or []
        p_tokens = self.prompt_tokens(messages)
        max_tokens = int(body.get("max_tokens") or 16)
        if p_tokens + max_tokens > self.max_context:
            return web.json_response(
                {"error": {"message": f"prompt too long: {p_tokens} tokens, limit {self.max_context}"}}, status=400
            )
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        if any((m.get("content") or "").startswith("<|think|>") for m in messages if m.get("role") == "system"):
            thinking = True
        n_reasoning = min(max_tokens // 2, 8) if thinking else 0
        text = self.canonical(messages)
        cached_tokens = min(p_tokens, self.cache_lookup(text))
        usage = {
            "prompt_tokens": p_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": p_tokens + max_tokens,
        }
        if self.report_cached:
            usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}

        async with self.sem:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            try:
                self.prefilling += 1
                try:
                    await asyncio.sleep((p_tokens - cached_tokens) / self.prefill_tokens_per_s)
                finally:
                    self.prefilling -= 1
                self.cache_store(text, p_tokens)
                if not body.get("stream"):
                    last = (messages[-1].get("content") or "") if messages else ""
                    if body.get("tools") and "Lager" in last:  # nachgestellter Werkzeugaufruf
                        call = {"type": "function", "function": {"name": "lagerbestand", "arguments": '{"artikel": "Harnstoff"}'}}
                        return web.json_response(
                            {"choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                                          "finish_reason": "tool_calls"}], "usage": usage}
                        )
                    await asyncio.sleep(min(max_tokens, 40) / self.tokens_per_s)
                    canned = next((answer for key, answer in self.canned.items() if key in last), None)
                    text = canned or " ".join(WORDS[i % len(WORDS)] for i in range(min(max_tokens, 40)))
                    return web.json_response(
                        {
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                         "finish_reason": "length"}],
                            "usage": usage,
                        }
                    )
                resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await resp.prepare(request)
                for i in range(max_tokens):
                    # Mehr gleichzeitige Anfragen bremsen jede einzelne etwas.
                    slow = 1 + 0.05 * (self.active - 1) + (self.prefill_slowdown if self.prefilling else 0.0)
                    await asyncio.sleep(slow / self.tokens_per_s)
                    word = WORDS[i % len(WORDS)] + " "
                    delta = {"reasoning_content": word} if i < n_reasoning else {"content": word}
                    last = i == max_tokens - 1
                    chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": "length" if last else None}]}
                    await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
                if self.send_usage and (body.get("stream_options") or {}).get("include_usage"):
                    await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
                await resp.write(b"data: [DONE]\n\n")
                await resp.write_eof()
                return resp
            finally:
                self.active -= 1
                self.requests += 1

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1024**3)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/v1/stats", self.stats)
        app.router.add_get("/v1/cache/status", self.cache_status)
        app.router.add_post("/v1/chat/completions", self.chat)
        return app


class Served:
    """Startet eine aiohttp-Anwendung auf einem freien Port (als ``async with``)."""

    def __init__(self, app: web.Application):
        self.app = app
        self.runner: web.AppRunner | None = None
        self.url = ""

    async def __aenter__(self) -> "Served":
        self.runner = web.AppRunner(self.app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self.runner is not None
        await self.runner.cleanup()


if __name__ == "__main__":
    # Als Programm: Attrappe auf Port 18999 für Probeläufe von Hand.
    web.run_app(
        MockEngine(
            max_context=300_000, max_running=8, kv_pool_tokens=16_000, prefill_slowdown=2.0, prefill_tokens_per_s=4000
        ).app(),
        host="127.0.0.1", port=18999, print=None,
    )
