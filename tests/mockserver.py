"""Attrappe einer OpenAI-kompatiblen Engine für Tests.

Verhält sich in den Punkten, die styrke misst, wie eine echte Engine: Der
Prompt wird mit begrenztem Tempo eingelesen, Tokens kommen mit einer Rate, die
bei mehr gleichzeitigen Anfragen sinkt, nur ``max_running`` Anfragen laufen
gleichzeitig, der Rest wartet. Zählwerte kommen im letzten Stück des Streams.
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
        usage = {
            "prompt_tokens": p_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": p_tokens + max_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
        }

        async with self.sem:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            try:
                await asyncio.sleep(p_tokens / self.prefill_tokens_per_s)
                if not body.get("stream"):
                    await asyncio.sleep(max_tokens / self.tokens_per_s)
                    text = " ".join(WORDS[i % len(WORDS)] for i in range(max_tokens))
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
                    await asyncio.sleep((1 + 0.05 * (self.active - 1)) / self.tokens_per_s)
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
    web.run_app(MockEngine(max_context=300_000).app(), host="127.0.0.1", port=18999, print=None)
