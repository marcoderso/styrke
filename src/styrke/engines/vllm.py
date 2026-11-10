"""Adapter für vLLM (``vllm serve``)."""

from __future__ import annotations

from typing import Any

import aiohttp

from .flags import FlagEngine
from .probes import engine_metrics, kv_capacity


class VllmEngine(FlagEngine):
    kind = "vllm"
    entrypoint = "vllm"
    # Neuere Versionen zeigen mit --help nur eine Übersicht; --help=all listet alle Optionen.
    help_args = [["serve", "--help=all"], ["serve", "--help"]]

    def fixed_args(self) -> dict[str, Any]:
        return {
            "served-model-name": self.model.served_name,
            "host": "0.0.0.0",
            "port": self.engine.container_port,
            "max-model-len": self.model.max_context,
        }

    def prefix_args(self) -> list[str]:
        return ["serve", f"/models/{self.model.dir}"]

    async def metrics(self, session: aiohttp.ClientSession) -> dict[str, Any] | None:
        return await engine_metrics(self.kind, self, session)

    async def kv_capacity_tokens(self, session: aiohttp.ClientSession) -> int | None:
        return await kv_capacity(self.kind, self, session)
