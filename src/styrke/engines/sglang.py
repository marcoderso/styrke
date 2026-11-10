"""Adapter für SGLang (``python3 -m sglang.launch_server``)."""

from __future__ import annotations

from typing import Any

import aiohttp

from .flags import FlagEngine
from .probes import engine_metrics, kv_capacity


class SglangEngine(FlagEngine):
    kind = "sglang"
    entrypoint = "python3"
    help_args = [["-m", "sglang.launch_server", "--help"]]

    def fixed_args(self) -> dict[str, Any]:
        return {
            "model-path": f"/models/{self.model.dir}",
            "served-model-name": self.model.served_name,
            "host": "0.0.0.0",
            "port": self.engine.container_port,
            "context-length": self.model.max_context,
        }

    def prefix_args(self) -> list[str]:
        return ["-m", "sglang.launch_server"]

    async def metrics(self, session: aiohttp.ClientSession) -> dict[str, Any] | None:
        return await engine_metrics(self.kind, self, session)

    async def kv_capacity_tokens(self, session: aiohttp.ClientSession) -> int | None:
        return await kv_capacity(self.kind, self, session)
