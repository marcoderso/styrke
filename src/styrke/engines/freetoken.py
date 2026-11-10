"""Adapter für FreeToken (``ft serve``)."""

from __future__ import annotations

from typing import Any

import aiohttp

from .flags import FlagEngine, resolve_flags as _resolve
from .probes import engine_metrics

# Optionen, die zwischen FreeToken-Versionen umbenannt wurden. Der Adapter
# nimmt die Schreibweise, die das installierte ``ft serve`` kennt.
ALIASES = {
    "moe-strategy": "moe-backend",
    "moe-backend": "moe-strategy",
}


def resolve_flags(args: dict[str, Any], help_text: str, image: str) -> dict[str, Any]:
    return _resolve(args, help_text, image, ALIASES)


class FreeTokenEngine(FlagEngine):
    kind = "freetoken"
    entrypoint = "ft"
    build_dir = "engines/freetoken"
    aliases = ALIASES
    help_args = [["serve", "--help"]]

    def fixed_args(self) -> dict[str, Any]:
        return {
            "model": f"/models/{self.model.dir}",
            "served-model-name": self.model.served_name,
            "host": "0.0.0.0",
            "port": self.engine.container_port,
        }

    def prefix_args(self) -> list[str]:
        return ["serve"]

    async def metrics(self, session: aiohttp.ClientSession) -> dict[str, Any] | None:
        return await engine_metrics(self.kind, self, session)
