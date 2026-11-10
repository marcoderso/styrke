"""Adapter für FreeToken (``ft serve``)."""

from __future__ import annotations

import re
from typing import Any

import aiohttp

from .base import EngineError, fetch_metric_paths
from .docker import DockerEngine, render_args, run_cmd

# Optionen, die zwischen FreeToken-Versionen umbenannt wurden. Der Adapter
# nimmt die Schreibweise, die das installierte ``ft serve`` kennt.
ALIASES = {
    "moe-strategy": "moe-backend",
    "moe-backend": "moe-strategy",
}


class FreeTokenEngine(DockerEngine):
    kind = "freetoken"
    entrypoint = "ft"
    build_dir = "engines/freetoken"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._serve_args: dict[str, Any] | None = None

    def merged_args(self) -> dict[str, Any]:
        """Feste Angaben, Engine-Grundwerte und Varianten-Werte, in dieser Reihenfolge."""
        args: dict[str, Any] = {
            "model": f"/models/{self.model.dir}",
            "served-model-name": self.model.served_name,
            "host": "0.0.0.0",
            "port": self.engine.container_port,
        }
        args.update(self.engine.base_args)
        args.update(self.variant.args)
        return args

    def server_args(self) -> list[str]:
        args = self._serve_args if self._serve_args is not None else self.merged_args()
        return ["serve", *render_args(args), *self.variant.extra_args]

    async def validate(self) -> None:
        """Gleicht die Optionen mit ``ft serve --help`` des Images ab."""
        code, out = await run_cmd(
            "docker", "run", "--rm", "--entrypoint", "ft", self.engine.image, "serve", "--help", timeout=180
        )
        args = self.merged_args()
        if code != 0 or "--" not in out:
            self._serve_args = args  # Hilfe nicht lesbar: ungeprüft weiter
            return
        self._serve_args = resolve_flags(args, out, self.engine.image)

    async def metrics(self, session: aiohttp.ClientSession) -> dict[str, Any] | None:
        return await fetch_metric_paths(self, session, ["/v1/stats", "/v1/cache/status"])


def resolve_flags(args: dict[str, Any], help_text: str, image: str) -> dict[str, Any]:
    """Ersetzt umbenannte Optionen und meldet unbekannte, bevor der Container startet."""
    known = set(re.findall(r"--([a-z0-9][a-z0-9-]*)", help_text))
    out: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in args.items():
        if key in known:
            out[key] = value
        elif ALIASES.get(key) in known:
            out[ALIASES[key]] = value
        else:
            unknown.append(key)
    if unknown:
        raise EngineError(
            f"{image} kennt diese Optionen von 'ft serve' nicht: "
            + ", ".join(f"--{k}" for k in unknown)
            + ". Variante in der Testmatrix anpassen."
        )
    return out
