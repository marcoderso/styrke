"""Engines, deren Server über Kommandozeilen-Optionen eingestellt wird.

Die Optionen kommen aus vier Quellen, spätere überschreiben frühere: feste
Angaben des Adapters, ``base_args`` der Engine, ``engine_args`` des Modells und
``args`` der Variante. Vor dem Start gleicht der Adapter sie mit der Hilfe des
Images ab: Eine unbekannte Option führt zu einer klaren Meldung statt zu einem
Container, der sofort wieder endet. Optionen mit Fragezeichen am Ende
(``enable-metrics?``) sind Wünsche: Kennt das Image sie nicht, entfallen sie.
"""

from __future__ import annotations

import re
from typing import Any

from .base import EngineError
from .docker import DockerEngine, render_args, run_cmd


def resolve_flags(
    args: dict[str, Any], help_text: str, image: str, aliases: dict[str, str] | None = None,
    dropped: list[str] | None = None,
) -> dict[str, Any]:
    """Ersetzt umbenannte Optionen, lässt unbekannte Wünsche weg und meldet unbekannte Pflichtangaben."""
    known = set(re.findall(r"--([a-z0-9][a-z0-9-]*)", help_text))
    aliases = aliases or {}
    out: dict[str, Any] = {}
    unknown: list[str] = []
    for raw_key, value in args.items():
        optional = raw_key.endswith("?")
        key = raw_key.rstrip("?")
        if key in known:
            out[key] = value
        elif aliases.get(key) in known:
            out[aliases[key]] = value
        elif optional:
            if dropped is not None:
                dropped.append(key)
        else:
            unknown.append(key)
    if unknown:
        raise EngineError(
            f"{image} kennt diese Optionen nicht: " + ", ".join(f"--{k}" for k in unknown)
            + ". Variante in der Testmatrix anpassen."
        )
    return out


class FlagEngine(DockerEngine):
    """Grundlage für FreeToken, vLLM und SGLang."""

    aliases: dict[str, str] = {}
    help_args: list[list[str]] = []  # Aufrufe hinter dem Entrypoint, die alle Optionen auflisten; der erste, der gelingt, zählt

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._serve_args: dict[str, Any] | None = None
        self.dropped_flags: list[str] = []

    def fixed_args(self) -> dict[str, Any]:
        raise NotImplementedError

    def prefix_args(self) -> list[str]:
        """Was vor den Optionen steht, etwa das Unterkommando."""
        return []

    def merged_args(self) -> dict[str, Any]:
        args = dict(self.fixed_args())
        for extra in (self.engine.base_args, self.model.engine_args.get(self.kind) or {}, self.variant.args):
            for key, value in extra.items():
                args.pop(key.rstrip("?"), None)
                args.pop(key.rstrip("?") + "?", None)
                args[key] = value
        return args

    def server_args(self) -> list[str]:
        args = self._serve_args
        if args is None:  # ungeprüft: Wünsche mitschicken, als wären sie bekannt
            args = {k.rstrip("?"): v for k, v in self.merged_args().items()}
        return [*self.prefix_args(), *render_args(args), *self.variant.extra_args]

    async def validate(self) -> None:
        args = self.merged_args()
        for help_cmd in self.help_args:
            cmd = ["docker", "run", "--rm"]
            if self.entrypoint:
                cmd += ["--entrypoint", self.entrypoint]
            code, out = await run_cmd(*cmd, self.engine.image, *help_cmd, timeout=600)
            if code == 0 and out.count("--") >= 5:
                self.dropped_flags = []
                self._serve_args = resolve_flags(args, out, self.engine.image, self.aliases, self.dropped_flags)
                return
        self._serve_args = None  # Hilfe nicht lesbar: ungeprüft weiter

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "dropped_flags": self.dropped_flags}
