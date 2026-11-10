"""Gemeinsame Grundlage aller Engine-Adapter."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

import aiohttp

from ..client import complete


class EngineError(RuntimeError):
    """Die Engine ließ sich nicht starten, wurde nicht bereit oder ist abgestürzt."""


class EngineAdapter:
    """Schnittstelle, über die die Ablaufsteuerung eine Engine bedient.

    ``managed`` sagt, ob styrke die Engine selbst startet und stoppt. Bei einer
    fremd gestarteten Engine (``ExternalEngine``) sind Start und Stopp leer.
    """

    kind = "base"
    managed = False

    def __init__(self, base_url: str, served_name: str):
        self.base_url = base_url.rstrip("/")
        self.served_name = served_name
        self.load_time_s: float | None = None

    async def start(self, store: Any) -> None:
        """Startet die Engine; kehrt zurück, sobald der Prozess läuft (nicht: bereit ist)."""

    async def stop(self) -> None:
        """Stoppt die Engine und räumt auf."""

    async def check_alive(self) -> None:
        """Wirft ``EngineError``, wenn der Engine-Prozess nicht mehr läuft."""

    async def metrics(self, session: aiohttp.ClientSession) -> dict[str, Any] | None:
        """Engine-eigene Kennzahlen (Cache-Belegung, Durchsatz), falls verfügbar."""
        return None

    def describe(self) -> dict[str, Any]:
        """Angaben für das Manifest."""
        return {"kind": self.kind, "managed": self.managed, "base_url": self.base_url}

    async def wait_ready(
        self,
        session: aiohttp.ClientSession,
        timeout_s: float,
        probe_body: dict[str, Any] | None = None,
        log: Callable[[str], None] = print,
    ) -> float:
        """Wartet, bis die Engine eine Anfrage beantwortet; liefert die Wartezeit in Sekunden.

        Bereit heißt: ``/v1/models`` antwortet und eine Anfrage mit einem Token
        Ausgabe gelingt. Das ist für alle Engines gleich und unabhängig von
        deren eigenen Statusmeldungen.
        """
        t0 = time.perf_counter()
        last_note = 0.0
        while True:
            waited = time.perf_counter() - t0
            if waited > timeout_s:
                raise EngineError(f"Engine nach {timeout_s:.0f} s nicht bereit")
            await self.check_alive()
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.get(self.base_url + "/v1/models", timeout=timeout) as resp:
                    models_ok = resp.status == 200
                    await resp.read()
                if models_ok:
                    status, _ = await complete(
                        session,
                        self.base_url,
                        self.served_name,
                        [{"role": "user", "content": "Antworte mit OK."}],
                        max_tokens=1,
                        body=probe_body,
                        timeout_s=max(30.0, min(600.0, timeout_s - waited)),
                    )
                    if status == 200:
                        self.load_time_s = round(time.perf_counter() - t0, 1)
                        return self.load_time_s
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                pass
            if waited - last_note >= 60:
                log(f"  Engine lädt noch ({waited:.0f} s)")
                last_note = waited
            await asyncio.sleep(3)


class ExternalEngine(EngineAdapter):
    """Eine bereits laufende Engine, die styrke nur anspricht."""

    kind = "external"
    managed = False

    def __init__(self, base_url: str, served_name: str):
        super().__init__(base_url, served_name)
        self._metric_paths: list[str] | None = None

    async def metrics(self, session: aiohttp.ClientSession) -> dict[str, Any] | None:
        return await fetch_metric_paths(self, session, ["/v1/stats", "/v1/cache/status"])


async def fetch_metric_paths(
    adapter: EngineAdapter, session: aiohttp.ClientSession, paths: list[str]
) -> dict[str, Any] | None:
    """Fragt JSON-Endpunkte ab; Pfade, die es nicht gibt, werden künftig ausgelassen."""
    active = getattr(adapter, "_metric_paths", None)
    if active is None:
        active = list(paths)
        adapter._metric_paths = active  # type: ignore[attr-defined]
    out: dict[str, Any] = {}
    for path in list(active):
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with session.get(adapter.base_url + path, timeout=timeout) as resp:
                if resp.status != 200:
                    active.remove(path)
                    continue
                out[path.strip("/").replace("/", "_")] = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError):
            continue
    return out or None
