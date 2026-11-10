"""Kennzahlen und Größe des KV-Pools einer laufenden Engine, je Engine-Art.

Gilt für selbst gestartete und für fremd gestartete Engines gleichermaßen.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import aiohttp

JSON_PATHS = {"freetoken": ["/v1/stats", "/v1/cache/status"]}
PROMETHEUS_PREFIX = {"vllm": "vllm:", "sglang": "sglang:"}
SERVER_INFO_PATHS = ["/get_server_info", "/server_info"]
ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError)


async def _get(adapter: Any, session: aiohttp.ClientSession, path: str, as_json: bool) -> Any:
    timeout = aiohttp.ClientTimeout(total=5)
    async with session.get(adapter.base_url + path, timeout=timeout) as resp:
        if resp.status != 200:
            return None
        return await resp.json(content_type=None) if as_json else await resp.text()


def parse_prometheus(text: str, prefix: str) -> dict[str, float]:
    """Liest Zeilen ``name{labels} wert``; Histogramm-Klassen (``_bucket``) bleiben weg, sie blähen nur auf."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(prefix) or "_bucket{" in line:
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name[len(prefix):]] = float(value)
        except ValueError:
            continue
    return out


async def engine_metrics(kind: str, adapter: Any, session: aiohttp.ClientSession) -> dict[str, Any] | None:
    """Engine-eigene Kennzahlen; Pfade, die es nicht gibt, werden künftig ausgelassen."""
    if kind in PROMETHEUS_PREFIX:
        try:
            text = await _get(adapter, session, "/metrics", as_json=False)
        except ERRORS:
            return None
        return {"metrics": parse_prometheus(text, PROMETHEUS_PREFIX[kind])} if text else None
    active = getattr(adapter, "_metric_paths", None)
    if active is None:
        active = list(JSON_PATHS.get(kind) or JSON_PATHS["freetoken"])
        adapter._metric_paths = active
    out: dict[str, Any] = {}
    for path in list(active):
        try:
            data = await _get(adapter, session, path, as_json=True)
        except ERRORS:
            continue
        if data is None:
            active.remove(path)
            continue
        out[path.strip("/").replace("/", "_")] = data
    return out or None


def _find_key(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def vllm_kv_tokens(metrics_text: str) -> int | None:
    """Poolgröße aus ``vllm:cache_config_info``: Zahl der Blöcke mal Blockgröße."""
    match = re.search(r"vllm:cache_config_info\{([^}]*)\}", metrics_text)
    if not match:
        return None
    labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
    try:
        return int(float(labels["num_gpu_blocks"])) * int(float(labels.get("block_size") or 16))
    except (KeyError, ValueError):
        return None


async def kv_capacity(kind: str, adapter: Any, session: aiohttp.ClientSession) -> int | None:
    """Größe des KV-Pools in Tokens, wie die laufende Engine sie angibt."""
    try:
        if kind == "vllm":
            text = await _get(adapter, session, "/metrics", as_json=False)
            return vllm_kv_tokens(text) if text else None
        if kind == "sglang":
            for path in SERVER_INFO_PATHS:
                info = await _get(adapter, session, path, as_json=True)
                value = _find_key(info, "max_total_num_tokens") if info else None
                if value:
                    return int(value)
    except ERRORS:
        return None
    return None
