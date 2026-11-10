"""Engine-Adapter."""

from __future__ import annotations

from ..config import EngineConfig, HostConfig, ModelConfig, VariantConfig
from .base import EngineAdapter, EngineError, ExternalEngine, SetupError
from .freetoken import FreeTokenEngine

__all__ = [
    "EngineAdapter", "EngineError", "ExternalEngine", "FreeTokenEngine", "SetupError", "kv_pool_tokens", "make_adapter",
]


def make_adapter(
    host: HostConfig,
    engine: EngineConfig,
    model: ModelConfig,
    variant: VariantConfig,
    allow_busy_gpu: bool = False,
) -> EngineAdapter:
    """Liefert den Adapter, der die Engine einer Zelle startet."""
    if engine.kind == "freetoken":
        return FreeTokenEngine(host, engine, model, variant, allow_busy_gpu=allow_busy_gpu)
    if engine.kind in ("vllm", "sglang"):
        raise EngineError(f"Der Adapter für {engine.kind} folgt in Schritt 2")
    raise EngineError(f"Unbekannte Engine-Art: {engine.kind}")


def kv_pool_tokens(engine: EngineConfig, variant: VariantConfig) -> int | None:
    """Größe des KV-Pools in Tokens, soweit sie aus den Startoptionen hervorgeht.

    Bei FreeToken legt ``kv-reserve-tokens`` sie fest. vLLM und SGLang bemessen
    ihren Pool selbst nach dem freien Speicher; dort liefert erst die laufende
    Engine den Wert (folgt mit den Adaptern).
    """
    if engine.kind == "freetoken":
        value = {**engine.base_args, **variant.args}.get("kv-reserve-tokens")
        return int(value) if value else None
    return None
