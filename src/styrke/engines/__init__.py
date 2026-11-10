"""Engine-Adapter."""

from __future__ import annotations

from ..config import EngineConfig, HostConfig, ModelConfig, VariantConfig
from .base import EngineAdapter, EngineError, ExternalEngine
from .freetoken import FreeTokenEngine

__all__ = ["EngineAdapter", "EngineError", "ExternalEngine", "FreeTokenEngine", "make_adapter"]


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
