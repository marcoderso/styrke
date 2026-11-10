"""Testmatrix laden und prüfen.

Die Matrix ist eine YAML-Datei mit fünf Abschnitten: ``host`` (Pfade und
Adressen der Maschine), ``engines``, ``models``, ``cells`` (welches Modell auf
welcher Engine in welchen Varianten) und ``stages`` (Parameter der Teststufen).
Eine Datei ``local.yaml`` neben der Matrix überschreibt einzelne Werte, ohne
dass maschinenspezifische Angaben im Repository landen.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Die Testmatrix ist fehlerhaft."""


# Standardwerte der Stufen. Die Matrix kann jeden Wert unter ``stages``
# überschreiben; ``--smoke`` verkleinert sie für einen schnellen Funktionstest.
DEFAULT_STAGES: dict[str, dict[str, Any]] = {
    "common": {
        "temperature": 0.7,
        "top_p": 0.95,
        "request_timeout_s": 1800,
        "warmup_requests": 3,
        "restart_between_stages": True,
        "sample_interval_s": 2.0,
    },
    "s0": {
        "context_probes": [32000, 128000, 250000],
        "sanity_max_tokens": 128,
    },
    "s1": {
        "lengths": [100, 1000, 8000, 32000, 128000, 250000],
        "reps": 3,
        "thinking": ["off", "on"],
        "max_tokens": {"off": 256, "on": 1024},
        "pause_s": 1.0,
    },
    "s2": {
        "levels": [1, 2, 4, 8, 10, 15, 20, 25, 30, 35, 40, 48, 64, 96],
        "prompt_tokens": 1000,
        "max_tokens": 300,
        "thinking": "off",
        "warmup_s": 10,
        "window_s": 30,
        "windows": 3,
        "drain_s": 120,
        "pause_s": 5,
        "saturation_gain": 0.05,
        "extra_levels_after_saturation": 2,
        "max_error_rate": 0.2,
    },
}

SMOKE_STAGES: dict[str, dict[str, Any]] = {
    "common": {"warmup_requests": 1},
    "s0": {"context_probes": [2000]},
    "s1": {"lengths": [100, 1000], "reps": 1, "max_tokens": {"off": 48, "on": 96}, "pause_s": 0.1},
    "s2": {
        "levels": [1, 2, 4, 8],
        "prompt_tokens": 300,
        "max_tokens": 48,
        "warmup_s": 1,
        "window_s": 3,
        "windows": 2,
        "drain_s": 30,
        "pause_s": 0.5,
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    """Legt ``over`` über ``base``; verschachtelte Wörterbücher werden gemischt."""
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class HostConfig:
    models_dir: str = "/srv/models"
    cache_dir: str = "/srv/styrke/cache"
    corpus_dir: str = "corpus"
    runs_dir: str = "runs"
    bind: str = "127.0.0.1"
    port: int = 18000
    gpus: str = "all"


@dataclass
class EngineConfig:
    name: str
    kind: str
    image: str = ""
    container_port: int = 8000
    ready_timeout_s: int = 2700
    base_args: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    docker: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str
    repo: str
    dir: str = ""
    served_name: str = "styrke"
    max_context: int = 131072
    tokenizer: str = "tokenizer.json"
    thinking: dict[str, dict[str, Any]] = field(default_factory=dict)

    def thinking_mode(self, mode: str) -> dict[str, Any]:
        """Liefert ``{"body": {...}, "system_prefix": "..."}`` für ``off`` oder ``on``."""
        return self.thinking.get(mode) or {}


@dataclass
class VariantConfig:
    id: str
    args: dict[str, Any] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""


@dataclass
class CellConfig:
    id: str
    model: str
    engine: str
    enabled: bool = True
    note: str = ""
    variants: list[VariantConfig] = field(default_factory=list)

    def variant(self, variant_id: str | None) -> VariantConfig:
        if variant_id is None:
            return self.variants[0]
        for v in self.variants:
            if v.id == variant_id:
                return v
        known = ", ".join(v.id for v in self.variants)
        raise ConfigError(f"Zelle {self.id!r} hat keine Variante {variant_id!r} (vorhanden: {known})")


@dataclass
class Matrix:
    host: HostConfig
    engines: dict[str, EngineConfig]
    models: dict[str, ModelConfig]
    cells: list[CellConfig]
    stages: dict[str, dict[str, Any]]
    load_model: dict[str, Any]
    path: Path
    raw: dict[str, Any]

    def cell(self, cell_id: str) -> CellConfig:
        for c in self.cells:
            if c.id == cell_id:
                return c
        known = ", ".join(c.id for c in self.cells)
        raise ConfigError(f"Zelle {cell_id!r} steht nicht in der Matrix (vorhanden: {known})")

    def stage_params(self, smoke: bool = False) -> dict[str, dict[str, Any]]:
        params = deep_merge(DEFAULT_STAGES, self.stages)
        if smoke:
            params = deep_merge(params, SMOKE_STAGES)
        return params


def _only_known(section: str, data: dict, cls: type) -> dict:
    known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unbekannte Angabe in {section}: {', '.join(unknown)}")
    return data


def load_matrix(path: str | os.PathLike, local: str | os.PathLike | None = None) -> Matrix:
    """Liest die Matrix und legt, falls vorhanden, ``local.yaml`` darüber."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Testmatrix nicht gefunden: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: oberste Ebene muss ein Wörterbuch sein")

    local_path = Path(local) if local else Path(os.environ.get("STYRKE_LOCAL", path.with_name("local.yaml")))
    if local_path.is_file():
        over = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
        if not isinstance(over, dict):
            raise ConfigError(f"{local_path}: oberste Ebene muss ein Wörterbuch sein")
        raw = deep_merge(raw, over)

    host = HostConfig(**_only_known("host", raw.get("host") or {}, HostConfig))

    engines: dict[str, EngineConfig] = {}
    for name, data in (raw.get("engines") or {}).items():
        data = dict(data or {})
        if "kind" not in data:
            raise ConfigError(f"Engine {name!r}: Angabe 'kind' fehlt")
        engines[name] = EngineConfig(name=name, **_only_known(f"engines.{name}", data, EngineConfig))

    models: dict[str, ModelConfig] = {}
    for name, data in (raw.get("models") or {}).items():
        data = dict(data or {})
        if "repo" not in data:
            raise ConfigError(f"Modell {name!r}: Angabe 'repo' fehlt")
        model = ModelConfig(name=name, **_only_known(f"models.{name}", data, ModelConfig))
        if not model.dir:
            model.dir = model.repo.rsplit("/", 1)[-1]
        for mode in model.thinking:
            if mode not in ("off", "on"):
                raise ConfigError(f"Modell {name!r}: thinking kennt nur 'off' und 'on', nicht {mode!r}")
        models[name] = model

    cells: list[CellConfig] = []
    seen: set[str] = set()
    for data in raw.get("cells") or []:
        data = dict(data or {})
        for key in ("id", "model", "engine"):
            if key not in data:
                raise ConfigError(f"Zelle {data.get('id', '?')!r}: Angabe {key!r} fehlt")
        variants_raw = data.pop("variants", None) or [{"id": "basis"}]
        cell = CellConfig(variants=[], **_only_known(f"cells.{data['id']}", data, CellConfig))
        if cell.id in seen:
            raise ConfigError(f"Zelle {cell.id!r} ist doppelt vorhanden")
        seen.add(cell.id)
        if cell.model not in models:
            raise ConfigError(f"Zelle {cell.id!r} nennt unbekanntes Modell {cell.model!r}")
        if cell.engine not in engines:
            raise ConfigError(f"Zelle {cell.id!r} nennt unbekannte Engine {cell.engine!r}")
        variant_ids: set[str] = set()
        for v in variants_raw:
            v = dict(v or {})
            if "id" not in v:
                raise ConfigError(f"Zelle {cell.id!r}: eine Variante hat keine 'id'")
            variant = VariantConfig(**_only_known(f"cells.{cell.id}.variants.{v['id']}", v, VariantConfig))
            if variant.id in variant_ids:
                raise ConfigError(f"Zelle {cell.id!r}: Variante {variant.id!r} ist doppelt vorhanden")
            variant_ids.add(variant.id)
            cell.variants.append(variant)
        cells.append(cell)
    if not cells:
        raise ConfigError(f"{path}: keine Zellen definiert")

    stages = raw.get("stages") or {}
    unknown = sorted(set(stages) - set(DEFAULT_STAGES))
    if unknown:
        raise ConfigError(f"Unbekannte Stufe unter stages: {', '.join(unknown)}")

    return Matrix(
        host=host,
        engines=engines,
        models=models,
        cells=cells,
        stages=stages,
        load_model=raw.get("load_model") or {},
        path=path,
        raw=raw,
    )
