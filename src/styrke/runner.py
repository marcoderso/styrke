"""Ablaufsteuerung: fährt die Stufen einer Zelle nacheinander.

Je Stufe: Engine starten (oder an eine laufende anhängen), auf Bereitschaft
warten, aufwärmen, messen. Zwischen den Stufen startet die Engine neu, damit
jede Stufe mit leerem Cache beginnt. Ein Fehlstart ist ein Ergebnis und wird
als solches festgehalten.
"""

from __future__ import annotations

import os
import platform
import secrets
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import aiohttp

from . import __version__
from .client import RequestSpec
from .config import ConfigError, Matrix, load_matrix
from .corpus import Corpus
from .engines import EngineAdapter, EngineError, ExternalEngine, SetupError, kv_pool_tokens, make_adapter
from .prompts import PromptBuilder, make_counter
from .sampler import GpuSampler, PeriodicSampler, gpu_snapshot, host_snapshot
from .stages import STAGES
from .stages.common import StageContext
from .store import RunStore, read_jsonl


@dataclass
class RunOptions:
    matrix_path: str = "matrix.yaml"
    cell_id: str = ""
    variant_id: str | None = None
    stages: list[int] = field(default_factory=lambda: sorted(STAGES))  # ohne Angabe alle Stufen
    endpoint: str | None = None  # an eine laufende Engine anhängen statt selbst zu starten
    model_name: str | None = None
    resume: str | None = None
    smoke: bool = False
    allow_busy_gpu: bool = False


def _git_commit() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def build_manifest(
    matrix: Matrix, opts: RunOptions, params: dict[str, Any], builder: PromptBuilder, adapter: Any
) -> dict[str, Any]:
    cell = matrix.cell(opts.cell_id)
    variant = cell.variant(opts.variant_id)
    model = matrix.models[cell.model]
    return {
        "styrke_version": __version__,
        "styrke_commit": _git_commit(),
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "smoke": opts.smoke,
        "cell": cell.id,
        "variant": {"id": variant.id, "note": variant.note, "args": variant.args, "extra_args": variant.extra_args},
        "model": {"name": model.name, "repo": model.repo, "dir": model.dir, "max_context": model.max_context},
        "engine": adapter.describe(),
        "stages": opts.stages,
        "stage_params": params,
        "prompt_counter": builder.counter.kind,
        "corpus": builder.corpus.info(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
            **host_snapshot(),
        },
        "gpu": gpu_snapshot(),
    }


async def _discover_model(session: aiohttp.ClientSession, base_url: str) -> str | None:
    try:
        async with session.get(base_url.rstrip("/") + "/v1/models", timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json(content_type=None)
        return data["data"][0]["id"]
    except Exception:
        return None


async def _warmup(ctx: StageContext, stage: int) -> None:
    for _ in range(ctx.params["common"]["warmup_requests"]):
        prompt = ctx.builder.build(200, seed=ctx.next_seed())
        await ctx.request(
            RequestSpec(
                messages=prompt.messages,
                max_tokens=32,
                body=ctx.body_for("off"),
                tags={"stage": stage, "what": "warmup"},
            )
        )


async def run_cell(opts: RunOptions, log: Callable[[str], None] = print) -> int:
    """Fährt die gewählten Stufen.

    Rückgabewert: 0 bei Erfolg, 1 wenn eine Stufe an einem Fehler in styrke
    selbst scheiterte, 2 bei einem Engine-Fehler (ein Messergebnis: die Zelle
    lädt oder hält nicht), 3 wenn die Maschine nicht vorbereitet ist (dann
    entsteht kein Laufordner). Scheitert eine Stufe, laufen die übrigen
    trotzdem, solange die Engine überhaupt lädt.
    """
    matrix = load_matrix(opts.matrix_path)
    cell = matrix.cell(opts.cell_id)
    variant = cell.variant(opts.variant_id)
    model = matrix.models[cell.model]
    engine_cfg = matrix.engines[cell.engine]
    params = matrix.stage_params(opts.smoke)
    unknown = [s for s in opts.stages if s not in STAGES]
    if unknown:
        raise ConfigError(f"Diese Stufen gibt es noch nicht: {unknown} (vorhanden: {sorted(STAGES)})")
    if not opts.endpoint and not cell.enabled:
        raise ConfigError(f"Zelle {cell.id!r} ist noch nicht freigeschaltet: {cell.note}")

    adapter: EngineAdapter | None = None
    if not opts.endpoint:
        adapter = make_adapter(matrix.host, engine_cfg, model, variant, allow_busy_gpu=opts.allow_busy_gpu)
        try:
            await adapter.check_setup()
        except SetupError as exc:
            log("Nicht gestartet, die Vorbereitung ist unvollständig:")
            for i, problem in enumerate(exc.problems, 1):
                log(f"  {i}. {problem}")
            return 3

    counter = make_counter(Path(matrix.host.models_dir) / model.dir / model.tokenizer)
    corpus = Corpus.load(matrix.host.corpus_dir)
    builder = PromptBuilder(corpus, counter)

    if opts.resume:
        store = RunStore(Path(opts.resume))
    else:
        label = variant.id + ("-extern" if opts.endpoint else "") + ("-smoke" if opts.smoke else "")
        store = RunStore.create(matrix.host.runs_dir, cell.id, label)
    console = log
    run_log = open(store.dir / "run.log", "a", encoding="utf-8")

    def log(message: str = "") -> None:  # dieselben Zeilen auch in den Laufordner, für `styrke watch`
        console(message)
        run_log.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
        run_log.flush()

    log(f"Lauf: {store.dir}")
    if counter.kind != "tokenizer":
        log("Hinweis: Tokenizer des Modells nicht gefunden; Promptlängen werden an der Engine geeicht.")
    if corpus.builtin:
        log("Hinweis: eingebauter Mini-Korpus aktiv. Für Messungen erst 'styrke corpus fetch' ausführen.")

    exit_code = 0
    started = False
    failed: list[int] = []  # Stufen, die nicht zu Ende liefen
    engine_failures = 0  # Engine-Fehler in Folge
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        if adapter is None:
            served = opts.model_name or await _discover_model(session, opts.endpoint) or model.served_name
            adapter = ExternalEngine(opts.endpoint, served, engine_cfg.kind)
            log(f"Hänge an laufende Engine an: {opts.endpoint} (Modell {served})")
        ctx = StageContext(
            store=store, adapter=adapter, session=session, model=model, params=params, builder=builder, log=log,
            load_model=matrix.load_model, kv_pool_tokens=kv_pool_tokens(engine_cfg, variant),
            base_dir=Path(matrix.path).resolve().parent,
            # Zufälliger Startwert: Prompts eines Laufs treffen nie den Präfix-Cache eines früheren Laufs.
            _seed=secrets.randbits(40) << 20,
        )
        store.write_json("manifest.json", build_manifest(matrix, opts, params, builder, adapter))
        store.event("run_start", cell=cell.id, variant=variant.id, stages=opts.stages, smoke=opts.smoke)

        gpu_sampler = GpuSampler(store.samples, interval_s=1.0)
        periodic = PeriodicSampler(
            store.samples, params["common"]["sample_interval_s"], engine_metrics=lambda: adapter.metrics(session)
        )
        try:
            for stage in opts.stages:
                name, stage_run = STAGES[stage]
                if opts.resume and store.stage_done(stage):
                    log(f"Stufe {stage} ({name}): schon abgeschlossen, übersprungen")
                    continue
                log(f"Stufe {stage} ({name})")
                earlier = read_jsonl(store.dir / "events.jsonl")
                ctx.attempt = 1 + sum(1 for e in earlier if e.get("kind") == "stage_start" and e.get("stage") == stage)
                store.event("stage_start", stage=stage, attempt=ctx.attempt)
                t_stage = time.perf_counter()
                try:
                    if adapter.managed:
                        if started and params["common"]["restart_between_stages"]:
                            log("  Engine startet neu (leerer Cache für die Stufe)")
                            await adapter.stop()
                            started = False
                        if not started:
                            await adapter.start(store)
                            started = True
                            store.write_json("manifest.json", build_manifest(matrix, opts, params, builder, adapter))
                    ready_timeout = engine_cfg.ready_timeout_s if adapter.managed else min(60, engine_cfg.ready_timeout_s)
                    await adapter.wait_ready(session, ready_timeout, probe_body=ctx.body_for("off"), log=log)
                    if not adapter.managed:
                        adapter.load_time_s = None
                    if ctx.kv_pool_tokens is None:  # vLLM und SGLang bemessen ihren Pool selbst
                        ctx.kv_pool_tokens = await adapter.kv_capacity_tokens(session)
                    await _warmup(ctx, stage)
                    gpu_sampler.start()
                    periodic.start()
                    try:
                        await stage_run(ctx)
                    finally:
                        await periodic.stop()
                        gpu_sampler.stop()
                    await adapter.check_alive()
                    store.mark_stage_done(stage)
                    store.event("stage_end", stage=stage, ok=True, seconds=round(time.perf_counter() - t_stage, 1))
                    engine_failures = 0
                except EngineError as exc:
                    log(f"  ENGINE-FEHLER: {exc}")
                    store.event("stage_end", stage=stage, ok=False, error=str(exc))
                    exit_code = 2
                    failed.append(stage)
                    engine_failures += 1
                    if not (store.read_json("feasibility.json") or {}).get("loaded"):
                        # Die Zelle läuft gar nicht: Das ist ihr Ergebnis, weitere Stufen wären sinnlos.
                        store.write_json("feasibility.json", {"loaded": False, "error": str(exc)})
                        break
                    if engine_failures >= 2:
                        log("  Zwei Stufen in Folge sind an der Engine gescheitert; der Lauf endet hier.")
                        break
                    # Die Engine lief schon einmal: Die nächste Stufe bekommt einen frischen Start.
                    if adapter.managed and started:
                        await adapter.stop()
                        started = False
                except Exception as exc:  # Fehler in styrke selbst: festhalten, die übrigen Stufen trotzdem fahren
                    with open(store.dir / "stage_errors.log", "a", encoding="utf-8") as fh:
                        fh.write(f"===== Stufe {stage}, {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{traceback.format_exc()}\n")
                    log(f"  FEHLER in Stufe {stage}: {type(exc).__name__}: {exc} (Einzelheiten in stage_errors.log)")
                    store.event("stage_end", stage=stage, ok=False, error=f"{type(exc).__name__}: {exc}")
                    exit_code = exit_code or 1
                    failed.append(stage)
        finally:
            if started:
                log("Engine stoppt")
                await adapter.stop()
            store.event("run_end", exit_code=exit_code)
            store.close()
    if failed:
        log(f"Nicht abgeschlossen: Stufe {', '.join(str(n) for n in failed)}. Nachholen mit --resume {store.dir}")
    log(f"Fertig. Rohdaten: {store.dir}")
    run_log.close()
    return exit_code
