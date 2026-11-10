"""Gesamtlauf über mehrere Zellen: ``styrke all``.

Fährt die gewählten Zellen nacheinander, jede als eigenen Lauf. Vorhandene
Läufe werden fortgesetzt und fertige übersprungen; nach einem Abbruch genügt
also derselbe Aufruf noch einmal. Der Stand steht in ``runs/batch.json``, am
Ende entsteht der Bericht.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Matrix
from .engines import SetupError, make_adapter
from .report import build_report, load_run
from .runner import RunOptions, run_cell


@dataclass
class Job:
    cell: str
    variant: str
    status: str = "offen"  # offen, läuft, fertig, Fehler
    run_dir: str | None = None
    exit_code: int | None = None
    note: str = ""


def plan(matrix: Matrix, cells: list[str] | None, all_variants: bool) -> list[Job]:
    """Je Zelle die erste Variante, mit ``all_variants`` alle."""
    jobs = []
    for cell in matrix.cells:
        if cells and cell.id not in cells:
            continue
        if not cell.enabled:
            continue
        for variant in cell.variants if all_variants else cell.variants[:1]:
            jobs.append(Job(cell.id, variant.id))
    return jobs


def existing_run(matrix: Matrix, job: Job, smoke: bool, external: bool) -> Path | None:
    """Neuester Laufordner derselben Zelle und Variante, der zur Art des Laufs passt."""
    found = None
    for path in sorted(Path(matrix.host.runs_dir).glob("*")):
        if not (path / "manifest.json").is_file():
            continue
        run = load_run(path)
        is_external = (run["manifest"].get("engine") or {}).get("kind") == "external"
        if (run["cell"], run["variant"], run["smoke"], is_external) == (job.cell, job.variant, smoke, external):
            found = path
    return found


def stages_done(run_dir: Path) -> list[int]:
    try:
        return json.loads((run_dir / "state.json").read_text(encoding="utf-8")).get("stages_done") or []
    except (OSError, ValueError):
        return []


def _hf() -> str | None:
    local = Path(sys.executable).parent / "hf"
    return str(local) if local.is_file() else shutil.which("hf") or shutil.which("huggingface-cli")


def preparation_steps(matrix: Matrix, jobs: list[Job]) -> list[tuple[str, list[str]]]:
    """Was vor dem Gesamtlauf noch fehlt: (Beschreibung, Kommando) für Images und Modelle."""
    steps: list[tuple[str, list[str]]] = []
    engines = {matrix.cell(j.cell).engine for j in jobs}
    models = {matrix.cell(j.cell).model for j in jobs}
    for name in sorted(engines):
        engine = matrix.engines[name]
        present = subprocess.run(["docker", "image", "inspect", engine.image], capture_output=True).returncode == 0
        if present:
            continue
        build_dir = Path(matrix.path).resolve().parent / "engines" / engine.kind
        if (build_dir / "Dockerfile").is_file():
            steps.append((f"Image {engine.image} bauen", ["docker", "build", "-t", engine.image, str(build_dir)]))
        else:
            steps.append((f"Image {engine.image} ziehen", ["docker", "pull", engine.image]))
    for name in sorted(models):
        model = matrix.models[name]
        target = Path(matrix.host.models_dir) / model.dir
        if not (target / "config.json").is_file():
            steps.append((f"Modell {model.repo} herunterladen", [_hf() or "hf", "download", model.repo, "--local-dir", str(target)]))
    return steps


async def gpu_problem(matrix: Matrix, jobs: list[Job]) -> str | None:
    """Ist die GPU belegt? Geprüft wird mit dem Adapter der ersten Zelle, alles andere ist dann schon vorhanden."""
    if not jobs:
        return None
    cell = matrix.cell(jobs[0].cell)
    adapter = make_adapter(matrix.host, matrix.engines[cell.engine], matrix.models[cell.model], cell.variant(jobs[0].variant))
    try:
        await adapter.check_setup()
    except SetupError as exc:
        return next((p for p in exc.problems if "GPU" in p), exc.problems[0])
    return None


def write_state(matrix: Matrix, jobs: list[Job], stages: list[int], started: float, finished: bool) -> None:
    path = Path(matrix.host.runs_dir)
    path.mkdir(parents=True, exist_ok=True)
    state = {"started": started, "updated": time.time(), "finished": finished, "stages": stages, "jobs": [asdict(j) for j in jobs]}
    tmp = path / "batch.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path / "batch.json")


async def run_batch(
    matrix: Matrix, jobs: list[Job], stages: list[int], *, log: Callable[[str], None], smoke: bool = False,
    endpoint: str | None = None, fresh: bool = False, report_path: str | None = "bericht.html",
) -> int:
    """Fährt alle Aufträge nacheinander. Rückgabewert 0, wenn jede Zelle alle Stufen abgeschlossen hat."""
    started = time.time()
    for job in jobs:  # vorab eintragen, was schon da ist
        found = None if fresh else existing_run(matrix, job, smoke, bool(endpoint))
        if found:
            job.run_dir = str(found)
            if all(s in stages_done(found) for s in stages):
                job.status, job.note = "fertig", "schon vorhanden"
    write_state(matrix, jobs, stages, started, False)

    for number, job in enumerate(jobs, 1):
        if job.status == "fertig":
            log(f"=== {number} von {len(jobs)}: {job.cell} / {job.variant}: schon abgeschlossen, übersprungen")
            continue
        log(f"=== {number} von {len(jobs)}: {job.cell} / {job.variant}" + (" (wird fortgesetzt)" if job.run_dir else ""))
        job.status = "läuft"
        write_state(matrix, jobs, stages, started, False)
        opts = RunOptions(
            matrix_path=str(matrix.path), cell_id=job.cell, variant_id=job.variant, stages=stages,
            endpoint=endpoint, resume=job.run_dir, smoke=smoke,
        )
        try:
            job.exit_code = await run_cell(opts, log=log)
        except BaseException:
            job.status, job.note = "offen", "abgebrochen"
            write_state(matrix, jobs, stages, started, False)
            raise
        found = existing_run(matrix, job, smoke, bool(endpoint))
        job.run_dir = str(found) if found else job.run_dir
        job.status = "fertig" if job.exit_code == 0 else "Fehler"
        job.note = {0: "", 1: "eine Stufe scheiterte an styrke", 2: "Engine-Fehler", 3: "Vorbereitung unvollständig"}.get(job.exit_code, "")
        write_state(matrix, jobs, stages, started, False)
    write_state(matrix, jobs, stages, started, True)

    log("")
    log("Gesamtlauf beendet:")
    for job in jobs:
        log(f"  {job.cell + ' / ' + job.variant:<34} {job.status}" + (f" ({job.note})" if job.note else ""))
    run_dirs = [Path(j.run_dir) for j in jobs if j.run_dir]
    if report_path and run_dirs:
        opts: dict[str, Any] = {"ttft_p95": None, "total_p95": None, "cost_month": None, "profile": "mittel"}
        if "mittel" not in (matrix.load_model.get("profiles") or {}):
            opts["profile"] = next(iter(matrix.load_model.get("profiles") or {"mittel": {}}))
        Path(report_path).write_text(build_report(run_dirs, opts), encoding="utf-8")
        log(f"Bericht: {report_path}")
    return 0 if all(j.status == "fertig" for j in jobs) else 2
