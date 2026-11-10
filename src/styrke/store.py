"""Rohdatenablage: ein Ordner je Lauf.

Alles wird zeilenweise als JSON geschrieben und sofort auf die Platte gebracht.
So bleibt ein Lauf auch nach einem Abbruch auswertbar und fortsetzbar.

    manifest.json     Hardware, Image, Modell, alle Startoptionen
    requests.jsonl    eine Zeile je Anfrage, mit Zeitstempeln der Antwortstücke
    samples.jsonl     GPU, Rechner und Engine in festen Abständen
    events.jsonl      Ereignisse für Protokoll und TUI
    engine.log        Ausgabe der Engine
    state.json        welche Stufen abgeschlossen sind
    summary_s*.json   Kurzauswertung je Stufe
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonlWriter:
    """Hängt JSON-Zeilen an eine Datei an; aus mehreren Threads nutzbar."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Liest eine JSONL-Datei; eine abgeschnittene letzte Zeile wird übergangen."""
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


class RunStore:
    """Der Ordner eines Laufs samt seiner Dateien."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "answers").mkdir(exist_ok=True)
        self.requests = JsonlWriter(self.dir / "requests.jsonl")
        self.samples = JsonlWriter(self.dir / "samples.jsonl")
        self.events = JsonlWriter(self.dir / "events.jsonl")

    @classmethod
    def create(cls, runs_dir: str | Path, cell: str, variant: str) -> "RunStore":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(Path(runs_dir) / f"{stamp}_{_slug(cell)}_{_slug(variant)}")

    # -- Ereignisse -----------------------------------------------------

    def event(self, kind: str, **fields: Any) -> None:
        self.events.write({"ts": round(time.time(), 3), "kind": kind, **fields})

    # -- einzelne JSON-Dateien -------------------------------------------

    def write_json(self, name: str, obj: Any) -> Path:
        path = self.dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path

    def read_json(self, name: str) -> Any | None:
        path = self.dir / name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    # -- Fortschritt -----------------------------------------------------

    def state(self) -> dict[str, Any]:
        return self.read_json("state.json") or {"stages_done": []}

    def stage_done(self, stage: int) -> bool:
        return stage in self.state().get("stages_done", [])

    def mark_stage_done(self, stage: int) -> None:
        state = self.state()
        done = set(state.get("stages_done", []))
        done.add(stage)
        state["stages_done"] = sorted(done)
        state["updated"] = datetime.now().isoformat(timespec="seconds")
        self.write_json("state.json", state)

    def close(self) -> None:
        for writer in (self.requests, self.samples, self.events):
            writer.close()
