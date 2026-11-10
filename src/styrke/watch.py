"""Betrachter für einen Lauf: ``styrke watch``.

Der Betrachter liest nur die Dateien des Laufordners und zeigt daraus den
Stand. Er hängt nicht am messenden Prozess: Man kann ihn jederzeit in einem
zweiten Fenster starten und beenden, auch für einen schon beendeten Lauf, und
er kostet die Messung nichts außer dem Lesen der Dateien.

Zahlen zu Anfragen stammen aus ``requests.jsonl``. Dort steht eine Anfrage
erst, wenn sie beendet ist; noch laufende lange Anfragen fehlen also in den
Werten der letzten Sekunden.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .stages import STAGES
from .stats import percentile

BARS = "▁▂▃▄▅▆▇█"
STALE_S = 150  # so lange ohne neue Daten gilt ein Lauf noch als laufend (beim Laden kommt eine Zeile je Minute)
BIN_S = 5  # Breite einer Zeitscheibe des Durchsatzverlaufs in Sekunden
STAGE_NAMES = {n: name for n, (name, _) in STAGES.items()}
ENGINE_HINTS = ("running", "waiting", "queue", "kv_cache", "token_usage", "used_tokens", "hit_rate", "evict")


class JsonlTail:
    """Liest aus einer wachsenden JSONL-Datei jeweils nur die neuen, vollständigen Zeilen."""

    def __init__(self, path: Path):
        self.path = path
        self.pos = 0
        self.rest = b""

    def read_new(self) -> list[dict[str, Any]]:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.pos)
                data = fh.read()
                self.pos = fh.tell()
        except OSError:
            return []
        lines = (self.rest + data).split(b"\n")
        self.rest = lines.pop()  # unvollständige letzte Zeile bleibt für das nächste Mal
        out = []
        for line in lines:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out


def sparkline(values: list[float]) -> str:
    top = max(values, default=0.0)
    if top <= 0:
        return BARS[0] * len(values)
    return "".join(BARS[min(len(BARS) - 1, int(v / top * (len(BARS) - 1) + 0.5))] for v in values)


def flatten_numbers(data: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(flatten_numbers(value, f"{prefix}{key}." if isinstance(value, dict) else f"{prefix}{key}"))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        out[prefix.rstrip(".")] = float(data)
    return out


def clock(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class RunState:
    """Stand eines Laufs, aus den Dateien des Laufordners fortgeschrieben."""

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self._events = JsonlTail(self.dir / "events.jsonl")
        self._requests = JsonlTail(self.dir / "requests.jsonl")
        self._samples = JsonlTail(self.dir / "samples.jsonl")
        self.manifest: dict[str, Any] = {}
        self.planned: list[int] = []
        self.stages: dict[int, dict[str, Any]] = {}
        self.current: int | None = None
        self.activity = "wartet auf den Start"
        self.t_run: float | None = None
        self.finished: int | None = None  # Rückgabewert, sobald der Lauf beendet ist
        self.counts: dict[int, dict[str, int]] = {}  # Stufe -> beendete Anfragen und Fehler (ohne Aufwärmen)
        self.total_requests = 0
        self.recent: deque[dict[str, Any]] = deque()  # beendete Anfragen der letzten Minuten
        self.bins: dict[int, float] = {}  # Zeitscheibe -> erzeugte Tokens
        self.gpus: dict[str, dict[str, Any]] = {}
        self.host: dict[str, Any] = {}
        self.load: dict[str, Any] | None = None
        self.engine: dict[str, float] = {}
        self.last_data = 0.0  # Zeitstempel des jüngsten Datensatzes; die Uhr des Betrachters für beendete Läufe

    # -- Fortschreiben ----------------------------------------------------

    def refresh(self) -> None:
        if not self.manifest:
            try:
                self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        for event in self._events.read_new():
            self._on_event(event)
        for rec in self._requests.read_new():
            self._on_request(rec)
        for sample in self._samples.read_new():
            self._on_sample(sample)
        try:  # während die Engine lädt, entstehen keine Messwerte, aber Zeilen im Protokoll
            self.last_data = max(self.last_data, (self.dir / "run.log").stat().st_mtime)
        except OSError:
            pass

    def _on_event(self, e: dict[str, Any]) -> None:
        kind, ts = e.get("kind"), e.get("ts", 0.0)
        self.last_data = max(self.last_data, ts)
        stage = e.get("stage")
        if kind == "run_start":
            self.planned = sorted(set(self.planned) | set(e.get("stages") or []))
            self.t_run = self.t_run or ts
            self.finished = None
        elif kind == "stage_start":
            self.current = stage
            self.stages[stage] = {"state": "läuft", "t0": ts, "t1": None, "attempt": e.get("attempt", 1), "error": None}
            self.counts[stage] = {"n": 0, "errors": 0}  # ein neuer Anlauf zählt von vorn
            self.activity = "Engine wird vorbereitet"
            self.load = None
        elif kind == "stage_end" and stage in self.stages:
            self.stages[stage].update(state="fertig" if e.get("ok") else "Fehler", t1=ts, error=e.get("error"))
            self.activity = "zwischen zwei Stufen"
        elif kind == "engine_start":
            self.activity = "Engine startet und lädt das Modell"
        elif kind == "level_start":
            level = e.get("level")
            if stage in (6, 7):
                self.activity = f"Ankunftsrate {level:g} Aufrufe/s"
            else:
                fill = f", Füllstand {e['fill']:.0%}" if "fill" in e else ""
                self.activity = f"Laststufe {level} parallel{fill}"
        elif kind == "fill_start":
            self.activity = f"füllt den KV-Pool auf {e.get('fill', 0):.0%}"
        elif kind == "flood_start":
            self.activity = f"Verdrängung: {e.get('tokens')} Tokens fremder Text"
        elif kind == "injection_start":
            self.activity = f"große Anfrage mit {e.get('big_target')} Tokens läuft"
        elif kind == "injection_end":
            self.activity = "Erholung nach der großen Anfrage"
        elif kind == "run_end":
            self.finished = e.get("exit_code", 0)
            self.current = None
            self.activity = "Lauf beendet"

    def _on_request(self, r: dict[str, Any]) -> None:
        self.total_requests += 1
        t_end = r.get("ts_submit", 0.0) + (r.get("total_s") or 0.0)
        self.last_data = max(self.last_data, t_end)
        if r.get("what") == "warmup":
            return
        if self.activity in ("Engine wird vorbereitet", "Engine startet und lädt das Modell"):
            self.activity = "misst"
        info = self.stages.get(r.get("stage")) or {}
        if r.get("attempt", 1) >= info.get("attempt", 1):  # Anfragen eines früheren Anlaufs zählen nicht mit
            count = self.counts.setdefault(r.get("stage"), {"n": 0, "errors": 0})
            count["n"] += 1
            if not r.get("ok") and r.get("error") != "abgebrochen":
                count["errors"] += 1
        self.recent.append(
            {"t": t_end, "ok": r.get("ok"), "ttft": r.get("ttft_s"), "tps": r.get("decode_tps"), "what": r.get("what")}
        )
        chunks = r.get("chunk_ms") or []
        if chunks:
            per_chunk = (r.get("completion_tokens") or len(chunks)) / len(chunks)
            base = r["ts_submit"]
            for ms in chunks:
                idx = int((base + ms / 1000.0) // BIN_S)
                self.bins[idx] = self.bins.get(idx, 0.0) + per_chunk

    def _on_sample(self, s: dict[str, Any]) -> None:
        self.last_data = max(self.last_data, s.get("ts", 0.0))
        kind = s.get("kind")
        if kind == "gpu":
            self.gpus[str(s.get("gpu"))] = s
        elif kind == "host":
            self.host = s
        elif kind == "load":
            self.load = s
        elif kind == "engine":
            flat = flatten_numbers({k: v for k, v in s.items() if k not in ("ts", "kind")})
            self.engine = {k: v for k, v in flat.items() if any(h in k.lower() for h in ENGINE_HINTS)}

    # -- Abgeleitete Werte ---------------------------------------------------

    def now(self) -> float:
        """Jetzt bei einem laufenden Lauf, sonst der Zeitpunkt des letzten Datensatzes."""
        return time.time() if self.finished is None and time.time() - self.last_data < STALE_S else self.last_data

    def window(self, seconds: float = 30.0) -> dict[str, Any]:
        """Kennzahlen der in den letzten ``seconds`` beendeten Anfragen."""
        now = self.now()
        while self.recent and self.recent[0]["t"] < now - 600:
            self.recent.popleft()
        rows = [r for r in self.recent if r["t"] >= now - seconds and r["ok"]]
        first = int((now - seconds) // BIN_S)
        tokens = sum(v for idx, v in self.bins.items() if first <= idx < int(now // BIN_S))
        span = max(BIN_S, (int(now // BIN_S) - first) * BIN_S)
        return {
            "n": len(rows),
            "tps": tokens / span,
            "ttft_p50": percentile([r["ttft"] for r in rows], 50),
            "ttft_p95": percentile([r["ttft"] for r in rows], 95),
            "decode_tps": percentile([r["tps"] for r in rows], 50),
        }

    def history(self, slices: int = 60) -> list[float]:
        """Durchsatz der letzten Zeitscheiben in Tokens pro Sekunde, älteste zuerst."""
        last = int(self.now() // BIN_S) - 1
        for idx in [i for i in self.bins if i < last - 2 * slices]:
            del self.bins[idx]
        return [self.bins.get(idx, 0.0) / BIN_S for idx in range(last - slices + 1, last + 1)]


# -- Darstellung ---------------------------------------------------------------------

SYMBOL = {"fertig": ("✓", "green"), "läuft": ("▶", "bold yellow"), "Fehler": ("✗", "bold red"), "offen": ("·", "dim")}


def _num(value: Any, digits: int = 1, unit: str = "") -> str:
    return "-" if value is None else f"{value:.{digits}f}{unit}"


def render(state: RunState, log_lines: int = 8) -> Group:
    m = state.manifest
    now = state.now()
    variant = (m.get("variant") or {}).get("id")
    status = (
        f"beendet, Rückgabewert {state.finished}" if state.finished is not None
        else f"läuft seit {clock(now - state.t_run) if state.t_run else '-'}"
    )
    if state.finished is None and state.last_data and time.time() - state.last_data >= STALE_S:
        status = f"keine neuen Daten seit {clock(time.time() - state.last_data)} (abgebrochen?)"
    head = Text.assemble(
        ("styrke", "bold"), "  ", f"{m.get('cell', '?')} / {variant or '?'}", "   ",
        ((m.get("model") or {}).get("repo") or "", "dim"), "   ", (status, "bold"),
    )
    if m.get("smoke"):
        head.append("   Funktionstest, keine Messung", style="yellow")

    stages = Table(box=None, padding=(0, 1), show_header=True, header_style="dim")
    for column, justify in (
        ("", "left"), ("Stufe", "left"), ("Stand", "left"), ("Dauer", "right"), ("Anfragen", "right"), ("Fehler", "right"),
    ):
        stages.add_column(column, justify=justify)  # type: ignore[arg-type]
    for n in sorted(set(state.planned) | set(state.stages)):
        info = state.stages.get(n) or {"state": "offen", "t0": None, "t1": None}
        symbol, style = SYMBOL[info["state"]]
        duration = None if info["t0"] is None else (info["t1"] or now) - info["t0"]
        count = state.counts.get(n)
        errors = Text(str(count["errors"]), style="bold red" if count["errors"] else "") if count else Text("")
        stages.add_row(
            Text(symbol, style=style), f"{n} {STAGE_NAMES.get(n, '?')}", Text(info["state"], style=style), clock(duration),
            str(count["n"]) if count else "", errors,
        )

    w = state.window(30)
    live = Table.grid(padding=(0, 2))
    live.add_column(style="dim")
    live.add_column()
    live.add_row("Gerade", state.activity)
    live.add_row("Letzte 30 s", f"{w['n']} beendet, {_num(w['tps'], 0)} tok/s gesamt")
    live.add_row("Erstes Token", f"p50 {_num(w['ttft_p50'], 2, ' s')}, p95 {_num(w['ttft_p95'], 2, ' s')}")
    live.add_row("Tempo je Anfrage", f"{_num(w['decode_tps'], 1)} tok/s (Median)")
    if state.load:
        live.add_row("An der Engine", f"{state.load.get('inflight')} Aufrufe laufen, {state.load.get('sessions')} Gespräche offen")
    history = state.history()
    live.add_row("Durchsatz, 5 min", f"{sparkline(history)}  Spitze {_num(max(history, default=0.0), 0)} tok/s")

    machine = Table.grid(padding=(0, 2))
    machine.add_column(style="dim")
    machine.add_column()
    for idx, gpu in sorted(state.gpus.items()):
        machine.add_row(
            f"GPU {idx}",
            f"{_num(gpu.get('util_pct'), 0, ' %')} Last, {_num(gpu.get('mem_used_mib'), 0)} von "
            f"{_num(gpu.get('mem_total_mib'), 0)} MiB, {_num(gpu.get('power_w'), 0, ' W')}, {_num(gpu.get('temp_c'), 0, ' °C')}",
        )
    if state.host:
        machine.add_row(
            "Rechner",
            f"{_num(state.host.get('mem_available_gib'), 1)} von {_num(state.host.get('mem_total_gib'), 0)} GiB RAM frei, "
            f"Last {_num(state.host.get('load1'), 1)}",
        )
    for key, value in list(sorted(state.engine.items()))[:6]:
        machine.add_row(key[-44:], f"{value:g}")
    if not (state.gpus or state.host or state.engine):
        machine.add_row("", "noch keine Messreihen")

    try:
        lines = (state.dir / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()[-log_lines:]
    except OSError:
        lines = []
    return Group(
        Panel(head, border_style="dim"),
        Panel(stages, title="Stufen", title_align="left", border_style="dim"),
        Panel(live, title="Messung", title_align="left", border_style="dim"),
        Panel(machine, title="Maschine und Engine", title_align="left", border_style="dim"),
        Panel(Text("\n".join(lines) or "noch keine Ausgabe", overflow="ellipsis", no_wrap=True), title="Protokoll",
              title_align="left", border_style="dim"),
    )


def newest_run(runs_dir: str | Path) -> Path | None:
    runs = [p for p in Path(runs_dir).glob("*") if (p / "events.jsonl").is_file()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def watch(run_dir: str | Path, refresh_s: float = 1.0, once: bool = False) -> int:
    state = RunState(run_dir)
    state.refresh()
    console = Console()
    if once:
        console.print(render(state))
        return 0
    try:
        with Live(render(state), console=console, screen=True, refresh_per_second=4) as live:
            while True:
                time.sleep(refresh_s)
                state.refresh()
                live.update(render(state))
    except KeyboardInterrupt:
        pass
    console.print(render(state))  # letzter Stand bleibt nach dem Beenden stehen
    return 0
