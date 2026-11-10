"""Sammler: GPU, Rechner und Engine in festen Abständen.

Alle Messreihen landen in ``samples.jsonl``, unterschieden durch ``kind``.
Fehlt ``nvidia-smi``, entfällt die GPU-Reihe ohne Fehler.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Awaitable, Callable

from .store import JsonlWriter

GPU_FIELDS = ["utilization.gpu", "memory.used", "memory.total", "power.draw", "temperature.gpu"]
GPU_KEYS = ["util_pct", "mem_used_mib", "mem_total_mib", "power_w", "temp_c"]


def _num(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def gpu_snapshot() -> list[dict[str, Any]]:
    """Einmalige Abfrage aller GPUs; leere Liste, wenn nvidia-smi fehlt."""
    if not shutil.which("nvidia-smi"):
        return []
    query = "index,name,driver_version," + ",".join(GPU_FIELDS)
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 + len(GPU_FIELDS):
            continue
        row: dict[str, Any] = {"index": parts[0], "name": parts[1], "driver": parts[2]}
        row.update({key: _num(val) for key, val in zip(GPU_KEYS, parts[3:])})
        gpus.append(row)
    return gpus


def gpu_processes() -> list[str]:
    """Prozesse, die gerade GPU-Speicher belegen (eine Zeile je Prozess)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


class GpuSampler:
    """Liest ``nvidia-smi -l`` in einem eigenen Thread mit."""

    def __init__(self, writer: JsonlWriter, interval_s: float = 1.0):
        self.writer = writer
        self.interval_s = max(1, int(round(interval_s)))
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        query = "index," + ",".join(GPU_FIELDS)
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-l", str(self.interval_s)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 1 + len(GPU_FIELDS):
                continue
            row: dict[str, Any] = {"ts": round(time.time(), 3), "kind": "gpu", "gpu": parts[0]}
            row.update({key: _num(val) for key, val in zip(GPU_KEYS, parts[1:])})
            try:
                self.writer.write(row)
            except ValueError:  # Datei schon geschlossen
                return

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)
        self._proc = None
        self._thread = None


def host_snapshot() -> dict[str, Any]:
    """Last des Rechners, freier Speicher und die CPU-Zeit von styrke selbst."""
    row: dict[str, Any] = {}
    try:
        with open("/proc/loadavg", encoding="ascii") as fh:
            row["load1"] = float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    row["mem_total_gib" if key == "MemTotal" else "mem_available_gib"] = round(
                        int(rest.split()[0]) / 1024 / 1024, 2
                    )
    except (OSError, ValueError, IndexError):
        pass
    times = os.times()
    row["styrke_cpu_s"] = round(times.user + times.system, 2)
    return row


class PeriodicSampler:
    """Ruft Rechner- und Engine-Werte regelmäßig ab (läuft in der Ereignisschleife)."""

    def __init__(
        self,
        writer: JsonlWriter,
        interval_s: float,
        engine_metrics: Callable[[], Awaitable[dict[str, Any] | None]] | None = None,
    ):
        self.writer = writer
        self.interval_s = interval_s
        self.engine_metrics = engine_metrics
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            ts = round(time.time(), 3)
            self.writer.write({"ts": ts, "kind": "host", **host_snapshot()})
            if self.engine_metrics is not None:
                try:
                    data = await self.engine_metrics()
                except Exception:  # Metriken sind Beiwerk; ein Fehler darf die Messung nicht stören
                    data = None
                if data:
                    self.writer.write({"ts": ts, "kind": "engine", **data})
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
