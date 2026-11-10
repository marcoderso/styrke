"""Gemischter Betrieb mit vorgegebener Ankunftsrate (Grundlage der Stufen 6 und 7).

Gespräche treffen zufällig ein, im Mittel so viele, dass sich die verlangte
Rate an Modellaufrufen ergibt (Poisson-Ankünfte). Jedes Gespräch gehört zu
einer Sitzungsart aus dem Lastmodell und läuft über mehrere Züge; zwischen
zwei Zügen liegt eine zufällige Denkpause des Nutzers. Wie viele Aufrufe
gleichzeitig laufen, ist nicht vorgegeben, sondern ergibt sich daraus, wie
schnell die Engine antwortet. Kommt sie nicht nach, wächst der Rückstau.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..client import CancelledWithRecord
from ..loadmodel import supported_users
from ..stats import mean
from .common import StageContext
from .load import Chat, latency_stats, throughput_between

FOLLOWUP_QUESTION_TOKENS = 20  # ungefähre Länge der Folgefrage selbst, ohne beigefügtes Material


@dataclass
class SessionShape:
    """Form einer Sitzungsart: wie lang der erste Prompt, wie viele Züge, wie lange Pausen."""

    kind: str
    share: float
    first_tokens: tuple[int, int]
    turns: int
    followup_tokens: tuple[int, int]
    max_tokens: int
    think_s: float


def session_shapes(ctx: StageContext, params: dict[str, Any]) -> tuple[list[SessionShape], list[dict[str, Any]]]:
    """Sitzungsarten mit ihren Anteilen aus dem Lastmodell; was nicht in den Kontext passt, wird gemeldet."""
    mix = ctx.load_model.get("session_mix") or {}
    shapes: list[SessionShape] = []
    dropped: list[dict[str, Any]] = []
    for kind, raw in (params.get("sessions") or {}).items():
        share = float((mix.get(kind) or {}).get("share", raw.get("share", 0.0)))
        if share <= 0:
            continue
        lo, hi = raw["first_tokens"]
        fu_lo, fu_hi = raw["followup_tokens"]
        growth = raw["turns"] * (fu_hi + raw["max_tokens"]) + 512
        cap = min(ctx.prompt_limit(), ctx.model.max_context - growth)
        if lo > cap:
            dropped.append({"kind": kind, "share": share, "reason": f"erster Prompt ab {lo} Tokens passt nicht in den Kontext"})
            continue
        shapes.append(
            SessionShape(kind, share, (lo, min(hi, cap)), raw["turns"], (fu_lo, fu_hi), raw["max_tokens"], raw["think_s"])
        )
    total = sum(s.share for s in shapes)
    for shape in shapes:
        shape.share = shape.share / total
    return shapes, dropped


def calls_per_session(shapes: list[SessionShape]) -> float:
    return sum(s.share * s.turns for s in shapes)


def log_uniform(rng: random.Random, lo: int, hi: int) -> int:
    """Zufallswert zwischen ``lo`` und ``hi``, bei dem kleine Werte häufiger sind als große."""
    if hi <= lo:
        return lo
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


class Scenario:
    """Ein Lastabschnitt mit fester Ankunftsrate."""

    def __init__(
        self,
        ctx: StageContext,
        *,
        stage: int,
        rate: float,
        shapes: list[SessionShape],
        thinking: str = "off",
        progress_s: float = 60.0,
    ):
        self.ctx = ctx
        self.stage = stage
        self.rate = rate
        self.shapes = shapes
        self.thinking = thinking
        self.progress_s = progress_s
        self.rng = random.Random(ctx.next_seed())
        self.records: list[dict[str, Any]] = []
        self.timeline: list[tuple[float, int, int]] = []  # (Zeit, laufende Aufrufe, offene Gespräche)
        self.inflight = 0
        self.sessions_active = 0
        self.sessions_started = 0
        self.t_start = 0.0
        self._stop = asyncio.Event()
        self._sessions: set[asyncio.Task] = set()
        self._helpers: list[asyncio.Task] = []

    async def _pause(self, seconds: float) -> bool:
        """Wartet; liefert ``True``, wenn der Abschnitt in der Zwischenzeit beendet wurde."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
            return True
        except asyncio.TimeoutError:
            return False

    async def _session(self, shape: SessionShape) -> None:
        rng = random.Random(self.ctx.next_seed())
        self.sessions_active += 1
        try:
            chat = Chat(self.ctx, stage=self.stage, thinking=self.thinking, tags={"kind": shape.kind, "rate": self.rate})
            await chat.open(log_uniform(rng, *shape.first_tokens))
            for turn in range(shape.turns):
                if turn > 0:
                    if shape.think_s > 0 and await self._pause(rng.expovariate(1.0 / shape.think_s)):
                        return
                    await chat.add_followup(log_uniform(rng, *shape.followup_tokens) - FOLLOWUP_QUESTION_TOKENS)
                if self._stop.is_set():
                    return
                self.inflight += 1
                try:
                    rec = await chat.send(shape.max_tokens)
                except CancelledWithRecord as exc:
                    self.records.append(exc.record)
                    raise
                finally:
                    self.inflight -= 1
                self.records.append(rec)
                if not rec["ok"]:
                    return  # nach einem Fehler gibt der Nutzer das Gespräch auf
        finally:
            self.sessions_active -= 1

    def _session_done(self, task: asyncio.Task) -> None:
        self._sessions.discard(task)
        if not task.cancelled() and task.exception() is not None:
            exc = task.exception()
            self.ctx.store.event("worker_error", stage=self.stage, error=f"{type(exc).__name__}: {exc}")
            self.ctx.log(f"  Gespräch abgebrochen: {type(exc).__name__}: {exc}")

    async def _arrivals(self) -> None:
        mean_gap = calls_per_session(self.shapes) / self.rate  # mittlerer Abstand zweier neuer Gespräche
        weights = [s.share for s in self.shapes]
        while not await self._pause(self.rng.expovariate(1.0 / mean_gap)):
            shape = self.rng.choices(self.shapes, weights=weights)[0]
            task = asyncio.create_task(self._session(shape))
            self._sessions.add(task)
            task.add_done_callback(self._session_done)
            self.sessions_started += 1

    async def _ticker(self) -> None:
        last_note = time.time()
        while True:
            await asyncio.sleep(1.0)
            now = time.time()
            self.timeline.append((now, self.inflight, self.sessions_active))
            self.ctx.store.samples.write(
                {
                    "ts": round(now, 3), "kind": "load", "stage": self.stage, "rate": self.rate,
                    "inflight": self.inflight, "sessions": self.sessions_active,
                }
            )
            if now - last_note >= self.progress_s:
                last_note = now
                done = [r for r in self.records if r["ok"]]
                recent = latency_stats(done[-50:])
                self.ctx.log(
                    f"    {(now - self.t_start) / 60:4.0f} min: {self.inflight} Aufrufe laufen, {len(done)} fertig, "
                    f"{len(self.records) - len(done)} Fehler, TTFT p50 zuletzt "
                    f"{'-' if recent['ttft_p50_s'] is None else format(recent['ttft_p50_s'], '.2f')} s"
                )

    async def run(self, duration_s: float, drain_s: float) -> None:
        """Lässt ``duration_s`` lang Gespräche eintreffen; laufende Aufrufe dürfen danach ``drain_s`` lang enden."""
        self.t_start = time.time()
        self._helpers = [asyncio.create_task(self._arrivals()), asyncio.create_task(self._ticker())]
        try:
            await asyncio.sleep(duration_s)
            self._stop.set()
            if self._sessions:
                await asyncio.wait(set(self._sessions), timeout=drain_s)
        finally:
            await self.abort()

    async def abort(self) -> None:
        self._stop.set()
        tasks = [*self._helpers, *self._sessions]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._helpers = []


# -- Auswertung ---------------------------------------------------------------------


def _r(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def _timeline_mean(timeline: list[tuple[float, int, int]], t0: float, t1: float) -> float | None:
    return _r(mean([inflight for t, inflight, _ in timeline if t0 <= t < t1]))


def summarize(
    scenario: Scenario, warmup_s: float, duration_s: float, load_model: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Kennzahlen eines Lastabschnitts; ausgewertet wird das Fenster nach der Anlaufzeit."""
    w0 = scenario.t_start + warmup_s
    w1 = scenario.t_start + duration_s
    window = max(1e-9, w1 - w0)
    calls = [r for r in scenario.records if w0 <= r["ts_submit"] < w1]
    ok = [r for r in calls if r["ok"]]
    unfinished = [r for r in calls if r["error"] == "abgebrochen"]
    errors = [r for r in calls if not r["ok"] and r["error"] != "abgebrochen"]
    quarter = window / 4
    prompt_sum = sum(r["prompt_tokens"] or 0 for r in ok)
    cached = [r["cached_tokens"] for r in ok if r.get("cached_tokens") is not None]
    inflight = [n for t, n, _ in scenario.timeline if w0 <= t < w1]

    row: dict[str, Any] = {
        "rate": scenario.rate,
        "achieved_rate": round(len(calls) / window, 4),
        "sessions_started": scenario.sessions_started,
        "calls": len(calls),
        "calls_ok": len(ok),
        "errors": len(errors),
        "unfinished": len(unfinished),
        "error_rate": round(len(errors) / len(calls), 4) if calls else 0.0,
        "unfinished_rate": round(len(unfinished) / len(calls), 4) if calls else 0.0,
        "error_examples": sorted({r["error"] for r in errors if r["error"]})[:3],
        "throughput_tps": round(throughput_between(scenario.records, w0, w1) or 0.0, 1),
        **latency_stats(ok),
        "service_mean_s": _r(mean([r["total_s"] for r in ok])),
        "prompt_tokens_mean": round(prompt_sum / len(ok)) if ok else None,
        "completion_tokens_mean": _r(mean([r["completion_tokens"] for r in ok]), 1),
        "cached_share": round(sum(cached) / prompt_sum, 4) if cached and prompt_sum else None,
        "inflight_mean": _r(mean(inflight)),
        "inflight_max": max(inflight) if inflight else None,
        "inflight_first_quarter": _timeline_mean(scenario.timeline, w0, w0 + quarter),
        "inflight_last_quarter": _timeline_mean(scenario.timeline, w1 - quarter, w1),
        "by_kind": {},
        "by_turn": {
            "first": latency_stats([r for r in ok if r.get("turn") == 0]),
            "followup": latency_stats([r for r in ok if (r.get("turn") or 0) > 0]),
        },
    }
    for shape in scenario.shapes:
        subset = [r for r in calls if r.get("kind") == shape.kind]
        row["by_kind"][shape.kind] = {"calls": len(subset), **latency_stats(subset)}
    if load_model:
        row["users_supported"] = users_supported(row["achieved_rate"], load_model)
    return row


def overload_reasons(row: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """Woran sich zeigt, dass die Engine die Rate nicht trägt. Leer heißt: getragen."""
    reasons: list[str] = []
    if row["error_rate"] > params["max_error_rate"]:
        reasons.append(f"Fehleranteil {row['error_rate']:.0%}")
    if row["unfinished_rate"] > params["max_unfinished_rate"]:
        reasons.append(f"{row['unfinished_rate']:.0%} der Aufrufe bis zum Ende des Abschnitts nicht fertig")
    first, last = row["inflight_first_quarter"], row["inflight_last_quarter"]
    if first is not None and last is not None and last > params["backlog_factor"] * first + params["backlog_slack"]:
        reasons.append(f"Rückstau wächst (laufende Aufrufe im Mittel {first:.1f} am Anfang, {last:.1f} am Ende)")
    return reasons


def users_supported(rate: float, load_model: dict[str, Any]) -> dict[str, int]:
    """Wie viele Berechtigte die Rate je Nutzungsprofil bedient."""
    out: dict[str, int] = {}
    for name, p in (load_model.get("profiles") or {}).items():
        out[name] = int(supported_users(rate, p["a"], p["t"], p["k"]))
    return out


def shapes_as_dicts(shapes: list[SessionShape]) -> list[dict[str, Any]]:
    return [{**asdict(s), "share": round(s.share, 4)} for s in shapes]
