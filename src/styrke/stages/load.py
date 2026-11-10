"""Bausteine, die mehrere Stufen teilen: Dauerlast, Gespräche, Auswertung über Zeitabschnitte."""

from __future__ import annotations

import asyncio
import itertools
import random
import uuid
from typing import Any

from ..client import CancelledWithRecord, RequestSpec
from ..stats import percentile
from .common import StageContext

# -- Dauerlast im geschlossenen Kreislauf ------------------------------------------


class ClosedLoop:
    """N Arbeiter schicken ununterbrochen kurze Anfragen, bis ``stop`` gerufen wird.

    Jeder Arbeiter wartet auf das Ende seiner Anfrage und schickt sofort die
    nächste. Die Zahl gleichzeitiger Anfragen ist damit fest vorgegeben.
    """

    def __init__(
        self,
        ctx: StageContext,
        n: int,
        *,
        stage: int,
        prompt_tokens: int,
        max_tokens: int,
        thinking: str = "off",
        tags: dict[str, Any] | None = None,
    ):
        self.ctx = ctx
        self.n = n
        self.stage = stage
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.tags = dict(tags or {})
        self.records: list[dict[str, Any]] = []
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._seq = itertools.count()

    async def _worker(self, wid: int) -> None:
        ctx = self.ctx
        body = ctx.body_for(self.thinking)
        prefix = ctx.system_prefix_for(self.thinking)
        await asyncio.sleep(0.5 * wid / max(1, self.n))  # Starts leicht versetzen
        while not self._stop.is_set():
            prompt = await ctx.build(self.prompt_tokens, system_prefix=prefix)
            spec = RequestSpec(
                messages=prompt.messages,
                max_tokens=self.max_tokens,
                body=body,
                tags={
                    "stage": self.stage,
                    **self.tags,
                    "level": self.n,
                    "worker": wid,
                    "seq": next(self._seq),
                    "thinking": self.thinking,
                    "prompt_target": self.prompt_tokens,
                    "prompt_local": prompt.local_tokens,
                },
            )
            try:
                rec = await ctx.request(spec)
            except CancelledWithRecord as exc:
                self.records.append(exc.record)
                raise
            self.records.append(rec)
            if not rec["ok"]:
                await asyncio.sleep(1.0)  # bei Fehlern nicht im Kreis rennen

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._worker(w)) for w in range(self.n)]

    async def stop(self, drain_s: float) -> list[dict[str, Any]]:
        """Keine neuen Anfragen mehr; laufende dürfen ``drain_s`` lang zu Ende gehen."""
        self._stop.set()
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=drain_s)
        await self.abort()
        return self.records

    async def abort(self) -> None:
        """Bricht ab, was noch läuft; die bis dahin gemessenen Datensätze bleiben erhalten."""
        self._stop.set()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        outcomes = await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        for outcome in outcomes:
            if isinstance(outcome, Exception) and not isinstance(outcome, asyncio.CancelledError):
                self.ctx.store.event(
                    "worker_error", stage=self.stage, level=self.n, error=f"{type(outcome).__name__}: {outcome}"
                )
                self.ctx.log(f"  Arbeiter abgebrochen: {type(outcome).__name__}: {outcome}")


# -- Gespräche über mehrere Züge -----------------------------------------------------

FOLLOWUPS = [
    "Fasse das in drei Sätzen zusammen.",
    "Welche zwei Punkte davon sind am wichtigsten, und warum?",
    "Erkläre den schwierigsten Teil noch einmal mit einfacheren Worten.",
    "Nenne fünf Stichworte, unter denen man das ablegen könnte.",
    "Was davon ist für jemanden ohne Vorwissen am schwersten zu verstehen?",
    "Formuliere daraus eine kurze Notiz für einen Kollegen, der wenig Zeit hat.",
    "Welche Fragen bleiben dabei offen?",
    "Gib mir eine Gliederung mit Zwischenüberschriften dazu.",
    "Welche Zahlen und Namen kommen darin vor? Liste sie auf.",
    "Schreibe dazu einen Absatz, der die Gegenposition vertritt.",
    "Woran würde man merken, dass eine Aussage darin nicht stimmt?",
    "Mach daraus eine Liste mit Aufgaben, die sich daraus ergeben.",
]
MATERIAL_HEADER = "\n\nDazu liegt jetzt dieses Material vor:\n\n"


class Chat:
    """Ein Gespräch: hält den Verlauf und schickt Zug um Zug den ganzen Verlauf mit.

    So arbeitet jede Chat-Oberfläche. Für die Engine ist der Anfang einer
    Folgefrage deshalb schon bekannt, und der Präfix-Cache kann greifen.
    Die Antworten des Modells bleiben nur im Speicher, bis das Gespräch endet.
    """

    def __init__(self, ctx: StageContext, *, stage: int, thinking: str = "off", tags: dict[str, Any] | None = None):
        self.ctx = ctx
        self.stage = stage
        self.thinking = thinking
        self.tags = dict(tags or {})
        self.id = uuid.uuid4().hex[:10]
        self.messages: list[dict[str, Any]] = []
        self.turn = 0
        self.first_target = 0
        self.rng = random.Random(ctx.next_seed())
        self.last_prompt_tokens: int | None = None
        self.cold_ttft_s: float | None = None  # Zeit bis zum ersten Token beim ersten, kalten Zug

    async def open(self, target_tokens: int) -> None:
        """Erste Nachricht mit ``target_tokens`` Tokens (Text aus dem Korpus mit Aufgabe)."""
        prompt = await self.ctx.build(target_tokens, system_prefix=self.ctx.system_prefix_for(self.thinking))
        self.messages = list(prompt.messages)
        self.first_target = target_tokens

    async def add_followup(self, material_tokens: int = 0) -> None:
        """Hängt eine Folgefrage an, auf Wunsch mit zusätzlichem Text (etwa einem Werkzeugergebnis)."""
        text = self.rng.choice(FOLLOWUPS)
        if material_tokens >= 20:
            rng = random.Random(self.ctx.next_seed())
            doc = await asyncio.to_thread(self.ctx.builder.document, rng, material_tokens)
            text += MATERIAL_HEADER + doc
        self.messages.append({"role": "user", "content": text})

    async def send(self, max_tokens: int, **tags: Any) -> dict[str, Any]:
        """Schickt den Verlauf ab und hängt die Antwort an. Liefert den Messdatensatz."""
        spec = RequestSpec(
            messages=list(self.messages),
            max_tokens=max_tokens,
            body=self.ctx.body_for(self.thinking),
            tags={
                "stage": self.stage,
                **self.tags,
                "session": self.id,
                "turn": self.turn,
                "thinking": self.thinking,
                "prompt_target": self.first_target,
                **tags,
            },
            keep_text=True,
        )
        rec = await self.ctx.request(spec)
        answer = rec.pop("_text", None)
        if rec["ok"]:
            if self.turn == 0:
                self.cold_ttft_s = rec["ttft_s"]
            # Ein leerer Beitrag des Assistenten wird von manchen Chat-Vorlagen abgelehnt.
            self.messages.append({"role": "assistant", "content": answer or "(keine Antwort)"})
            self.turn += 1
            self.last_prompt_tokens = rec.get("prompt_tokens")
        return rec


def cached_share(rec: dict[str, Any]) -> float | None:
    """Anteil des Prompts, der laut Engine aus dem Präfix-Cache kam; ``None``, wenn sie nichts meldet."""
    cached, prompt = rec.get("cached_tokens"), rec.get("prompt_tokens")
    if cached is None or not prompt:
        return None
    return round(cached / prompt, 4)


MIN_COLD_TTFT_S = 0.25  # darunter lässt sich ein Cache-Treffer an der Antwortzeit nicht sicher erkennen


def looks_cached(ttft_s: float | None, cold_ttft_s: float | None) -> bool | None:
    """Urteil nach Antwortzeit: Kam das erste Token in höchstens der halben Zeit des kalten Prompts?

    Das Urteil braucht keine Angaben der Engine und gilt deshalb für alle
    Engines gleich. ``None``, wenn schon der kalte Prompt so schnell war, dass
    sich kalt und warm nicht sicher unterscheiden lassen.
    """
    if ttft_s is None or not cold_ttft_s or cold_ttft_s < MIN_COLD_TTFT_S:
        return None
    return ttft_s <= 0.5 * cold_ttft_s


# -- Auswertung über Zeitabschnitte -----------------------------------------------------


def token_events(records: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """(Zeitpunkt, Tokens) für jedes Antwortstück aller Anfragen, nach Zeit sortiert.

    Die Tokenzahl einer Anfrage wird gleichmäßig auf ihre Stücke verteilt;
    Engines, die mehrere Tokens je Stück senden, werden so richtig gezählt.
    """
    events: list[tuple[float, float]] = []
    for rec in records:
        chunks = rec.get("chunk_ms") or []
        if not chunks:
            continue
        per_chunk = (rec.get("completion_tokens") or len(chunks)) / len(chunks)
        base = rec["ts_submit"]
        events.extend((base + ms / 1000.0, per_chunk) for ms in chunks)
    events.sort()
    return events


def throughput_between(records: list[dict[str, Any]], t0: float, t1: float) -> float | None:
    """Erzeugte Tokens pro Sekunde im Abschnitt ``[t0, t1)``."""
    if t1 <= t0:
        return None
    return sum(tok for t, tok in token_events(records) if t0 <= t < t1) / (t1 - t0)


def gaps_between(records: list[dict[str, Any]], t0: float, t1: float) -> list[float]:
    """Abstände in Sekunden zwischen zwei Stücken derselben Anfrage, die in ``[t0, t1)`` enden."""
    gaps: list[float] = []
    for rec in records:
        chunks = rec.get("chunk_ms") or []
        base = rec["ts_submit"]
        for prev, cur in zip(chunks, chunks[1:]):
            if t0 <= base + cur / 1000.0 < t1:
                gaps.append((cur - prev) / 1000.0)
    return gaps


def longest_silence(records: list[dict[str, Any]], t0: float, t1: float) -> float | None:
    """Längste Zeit im Abschnitt, in der bei keiner der Anfragen ein Stück ankam."""
    if t1 <= t0:
        return None
    times = [t for t, _ in token_events(records) if t0 <= t < t1]
    edges = [t0, *times, t1]
    return max(b - a for a, b in zip(edges, edges[1:]))


def latency_stats(records: list[dict[str, Any]], digits: int = 3) -> dict[str, Any]:
    """Antwortzeiten einer Gruppe von Anfragen; nur vollständig beantwortete zählen."""
    ok = [r for r in records if r.get("ok")]

    def pct(key: str, q: float) -> float | None:
        value = percentile([r[key] for r in ok if r.get(key) is not None], q)
        return None if value is None else round(value, digits)

    return {
        "n": len(ok),
        "ttft_p50_s": pct("ttft_s", 50),
        "ttft_p95_s": pct("ttft_s", 95),
        "ttft_p99_s": pct("ttft_s", 99),
        "ttfc_p50_s": pct("ttfc_s", 50),
        "ttfc_p95_s": pct("ttfc_s", 95),
        "total_p50_s": pct("total_s", 50),
        "total_p95_s": pct("total_s", 95),
        "decode_tps_p50": pct("decode_tps", 50),
    }
