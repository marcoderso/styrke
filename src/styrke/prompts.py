"""Prompts in abgezählter Länge.

Der Zähler nutzt, wenn vorhanden, den Tokenizer des Modells (``tokenizer.json``
im Modellordner). Sonst rechnet er mit einem Zeichen-pro-Token-Verhältnis, das
Stufe 0 an der Engine eicht. Die tatsächliche Länge steht in jedem Fall im
Messdatensatz, weil die Engine sie zurückmeldet.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .corpus import Corpus

TOPICS = [
    "die Funktionsweise eines Wärmetauschers",
    "den Aufbau eines Stromnetzes",
    "die Geschichte des Buchdrucks",
    "den Wasserkreislauf",
    "die Planung eines Umzugs",
    "die Wartung von Pumpen",
    "die Entstehung von Gebirgen",
    "den Aufbau eines Rechenzentrums",
    "die Arbeit einer Kläranlage",
    "die Grundlagen der Buchhaltung",
    "die Herstellung von Papier",
    "den Betrieb einer Eisenbahnstrecke",
    "die Lagerung von Lebensmitteln",
    "die Grundlagen der Statistik",
    "die Organisation einer Bibliothek",
    "den Bau einer Brücke",
    "die Funktionsweise eines Kühlschranks",
    "die Einarbeitung neuer Kolleginnen und Kollegen",
    "die Vorbereitung einer Besprechung",
    "die Grundlagen der Netzwerktechnik",
    "die Pflege eines Gartens",
    "die Sicherung von Daten",
    "die Arbeit eines Stellwerks",
    "die Planung einer Schicht",
]

SUMMARY_PREFIX = "Lies den folgenden Text aufmerksam.\n\n=== TEXT ===\n"
SUMMARY_SUFFIX = (
    "\n=== ENDE ===\n\nAufgabe: Schreibe eine ausführliche Zusammenfassung dieses Textes mit mindestens "
    "600 Wörtern. Gehe auf Einzelheiten ein und gliedere die Zusammenfassung in Absätze."
)
ESSAY_TASK = "Schreibe einen ausführlichen, sachlichen Text von mindestens 600 Wörtern über {topic}."
ESSAY_CONTEXT = "\n\nBerücksichtige dabei diesen Hintergrund:\n\n"


class RatioCounter:
    """Zählt Tokens über ein festes Zeichen-pro-Token-Verhältnis."""

    kind = "ratio"

    def __init__(self, chars_per_token: float = 3.6):
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token) if text else 0

    def truncate(self, text: str, n_tokens: int) -> str:
        return text[: max(0, int(n_tokens * self.chars_per_token))]


class HFCounter:
    """Zählt Tokens mit dem Tokenizer des Modells."""

    kind = "tokenizer"

    def __init__(self, tokenizer_file: str | Path):
        from tokenizers import Tokenizer

        self._tk = Tokenizer.from_file(str(tokenizer_file))
        self._tk.no_truncation()
        self._tk.no_padding()

    def count(self, text: str) -> int:
        return len(self._tk.encode(text, add_special_tokens=False).ids) if text else 0

    def truncate(self, text: str, n_tokens: int) -> str:
        if n_tokens <= 0:
            return ""
        enc = self._tk.encode(text, add_special_tokens=False)
        if len(enc.ids) <= n_tokens:
            return text
        end = enc.offsets[n_tokens - 1][1]
        return text[:end]


def make_counter(tokenizer_file: str | Path | None) -> RatioCounter | HFCounter:
    """Tokenizer des Modells, falls die Datei lesbar ist; sonst das Verhältnis."""
    if tokenizer_file and Path(tokenizer_file).is_file():
        try:
            return HFCounter(tokenizer_file)
        except Exception:  # defekte Datei oder Bibliothek fehlt: mit Verhältnis weiter
            pass
    return RatioCounter()


@dataclass
class Prompt:
    messages: list[dict[str, Any]]
    target_tokens: int
    local_tokens: int
    salt: str


class PromptBuilder:
    """Baut Chat-Nachrichten, deren Länge möglichst genau ``target_tokens`` beträgt.

    Jeder Prompt beginnt mit einer Vorgangsnummer. Verschiedene Nummern ergeben
    verschiedene Prompt-Anfänge, also keinen Treffer im Präfix-Cache; dieselbe
    Nummer mit demselben Text ergibt gezielt einen Treffer.
    """

    def __init__(self, corpus: Corpus, counter: RatioCounter | HFCounter, overhead_tokens: int = 0):
        self.corpus = corpus
        self.counter = counter
        self.overhead_tokens = overhead_tokens  # Tokens, die das Chat-Template selbst hinzufügt

    def _document(self, rng: random.Random, n_tokens: int) -> str:
        """Korpustext mit genau ``n_tokens`` Tokens (soweit der Zähler es hergibt)."""
        if n_tokens <= 0:
            return ""
        chars = int(n_tokens * 4.5) + 2000
        raw = self.corpus.text(rng, chars)
        doc = self.counter.truncate(raw, n_tokens)
        while len(doc) == len(raw):  # nichts abgeschnitten: der Rohtext war zu kurz
            chars *= 2
            raw = self.corpus.text(rng, chars)
            doc = self.counter.truncate(raw, n_tokens)
        return doc

    def build(self, target_tokens: int, seed: int, system_prefix: str | None = None, salt: str | None = None) -> Prompt:
        rng = random.Random(seed)
        salt = salt or f"{rng.getrandbits(48):012x}"
        header = f"Vorgang {salt}\n\n"
        system = [{"role": "system", "content": system_prefix}] if system_prefix else []
        fixed = self.overhead_tokens + (self.counter.count(system_prefix) if system_prefix else 0)

        summary_fixed = fixed + self.counter.count(header + SUMMARY_PREFIX + SUMMARY_SUFFIX)
        doc_tokens = target_tokens - summary_fixed
        if doc_tokens >= 60:
            user = header + SUMMARY_PREFIX + self._document(rng, doc_tokens) + SUMMARY_SUFFIX
        else:
            task = ESSAY_TASK.format(topic=rng.choice(TOPICS))
            user = header + task
            rest = target_tokens - fixed - self.counter.count(user + ESSAY_CONTEXT)
            if rest >= 20:
                user += ESSAY_CONTEXT + self._document(rng, rest)

        local = fixed + self.counter.count(user)
        return Prompt(
            messages=system + [{"role": "user", "content": user}],
            target_tokens=target_tokens,
            local_tokens=local,
            salt=salt,
        )
