"""Textkorpus für die Prompts.

Die Messungen brauchen natürliche deutsche Texte in großer Menge und Vielfalt.
Zufallstext wäre irreführend: große MoE-Modelle wählen ihre Experten nach
Inhalt, und eintöniger Text würde einen Experten-Cache zu gut aussehen lassen.

Quelle ist ein Ordner mit ``*.txt``-Dateien. ``styrke corpus fetch`` füllt ihn
mit Artikeln der deutschsprachigen Wikipedia; eigene Textdateien gehen ebenso.
Fehlt der Ordner, greift ein kleiner eingebauter Korpus, der nur für
Funktionstests taugt.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


class CorpusError(RuntimeError):
    """Der Korpus konnte nicht geladen oder beschafft werden."""


# Eingebauter Mini-Korpus: eigene, neutrale Sachtexte. Reicht für Funktionstests,
# nicht für Messungen (zu klein, zu gleichförmig).
BUILTIN_PARAGRAPHS = [
    "Ein Wärmetauscher überträgt Wärme von einem strömenden Medium auf ein anderes, ohne dass sich beide vermischen. "
    "In der Praxis unterscheidet man Bauformen nach der Führung der Ströme: Im Gleichstrom fließen beide Medien in "
    "dieselbe Richtung, im Gegenstrom entgegengesetzt, im Kreuzstrom rechtwinklig zueinander. Der Gegenstrom erreicht "
    "bei gleicher Fläche die höchste Übertragungsleistung, weil das Temperaturgefälle über die gesamte Länge erhalten "
    "bleibt. Ablagerungen auf den Flächen verschlechtern den Wärmeübergang mit der Zeit und müssen bei der Auslegung "
    "berücksichtigt werden.",
    "Die Gezeiten entstehen durch das Zusammenspiel der Anziehungskräfte von Mond und Sonne mit der Drehung der Erde. "
    "Auf der dem Mond zugewandten Seite und auf der gegenüberliegenden Seite wölbt sich das Wasser auf, dazwischen "
    "liegt Niedrigwasser. Stehen Sonne, Mond und Erde in einer Linie, verstärken sich die Wirkungen zur Springflut; "
    "bilden sie einen rechten Winkel, fällt der Tidenhub als Nippflut geringer aus. Die Form der Küste bestimmt, wie "
    "stark sich der Tidenhub vor Ort ausprägt, weshalb er in Buchten und Trichtermündungen besonders groß wird.",
    "Eine Bibliothek ordnet ihren Bestand so, dass ein Werk auffindbar bleibt, auch wenn niemand seinen Standort im "
    "Kopf hat. Dafür erhält jedes Exemplar eine Signatur, die sich aus einer Systematik und einer laufenden Kennung "
    "zusammensetzt. Der Katalog verknüpft Signatur, Titel, Verfasser und Schlagwörter. Wird ein Buch falsch "
    "eingestellt, ist es für die Nutzer praktisch verloren, obwohl es im Haus steht. Deshalb gehört das regelmäßige "
    "Prüfen der Regale ebenso zur Arbeit wie das Erfassen neuer Titel.",
    "Beim Brückenbau entscheidet die Spannweite maßgeblich über die Bauform. Kurze Strecken überbrückt ein einfacher "
    "Balken, mittlere ein Bogen oder ein Fachwerk, große eine Schrägseil- oder Hängebrücke. Bei der Hängebrücke tragen "
    "zwei Hauptseile die Fahrbahn über senkrechte Hänger und leiten die Last in die Türme und in massive "
    "Verankerungen an den Ufern. Wind kann eine leichte Fahrbahn zum Schwingen anregen; deshalb wird ihr Querschnitt "
    "im Windkanal geprüft und so geformt, dass sich keine gefährlichen Schwingungen aufschaukeln.",
    "Sauerteig ist ein Gemisch aus Mehl und Wasser, in dem Milchsäurebakterien und Hefen leben. Die Bakterien bilden "
    "Säuren, die dem Brot Geschmack geben und es haltbarer machen; die Hefen erzeugen Gas, das den Teig lockert. "
    "Roggenmehl lässt sich ohne Säuerung kaum backen, weil erst die Säure bestimmte Enzyme hemmt, die sonst die Krume "
    "zerstören würden. Ein Sauerteig wird über mehrere Stufen geführt, bei denen Temperatur und Festigkeit steuern, "
    "welche Mikroorganismen sich durchsetzen.",
    "Ein Stellwerk sichert den Zugverkehr, indem es Weichen und Signale so voneinander abhängig macht, dass sich "
    "widersprechende Fahrten ausgeschlossen sind. Bevor ein Signal Fahrt zeigen kann, müssen alle Weichen der "
    "Fahrstraße in der richtigen Lage verschlossen und der Abschnitt als frei gemeldet sein. Ältere Anlagen lösen das "
    "mechanisch über Riegel und Gestänge, neuere über Relais oder Rechner. Das Grundprinzip ist in allen Bauformen "
    "gleich: Im Zweifel fällt die Anlage in den sicheren Zustand, also auf Halt.",
    "Die Photosynthese wandelt Lichtenergie in chemisch gebundene Energie um. In den Chloroplasten der Pflanzenzellen "
    "spaltet das Licht Wasser, wobei Sauerstoff frei wird; die gewonnene Energie dient dazu, aus Kohlendioxid Zucker "
    "aufzubauen. Wie viel eine Pflanze leistet, hängt von Licht, Temperatur, Wasser und Kohlendioxid ab, und stets "
    "begrenzt der knappste Faktor den Ertrag. Bei großer Hitze schließen viele Pflanzen ihre Spaltöffnungen, um "
    "Wasser zu sparen, und nehmen dann auch weniger Kohlendioxid auf.",
    "Eine Datenbank hält ihre Daten auch dann widerspruchsfrei, wenn viele Vorgänge gleichzeitig darauf zugreifen. "
    "Dazu fasst sie zusammengehörige Änderungen zu Transaktionen zusammen, die entweder ganz oder gar nicht wirksam "
    "werden. Sperren oder Versionen der Datensätze verhindern, dass sich parallele Transaktionen gegenseitig stören. "
    "Vor jeder Änderung schreibt das System einen Eintrag in ein Protokoll; nach einem Absturz lässt sich daraus der "
    "letzte gültige Stand wiederherstellen.",
    "Glas entsteht, wenn eine Schmelze aus Quarzsand, Soda und Kalk so abkühlt, dass sich keine Kristalle bilden. Beim "
    "Floatverfahren fließt die Schmelze auf ein Bad aus flüssigem Zinn und breitet sich dort zu einem ebenen Band "
    "aus. Die Dicke ergibt sich aus der Geschwindigkeit, mit der das Band abgezogen wird. Anschließend durchläuft das "
    "Glas einen Kühlkanal, in dem es langsam und gleichmäßig abkühlt, damit keine inneren Spannungen zurückbleiben, "
    "die später zum Bruch führen könnten.",
    "Ein Deich schützt das Hinterland, solange er nicht überströmt oder durchweicht wird. Sein Querschnitt ist auf der "
    "Wasserseite flach geneigt, damit auflaufende Wellen ihre Energie verlieren. Im Inneren liegt ein Kern aus Sand, "
    "darüber eine dichte Schicht aus Klei, die das Eindringen von Wasser bremst. Eine geschlossene Grasnarbe hält die "
    "Oberfläche zusammen; Schafe, die das Gras kurz halten und den Boden festtreten, sind deshalb ein fester "
    "Bestandteil der Deichpflege.",
    "Die Druckerpresse mit beweglichen Lettern machte es möglich, Texte in großer Zahl und gleicher Gestalt "
    "herzustellen. Der Setzer fügte einzelne Metalllettern zu Zeilen und Seiten, die eingefärbt und auf Papier "
    "gepresst wurden. Nach dem Druck ließen sich die Lettern wieder trennen und neu verwenden. Weil ein Fehler im Satz "
    "in jedem Exemplar auftauchte, entstand der Beruf des Korrektors, der die Abzüge vor dem Druck der Auflage prüfte.",
    "Ein Gletscher bildet sich dort, wo über viele Jahre mehr Schnee fällt als abschmilzt. Unter dem Druck der "
    "aufliegenden Schichten verdichtet sich der Schnee zu Firn und schließlich zu Eis, das unter seinem eigenen "
    "Gewicht langsam talwärts fließt. Im oberen Bereich gewinnt der Gletscher Masse, im unteren verliert er sie durch "
    "Schmelzen. Verschiebt sich die Grenze zwischen beiden Bereichen nach oben, zieht sich die Gletscherzunge mit "
    "einigen Jahren Verzögerung zurück.",
]


class Corpus:
    """Eine Liste von Absätzen, aus der sich beliebig lange Texte ziehen lassen."""

    def __init__(self, paragraphs: list[str], source: str, builtin: bool = False):
        if not paragraphs:
            raise CorpusError("Der Korpus enthält keinen Text")
        self.paragraphs = paragraphs
        self.source = source
        self.builtin = builtin
        self.total_chars = sum(len(p) for p in paragraphs)

    @classmethod
    def load(cls, corpus_dir: str | os.PathLike | None) -> "Corpus":
        """Liest alle ``*.txt`` im Ordner; ohne Dateien greift der eingebaute Mini-Korpus."""
        files = sorted(Path(corpus_dir).glob("*.txt")) if corpus_dir and Path(corpus_dir).is_dir() else []
        paragraphs: list[str] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            for block in text.split("\n\n"):
                block = " ".join(block.split())
                if len(block) >= 200 and not block.startswith("#"):
                    paragraphs.append(block)
        if paragraphs:
            return cls(paragraphs, source=str(corpus_dir))
        return cls(list(BUILTIN_PARAGRAPHS), source="eingebaut", builtin=True)

    def text(self, rng: random.Random, min_chars: int) -> str:
        """Zusammenhängender Text ab einer zufälligen Stelle, mindestens ``min_chars`` lang."""
        n = len(self.paragraphs)
        idx = rng.randrange(n)
        parts: list[str] = []
        size = 0
        while size < min_chars:
            para = self.paragraphs[idx % n]
            parts.append(para)
            size += len(para) + 2
            idx += 1
        return "\n\n".join(parts)

    def info(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "builtin": self.builtin,
            "paragraphs": len(self.paragraphs),
            "chars": self.total_chars,
        }


# -- Beschaffung --------------------------------------------------------------

ROWS_API = "https://datasets-server.huggingface.co/rows"
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.de"


def _http_json(url: str, timeout: float = 60) -> dict[str, Any]:
    headers = {"User-Agent": "styrke-corpus/0.1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


# Fehler, bei denen ein späterer Versuch Aussicht hat: Netz, Zeitüberschreitung,
# abgeschnittene oder unlesbare Antwort, Störung auf der Gegenseite.
FETCH_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError, ValueError)
# Statuscodes, bei denen Warten nichts ändert (falsche Adresse, kein Zugriff).
PERMANENT_STATUS = {400, 401, 403, 404}
FILE_PREFIX = "wikipedia-de-"


def _retry_after(exc: BaseException) -> float | None:
    """Wartezeit, die die Gegenseite selbst nennt (Kopfzeile ``Retry-After``)."""
    headers = getattr(exc, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    try:
        return max(1.0, min(600.0, float(value))) if value is not None else None
    except (TypeError, ValueError):
        return None


def _existing(out: Path) -> dict[int, int]:
    """Schon geladene Dateien: Stelle im Datensatz und Zeichenzahl."""
    found: dict[int, int] = {}
    for path in out.glob(f"{FILE_PREFIX}*.txt"):
        try:
            offset = int(path.stem[len(FILE_PREFIX) :])
        except ValueError:
            continue
        found[offset] = len(path.read_text(encoding="utf-8", errors="replace"))
    return found


def fetch_wikipedia(
    corpus_dir: str | os.PathLike,
    target_mb: float = 40.0,
    seed: int = 20261110,
    fetch_json: Callable[[str], dict[str, Any]] = _http_json,
    log: Callable[[str], None] = print,
    pause_s: float = 1.0,
    max_failures: int = 10,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Lädt Artikel der deutschsprachigen Wikipedia in den Korpus-Ordner.

    Genutzt wird die Zeilen-Schnittstelle des Hugging-Face-Datensatzes
    ``wikimedia/wikipedia``. Die Artikel werden an zufälligen Stellen des
    Datensatzes entnommen, damit die Themen breit gestreut sind. Lizenz der
    Texte: CC BY-SA; sie bleiben auf der Maschine und werden nicht weitergegeben.

    Der Abruf lässt sich fortsetzen: Was schon im Ordner liegt, zählt mit und
    wird nicht erneut geladen. Antwortet die Quelle nicht, wartet der Abruf mit
    wachsenden Pausen und gibt erst nach ``max_failures`` Fehlversuchen in Folge
    auf (bei den Standardwerten nach gut zehn Minuten).
    """
    out = Path(corpus_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = {"dataset": WIKI_DATASET, "config": WIKI_CONFIG, "split": "train"}
    target_chars = int(target_mb * 1_000_000)
    have = _existing(out)
    got_chars = sum(have.values())
    new_files = 0

    def result() -> dict[str, Any]:
        return {"files": len(have), "new_files": new_files, "chars": got_chars, "dir": str(out)}

    if got_chars >= target_chars:
        log(f"  {got_chars / 1e6:.1f} MB sind schon vorhanden, nichts zu tun")
        return result()
    if have:
        log(f"  {got_chars / 1e6:.1f} MB sind schon vorhanden, der Abruf setzt dort fort")

    def rows_url(offset: int, length: int) -> str:
        return ROWS_API + "?" + urllib.parse.urlencode({**base, "offset": offset, "length": length})

    failures = 0

    def failed(exc: BaseException) -> None:
        """Zählt einen Fehlversuch, wartet oder gibt auf."""
        nonlocal failures
        failures += 1
        stand = (
            f"Im Ordner {out}/ liegen schon {got_chars / 1e6:.1f} MB ({len(have)} "
            f"{'Datei' if len(have) == 1 else 'Dateien'}); sie sind nutzbar, und derselbe Aufruf setzt später dort fort."
            if have
            else f"Alternative: eigene deutsche Textdateien (*.txt) nach {out}/ legen."
        )
        code = getattr(exc, "code", None)
        if code in PERMANENT_STATUS:
            raise CorpusError(f"Die Wikipedia-Quelle lehnt die Anfrage ab ({exc}). {stand}") from exc
        if failures >= max_failures:
            raise CorpusError(
                f"Die Wikipedia-Quelle antwortet seit {failures} Versuchen nicht ({exc}). {stand}"
            ) from exc
        wait = _retry_after(exc) or min(120.0, 5.0 * 2 ** (failures - 1))
        log(f"  Quelle antwortet nicht ({exc}); Versuch {failures} von {max_failures}, weiter in {wait:.0f} s")
        sleep(wait)

    total_rows = 0
    while not total_rows:
        try:
            first = fetch_json(rows_url(0, 1))
        except FETCH_ERRORS as exc:
            failed(exc)
            continue
        total_rows = int(first.get("num_rows_total") or 0)
        if total_rows < 1000:
            raise CorpusError("Unerwartete Antwort der Wikipedia-Quelle (num_rows_total fehlt)")
    failures = 0

    rng = random.Random(seed)
    shown_mb = int(got_chars / 1e6)
    while got_chars < target_chars:
        offset = rng.randrange(0, max(1, total_rows - 100))
        if offset in have:  # aus einem früheren Abruf
            continue
        try:
            data = fetch_json(rows_url(offset, 100))
        except FETCH_ERRORS as exc:
            failed(exc)  # danach eine andere Stelle versuchen
            continue
        failures = 0
        parts: list[str] = []
        for item in data.get("rows") or []:
            row = item.get("row") or {}
            text = (row.get("text") or "").strip()
            if len(text) < 1500:  # Stummel und Weiterleitungen auslassen
                continue
            parts.append(f"# {row.get('title', '')}\n\n{text}")
        if parts:
            blob = "\n\n".join(parts) + "\n"
            (out / f"{FILE_PREFIX}{offset:08d}.txt").write_text(blob, encoding="utf-8")
            have[offset] = len(blob)
            got_chars += len(blob)
            new_files += 1
            if int(got_chars / 1e6) > shown_mb or got_chars >= target_chars:
                shown_mb = int(got_chars / 1e6)
                log(f"  {got_chars / 1e6:6.1f} von {target_mb:g} MB")
        if pause_s:
            sleep(pause_s)
    return result()
