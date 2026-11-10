"""Qualitätsstufe: Fragensatz, automatische Prüfungen und die Blindbewertung von Hand."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import yaml

# Schriftzeichen, die in einer deutschen Antwort nichts zu suchen haben: Kyrillisch, Hebräisch, Arabisch,
# indische Schriften, Thai, CJK mit Kana und Hangul.
FOREIGN = re.compile(r"[Ѐ-ӿ֐-ۿऀ-෿฀-๿぀-ヿ㐀-鿿가-힯]")
SCALE = (
    "0 = falsch oder unbrauchbar; 1 = teilweise richtig, wesentliche Mängel; "
    "2 = richtig mit kleinen Mängeln; 3 = richtig, vollständig und gut formuliert"
)


def load_questions(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    ids = [q["id"] for q in data.get("fragen") or []]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Doppelte Kennungen im Fragensatz {path}")
    return {"fragen": data.get("fragen") or [], "werkzeuge": data.get("werkzeuge") or []}


def tools_payload(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": tool} for tool in tools]


def foreign_chars(text: str) -> int:
    return len(FOREIGN.findall(text or ""))


def extract_json(text: str) -> Any:
    """Liest JSON aus einer Antwort, auch wenn es in einem Code-Block steht."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        return None


def check(question: dict[str, Any], answer: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Automatische Prüfung einer Antwort. ``ok`` ist ``None``, wenn nur von Hand bewertet werden kann."""
    low = (answer or "").lower()
    results: dict[str, Any] = {}
    if question.get("muss"):
        missing = [group for group in question["muss"] if not any(str(term).lower() in low for term in group)]
        results["muss"] = not missing
        if missing:
            results["fehlt"] = [" / ".join(str(t) for t in group) for group in missing]
    if question.get("regex"):
        results["regex"] = all(re.search(p, answer or "", flags=re.I | re.S | re.M) for p in question["regex"])
    if question.get("darf_nicht"):
        results["darf_nicht"] = not any(str(term).lower() in low for term in question["darf_nicht"])
    if question.get("json"):
        spec, data = question["json"], extract_json(answer)
        ok = isinstance(data, dict) and all(key in data for key in spec.get("keys") or [])
        ok = ok and all(data.get(key) == value for key, value in (spec.get("werte") or {}).items())
        results["json"] = bool(ok)
    if "werkzeug" in question:
        want = question["werkzeug"]
        if want == "keines":
            results["werkzeug"] = not tool_calls
        else:
            results["werkzeug"] = any(_tool_matches(call, want) for call in tool_calls)
    manual = question.get("pruefung") == "manuell" or not results
    return {"ok": None if manual else all(v for k, v in results.items() if k != "fehlt"), **results}


def _tool_matches(call: dict[str, Any], want: dict[str, Any]) -> bool:
    function = call.get("function") or {}
    if function.get("name") != want.get("name"):
        return False
    args = function.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return False
    if not isinstance(args, dict):
        return False
    for key, alternatives in (want.get("args") or {}).items():
        value = str(args.get(key, "")).lower()
        if not any(str(alt).lower() in value for alt in alternatives):
            return False
    return True


# -- Blindbewertung -------------------------------------------------------------------


def export_blind(run_dirs: list[str | Path], out_dir: str | Path, thinking: str = "off", seed: int | None = None) -> dict[str, int]:
    """Schreibt den Bewertungsbogen: alle Antworten gemischt und ohne Herkunft, dazu getrennt den Schlüssel."""
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "answers" / "s8.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("thinking") != thinking or rec.get("kategorie") in ("werkzeuge", "langer_kontext", "struktur"):
                continue  # diese Kategorien prüft styrke selbst
            rows.append({**rec, "run": Path(run_dir).name})
    random.Random(seed).shuffle(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict[str, Any]] = {}
    with open(out / "bogen.csv", "w", encoding="utf-8-sig", newline="") as fh:  # mit BOM, damit Excel Umlaute erkennt
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["nr", "frage", "referenz", "antwort", "punkte", "bemerkung"])
        for nr, row in enumerate(rows, 1):
            key[str(nr)] = {"run": row["run"], "id": row["id"], "kategorie": row["kategorie"], "thinking": row["thinking"]}
            writer.writerow([nr, row["frage"], row.get("referenz", ""), row.get("antwort") or "(keine Antwort)", "", ""])
    (out / "schluessel.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "anleitung.md").write_text(
        "# Bewertung der Antworten\n\n"
        "Die Datei `bogen.csv` enthält Fragen, je eine Referenzantwort und die Antwort eines Modells. Von welchem "
        "Modell eine Antwort stammt, steht dort nicht; die Reihenfolge ist gemischt.\n\n"
        "Bitte in jeder Zeile die Spalte `punkte` ausfüllen:\n\n"
        + "".join(f"- {part.strip()}\n" for part in SCALE.split(";"))
        + "\nMaßstab ist die Referenzantwort. Sie beschreibt, was eine gute Antwort enthält; der Wortlaut darf "
        "abweichen. Fremde Schriftzeichen oder Wörter in anderen Sprachen mitten im deutschen Text sind ein Mangel. "
        "Die Spalte `bemerkung` ist freiwillig.\n\n"
        "Die Datei `schluessel.json` ordnet die Zeilen den Modellen zu. Wer bewertet, bekommt sie nicht.\n",
        encoding="utf-8",
    )
    return {"antworten": len(rows), "laeufe": len({r["run"] for r in rows})}


def import_blind(sheet: str | Path, key_file: str | Path) -> dict[str, Any]:
    """Liest den ausgefüllten Bogen und rechnet mittlere Punkte je Lauf und Kategorie."""
    key = json.loads(Path(key_file).read_text(encoding="utf-8"))
    scores: dict[str, dict[str, list[float]]] = {}
    missing = 0
    with open(sheet, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            info = key.get(str(row.get("nr", "")).strip())
            points = str(row.get("punkte") or "").strip().replace(",", ".")
            if not info:
                continue
            try:
                value = float(points)
            except ValueError:
                missing += 1
                continue
            scores.setdefault(info["run"], {}).setdefault(info["kategorie"], []).append(value)
    result: dict[str, Any] = {"skala": SCALE, "ohne_punkte": missing, "laeufe": {}}
    for run, categories in sorted(scores.items()):
        every = [v for values in categories.values() for v in values]
        result["laeufe"][run] = {
            "mittel": round(sum(every) / len(every), 2),
            "n": len(every),
            "kategorien": {cat: {"mittel": round(sum(v) / len(v), 2), "n": len(v)} for cat, v in sorted(categories.items())},
        }
    return result
