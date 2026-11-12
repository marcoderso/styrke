"""Qualitätsstufe: Fragensatz, automatische Prüfungen und die Blindbewertung von Hand."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import yaml

from .rating_page import build_page

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
        if not results["werkzeug"]:  # festhalten, was stattdessen kam
            calls = [f"{(c.get('function') or {}).get('name')}({(c.get('function') or {}).get('arguments')})" for c in tool_calls]
            results["fehlt"] = results.get("fehlt", []) + ["Werkzeugaufruf: " + ("; ".join(calls)[:200] if calls else "keiner")]
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

VERDICTS = {"gut": 2, "teils": 1, "schlecht": 0}


def collect_manual(run_dirs: list[str | Path], thinking: str = "off") -> list[dict[str, Any]]:
    """Die von Hand zu bewertenden Antworten, je Modell aus einem Lauf.

    Die Antwortgüte hängt am Modell, nicht an der Engine. Liegen für ein Modell
    mehrere Läufe vor, zählt der zuletzt genannte, also der neueste.
    """
    by_model: dict[str, list[dict[str, Any]]] = {}
    for run_dir in run_dirs:
        path = Path(run_dir) / "answers" / "s8.jsonl"
        if not path.is_file():
            continue
        try:
            manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        model = (manifest.get("model") or {}).get("repo") or Path(run_dir).name
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            manual = rec.get("manuell", (rec.get("check") or {}).get("ok") is None and not (rec.get("check") or {}).get("skipped"))
            if rec.get("thinking") == thinking and manual and not rec.get("error"):
                rows.append({**rec, "run": Path(run_dir).name, "model": model})
        if rows:
            by_model[model] = rows
    return [row for rows in by_model.values() for row in rows]


def export_blind(run_dirs: list[str | Path], out_dir: str | Path, thinking: str = "off", seed: int | None = None) -> dict[str, int]:
    """Schreibt die Bewertungsseite: je Frage die Antworten der Modelle ohne Herkunft, dazu getrennt den Schlüssel."""
    rows = collect_manual(run_dirs, thinking)
    random.Random(seed).shuffle(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict[str, Any]] = {}
    questions: dict[str, dict[str, Any]] = {}
    for nr, row in enumerate(rows, 1):
        key[str(nr)] = {"model": row["model"], "run": row["run"], "id": row["id"], "kategorie": row["kategorie"]}
        question = questions.setdefault(
            row["id"], {"id": row["id"], "frage": row["frage"], "referenz": row.get("referenz", ""), "cards": []}
        )
        text = (row.get("antwort") or "").strip()
        card = next((c for c in question["cards"] if c["text"] == text), None)
        if card is None:
            question["cards"].append({"nrs": [nr], "text": text})  # wortgleiche Antworten sind eine Karte
        else:
            card["nrs"].append(nr)
    (out / "schluessel.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    sheet_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    ordered = [questions[qid] for qid in sorted(questions)]
    (out / "bewertung.html").write_text(build_page(sheet_id, ordered), encoding="utf-8")
    return {"antworten": len(rows), "modelle": len({r["model"] for r in rows}), "fragen": len(questions)}


def import_verdicts(files: list[str | Path], key_file: str | Path) -> dict[str, Any]:
    """Liest die Urteile aus der Bewertungsseite (eine Datei je Person) und zählt sie je Modell aus.

    ``punkte`` ist der Anteil der erreichbaren Punkte in Prozent: gut zählt 2,
    teils 1, schlecht 0.
    """
    key = json.loads(Path(key_file).read_text(encoding="utf-8"))
    counts: dict[str, dict[str, dict[str, int]]] = {}
    raters: list[str] = []
    total = 0
    for file in files:
        data = json.loads(Path(file).read_text(encoding="utf-8"))
        if data.get("format") != "styrke-urteile-1":
            raise ValueError(f"{file} ist keine Ergebnisdatei der Bewertungsseite")
        raters.append(data.get("bewerter") or Path(file).stem)
        for verdicts in (data.get("urteile") or {}).values():
            for nr, verdict in verdicts.items():
                info = key.get(str(nr))
                if not info or verdict not in VERDICTS:
                    continue
                total += 1
                per = counts.setdefault(info["model"], {}).setdefault(info["kategorie"], {v: 0 for v in VERDICTS})
                per[verdict] += 1

    def score(c: dict[str, int]) -> float:
        n = sum(c.values())
        return round(100 * sum(VERDICTS[v] * k for v, k in c.items()) / (2 * n), 1) if n else 0.0

    result: dict[str, Any] = {"bewerter": raters, "urteile": total, "modelle": {}}
    for model, categories in counts.items():
        every = {v: sum(c[v] for c in categories.values()) for v in VERDICTS}
        result["modelle"][model] = {
            "punkte": score(every), **every, "n": sum(every.values()),
            "kategorien": {cat: {"punkte": score(c), **c} for cat, c in sorted(categories.items())},
        }
    result["modelle"] = dict(sorted(result["modelle"].items(), key=lambda item: -item[1]["punkte"]))
    return result
