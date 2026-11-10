"""Stufe 8, Qualität: Antwortgüte an einem festen Fragensatz.

Jede Frage des Fragensatzes (``quality/fragen.yaml``) geht ohne und mit
Nachdenken an das Modell. Die Antworten stehen mit Frage und Referenzantwort
in ``answers/s8.jsonl``; anders als in den Leistungsstufen werden sie hier
aufbewahrt, denn um sie geht es.

Was sich eindeutig prüfen lässt, prüft styrke selbst: Rechenaufgaben,
wiedergefundene Angaben in langen Texten, Werkzeugaufrufe, JSON. Alles andere
bewertet ein Mensch blind (``styrke bewertung``). Die Werkzeuge sind
nachgebaut: Geprüft wird, ob das Modell das richtige Werkzeug mit den
richtigen Angaben verlangt; ausgeführt wird nichts.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from ..client import complete
from ..quality import check, foreign_chars, load_questions, tools_payload
from .common import StageContext, ensure_calibrated

NEEDLE_PREFIX = "Lies den folgenden Text aufmerksam.\n\n=== TEXT ===\n"
NEEDLE_SUFFIX = "\n=== ENDE ===\n\n"


async def build_messages(ctx: StageContext, question: dict[str, Any], thinking: str) -> list[dict[str, Any]] | None:
    """Nachrichten für eine Frage; ``None``, wenn ihr langer Text nicht in den Kontext passt."""
    content = question["frage"]
    if question.get("nadel"):
        tokens = min(question["kontext_tokens"], ctx.params["s8"].get("context_cap") or question["kontext_tokens"])
        if not ctx.fits(tokens, ctx.params["s8"]["max_tokens"][thinking] + 1024):
            return None
        doc = await asyncio.to_thread(ctx.builder.document, random.Random(ctx.next_seed()), tokens)
        paragraphs = doc.split("\n\n")
        paragraphs.insert(int(len(paragraphs) * 0.6), question["nadel"])  # mitten im Text, nicht am Rand
        content = NEEDLE_PREFIX + "\n\n".join(paragraphs) + NEEDLE_SUFFIX + question["frage"]
    prefix = ctx.system_prefix_for(thinking)
    system = [{"role": "system", "content": prefix}] if prefix else []
    return system + [{"role": "user", "content": content}]


async def ask(ctx: StageContext, question: dict[str, Any], thinking: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    params = ctx.params["s8"]
    rec: dict[str, Any] = {
        "id": question["id"], "kategorie": question["kategorie"], "thinking": thinking, "frage": question["frage"],
        "referenz": question.get("referenz", ""), "antwort": None, "tool_calls": [], "error": None,
    }
    messages = await build_messages(ctx, question, thinking)
    if messages is None:
        rec["error"] = "langer Text passt nicht in den Kontext"
        rec["check"] = {"ok": None, "skipped": True}
        return rec
    body = {**ctx.sampling(), **ctx.body_for(thinking)}
    if question["kategorie"] == "werkzeuge":
        body.update(tools=tools_payload(tools), tool_choice="auto")
    status, data = await complete(
        ctx.session, ctx.adapter.base_url, ctx.adapter.served_name, messages, params["max_tokens"][thinking],
        body=body, timeout_s=ctx.params["common"]["request_timeout_s"],
    )
    if status != 200 or not isinstance(data, dict):
        rec["error"] = f"HTTP {status}: {str(data)[:300]}"
        rec["check"] = {"ok": False}
        return rec
    message = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
    usage = data.get("usage") or {}
    rec["antwort"] = message.get("content") or ""
    rec["tool_calls"] = message.get("tool_calls") or []
    rec["reasoning_chars"] = len(message.get("reasoning_content") or message.get("reasoning") or "")
    rec["completion_tokens"] = usage.get("completion_tokens")
    rec["prompt_tokens"] = usage.get("prompt_tokens")
    rec["finish_reason"] = ((data.get("choices") or [{}])[0] or {}).get("finish_reason")
    rec["foreign_chars"] = foreign_chars(rec["antwort"])
    rec["check"] = check(question, rec["antwort"], rec["tool_calls"])
    return rec


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for thinking in sorted({r["thinking"] for r in records}):
        for category in sorted({r["kategorie"] for r in records}):
            group = [r for r in records if r["thinking"] == thinking and r["kategorie"] == category]
            checked = [r for r in group if r["check"].get("ok") is not None]
            rows.append(
                {
                    "thinking": thinking, "kategorie": category, "n": len(group),
                    "auto_checked": len(checked), "auto_ok": sum(1 for r in checked if r["check"]["ok"]),
                    "manual": sum(1 for r in group if r["check"].get("ok") is None and not r["check"].get("skipped")),
                    "skipped": sum(1 for r in group if r["check"].get("skipped")),
                    "errors": sum(1 for r in group if r["error"] and not r["check"].get("skipped")),
                    "with_foreign_chars": sum(1 for r in group if r.get("foreign_chars")),
                }
            )
    failed = [
        {"id": r["id"], "thinking": r["thinking"], "fehlt": r["check"].get("fehlt"), "error": r["error"]}
        for r in records if r["check"].get("ok") is False
    ]
    totals = {}
    for thinking in sorted({r["thinking"] for r in records}):
        sub = [row for row in rows if row["thinking"] == thinking]
        checked = sum(row["auto_checked"] for row in sub)
        totals[thinking] = {
            "auto_checked": checked, "auto_ok": sum(row["auto_ok"] for row in sub),
            "auto_share": round(sum(row["auto_ok"] for row in sub) / checked, 3) if checked else None,
            "with_foreign_chars": sum(row["with_foreign_chars"] for row in sub),
        }
    return {"rows": rows, "totals": totals, "failed": failed}


async def run(ctx: StageContext) -> dict[str, Any]:
    params = ctx.params["s8"]
    await ensure_calibrated(ctx)
    path = Path(params["questions"])
    if not path.is_absolute():
        path = ctx.base_dir / path
    if not path.is_file():
        summary = {"skipped": f"Fragensatz nicht gefunden: {path}. Pfad in der Testmatrix unter stages.s8.questions angeben."}
        ctx.store.write_json("summary_s8.json", summary)
        ctx.log(f"  ausgelassen: {summary['skipped']}")
        return summary
    questions = load_questions(path)
    modes = [m for m in params["thinking"] if m == "off" or ctx.model.thinking_mode("on")]
    records: list[dict[str, Any]] = []
    out = ctx.store.dir / "answers" / "s8.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for thinking in modes:
            for question in questions["fragen"]:
                rec = await ask(ctx, question, thinking, questions["werkzeuge"])
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                verdict = {True: "bestanden", False: "NICHT bestanden", None: "von Hand zu bewerten"}[rec["check"].get("ok")]
                if rec["check"].get("skipped"):
                    verdict = "ausgelassen: " + str(rec["error"])
                ctx.log(f"  Nachdenken {thinking:<3} {question['id']:<12} {verdict}" + (f" ({rec['error']})" if rec["error"] and rec["check"].get("ok") is False else ""))
    summary = {"questions": str(path), **summarize(records)}
    ctx.store.write_json("summary_s8.json", summary)
    for thinking, total in summary["totals"].items():
        ctx.log(
            f"  Nachdenken {thinking}: {total['auto_ok']} von {total['auto_checked']} automatisch geprüften Fragen bestanden, "
            f"{total['with_foreign_chars']} Antworten mit fremden Schriftzeichen"
        )
    return summary
