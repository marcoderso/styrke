"""Bericht über einen oder mehrere Läufe: ``styrke report``.

Der Bericht ist eine einzelne HTML-Datei ohne äußere Abhängigkeiten. Er stellt
die Läufe nebeneinander: zuerst die Kennzahlen für die Entscheidung, dann
Diagramme, dann je Lauf die Tabellen aller Stufen. Zielwerte für
Antwortzeiten und die Monatskosten der Maschine kommen als Optionen dazu und
werden nachträglich auf die Messwerte angewandt; neu gemessen wird nichts.
"""

from __future__ import annotations

import html
import json
import math
import time
from pathlib import Path
from typing import Any

# Feste Farbfolge für die Läufe (hell, dunkel). Ein Lauf behält seine Farbe in allen Diagrammen.
PALETTE = [
    ("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500"),
    ("#e87ba4", "#d55181"), ("#008300", "#008300"), ("#6250d6", "#9085e9"), ("#e34948", "#e66767"),
]
SECONDS_PER_MONTH = 30 * 24 * 3600


# -- Zahlen in deutscher Schreibweise -------------------------------------------------


def de(value: Any, digits: int = 1, unit: str = "") -> str:
    """Zahl mit Dezimalkomma und Tausenderpunkt; Strich, wenn nichts gemessen wurde."""
    if value is None:
        return "–"
    text = f"{value:,.{digits}f}".replace(",", " ").replace(".", ",").replace(" ", ".")
    return text + unit


def esc(value: Any) -> str:
    return html.escape(str(value))


# -- Läufe laden ---------------------------------------------------------------------


def load_run(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir)

    def load(name: str) -> Any:
        try:
            return json.loads((path / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    manifest = load("manifest.json") or {}
    run = {
        "dir": str(path),
        "name": path.name,
        "manifest": manifest,
        "cell": manifest.get("cell") or path.name,
        "variant": (manifest.get("variant") or {}).get("id") or "",
        "smoke": bool(manifest.get("smoke")),
        "feasibility": load("feasibility.json"),
    }
    for n in range(1, 9):
        run[f"s{n}"] = load(f"summary_s{n}.json")
    run["label"] = f"{run['cell']} / {run['variant']}" if run["variant"] else run["cell"]
    return run


def discover(runs_dir: str | Path, include_smoke: bool = False) -> list[Path]:
    """Je Zelle und Variante der neueste Laufordner; Funktionstests bleiben ohne ``include_smoke`` weg."""
    newest: dict[tuple[str, str], Path] = {}
    for path in sorted(Path(runs_dir).glob("*")):
        if not (path / "manifest.json").is_file():
            continue
        run = load_run(path)
        if run["smoke"] and not include_smoke:
            continue
        newest[(run["cell"], run["variant"])] = path  # Ordnernamen beginnen mit dem Zeitstempel
    return sorted(newest.values())


# -- Kennzahlen für die Entscheidung ----------------------------------------------------


def _s1_row(run: dict[str, Any], target: int | None = None, largest: bool = False) -> dict[str, Any] | None:
    rows = [r for r in (run["s1"] or {}).get("rows", []) if r.get("thinking") == "off" and r.get("n_ok")]
    if not rows:
        return None
    if largest:
        return max(rows, key=lambda r: r["prompt_target"])
    return min(rows, key=lambda r: abs(r["prompt_target"] - (target or 1000)))


def rate_at_target(run: dict[str, Any], ttft_p95: float | None, total_p95: float | None) -> dict[str, Any] | None:
    """Höchste getragene Rate der Stufe 6, bei der die Zielwerte eingehalten sind."""
    best = None
    for row in (run["s6"] or {}).get("rows", []):
        if row.get("overload"):
            break
        if ttft_p95 is not None and (row.get("ttft_p95_s") is None or row["ttft_p95_s"] > ttft_p95):
            break
        if total_p95 is not None and (row.get("total_p95_s") is None or row["total_p95_s"] > total_p95):
            break
        best = row
    return best


def key_figures(run: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    feas = run["feasibility"] or {}
    s2, s4, s5, s6, s7 = (run[k] or {} for k in ("s2", "s4", "s5", "s6", "s7"))
    profile = opts["profile"]
    short, large = _s1_row(run, 1000), _s1_row(run, largest=True)
    followup = [r for r in s4.get("rows", []) if r.get("speedup")]
    big = next((r for r in s5.get("rows", []) if r.get("big_ok")), None)
    sustained = next((r for r in reversed(s6.get("rows", [])) if r["rate"] == s6.get("max_sustained_rate")), None)
    has_target = opts["ttft_p95"] is not None or opts["total_p95"] is not None
    at_target = rate_at_target(run, opts["ttft_p95"], opts["total_p95"]) if has_target else None
    users = (s6.get("users_supported") or {}).get(profile)
    users_target = ((at_target or {}).get("users_supported") or {}).get(profile)
    cost = opts["cost_month"]
    decisive_users = users_target if has_target else users
    return {
        "loaded": feas.get("loaded"),
        "error": feas.get("error") or feas.get("sanity_error"),
        "load_time_s": feas.get("load_time_s"),
        "max_context": feas.get("max_context_ok"),
        "single_tps": (short or {}).get("decode_tps"),
        "ttft_short_s": (short or {}).get("ttft_s"),
        "large_target": (large or {}).get("prompt_target"),
        "ttft_large_s": (large or {}).get("ttft_s"),
        "max_tps": s2.get("max_throughput_tps"),
        "max_tps_level": s2.get("max_throughput_level"),
        "saturated_at": s2.get("saturated_at"),
        "followup_target": followup[-1]["prompt_target"] if followup else None,
        "followup_speedup": followup[-1]["speedup"] if followup else None,
        "big_target": (big or {}).get("big_target"),
        "slowdown_pct": (big or {}).get("slowdown_prefill_pct"),
        "silence_s": ((big or {}).get("prefill") or {}).get("silence_max_s"),
        "sustained_rate": s6.get("max_sustained_rate"),
        "sustained_ttft_p95_s": (sustained or {}).get("ttft_p95_s"),
        "users": users,
        "target_rate": (at_target or {}).get("rate") if has_target else None,
        "users_target": users_target,
        "s6_run": bool(s6.get("rows")),
        "dropped_kinds": [d["kind"] for d in s6.get("sessions_dropped") or []],
        "endurance_errors": (s7.get("overall") or {}).get("errors"),
        "endurance_drift_pct": s7.get("ttft_p95_drift_pct"),
        "cost_per_user": cost / decisive_users if cost and decisive_users else None,
        "cost_per_mtok": cost / (s2["max_throughput_tps"] * SECONDS_PER_MONTH / 1e6)
        if cost and s2.get("max_throughput_tps") else None,
    }


# -- Diagramme ----------------------------------------------------------------------------


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / count
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw)
    first = math.ceil(lo / step) * step
    return [round(first + i * step, 10) for i in range(int((hi - first) / step + 1e-9) + 1)]


def log_ticks(lo: float, hi: float) -> list[float]:
    ticks = [m * 10**e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1) for m in (1, 2, 5)]
    inside = [t for t in ticks if lo * 0.999 <= t <= hi * 1.001]
    return inside if len(inside) <= 9 else [t for t in inside if str(t).lstrip("0.").startswith("1")]


def _tick_label(value: float) -> str:
    return de(value, 0 if value >= 10 or value == int(value) else (1 if value >= 1 else 2))


def line_chart(
    series: list[dict[str, Any]], *, x_label: str, y_label: str, log_x: bool = False, log_y: bool = False,
    target: float | None = None, target_label: str = "",
) -> str:
    """Liniendiagramm als SVG. ``series``: ``{"label", "slot", "points": [(x, y), ...]}``; eine Achse, keine zweite."""
    series = [{**s, "points": [(x, y) for x, y in s["points"] if x is not None and y is not None]} for s in series]
    if log_x:
        series = [{**s, "points": [p for p in s["points"] if p[0] > 0]} for s in series]
    if log_y:
        series = [{**s, "points": [p for p in s["points"] if p[1] > 0]} for s in series]
    series = [s for s in series if s["points"]]
    if not series:
        return '<p class="muted">Keine Messwerte für dieses Diagramm.</p>'
    width, height, left, right, top, bottom = 800, 340, 64, 220, 16, 46
    xs = [x for s in series for x, _ in s["points"]]
    ys = [y for s in series for _, y in s["points"]] + ([target] if target else [])
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = (min(ys), max(ys)) if log_y else (0.0, max(ys) * 1.08 or 1.0)
    if x_lo == x_hi:
        x_lo, x_hi = (x_lo / 2, x_hi * 2) if log_x else (x_lo - 1, x_hi + 1)
    if y_lo == y_hi:
        y_lo, y_hi = y_lo / 2, y_hi * 2

    def scale(value: float, lo: float, hi: float, log: bool) -> float:
        return (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) if log else (value - lo) / (hi - lo)

    def px(x: float) -> float:
        return left + scale(x, x_lo, x_hi, log_x) * (width - left - right)

    def py(y: float) -> float:
        return height - bottom - scale(y, y_lo, y_hi, log_y) * (height - top - bottom)

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" aria-label="{esc(y_label)} über {esc(x_label)}">']
    for tick in log_ticks(y_lo, y_hi) if log_y else nice_ticks(y_lo, y_hi):
        y = py(tick)
        out.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{_tick_label(tick)}</text>')
    x_ticks = log_ticks(x_lo, x_hi) if log_x else (sorted(set(xs)) if len(set(xs)) <= 8 else nice_ticks(x_lo, x_hi))
    for tick in x_ticks:
        out.append(f'<text class="tick" x="{px(tick):.1f}" y="{height - bottom + 18}" text-anchor="middle">{_tick_label(tick)}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}"/>')
    out.append(f'<text class="label" x="{(left + width - right) / 2:.0f}" y="{height - 6}" text-anchor="middle">{esc(x_label)}</text>')
    out.append(f'<text class="label" x="{left}" y="{top - 4}" text-anchor="start">{esc(y_label)}</text>')
    if target and y_lo <= target <= y_hi:
        y = py(target)
        out.append(f'<line class="target" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{width - right - 4}" y="{y - 5:.1f}" text-anchor="end">{esc(target_label)}</text>')
    label_ys: list[float] = []
    for s in series:
        color = f"var(--s{s['slot'] % len(PALETTE)})"
        points = sorted(s["points"])
        path = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(points))
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        for x, y in points:
            tip = f"{s['label']}: {de(y, 2)} bei {de(x, 2 if x < 10 else 0)}"
            out.append(
                f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="{color}" stroke="var(--surface)" stroke-width="2"/>'
                f'<circle class="hit" cx="{px(x):.1f}" cy="{py(y):.1f}" r="11" data-tip="{esc(tip)}"/>'
            )
        if len(series) <= 4:  # direkte Beschriftung am Linienende, ohne Überlappung
            y = py(points[-1][1])
            while any(abs(y - other) < 13 for other in label_ys):
                y += 13
            label_ys.append(y)
            label = s["label"] if len(s["label"]) <= 32 else s["label"][:31] + "…"
            out.append(f'<text class="direct" x="{px(points[-1][0]) + 9:.1f}" y="{y + 4:.1f}">{esc(label)}</text>')
    out.append("</svg>")
    legend = "".join(
        f'<span><i style="background:var(--s{s["slot"] % len(PALETTE)})"></i>{esc(s["label"])}</span>' for s in series
    )
    return f'<div class="legend">{legend}</div>' + "".join(out)


# -- HTML ----------------------------------------------------------------------------------

CSS = """
:root{--surface:#fcfcfb;--ink:#1f1e1d;--ink2:#5c5b56;--line:#deddd6;--panel:#f3f2ee;--bad:#b3261e;%LIGHT%}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--panel:#242422;--bad:#f2928c;%DARK%}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px}
h3{font-size:16px;margin:28px 0 6px}p{margin:6px 0 12px;max-width:78ch}.muted{color:var(--ink2)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin:8px 0 16px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-weight:600;color:var(--ink2);font-size:13px;vertical-align:bottom;white-space:normal}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:150px}table{font-size:14px}td.text{text-align:left;white-space:normal}.bad{color:var(--bad);font-weight:600}
.note{background:var(--panel);border-radius:8px;padding:10px 14px;margin:12px 0}
.chart{width:100%;height:auto;max-width:860px;display:block}.grid{stroke:var(--line);stroke-width:1}.axis{stroke:var(--ink2);stroke-width:1}
.tick{fill:var(--ink2);font-size:12px}.label{fill:var(--ink2);font-size:12px}.direct{fill:var(--ink);font-size:12px}
.target{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 4}.hit{fill:transparent;cursor:pointer}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;margin:6px 0}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
#tip{position:fixed;display:none;background:var(--ink);color:var(--surface);padding:5px 9px;border-radius:6px;font-size:13px;pointer-events:none;z-index:9}
details{margin:8px 0}summary{cursor:pointer;font-weight:600}
"""
JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('.hit').forEach(el=>{
  el.addEventListener('mousemove',e=>{tip.textContent=el.dataset.tip;tip.style.display='block';
    tip.style.left=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=(e.clientY-34)+'px';});
  el.addEventListener('mouseleave',()=>{tip.style.display='none';});});
"""


def table(headers: list[str], rows: list[list[Any]], text_columns: tuple[int, ...] = ()) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = ""
        for i, cell in enumerate(row):
            cls = ' class="text"' if i in text_columns else ""
            if isinstance(cell, tuple):  # (Text, Klasse)
                cells += f'<td class="{cell[1]}">{esc(cell[0])}</td>'
            else:
                cells += f"<td{cls}>{esc(cell)}</td>"
        body += f"<tr>{cells}</tr>"
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def decision_tables(runs: list[dict[str, Any]], figures: list[dict[str, Any]], opts: dict[str, Any]) -> str:
    perf_rows, cap_rows = [], []
    has_target = opts["ttft_p95"] is not None or opts["total_p95"] is not None
    for run, f in zip(runs, figures):
        name = run["label"] + (" (Funktionstest)" if run["smoke"] else "")
        if f["loaded"] is False:
            perf_rows.append([name, ("lädt nicht: " + str(f["error"])[:120], "bad text")] + [""] * 7)
            cap_rows.append([name, ("lädt nicht", "bad")] + [""] * (3 + (2 if has_target else 0) + (2 if opts["cost_month"] else 0)))
            continue
        perf_rows.append([
            name, de(f["load_time_s"], 0, " s"), de(f["max_context"], 0), de(f["single_tps"], 1), de(f["ttft_short_s"], 2, " s"),
            f"{de(f['ttft_large_s'], 1, ' s')} ({de(f['large_target'], 0)})" if f["large_target"] else "–",
            f"{de(f['max_tps'], 0)} (bei {f['max_tps_level']})" if f["max_tps"] else "–",
            f"{de(f['followup_speedup'], 1)}-fach ({de(f['followup_target'], 0)})" if f["followup_speedup"] else "–",
            f"{de(f['slowdown_pct'], 0, ' %')}, {de(f['silence_s'], 1, ' s')} Stille" if f["slowdown_pct"] is not None else "–",
        ])
        if not f["s6_run"]:
            sustained = "nicht gemessen"
        elif f["sustained_rate"] is None:
            sustained = ("keine Rate getragen", "bad")
        else:
            sustained = de(f["sustained_rate"], 2)
        if f["dropped_kinds"] and f["s6_run"]:
            sustained = (f"{sustained if isinstance(sustained, str) else sustained[0]} (ohne {', '.join(f['dropped_kinds'])})", "bad")
        row = [name, sustained, de(f["sustained_ttft_p95_s"], 2, " s"), de(f["users"], 0)]
        if has_target:
            row += [de(f["target_rate"], 2) if f["target_rate"] is not None else ("keine", "bad") if f["s6_run"] else "–",
                    de(f["users_target"], 0)]
        endurance = "–" if f["endurance_errors"] is None else (
            f"{f['endurance_errors']} Fehler, TTFT p95 {de(f['endurance_drift_pct'], 0, ' %')}"
        )
        row.append(endurance)
        if opts["cost_month"]:
            row += [de(f["cost_per_user"], 2, " €"), de(f["cost_per_mtok"], 2, " €")]
        cap_rows.append(row)
    perf = table(
        ["Zelle / Variante", "Ladezeit", "größter Kontext (Tokens)", "tok/s einzeln", "erstes Token, kurzer Prompt",
         "erstes Token, längster Prompt", "Durchsatz max. in tok/s (parallel)", "Folgefrage schneller (Prompt)",
         "Einbruch der Chats bei großer Anfrage"],
        perf_rows, text_columns=(1,),
    )
    headers = ["Zelle / Variante", "getragene Rate (Aufrufe/s)", "TTFT p95 dabei", f"Berechtigte ({opts['profile']})"]
    if has_target:
        headers += ["Rate mit Zielwert", f"Berechtigte mit Zielwert ({opts['profile']})"]
    headers.append("Dauerlauf: Fehler und Veränderung")
    if opts["cost_month"]:
        headers += ["Kosten je Berechtigtem und Monat", "Kosten je Mio. Ausgabe-Tokens bei Dauerlast"]
    return "<h3>Leistung</h3>" + perf + "<h3>Tragfähigkeit</h3>" + table(headers, cap_rows) + quality_table(runs)


def quality_table(runs: list[dict[str, Any]]) -> str:
    """Automatisch geprüfte Fragen der Stufe 8 je Lauf; die Bewertung von Hand kommt aus ``styrke bewertung``."""
    rows = []
    for run in runs:
        totals = (run["s8"] or {}).get("totals") or {}
        if not totals:
            continue
        cells = [run["label"]]
        for mode in ("off", "on"):
            t = totals.get(mode)
            cells.append(f"{t['auto_ok']} von {t['auto_checked']}" if t else "–")
        cells.append(sum(t["with_foreign_chars"] for t in totals.values()))
        failed = sorted({f["id"] for f in (run["s8"] or {}).get("failed", [])})
        cells.append(", ".join(failed) or "keine")
        rows.append(cells)
    if not rows:
        return ""
    return (
        "<h3>Qualität</h3><p>Automatisch prüfbare Fragen: Rechenaufgaben, Firmenwissen mit festen Begriffen, Angaben "
        "aus langen Texten, Werkzeugaufrufe, JSON. Die übrigen Fragen bewertet ein Mensch blind.</p>"
        + table(["Zelle / Variante", "bestanden ohne Nachdenken", "bestanden mit Nachdenken",
                 "Antworten mit fremden Schriftzeichen", "nicht bestanden"], rows, text_columns=(4,))
    )


def charts(runs: list[dict[str, Any]], opts: dict[str, Any]) -> str:
    shown = runs[: len(PALETTE)]
    note = "" if len(runs) <= len(PALETTE) else (
        f'<p class="muted">Die Diagramme zeigen die ersten {len(PALETTE)} Läufe; alle stehen in den Tabellen.</p>'
    )

    def collect(stage: str, x: str, y: str, pick: Any = None) -> list[dict[str, Any]]:
        out = []
        for slot, run in enumerate(shown):
            rows = [r for r in (run[stage] or {}).get("rows", []) if pick is None or pick(r)]
            out.append({"label": run["label"], "slot": slot, "points": [(r.get(x), r.get(y)) for r in rows]})
        return out

    parts = [
        "<h3>Durchsatz bei steigender Dauerlast (Stufe 2)</h3>",
        "<p>Erzeugte Tokens pro Sekunde über alle gleichzeitigen Anfragen. Wo die Linie abflacht, ist die Zelle gesättigt.</p>",
        line_chart(collect("s2", "level", "throughput_tps"), x_label="gleichzeitige Anfragen", y_label="Tokens pro Sekunde, gesamt"),
        "<h3>Zeit bis zum ersten Token nach Promptlänge (Stufe 1)</h3>",
        "<p>Eine einzelne Anfrage ohne Nachdenken, ohne Konkurrenz. Beide Achsen sind logarithmisch.</p>",
        line_chart(collect("s1", "prompt_target", "ttft_s", lambda r: r.get("thinking") == "off"),
                   x_label="Promptlänge in Tokens", y_label="Sekunden bis zum ersten Token", log_x=True, log_y=True),
        "<h3>Zeit bis zum ersten Token im gemischten Betrieb (Stufe 6)</h3>",
        "<p>95. Perzentil je Ankunftsrate. Gezeigt sind auch überlastete Abschnitte; dort steigt die Linie steil."
        + (" Die gestrichelte Linie ist der Zielwert." if opts["ttft_p95"] else "") + "</p>",
        line_chart(collect("s6", "rate", "ttft_p95_s"), x_label="Modellaufrufe pro Sekunde", y_label="Sekunden bis zum ersten Token, p95",
                   log_y=True, target=opts["ttft_p95"], target_label=f"Zielwert {de(opts['ttft_p95'], 1)} s" if opts["ttft_p95"] else ""),
    ]
    return note + "".join(parts)


def run_details(run: dict[str, Any]) -> str:
    """Tabellen aller Stufen eines Laufs."""
    m, feas = run["manifest"], run["feasibility"] or {}
    engine = m.get("engine") or {}
    parts = [f"<details><summary>{esc(run['label'])}</summary>"]
    parts.append(
        f'<p class="muted">Laufordner {esc(run["name"])}; Modell {esc((m.get("model") or {}).get("repo"))}; '
        f'Image {esc(engine.get("image") or "fremd gestartet")} {esc((engine.get("image_id") or "")[:19])}</p>'
    )
    if engine.get("command"):
        parts.append(f'<p class="muted">Start: <code>{esc(engine["command"])}</code></p>')
    if feas.get("loaded") is False:
        parts.append(f'<p class="bad">Lädt nicht: {esc(feas.get("error") or feas.get("sanity_error"))}</p></details>')
        return "".join(parts)
    s1, s2, s3, s4, s5, s6, s7 = (run[f"s{n}"] or {} for n in range(1, 8))
    if s1.get("rows"):
        parts.append("<h3>Stufe 1, Einzelstrom</h3>" + table(
            ["Nachdenken", "Prompt (Tokens)", "erstes Token", "erste Antwort", "tok/s", "Einlesen tok/s"],
            [[r["thinking"], de(r["prompt_target"], 0), de(r["ttft_s"], 2, " s"), de(r["ttfc_s"], 2, " s"),
              de(r["decode_tps"], 1), de(r["prefill_tps"], 0)] for r in s1["rows"]]))
    if s2.get("rows"):
        parts.append(f"<h3>Stufe 2, Parallelität</h3><p class='muted'>{esc(s2.get('stop_reason'))}</p>" + table(
            ["parallel", "tok/s gesamt", "tok/s je Anfrage", "TTFT p50", "TTFT p95", "Fehler"],
            [[r["level"], de(r["throughput_tps"], 0), de(r["decode_tps_p50"], 1), de(r["ttft_p50_s"], 2, " s"),
              de(r["ttft_p95_s"], 2, " s"), r["errors"]] for r in s2["rows"]]))
    if s3.get("rows"):
        kept = {f["fill"]: f for f in s3.get("fills", [])}

        def retained(fill: float) -> str:
            info = kept.get(fill) or {}
            value = info.get("retained_by_ttft") if info.get("retained_by_ttft") is not None else info.get("retained_by_cache_report")
            return "–" if value is None else de(100 * value, 0, " %")

        parts.append("<h3>Stufe 3, KV-Füllstand</h3>" + table(
            ["Füllstand", "parallel", "tok/s gesamt", "TTFT p50", "TTFT p95", "Fehler", "gehaltene Gespräche danach im Cache"],
            [[de(100 * r["fill"], 0, " %"), r["level"], de(r["throughput_tps"], 0), de(r["ttft_p50_s"], 2, " s"),
              de(r["ttft_p95_s"], 2, " s"), r["errors"], retained(r["fill"])] for r in s3["rows"]]))
    elif s3.get("skipped"):
        parts.append(f"<h3>Stufe 3, KV-Füllstand</h3><p class='muted'>Ausgelassen: {esc(s3['skipped'])}</p>")
    if s4.get("rows"):
        parts.append("<h3>Stufe 4, Präfix-Cache</h3>" + table(
            ["Prompt (Tokens)", "erstes Token kalt", "Folgefrage sofort", "Faktor", "aus dem Cache",
             "Folgefrage kurz danach", "nach Pause"],
            [[de(r["prompt_target"], 0), de(r["cold_ttft_s"], 2, " s"), de(r["followup_ttft_s"], 2, " s"), de(r["speedup"], 1),
              "–" if r["followup_cached_share"] is None else de(100 * r["followup_cached_share"], 0, " %"),
              de(r.get("followup_delayed_ttft_s"), 2, " s"), de(r["after_pause_ttft_s"], 2, " s")] for r in s4["rows"]]))
        ev = (s4.get("eviction") or {}).get("rows") or []
        if ev:
            parts.append(table(
                ["Verdrängung: Prompt (Tokens)", "kalt", "Folgefrage vorher", "Folgefrage danach"],
                [[de(r["prompt_target"], 0), de(r["cold_ttft_s"], 2, " s"), de(r["before_ttft_s"], 2, " s"),
                  de(r["after_ttft_s"], 2, " s")] for r in ev]))
    if s5.get("rows"):
        rows = []
        for r in s5["rows"]:
            if not r["big_ok"]:
                rows.append([de(r["big_target"], 0), ("Fehler: " + str(r["big_error"])[:80], "bad"), "", "", "", ""])
                continue
            base, pre = r["baseline"] or {}, r["prefill"] or {}
            rows.append([de(r["big_target"], 0), de(r["big_ttft_s"], 1, " s"), de(base.get("tps"), 0), de(pre.get("tps"), 0),
                         de(r["slowdown_prefill_pct"], 0, " %"), de(pre.get("silence_max_s"), 2, " s")])
        parts.append(f"<h3>Stufe 5, Störwirkung ({s5.get('chats')} laufende Chats)</h3>" + table(
            ["große Anfrage (Tokens)", "ihr erstes Token", "Chats tok/s vorher", "beim Einlesen", "Einbruch", "längste Stille"], rows))
    if s6.get("rows"):
        parts.append("<h3>Stufe 6, Szenario</h3>" + table(
            ["Aufrufe/s", "erreicht", "Aufrufe", "laufend im Mittel", "TTFT p50", "TTFT p95", "Dauer p50", "Dauer p95", "tok/s", "Fehler", "Urteil"],
            [[de(r["rate"], 2), de(r["achieved_rate"], 2), r["calls"], de(r["inflight_mean"], 1), de(r["ttft_p50_s"], 2, " s"),
              de(r["ttft_p95_s"], 2, " s"), de(r["total_p50_s"], 1, " s"), de(r["total_p95_s"], 1, " s"), de(r["throughput_tps"], 0),
              r["errors"], ("Überlast: " + "; ".join(r["overload"]), "bad text") if r.get("overload") else "getragen"]
             for r in s6["rows"]], text_columns=(10,)))
    if s7.get("buckets"):
        res = s7.get("resources") or {}
        parts.append(
            f"<h3>Stufe 7, Dauerlauf</h3><p class='muted'>{de(s7['duration_s'] / 60, 0)} Minuten mit {de(s7['rate'], 2)} Aufrufen/s. "
            f"GPU-Speicher {de(res.get('gpu_mem_first_mib'), 0)} → {de(res.get('gpu_mem_last_mib'), 0)} MiB, freier Arbeitsspeicher "
            f"{de(res.get('host_mem_available_first_gib'), 1)} → {de(res.get('host_mem_available_last_gib'), 1)} GiB.</p>"
            + table(["Minute", "Aufrufe", "laufend im Mittel", "TTFT p50", "TTFT p95", "tok/s", "Fehler"],
                    [[de(b["minute"], 0), b["calls"], de(b["inflight_mean"], 1), de(b["ttft_p50_s"], 2, " s"),
                      de(b["ttft_p95_s"], 2, " s"), de(b["throughput_tps"], 0), b["errors"]] for b in s7["buckets"]]))
    parts.append("</details>")
    return "".join(parts)


def build_report(run_dirs: list[str | Path], opts: dict[str, Any]) -> str:
    runs = [load_run(path) for path in run_dirs]
    figures = [key_figures(run, opts) for run in runs]
    light = "".join(f"--s{i}:{pair[0]};" for i, pair in enumerate(PALETTE))
    dark = "".join(f"--s{i}:{pair[1]};" for i, pair in enumerate(PALETTE))
    gpu_list = next((r["manifest"].get("gpu") for r in runs if r["manifest"].get("gpu")), None) or []
    gpus = ", ".join(f"{g.get('name')} ({de(g.get('mem_total_mib'), 0)} MiB)" for g in gpu_list) or "unbekannt"
    notes = []
    if any(r["smoke"] for r in runs):
        notes.append("Mindestens ein Lauf ist ein Funktionstest mit winzigen Parametern. Seine Zahlen sind keine Messung.")
    thinking = {(r["s6"] or {}).get("thinking") for r in runs if (r["s6"] or {}).get("rows")}
    if thinking:
        notes.append(
            "Das Szenario lief mit Nachdenken: " + ", ".join(sorted(str(t) for t in thinking)) + ". Mit eingeschaltetem "
            "Nachdenken schreibt ein Modell je Aufruf deutlich mehr Tokens; die getragene Rate liegt dann niedriger."
        )
    if any(f["dropped_kinds"] for f in figures):
        notes.append(
            "Nicht jede Zelle fuhr im Szenario denselben Sitzungsmix: Wo eine Sitzungsart nicht in den Kontext passt, "
            "entfällt sie, und die getragene Rate ist mit den anderen Zellen nicht vergleichbar. Die Tabelle nennt das "
            "bei der Rate. Eine solche Zelle kann die entfallene Arbeit, etwa große Dokumente, gar nicht leisten."
        )
    notes.append(
        f"Die Zahl der Berechtigten folgt aus der getragenen Rate und dem Nutzungsprofil „{opts['profile']}“ des Lastmodells. "
        "Anteil aktiver Nutzer, Aktionen je Stunde und die Form der Gespräche sind Annahmen, solange keine Mitschrift aus dem Betrieb vorliegt."
    )
    if opts["ttft_p95"] is not None or opts["total_p95"] is not None:
        targets = [f"erstes Token p95 höchstens {de(opts['ttft_p95'], 1)} s" if opts["ttft_p95"] is not None else "",
                   f"Gesamtdauer p95 höchstens {de(opts['total_p95'], 1)} s" if opts["total_p95"] is not None else ""]
        notes.append("Zielwerte, nachträglich angewandt: " + " und ".join(t for t in targets if t) + ".")
    if opts["cost_month"]:
        notes.append(
            f"Kosten: {de(opts['cost_month'], 0)} € je Monat für die Maschine. „Je Mio. Ausgabe-Tokens bei Dauerlast“ unterstellt, "
            "dass die Maschine den höchsten Durchsatz der Stufe 2 rund um die Uhr liefert; im echten Betrieb liegt der Wert höher."
        )
    body = [
        "<h1>styrke: Vergleich der Messläufe</h1>",
        f'<p class="muted">Erstellt am {time.strftime("%d.%m.%Y um %H:%M")}. {len(runs)} Läufe. GPU: {esc(gpus)}.</p>',
        "<h2>Kennzahlen für die Entscheidung</h2>",
        decision_tables(runs, figures, opts),
        "".join(f'<div class="note">{esc(n)}</div>' for n in notes),
        "<h2>Diagramme</h2>", charts(runs, opts),
        "<h2>Alle Stufen je Lauf</h2>", "".join(run_details(r) for r in runs),
    ]
    css = CSS.replace("%LIGHT%", light).replace("%DARK%", dark)
    return (
        '<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>styrke Bericht</title><style>{css}</style></head><body><main>{''.join(body)}</main>"
        f'<div id="tip"></div><script>{JS}</script></body></html>'
    )
