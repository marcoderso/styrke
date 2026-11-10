"""Kommandozeile von styrke."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, load_matrix


def log(message: str = "") -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}" if message else "", flush=True)


# -- matrix -------------------------------------------------------------------


def cmd_matrix(args: argparse.Namespace) -> int:
    matrix = load_matrix(args.matrix)
    print(f"Testmatrix: {matrix.path}")
    print(f"Modelle unter: {matrix.host.models_dir}\n")
    print(f"{'Zelle':<22} {'Modell':<32} {'Engine':<10} {'Varianten'}")
    for cell in matrix.cells:
        model = matrix.models[cell.model]
        variants = ", ".join(v.id for v in cell.variants)
        state = "" if cell.enabled else f"  (noch nicht freigeschaltet: {cell.note})"
        print(f"{cell.id:<22} {model.repo:<32} {cell.engine:<10} {variants}{state}")
    return 0


# -- run ----------------------------------------------------------------------


def parse_stages(text: str) -> list[int]:
    """Macht aus ``0,2,4-6`` die Liste ``[0, 2, 4, 5, 6]``."""
    stages: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            stages.update(range(int(lo), int(hi) + 1))
        else:
            stages.add(int(part))
    return sorted(stages)


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import RunOptions, run_cell

    try:
        stages = parse_stages(args.stages)
    except ValueError:
        print(f"Fehler: --stages versteht {args.stages!r} nicht (Beispiele: 0,1,2 oder 3-7)", file=sys.stderr)
        return 1
    opts = RunOptions(
        matrix_path=args.matrix,
        cell_id=args.cell,
        variant_id=args.variant,
        stages=stages,
        endpoint=args.endpoint,
        model_name=args.model_name,
        resume=args.resume,
        smoke=args.smoke,
        allow_busy_gpu=args.allow_busy_gpu,
    )
    async def main_task() -> int:
        task = asyncio.current_task()
        assert task is not None
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        return await run_cell(opts, log=log)

    try:
        return asyncio.run(main_task())
    except (KeyboardInterrupt, asyncio.CancelledError):
        log("Abgebrochen. Die bis dahin gemessenen Daten bleiben erhalten; mit --resume fortsetzen.")
        return 130


# -- show ---------------------------------------------------------------------


def _fmt(value: Any, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def cmd_show(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Kein Laufordner: {run_dir}", file=sys.stderr)
        return 1

    def load(name: str) -> Any:
        path = run_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    manifest = load("manifest.json") or {}
    print(f"Lauf:     {run_dir}")
    print(f"Zelle:    {manifest.get('cell')} / Variante {(manifest.get('variant') or {}).get('id')}")
    print(f"Modell:   {(manifest.get('model') or {}).get('repo')}")
    print(f"Engine:   {(manifest.get('engine') or {}).get('command') or (manifest.get('engine') or {}).get('base_url')}")
    if manifest.get("smoke"):
        print("Hinweis:  Funktionstest (--smoke), keine Messung")

    feas = load("feasibility.json")
    if feas:
        print("\nStufe 0, Machbarkeit")
        if not feas.get("loaded"):
            print(f"  nicht lauffähig: {feas.get('error') or feas.get('sanity_error')}")
        else:
            cal = feas.get("calibration") or {}
            print(f"  Ladezeit: {_fmt(feas.get('load_time_s'), 0)} s")
            print(f"  größter geprüfter Kontext: {feas.get('max_context_ok')} Tokens")
            print(f"  Promptlänge: Abweichung {_fmt(cal.get('check_deviation_pct'), 2)} % (Zähler: {cal.get('counter')})")
            for gpu in feas.get("gpu") or []:
                print(f"  GPU {gpu.get('name')}: {_fmt(gpu.get('mem_used_mib'), 0)} von {_fmt(gpu.get('mem_total_mib'), 0)} MiB belegt")

    s1 = load("summary_s1.json")
    if s1:
        print("\nStufe 1, Einzelstrom (Mediane)")
        print(f"  {'Thinking':<8} {'Prompt':>8} {'TTFT s':>8} {'Antwort s':>10} {'tok/s':>7} {'Einlesen tok/s':>15} {'ok':>5}")
        for row in s1["rows"]:
            print(
                f"  {row['thinking']:<8} {row['prompt_target']:>8} {_fmt(row['ttft_s'], 2):>8} {_fmt(row['ttfc_s'], 2):>10} "
                f"{_fmt(row['decode_tps']):>7} {_fmt(row['prefill_tps'], 0):>15} {row['n_ok']}/{row['n']:<3}"
            )
        for skip in s1.get("skipped") or []:
            print(f"  ausgelassen: {skip}")

    s2 = load("summary_s2.json")
    if s2:
        print("\nStufe 2, Parallelität")
        print(f"  {'parallel':>8} {'tok/s gesamt':>13} {'tok/s je Anfrage':>17} {'TTFT p50':>9} {'TTFT p95':>9} {'Fehler':>7}")
        for row in s2["rows"]:
            print(
                f"  {row['level']:>8} {row['throughput_tps']:>13.1f} {_fmt(row['decode_tps_p50']):>17} "
                f"{_fmt(row['ttft_p50_s'], 2):>9} {_fmt(row['ttft_p95_s'], 2):>9} {row['errors']:>7}"
            )
        print(f"  Ende der Stufe: {s2['stop_reason']}")

    _show_later_stages(load)
    return 0


def _pct(value: Any, digits: int = 0) -> str:
    return "-" if value is None else f"{100 * value:.{digits}f} %"


def _show_later_stages(load: Any) -> None:
    """Kurzauswertungen der Stufen 3 bis 7."""
    s3 = load("summary_s3.json")
    if s3:
        print("\nStufe 3, KV-Füllstand")
        if s3.get("skipped"):
            print(f"  ausgelassen: {s3['skipped']}")
        else:
            print(f"  KV-Pool {s3['pool_tokens']} Tokens, gehaltene Gespräche zu je {s3['session_tokens']} Tokens")
            print(f"  {'Füllstand':>9} {'parallel':>9} {'tok/s gesamt':>13} {'TTFT p50':>9} {'TTFT p95':>9} {'Fehler':>7}")
            for row in s3["rows"]:
                print(
                    f"  {_pct(row['fill']):>9} {row['level']:>9} {row['throughput_tps']:>13.1f} "
                    f"{_fmt(row['ttft_p50_s'], 2):>9} {_fmt(row['ttft_p95_s'], 2):>9} {row['errors']:>7}"
                )
            for fill in s3["fills"]:
                if not fill.get("probe_n"):
                    continue
                kept = fill["retained_by_ttft"] if fill["retained_by_ttft"] is not None else fill["retained_by_cache_report"]
                print(
                    f"  bei {_pct(fill['fill'])}: {fill['held_sessions']} Gespräche gehalten, nach der Last noch im Cache: "
                    f"{_pct(kept)} (Folgefrage {_fmt(fill['probe_ttft_p50_s'], 2)} s, kalt {_fmt(fill['cold_ttft_p50_s'], 2)} s)"
                )

    s4 = load("summary_s4.json")
    if s4:
        print("\nStufe 4, Präfix-Cache (Zeit bis zum ersten Token, Mediane)")
        print(f"  {'Prompt':>8} {'kalt s':>8} {'Folgefrage s':>13} {'Faktor':>7} {'aus Cache':>10} {'nach Pause s':>13}")
        for row in s4["rows"]:
            print(
                f"  {row['prompt_target']:>8} {_fmt(row['cold_ttft_s'], 2):>8} {_fmt(row['followup_ttft_s'], 2):>13} "
                f"{_fmt(row['speedup']):>7} {_pct(row['followup_cached_share']):>10} {_fmt(row['after_pause_ttft_s'], 2):>13}"
            )
        for skip in s4.get("skipped") or []:
            print(f"  ausgelassen: {skip['prompt_target']} Tokens, {skip['reason']}")
        eviction = s4.get("eviction") or {}
        if eviction.get("skipped"):
            print(f"  Verdrängung ausgelassen: {eviction['skipped']}")
        for row in eviction.get("rows") or []:
            print(
                f"  Verdrängung, {row['prompt_target']} Tokens: kalt {_fmt(row['cold_ttft_s'], 2)} s, Folgefrage vorher "
                f"{_fmt(row['before_ttft_s'], 2)} s, danach {_fmt(row['after_ttft_s'], 2)} s"
            )

    s5 = load("summary_s5.json")
    if s5:
        print(f"\nStufe 5, Störwirkung ({s5['chats']} laufende Chats)")
        print(
            f"  {'große Anfrage':>14} {'ihr TTFT s':>11} {'Chats tok/s vorher':>19} {'beim Einlesen':>14} "
            f"{'Einbruch':>9} {'längste Pause s':>16}"
        )
        for row in s5["rows"]:
            if not row["big_ok"]:
                print(f"  {row['big_target']:>14} FEHLER: {row['big_error']}")
                continue
            base, prefill = row["baseline"] or {}, row["prefill"] or {}
            slow = row["slowdown_prefill_pct"]
            print(
                f"  {row['big_target']:>14} {_fmt(row['big_ttft_s']):>11} {_fmt(base.get('tps')):>19} "
                f"{_fmt(prefill.get('tps')):>14} {('-' if slow is None else f'{slow:.0f} %'):>9} "
                f"{_fmt(prefill.get('silence_max_s'), 2):>16}"
            )
        for skip in s5.get("skipped") or []:
            print(f"  ausgelassen: {skip['big_target']} Tokens, {skip['reason']}")

    s6 = load("summary_s6.json")
    if s6:
        print("\nStufe 6, Szenario")
        if s6.get("skipped"):
            print(f"  ausgelassen: {s6['skipped']}")
        else:
            print(
                f"  {'Aufrufe/s':>10} {'erreicht':>9} {'Aufrufe':>8} {'laufend':>8} {'TTFT p50':>9} {'TTFT p95':>9} "
                f"{'Dauer p50':>10} {'tok/s':>7} {'Fehler':>7}  Urteil"
            )
            for row in s6["rows"]:
                verdict = "getragen" if not row.get("overload") else "Überlast: " + "; ".join(row["overload"])
                print(
                    f"  {row['rate']:>10.2f} {row['achieved_rate']:>9.2f} {row['calls']:>8} {_fmt(row['inflight_mean']):>8} "
                    f"{_fmt(row['ttft_p50_s'], 2):>9} {_fmt(row['ttft_p95_s'], 2):>9} {_fmt(row['total_p50_s']):>10} "
                    f"{row['throughput_tps']:>7.0f} {row['errors']:>7}  {verdict}"
                )
            if s6.get("max_sustained_rate") is None:
                print("  Ergebnis: schon die kleinste Rate wird nicht getragen")
            else:
                users = ", ".join(f"{name} {n}" for name, n in (s6.get("users_supported") or {}).items())
                print(f"  Ergebnis: {s6['max_sustained_rate']:g} Aufrufe/s getragen. Berechtigte je Profil: {users}")
            for drop in s6.get("sessions_dropped") or []:
                print(f"  Sitzungsart {drop['kind']} entfiel: {drop['reason']}")

    s7 = load("summary_s7.json")
    if s7:
        print("\nStufe 7, Dauerlauf")
        if s7.get("skipped"):
            print(f"  ausgelassen: {s7['skipped']}")
        else:
            overall, res = s7["overall"], s7["resources"]
            print(f"  {s7['duration_s'] / 60:g} Minuten mit {s7['rate']:g} Aufrufen/s ({s7['rate_source']})")
            print(
                f"  Aufrufe {overall['calls']}, Fehler {overall['errors']}, TTFT p50 {_fmt(overall['ttft_p50_s'], 2)} s, "
                f"p95 {_fmt(overall['ttft_p95_s'], 2)} s"
            )
            print(
                f"  Veränderung vom ersten zum letzten Drittel: TTFT p95 {_fmt(s7['ttft_p95_drift_pct'], 0)} %, "
                f"Durchsatz {_fmt(s7['throughput_drift_pct'], 0)} %"
            )
            print(
                f"  GPU-Speicher {_fmt(res['gpu_mem_first_mib'], 0)} -> {_fmt(res['gpu_mem_last_mib'], 0)} MiB, "
                f"freier Arbeitsspeicher {_fmt(res['host_mem_available_first_gib'])} -> "
                f"{_fmt(res['host_mem_available_last_gib'])} GiB"
            )
            if overall.get("overload"):
                print("  Hinweis: Überlast im Dauerlauf: " + "; ".join(overall["overload"]))

    s8 = load("summary_s8.json")
    if s8:
        print("\nStufe 8, Qualität")
        if s8.get("skipped"):
            print(f"  ausgelassen: {s8['skipped']}")
        else:
            print(f"  {'Nachdenken':<11} {'Kategorie':<16} {'Fragen':>7} {'geprüft':>8} {'bestanden':>10} {'von Hand':>9} {'Fehler':>7}")
            for row in s8["rows"]:
                print(
                    f"  {row['thinking']:<11} {row['kategorie']:<16} {row['n']:>7} {row['auto_checked']:>8} "
                    f"{row['auto_ok']:>10} {row['manual']:>9} {row['errors']:>7}"
                )
            for item in s8.get("failed") or []:
                reason = item.get("error") or ("fehlt: " + "; ".join(item["fehlt"]) if item.get("fehlt") else "Prüfung nicht bestanden")
                print(f"  nicht bestanden: {item['id']} (Nachdenken {item['thinking']}): {reason}")


# -- watch --------------------------------------------------------------------


def cmd_watch(args: argparse.Namespace) -> int:
    from .watch import newest_run, watch

    run_dir = Path(args.run_dir) if args.run_dir else newest_run(load_matrix(args.matrix).host.runs_dir)
    if run_dir is None or not run_dir.is_dir():
        print("Kein Laufordner gefunden. Angeben mit: styrke watch runs/<laufordner>", file=sys.stderr)
        return 1
    return watch(run_dir, refresh_s=args.interval, once=args.once)


# -- report -------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report, discover

    matrix = load_matrix(args.matrix)
    run_dirs = [Path(p) for p in args.run_dirs] or discover(matrix.host.runs_dir, include_smoke=args.mit_funktionstests)
    missing = [str(p) for p in run_dirs if not (p / "manifest.json").is_file()]
    if missing or not run_dirs:
        print("Fehler: " + (f"kein Laufordner: {', '.join(missing)}" if missing else "keine Läufe gefunden"), file=sys.stderr)
        return 1
    if args.profil not in (matrix.load_model.get("profiles") or {}):
        print(f"Fehler: Profil {args.profil!r} steht nicht im Lastmodell der Testmatrix", file=sys.stderr)
        return 1
    opts = {"ttft_p95": args.ttft_p95, "total_p95": args.dauer_p95, "cost_month": args.kosten_monat, "profile": args.profil}
    Path(args.out).write_text(build_report(run_dirs, opts), encoding="utf-8")
    print(f"Bericht über {len(run_dirs)} Läufe geschrieben: {args.out}")
    return 0


# -- bewertung ----------------------------------------------------------------


def cmd_bewertung(args: argparse.Namespace) -> int:
    from .quality import export_blind, import_blind
    from .report import discover

    if args.action == "export":
        run_dirs = [Path(p) for p in args.paths] or discover(load_matrix(args.matrix).host.runs_dir)
        result = export_blind(run_dirs, args.out, thinking=args.thinking)
        if not result["antworten"]:
            print("Fehler: In den Läufen liegen keine Antworten der Stufe 8 (answers/s8.jsonl).", file=sys.stderr)
            return 1
        print(f"{result['antworten']} Antworten aus {result['laeufe']} Läufen nach {args.out}/ geschrieben.")
        print("bogen.csv und anleitung.md gehen an die bewertende Person, schluessel.json bleibt bei dir.")
        return 0
    if len(args.paths) != 2:
        print("Aufruf: styrke bewertung import <bogen.csv> <schluessel.json>", file=sys.stderr)
        return 1
    result = import_blind(args.paths[0], args.paths[1])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "bewertung.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Skala: {result['skala']}")
    for run, info in result["laeufe"].items():
        cats = ", ".join(f"{cat} {c['mittel']:.2f}" for cat, c in info["kategorien"].items())
        print(f"  {run}: Mittel {info['mittel']:.2f} aus {info['n']} Antworten ({cats})")
    if result["ohne_punkte"]:
        print(f"  {result['ohne_punkte']} Zeilen ohne Punkte blieben unberücksichtigt.")
    return 0


# -- corpus -------------------------------------------------------------------


def cmd_corpus(args: argparse.Namespace) -> int:
    from .corpus import Corpus, CorpusError, fetch_wikipedia

    matrix = load_matrix(args.matrix)
    corpus_dir = matrix.host.corpus_dir
    code = 0
    if args.action == "fetch":
        print(f"Lade deutsche Wikipedia-Artikel nach {corpus_dir}/ (Ziel: {args.mb:g} MB)")
        try:
            result = fetch_wikipedia(corpus_dir, target_mb=args.mb)
            print(f"Fertig: {result['new_files']} neue Dateien, zusammen {result['chars'] / 1e6:.1f} MB")
        except CorpusError as exc:
            print(f"Fehler: {exc}", file=sys.stderr)
            code = 1
        except KeyboardInterrupt:
            print("Abgebrochen. Derselbe Aufruf setzt den Abruf fort.", file=sys.stderr)
            code = 130
    info = Corpus.load(corpus_dir).info()
    print(f"Korpus: {info['source']}, {info['paragraphs']} Absätze, {info['chars'] / 1e6:.2f} MB")
    if info["builtin"]:
        print("Das ist der eingebaute Mini-Korpus. Er taugt nur für Funktionstests.")
    return code


# -- lastmodell ---------------------------------------------------------------


def cmd_lastmodell(args: argparse.Namespace) -> int:
    from .loadmodel import table

    matrix = load_matrix(args.matrix)
    lm = matrix.load_model
    print(f"Berechtigte: {lm.get('entitled')}   (Gleichzeitigkeit, die in {args.quantile} % der Zeit reicht)\n")
    print(f"{'Profil':<16} {'a':>6} {'t':>5} {'k':>5} {'S in s':>7} {'lambda 1/s':>11} {'A':>7} {'C':>5}")
    for row in table(lm, args.quantile):
        print(
            f"{row['profile']:<16} {row['a']:>6.0%} {row['t']:>5} {row['k']:>5} {row['s']:>7} "
            f"{row['lambda']:>11.3f} {row['mean']:>7.1f} {row['peak']:>5}"
        )
    return 0


# -- mitschreiber -------------------------------------------------------------


def cmd_mitschreiber(args: argparse.Namespace) -> int:
    if args.action == "serve":
        from aiohttp import web

        from .prompts import HFCounter
        from .recorder import make_app

        counter = HFCounter(args.tokenizer) if args.tokenizer else None
        host, _, port = args.listen.rpartition(":")
        app = make_app(args.upstream, args.out, counter=counter, inject_usage=args.inject_usage)
        print(f"Mitschreiber: {args.listen} -> {args.upstream}, Mitschrift in {args.out}/ (keine Inhalte)")
        web.run_app(app, host=host or "127.0.0.1", port=int(port), print=None, access_log=None)
        return 0

    from .recorder import report

    result = report(args.out, entitled=args.berechtigte)
    if not result["requests"]:
        print(f"Keine mitgeschriebenen Anfragen in {args.out}/")
        return 0
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(f"Anfragen: {result['requests']} an {result['days']} Tag(en), {result['clients']} Absender, {result['users']} Nutzerkennungen")
    print(f"Gestreamt: {result['share_stream']:.0%}, mit Werkzeugen: {result['share_with_tools']:.0%}")
    print(f"Meiste gleichzeitige Anfragen: {result['peak_concurrency']}\n")
    print(f"{'':<20} {'Median':>10} {'95 %':>10} {'Maximum':>10}")
    for key, label in (
        ("prompt_tokens", "Prompt-Tokens"), ("prompt_chars", "Prompt-Zeichen"),
        ("completion_tokens", "Antwort-Tokens"), ("duration_s", "Dauer in s"), ("ttft_s", "Erstes Token in s"),
    ):
        d = result[key]
        if d["n"]:
            print(f"{label:<20} {d['p50']:>10} {d['p95']:>10} {d['max']:>10}")
    ph = result["peak_hour"]
    print(f"\nSpitzenstunde {ph['hour']} Uhr: {ph['requests']} Anfragen von {ph['active']} Aktiven")
    print(f"  Aufrufe je Aktivem (t x k): {ph['requests_per_active']}")
    print(f"  mittlere Dauer (S):         {ph['mean_duration_s']} s")
    if "share_active" in ph:
        print(f"  Anteil aktiv (a):           {ph['share_active']:.1%} von {args.berechtigte} Berechtigten")
    return 0


# -- Aufbau -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="styrke", description="Mess-Harness für lokale LLM-Inferenz")
    parser.add_argument("--version", action="version", version=f"styrke {__version__}")
    parser.add_argument("--matrix", default="matrix.yaml", help="Pfad der Testmatrix (Standard: matrix.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("matrix", help="Zellen und Varianten der Testmatrix anzeigen")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("run", help="Stufen einer Zelle fahren")
    p.add_argument("--cell", required=True, help="Zelle aus der Testmatrix")
    p.add_argument("--variant", help="Variante der Zelle (Standard: die erste)")
    p.add_argument("--stages", default="0-8", help="Stufen, z. B. 0,1,2 oder 3-7 (Standard: alle, 0-8)")
    p.add_argument("--endpoint", help="an eine laufende Engine anhängen, z. B. http://127.0.0.1:8000")
    p.add_argument("--model-name", help="Modellname an der Schnittstelle (bei --endpoint; sonst automatisch)")
    p.add_argument("--resume", metavar="LAUFORDNER", help="abgebrochenen Lauf fortsetzen")
    p.add_argument("--smoke", action="store_true", help="kurzer Funktionstest mit kleinen Parametern")
    p.add_argument("--allow-busy-gpu", action="store_true", help="auch starten, wenn die GPU schon belegt ist")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("show", help="Kurzauswertung eines Laufs anzeigen")
    p.add_argument("run_dir", help="Laufordner unter runs/")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("watch", help="laufenden oder beendeten Lauf im Terminal verfolgen")
    p.add_argument("run_dir", nargs="?", help="Laufordner; ohne Angabe der neueste unter runs/")
    p.add_argument("--interval", type=float, default=1.0, help="Sekunden zwischen zwei Aktualisierungen (Standard: 1)")
    p.add_argument("--once", action="store_true", help="den Stand einmal ausgeben und enden")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("report", help="Bericht (HTML) über einen oder mehrere Läufe schreiben")
    p.add_argument("run_dirs", nargs="*", help="Laufordner; ohne Angabe je Zelle und Variante der neueste unter runs/")
    p.add_argument("--out", default="bericht.html", help="Zieldatei (Standard: bericht.html)")
    p.add_argument("--ttft-p95", type=float, help="Zielwert: Sekunden bis zum ersten Token, 95. Perzentil")
    p.add_argument("--dauer-p95", type=float, help="Zielwert: Gesamtdauer eines Aufrufs in Sekunden, 95. Perzentil")
    p.add_argument("--kosten-monat", type=float, help="Monatskosten der Maschine in Euro")
    p.add_argument("--profil", default="mittel", help="Nutzungsprofil des Lastmodells für die Zahl der Berechtigten (Standard: mittel)")
    p.add_argument("--mit-funktionstests", action="store_true", help="auch Läufe mit --smoke aufnehmen")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("bewertung", help="Blindbewertung der Antworten: Bogen ausgeben und ausgefüllt einlesen")
    p.add_argument("action", choices=["export", "import"])
    p.add_argument("paths", nargs="*", help="export: Laufordner (ohne Angabe die neuesten); import: bogen.csv schluessel.json")
    p.add_argument("--out", default="bewertung", help="Ordner für Bogen und Ergebnis (Standard: bewertung)")
    p.add_argument("--thinking", default="off", choices=["off", "on"], help="welche Antworten in den Bogen kommen (Standard: off)")
    p.set_defaults(func=cmd_bewertung)

    p = sub.add_parser("corpus", help="Textkorpus beschaffen oder anzeigen")
    p.add_argument("action", choices=["fetch", "info"])
    p.add_argument("--mb", type=float, default=40.0, help="Zielgröße beim Beschaffen in MB (Standard: 40)")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("lastmodell", help="Gleichzeitige Anfragen je Nutzungsprofil berechnen")
    p.add_argument("--quantile", type=int, choices=[95, 99], default=99)
    p.set_defaults(func=cmd_lastmodell)

    p = sub.add_parser("mitschreiber", help="Längen und Zeiten echter Anfragen mitschreiben")
    p.add_argument("action", choices=["serve", "report"])
    p.add_argument("--listen", default="127.0.0.1:8000", help="Adresse, auf der der Mitschreiber lauscht")
    p.add_argument("--upstream", default="http://127.0.0.1:8001", help="Adresse der Engine")
    p.add_argument("--out", default="usage", help="Ordner der Mitschrift (Standard: usage)")
    p.add_argument("--tokenizer", help="tokenizer.json des Modells, um Tokens zu zählen, wenn die Engine keine meldet")
    p.add_argument("--inject-usage", action="store_true", help="bei gestreamten Anfragen Zählwerte von der Engine anfordern")
    p.add_argument("--berechtigte", type=int, help="Zahl der Berechtigten für den Anteil der Aktiven")
    p.add_argument("--json", action="store_true", help="Auswertung als JSON ausgeben")
    p.set_defaults(func=cmd_mitschreiber)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ConfigError as exc:
        print(f"Fehler in der Testmatrix: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
