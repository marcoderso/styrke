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


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import RunOptions, run_cell

    stages = sorted({int(s) for s in args.stages.split(",") if s.strip() != ""})
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
    return 0


# -- corpus -------------------------------------------------------------------


def cmd_corpus(args: argparse.Namespace) -> int:
    from .corpus import Corpus, CorpusError, fetch_wikipedia

    matrix = load_matrix(args.matrix)
    corpus_dir = matrix.host.corpus_dir
    if args.action == "fetch":
        print(f"Lade deutsche Wikipedia-Artikel nach {corpus_dir}/ (Ziel: {args.mb:.0f} MB)")
        try:
            result = fetch_wikipedia(corpus_dir, target_mb=args.mb)
        except CorpusError as exc:
            print(f"Fehler: {exc}", file=sys.stderr)
            return 1
        print(f"Fertig: {result['files']} Dateien, {result['chars'] / 1e6:.1f} MB")
    info = Corpus.load(corpus_dir).info()
    print(f"Korpus: {info['source']}, {info['paragraphs']} Absätze, {info['chars'] / 1e6:.2f} MB")
    if info["builtin"]:
        print("Das ist der eingebaute Mini-Korpus. Er taugt nur für Funktionstests.")
    return 0


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
    p.add_argument("--stages", default="0,1,2", help="Stufen, durch Komma getrennt (Standard: 0,1,2)")
    p.add_argument("--endpoint", help="an eine laufende Engine anhängen, z. B. http://127.0.0.1:8000")
    p.add_argument("--model-name", help="Modellname an der Schnittstelle (bei --endpoint; sonst automatisch)")
    p.add_argument("--resume", metavar="LAUFORDNER", help="abgebrochenen Lauf fortsetzen")
    p.add_argument("--smoke", action="store_true", help="kurzer Funktionstest mit kleinen Parametern")
    p.add_argument("--allow-busy-gpu", action="store_true", help="auch starten, wenn die GPU schon belegt ist")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("show", help="Kurzauswertung eines Laufs anzeigen")
    p.add_argument("run_dir", help="Laufordner unter runs/")
    p.set_defaults(func=cmd_show)

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
