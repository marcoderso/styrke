"""Tests gegen die Attrappe: Client, Stufen 0 bis 2, Ablaufsteuerung, Mitschreiber. Stufen 3 bis 7: test_stages.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiohttp
import yaml

from mockserver import MockEngine, Served
from styrke.client import RequestSpec, run_request
from styrke.recorder import make_app, report
from styrke.runner import RunOptions, run_cell
from styrke.stages.s2_concurrency import window_throughput
from styrke.store import read_jsonl

REPO = Path(__file__).resolve().parents[1]


def _matrix_for_test(tmp_path: Path) -> Path:
    raw = yaml.safe_load((REPO / "matrix.yaml").read_text(encoding="utf-8"))
    raw["host"].update(
        {"runs_dir": str(tmp_path / "runs"), "corpus_dir": str(tmp_path / "kein-korpus"), "models_dir": str(tmp_path)}
    )
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


# -- Client -------------------------------------------------------------------


def test_client_measures_stream_with_reasoning_and_usage():
    async def go():
        engine = MockEngine()
        async with Served(engine.app()) as srv, aiohttp.ClientSession() as session:
            spec = RequestSpec([{"role": "user", "content": "Hallo " * 50}], max_tokens=20, tags={"stage": 9})
            return await run_request(session, srv.url, "attrappe", spec)

    rec = asyncio.run(go())
    assert rec["ok"] and rec["http_status"] == 200 and rec["stage"] == 9
    assert rec["n_chunks"] == 20 and rec["completion_tokens"] == 20 and rec["usage_source"] == "stream"
    assert rec["chunk_kinds"] == "r" * 8 + "c" * 12  # erst Nachdenken, dann Antwort
    assert rec["ttfr_s"] <= rec["ttfc_s"] <= rec["total_s"] and rec["ttft_s"] == rec["ttfr_s"]
    assert rec["decode_tps"] > 0 and rec["finish_reason"] == "length"
    assert "_text" not in rec  # Antworttext wird nicht aufbewahrt


def test_client_thinking_off_has_no_reasoning_and_keeps_text_on_request():
    async def go():
        async with Served(MockEngine().app()) as srv, aiohttp.ClientSession() as session:
            spec = RequestSpec(
                [{"role": "user", "content": "Hallo"}], max_tokens=6,
                body={"chat_template_kwargs": {"enable_thinking": False}}, keep_text=True,
            )
            return await run_request(session, srv.url, "attrappe", spec)

    rec = asyncio.run(go())
    assert rec["chunk_kinds"] == "cccccc" and rec["ttfr_s"] is None and rec["chars_reasoning"] == 0
    assert rec["_text"].startswith("Das ist ein Testtext")


def test_client_reports_http_error_and_missing_usage():
    async def go():
        engine = MockEngine(max_context=100, send_usage=False)
        async with Served(engine.app()) as srv, aiohttp.ClientSession() as session:
            too_long = RequestSpec([{"role": "user", "content": "x" * 4000}], max_tokens=10)
            short = RequestSpec([{"role": "user", "content": "kurz"}], max_tokens=10)
            return (
                await run_request(session, srv.url, "attrappe", too_long),
                await run_request(session, srv.url, "attrappe", short),
                await run_request(session, srv.url, "attrappe", short, count_tokens=lambda text: len(text.split())),
            )

    bad, chunks, counted = asyncio.run(go())
    assert not bad["ok"] and bad["http_status"] == 400 and "prompt too long" in bad["error"]
    assert chunks["ok"] and chunks["usage_source"] == "chunks" and chunks["completion_tokens"] == 10
    assert counted["usage_source"] == "tokenizer" and counted["completion_tokens"] == 10


def test_client_connection_error_is_recorded_not_raised():
    async def go():
        async with aiohttp.ClientSession() as session:
            spec = RequestSpec([{"role": "user", "content": "Hallo"}], max_tokens=4)
            return await run_request(session, "http://127.0.0.1:9", "attrappe", spec)

    rec = asyncio.run(go())
    assert not rec["ok"] and rec["error"].startswith("Verbindungsfehler")


# -- Durchsatz je Fenster -------------------------------------------------------


def test_window_throughput_counts_tokens_by_timestamp():
    # Eine Anfrage, 10 Stücke im Abstand von 1 s ab t=100, 20 Tokens: 2 Tokens je Stück.
    rec = {"ts_submit": 100.0, "chunk_ms": [i * 1000 for i in range(10)], "completion_tokens": 20}
    per_window = window_throughput([rec], w_start=102.0, window_s=4.0, windows=2)
    # Fenster 1: Stücke bei 102..105 (4 Stück), Fenster 2: 106..109 (4 Stück) -> je 8 Tokens in 4 s
    assert per_window == [2.0, 2.0]


# -- Stufen 0 bis 2 über die Ablaufsteuerung --------------------------------------


def _run_smoke(tmp_path: Path, engine: MockEngine, **kwargs) -> tuple[int, Path]:
    matrix = _matrix_for_test(tmp_path)

    async def go():
        async with Served(engine.app()) as srv:
            opts = RunOptions(
                matrix_path=str(matrix), cell_id="flashnext-freetoken", endpoint=srv.url, smoke=True, **kwargs
            )
            return await run_cell(opts, log=lambda _msg: None)

    code = asyncio.run(go())
    run_dir = next((tmp_path / "runs").iterdir()) if not kwargs.get("resume") else Path(kwargs["resume"])
    return code, run_dir


def test_stages_0_to_2_against_mock(tmp_path):
    engine = MockEngine(max_running=4, tokens_per_s=400)
    code, run_dir = _run_smoke(tmp_path, engine, stages=[0, 1, 2])
    assert code == 0
    load = lambda name: json.loads((run_dir / name).read_text(encoding="utf-8"))  # noqa: E731

    manifest = load("manifest.json")
    assert manifest["smoke"] and manifest["cell"] == "flashnext-freetoken" and manifest["corpus"]["builtin"]
    assert manifest["engine"]["kind"] == "external"

    feas = load("feasibility.json")
    assert feas["loaded"] and feas["sanity_ok"] and feas["usage_in_stream"]
    assert feas["thinking"] == {"off_has_reasoning": False, "on_has_reasoning": True, "on_ok": True}
    assert feas["calibration"]["ok"] and abs(feas["calibration"]["check_deviation_pct"]) < 2.0
    assert feas["context_probes"][0]["ok"] and feas["max_context_ok"] > 1900
    assert "Hauptst" in load("answers/s0_sanity.json")["question"]

    s1 = load("summary_s1.json")
    assert [(r["thinking"], r["prompt_target"]) for r in s1["rows"]] == [
        ("off", 100), ("off", 1000), ("on", 100), ("on", 1000),
    ]
    for row in s1["rows"]:
        assert row["n_ok"] == row["n"] == 1 and row["decode_tps"] > 0
        assert abs(row["prompt_tokens"] - row["prompt_target"]) <= max(5, row["prompt_target"] * 0.02)
    on_row = next(r for r in s1["rows"] if r["thinking"] == "on")
    assert on_row["ttfc_s"] > on_row["ttft_s"]  # die Antwort beginnt nach dem Nachdenken

    s2 = load("summary_s2.json")
    levels = {r["level"]: r for r in s2["rows"]}
    assert levels[2]["throughput_tps"] > levels[1]["throughput_tps"] * 1.5  # skaliert unter der Grenze
    assert levels[8]["throughput_tps"] < levels[4]["throughput_tps"] * 1.2  # die Attrappe lässt nur 4 gleichzeitig zu
    assert s2["saturated_at"] == 8 and engine.peak_active == 4
    assert levels[8]["ttft_p50_s"] > levels[4]["ttft_p50_s"]  # Warteschlange zeigt sich in der Zeit bis zum ersten Token
    assert all(r["errors"] == 0 for r in s2["rows"])

    requests = read_jsonl(run_dir / "requests.jsonl")
    assert {r["stage"] for r in requests} == {0, 1, 2}
    assert all("_text" not in r for r in requests)
    assert any(s["kind"] == "host" for s in read_jsonl(run_dir / "samples.jsonl"))
    assert any(s["kind"] == "engine" and "v1_stats" in s for s in read_jsonl(run_dir / "samples.jsonl"))
    assert load("state.json")["stages_done"] == [0, 1, 2]
    kinds = [e["kind"] for e in read_jsonl(run_dir / "events.jsonl")]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and kinds.count("stage_end") == 3

    # Fortsetzen: abgeschlossene Stufen werden nicht noch einmal gefahren.
    before = engine.requests
    code, _ = _run_smoke(tmp_path, engine, resume=str(run_dir), stages=[0, 1, 2])
    assert code == 0 and engine.requests == before


def test_engine_without_stream_options_and_usage_still_measures(tmp_path):
    engine = MockEngine(accept_stream_options=False, send_usage=False)
    code, run_dir = _run_smoke(tmp_path, engine, stages=[0, 1])
    assert code == 0
    feas = json.loads((run_dir / "feasibility.json").read_text(encoding="utf-8"))
    assert feas["loaded"] and not feas["usage_in_stream"]
    s1 = json.loads((run_dir / "summary_s1.json").read_text(encoding="utf-8"))
    assert all(r["n_ok"] == 1 and r["decode_tps"] > 0 for r in s1["rows"])
    assert all("stream_options" not in b for b in engine.bodies[-4:])


def test_unreachable_engine_is_recorded_as_result(tmp_path):
    matrix = _matrix_for_test(tmp_path)
    raw = yaml.safe_load(matrix.read_text(encoding="utf-8"))
    raw["engines"]["freetoken"]["ready_timeout_s"] = 2
    matrix.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    opts = RunOptions(
        matrix_path=str(matrix), cell_id="flashnext-freetoken", endpoint="http://127.0.0.1:9", smoke=True, stages=[0]
    )
    code = asyncio.run(run_cell(opts, log=lambda _msg: None))
    run_dir = next((tmp_path / "runs").iterdir())
    feas = json.loads((run_dir / "feasibility.json").read_text(encoding="utf-8"))
    assert code == 2 and feas["loaded"] is False and "nicht bereit" in feas["error"]


# -- Mitschreiber ---------------------------------------------------------------

SECRET = "GEHEIMWORT-im-Prompt"


def _proxy_roundtrip(tmp_path: Path, inject_usage: bool) -> tuple[list[dict], str, str, MockEngine]:
    out = tmp_path / "usage"

    async def go():
        engine = MockEngine()
        async with Served(engine.app()) as upstream, Served(make_app(upstream.url, out, inject_usage=inject_usage)) as proxy:
            async with aiohttp.ClientSession() as session:
                body = {
                    "model": "attrappe", "max_tokens": 12, "temperature": 0.3, "user": "erika",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "tools": [{"type": "function", "function": {"name": "suche"}}],
                    "messages": [
                        {"role": "system", "content": "Du bist hilfreich."},
                        {"role": "user", "content": f"Bitte merke dir {SECRET}."},
                    ],
                }
                async with session.post(proxy.url + "/v1/chat/completions", json={**body, "stream": True}) as resp:
                    assert resp.status == 200 and "text/event-stream" in resp.headers["Content-Type"]
                    streamed = (await resp.read()).decode()
                async with session.post(proxy.url + "/v1/chat/completions", json=body) as resp:
                    plain = await resp.json()
                async with session.get(proxy.url + "/v1/models") as resp:
                    assert (await resp.json())["data"][0]["id"] == "attrappe"
                async with session.post(proxy.url + "/v1/chat/completions", json={**body, "max_tokens": 99999}) as resp:
                    assert resp.status == 400  # Fehler der Engine kommen unverändert durch
        return streamed, plain["choices"][0]["message"]["content"], engine

    streamed, plain, engine = asyncio.run(go())
    rows = [r for path in sorted(out.glob("usage-*.jsonl")) for r in read_jsonl(path)]
    raw_log = "".join(p.read_text(encoding="utf-8") for p in out.glob("usage-*.jsonl"))
    return rows, raw_log, streamed + plain, engine


def test_recorder_passes_through_and_logs_lengths_only(tmp_path):
    rows, raw_log, answers, engine = _proxy_roundtrip(tmp_path, inject_usage=False)
    assert "Testtext" in answers and answers.count("data: ") >= 12  # Antworten kommen vollständig an
    assert len(rows) == 3  # zwei Erfolge, ein Fehler; GET /v1/models wird nicht mitgeschrieben
    stream, plain, failed = rows

    assert SECRET not in raw_log and "Testtext" not in raw_log and "erika" not in raw_log
    assert "hilfreich" not in raw_log  # auch der System-Prompt bleibt draußen

    assert stream["stream"] is True and stream["status"] == 200 and stream["n_messages"] == 2
    assert stream["prompt_chars"] == len("Du bist hilfreich.") + len(f"Bitte merke dir {SECRET}.")
    assert stream["system_chars"] == len("Du bist hilfreich.") and stream["n_tools"] == 1
    assert stream["temperature"] == 0.3 and stream["enable_thinking"] is False and stream["max_tokens"] == 12
    assert stream["out_chars"] > 0 and stream["ttft_s"] is not None and stream["duration_s"] >= stream["ttft_s"]
    assert stream["finish_reason"] == "length"
    assert "completion_tokens" not in stream  # ohne Zählwerte der Engine und ohne Tokenizer
    assert "stream_options" not in engine.bodies[0]  # Anfrage unverändert weitergereicht

    assert plain["stream"] is False and plain["completion_tokens"] == 12 and plain["tokens_source"] == "usage"
    assert plain["prompt_tokens"] > 0 and plain["out_chars"] > 0
    assert failed["status"] == 400
    assert stream["client"] == plain["client"] and len(stream["client"]) == 12
    assert stream["user"] and stream["user"] != "erika"


def test_recorder_can_request_usage_and_report_summarizes(tmp_path):
    rows, _, _, engine = _proxy_roundtrip(tmp_path, inject_usage=True)
    assert engine.bodies[0]["stream_options"] == {"include_usage": True}
    assert rows[0]["usage_injected"] and rows[0]["completion_tokens"] == 12 and rows[0]["prompt_tokens"] > 0

    result = report(tmp_path / "usage", entitled=400)
    assert result["requests"] == 2 and result["users"] == 1 and result["share_stream"] == 0.5
    assert result["share_with_tools"] == 1.0 and result["completion_tokens"]["p50"] == 12
    assert result["peak_hour"]["requests"] == 2 and result["peak_hour"]["active"] == 1
    assert result["peak_hour"]["share_active"] == 0.003  # 1 von 400, gerundet
    assert report(tmp_path / "leer")["requests"] == 0


def test_level_cancels_stragglers_and_keeps_their_partial_records(tmp_path):
    """Am Ende einer Laststufe werden hängende Anfragen beendet, ihre Daten aber behalten."""
    from styrke.config import DEFAULT_STAGES, ModelConfig, deep_merge
    from styrke.corpus import Corpus
    from styrke.engines import ExternalEngine
    from styrke.prompts import PromptBuilder, RatioCounter
    from styrke.stages.common import StageContext
    from styrke.stages.s2_concurrency import run_level
    from styrke.store import RunStore

    params = deep_merge(
        DEFAULT_STAGES,
        {"s2": {"prompt_tokens": 100, "max_tokens": 40, "warmup_s": 0.2, "window_s": 0.5, "windows": 2, "drain_s": 0.1}},
    )

    async def go():
        engine = MockEngine(max_running=1, tokens_per_s=40)  # eine Anfrage dauert rund eine Sekunde
        store = RunStore(tmp_path / "run")
        async with Served(engine.app()) as srv, aiohttp.ClientSession() as session:
            ctx = StageContext(
                store=store, adapter=ExternalEngine(srv.url, "attrappe"), session=session,
                model=ModelConfig(name="m", repo="x/y"), params=params,
                builder=PromptBuilder(Corpus.load(None), RatioCounter()), log=lambda _msg: None,
            )
            row = await run_level(ctx, 3, params["s2"])
        store.close()
        return row

    row = asyncio.run(go())
    assert row["cancelled"] >= 2 and row["errors"] == 0 and row["error_rate"] == 0.0
    assert row["throughput_tps"] > 0  # auch die Stücke abgebrochener Anfragen zählen zum Durchsatz
    written = read_jsonl(tmp_path / "run" / "requests.jsonl")
    assert len(written) == row["requests"]
    assert sum(1 for r in written if r["error"] == "abgebrochen") == row["cancelled"]


def test_failing_stage_does_not_stop_the_remaining_stages(tmp_path, monkeypatch):
    import styrke.runner as runner

    async def broken(ctx):
        raise KeyError("absichtlich")

    monkeypatch.setitem(runner.STAGES, 1, ("Kaputt", broken))
    lines: list[str] = []
    matrix = _matrix_for_test(tmp_path)

    async def go():
        async with Served(MockEngine().app()) as srv:
            opts = RunOptions(
                matrix_path=str(matrix), cell_id="flashnext-freetoken", endpoint=srv.url, smoke=True, stages=[0, 1, 2]
            )
            return await run_cell(opts, log=lines.append)

    code = asyncio.run(go())
    run_dir = next((tmp_path / "runs").iterdir())
    assert code == 1
    assert json.loads((run_dir / "state.json").read_text(encoding="utf-8"))["stages_done"] == [0, 2]
    assert "KeyError" in (run_dir / "stage_errors.log").read_text(encoding="utf-8")
    assert any("Nicht abgeschlossen: Stufe 1" in line for line in lines)
    ends = [e for e in read_jsonl(run_dir / "events.jsonl") if e["kind"] == "stage_end"]
    assert [(e["stage"], e["ok"]) for e in ends] == [(0, True), (1, False), (2, True)]


def test_watch_shows_progress_of_a_run(tmp_path):
    from rich.console import Console

    from styrke.watch import JsonlTail, RunState, newest_run, render, sparkline

    code, run_dir = _run_smoke(tmp_path, MockEngine(max_running=4), stages=[0, 1, 2])
    assert code == 0 and newest_run(tmp_path / "runs") == run_dir
    state = RunState(run_dir)
    state.refresh()
    assert state.finished == 0 and state.planned == [0, 1, 2]
    assert [state.stages[n]["state"] for n in (0, 1, 2)] == ["fertig"] * 3
    assert state.counts[1] == {"n": 4, "errors": 0} and state.counts[2]["n"] > 20
    assert state.window(60)["tps"] > 0 and max(state.history()) > 0
    console = Console(record=True, width=120)
    console.print(render(state))
    text = console.export_text()
    for expected in ("flashnext-freetoken / basis", "2 Parallelität", "fertig", "beendet, Rückgabewert 0", "Stufe 2 (Parallelität)"):
        assert expected in text

    # Ein laufender Lauf: Zeilen kommen stückweise an, eine halbe Zeile zählt erst, wenn sie vollständig ist.
    path = tmp_path / "wachsend.jsonl"
    path.write_text('{"a": 1}\n{"a": ', encoding="utf-8")
    tail = JsonlTail(path)
    assert tail.read_new() == [{"a": 1}]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('2}\n')
    assert tail.read_new() == [{"a": 2}] and tail.read_new() == []
    assert sparkline([0, 5, 10]) == "▁▅█" and sparkline([0, 0]) == "▁▁"
