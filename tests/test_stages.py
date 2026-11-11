"""Tests der Stufen 3 bis 7: Auswertung an erfundenen Datensätzen und Läufe gegen die Attrappe."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mockserver import MockEngine, Served
from styrke.cli import parse_stages
from styrke.config import SMOKE_STAGES, deep_merge, load_matrix
from styrke.engines import kv_pool_tokens
from styrke.runner import RunOptions, run_cell
from styrke.stages import STAGES
from styrke.stages.load import cached_share, gaps_between, longest_silence, looks_cached, throughput_between
from styrke.stages.s5_interference import analyze, phase_stats
from styrke.stages.s7_endurance import _drift
from styrke.stages.scenario import calls_per_session, log_uniform, overload_reasons, session_shapes, users_supported
from styrke.store import read_jsonl

REPO = Path(__file__).resolve().parents[1]


def _run(tmp_path: Path, engine: MockEngine, stages: list[int], overrides: dict, cell: str = "flashnext-freetoken", resume=None):
    """Fährt Stufen gegen die Attrappe; die Stufenparameter sind die des Funktionstests plus ``overrides``."""
    raw = yaml.safe_load((REPO / "matrix.yaml").read_text(encoding="utf-8"))
    raw["host"].update(
        {"runs_dir": str(tmp_path / "runs"), "corpus_dir": str(tmp_path / "kein-korpus"), "models_dir": str(tmp_path)}
    )
    raw["stages"] = deep_merge(SMOKE_STAGES, overrides)
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    lines: list[str] = []

    async def go():
        async with Served(engine.app()) as srv:
            opts = RunOptions(matrix_path=str(matrix), cell_id=cell, endpoint=srv.url, stages=stages, resume=resume)
            return await run_cell(opts, log=lines.append)

    code = asyncio.run(go())
    run_dir = Path(resume) if resume else sorted((tmp_path / "runs").iterdir())[-1]
    load = lambda name: json.loads((run_dir / name).read_text(encoding="utf-8"))  # noqa: E731
    return code, run_dir, load, lines


# -- Bausteine ------------------------------------------------------------------


def test_all_stages_registered_and_stage_ranges_parse():
    assert sorted(STAGES) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert parse_stages("0,2,4-6") == [0, 2, 4, 5, 6] and parse_stages("3-7") == [3, 4, 5, 6, 7]
    with pytest.raises(ValueError):
        parse_stages("drei")


def test_kv_pool_size_comes_from_freetoken_start_options():
    matrix = load_matrix(REPO / "matrix.yaml", local="/nonexistent")
    cell = matrix.cell("flashnext-freetoken")
    engine = matrix.engines[cell.engine]
    assert kv_pool_tokens(engine, cell.variant("basis")) == 1048576
    assert kv_pool_tokens(engine, cell.variant("kv-gross")) == 2097152
    vllm = matrix.cell("gemma4-vllm")
    assert kv_pool_tokens(matrix.engines[vllm.engine], vllm.variants[0]) is None


def test_time_slices_of_token_stream():
    # Zwei Anfragen ab t=100: eine mit einem Stück je Sekunde (2 Tokens je Stück), eine mit Lücke von 4 s.
    steady = {"ts_submit": 100.0, "chunk_ms": [i * 1000 for i in range(10)], "completion_tokens": 20}
    gappy = {"ts_submit": 100.0, "chunk_ms": [0, 500, 4500, 5000], "completion_tokens": 4}
    assert throughput_between([steady], 102.0, 106.0) == 2.0
    assert throughput_between([steady, gappy], 100.0, 101.0) == 4.0  # 1 Stück zu 2 Tokens + 2 Stücke zu 1 Token
    assert throughput_between([steady], 106.0, 106.0) is None
    assert gaps_between([gappy], 100.0, 110.0) == [0.5, 4.0, 0.5]
    assert gaps_between([gappy], 104.0, 104.6) == [4.0]  # die Lücke zählt zu dem Abschnitt, in dem sie endet
    assert longest_silence([gappy], 100.0, 106.0) == 4.0
    assert longest_silence([], 100.0, 103.0) == 3.0  # gar kein Token: der ganze Abschnitt ist Stille


def test_cache_verdicts():
    assert cached_share({"cached_tokens": 900, "prompt_tokens": 1000}) == 0.9
    assert cached_share({"cached_tokens": None, "prompt_tokens": 1000}) is None  # Engine meldet nichts
    assert looks_cached(0.2, 4.0) is True and looks_cached(3.9, 4.0) is False
    assert looks_cached(0.01, 0.05) is None  # kalt schon zu schnell für ein Urteil


def test_interference_analysis_finds_slowdown_and_stall():
    # Hintergrund: ein Token alle 100 ms von t=0 bis 30, aber zwischen t=10 und t=14 kommt nichts.
    times = [i * 100 for i in range(300) if not 10_000 <= i * 100 < 14_000]
    background = [{"ts_submit": 0.0, "chunk_ms": times, "completion_tokens": len(times), "ok": True, "ttft_s": 0.0}]
    big = {"ok": True, "error": None, "ttft_s": 4.0, "total_s": 8.0, "prompt_tokens": 128000, "decode_tps": 50.0}
    row = analyze(
        background,
        {"target": 128000, "rep": 0, "record": big, "t_baseline": 0.0, "t_inject": 10.0, "t_end": 18.0, "t_recovered": 28.0},
    )
    assert row["baseline"]["tps"] == 10.0 and row["prefill"]["tps"] == 0.0 and row["decode"]["tps"] == 10.0
    assert row["slowdown_prefill_pct"] == 100.0 and row["slowdown_decode_pct"] == 0.0
    assert row["prefill"]["silence_max_s"] == 4.0 and row["baseline"]["silence_max_s"] <= 0.11
    assert row["decode"]["gap_max_ms"] == 4100.0  # die lange Lücke endet mit dem ersten Token danach
    assert row["recovery"]["tps"] == 10.0
    assert phase_stats(background, 5.0, 5.0) is None


def test_session_shapes_follow_load_model_and_context():
    params = {
        "sessions": {
            "kurz": {"first_tokens": [200, 1500], "turns": 4, "followup_tokens": [20, 300], "max_tokens": 400, "think_s": 30},
            "dokument": {"first_tokens": [64000, 250000], "turns": 3, "followup_tokens": [20, 200], "max_tokens": 600, "think_s": 60},
            "ohne_anteil": {"first_tokens": [10, 20], "turns": 1, "followup_tokens": [20, 30], "max_tokens": 10, "think_s": 0},
        }
    }
    load_model = {"session_mix": {"kurz": {"share": 0.6}, "dokument": {"share": 0.2}}}

    def ctx(limit: int, max_context: int = 262144):
        return SimpleNamespace(load_model=load_model, model=SimpleNamespace(max_context=max_context), prompt_limit=lambda: limit)

    shapes, dropped = session_shapes(ctx(262144), params)
    assert [(s.kind, round(s.share, 2)) for s in shapes] == [("kurz", 0.75), ("dokument", 0.25)]  # Anteile auf 1 gebracht
    assert shapes[1].first_tokens == (64000, 250000) and not dropped
    assert calls_per_session(shapes) == pytest.approx(0.75 * 4 + 0.25 * 3)

    # Kontext bis 128.000: Dokumente werden gekürzt. Kontext bis 32.000: Dokumentarbeit entfällt und wird gemeldet.
    shapes, _ = session_shapes(ctx(128000), params)
    assert shapes[1].first_tokens == (64000, 128000)
    shapes, dropped = session_shapes(ctx(32000), params)
    assert [s.kind for s in shapes] == ["kurz"] and shapes[0].share == 1.0
    assert dropped[0]["kind"] == "dokument" and "passt nicht" in dropped[0]["reason"]

    rng = random.Random(1)
    values = [log_uniform(rng, 64000, 250000) for _ in range(500)]
    assert min(values) >= 64000 and max(values) <= 250000
    assert sum(v < 126000 for v in values) > 200  # kleine Werte sind häufiger: Hälfte unter dem geometrischen Mittel
    assert log_uniform(rng, 500, 500) == 500


def test_overload_verdict_and_supported_users():
    params = {"max_error_rate": 0.05, "max_unfinished_rate": 0.1, "backlog_factor": 1.5, "backlog_slack": 4}
    fine = {"error_rate": 0.0, "unfinished_rate": 0.02, "inflight_first_quarter": 10.0, "inflight_last_quarter": 14.0}
    assert overload_reasons(fine, params) == []
    growing = {**fine, "inflight_last_quarter": 30.0}
    assert "Rückstau" in overload_reasons(growing, params)[0]
    broken = {**fine, "error_rate": 0.2, "unfinished_rate": 0.5}
    assert len(overload_reasons(broken, params)) == 2
    idle = {"error_rate": 0.0, "unfinished_rate": 0.0, "inflight_first_quarter": None, "inflight_last_quarter": None}
    assert overload_reasons(idle, params) == []

    # 0,42 Aufrufe/s sind beim mittleren Profil (25 % aktiv, 10 Aktionen, 1,5 Aufrufe) gut 400 Berechtigte.
    load_model = {"profiles": {"mittel": {"a": 0.25, "t": 10, "k": 1.5, "s": 30}}}
    assert users_supported(0.42, load_model) == {"mittel": 403}


def test_drift_between_first_and_last_third():
    rows = [{"x": v} for v in (1.0, 1.0, 1.0, 1.2, 1.5, 1.5)]
    assert _drift(rows, "x") == 50.0
    assert _drift(rows[:2], "x") is None  # zu wenige Zeitscheiben
    assert _drift([{"x": None}] * 6, "x") is None


# -- Stufe 3 -------------------------------------------------------------------

S3 = {
    "fills": [0.0, 0.5], "pool_tokens": 16000, "session_tokens": 2000, "probe_sessions": 4, "probe_max_tokens": 8,
    "levels": [2], "prompt_tokens": 300, "max_tokens": 24, "warmup_s": 0.3, "window_s": 1.5, "windows": 1,
}


def test_stage3_held_sessions_survive_when_pool_is_large_enough(tmp_path):
    engine = MockEngine(max_running=4, prefill_tokens_per_s=2000, kv_pool_tokens=500_000, max_context=20_000)
    code, run_dir, load, _ = _run(tmp_path, engine, [3], {"s3": S3})
    assert code == 0
    s3 = load("summary_s3.json")
    empty, half = s3["fills"]
    assert s3["pool_tokens"] == 16000 and empty["held_sessions"] == 0 and empty["probe_n"] == 0
    # Bei kleinem Pool werden die Gespräche verkleinert (hier auf 1.000 Tokens); 50 % von 16.000 werden nicht überschritten.
    assert s3["session_tokens"] == 1000 and half["held_sessions"] == 8 and 7900 <= half["held_tokens"] <= 8160
    assert half["cold_ttft_p50_s"] > 0.3 and half["probe_ttft_p50_s"] < 0.1
    assert half["retained_by_ttft"] == 1.0 and half["retained_by_cache_report"] == 1.0
    assert half["engine_metrics_after_fill"]["v1_cache_status"]["kv_pool"]["used_tokens"] > 8000
    assert [(r["fill"], r["level"]) for r in s3["rows"]] == [(0.0, 2), (0.5, 2)]
    assert all(r["throughput_tps"] > 0 and r["errors"] == 0 for r in s3["rows"])

    requests = [r for r in read_jsonl(run_dir / "requests.jsonl") if r["stage"] == 3]
    assert sum(r.get("what") == "fill" for r in requests) == 8 and sum(r.get("what") == "probe" for r in requests) == 4
    probes = [r for r in requests if r.get("what") == "probe"]
    assert all(r["turn"] == 1 and r["cached_tokens"] > 950 for r in probes)
    assert {r["fill"] for r in requests if "level" in r} == {0.0, 0.5}


def test_stage3_shows_eviction_when_pool_overflows(tmp_path):
    # Der Pool der Attrappe fasst die gehaltenen Gespräche gerade so; die Last danach verdrängt sie.
    engine = MockEngine(max_running=4, prefill_tokens_per_s=2000, kv_pool_tokens=9000, max_context=20_000)
    code, _, load, _ = _run(tmp_path, engine, [3], {"s3": S3})
    half = load("summary_s3.json")["fills"][1]
    assert code == 0 and engine.evictions > 0
    assert half["retained_by_ttft"] == 0.0 and half["retained_by_cache_report"] == 0.0
    assert half["probe_ttft_p50_s"] > 0.3  # die Folgefrage ist wieder so langsam wie der kalte Prompt


def test_stage3_is_skipped_when_pool_size_is_unknown(tmp_path):
    engine = MockEngine()
    code, _, load, lines = _run(tmp_path, engine, [3], {"s3": {**S3, "pool_tokens": None}}, cell="gemma4-vllm")
    assert code == 0 and "pool_tokens" in load("summary_s3.json")["skipped"]
    assert any("ausgelassen" in line for line in lines)


# -- Stufe 4 -------------------------------------------------------------------


def test_stage4_measures_cache_benefit_pause_and_eviction(tmp_path):
    engine = MockEngine(max_running=4, prefill_tokens_per_s=5000, kv_pool_tokens=10_000, max_context=20_000)
    params = {
        "lengths": [1000, 3000, 300000], "reps": 2, "max_tokens": 12, "pause_s": 0.3, "followup_delays_s": [0, 0.2],
        "evict_lengths": [2000], "evict_chunk_tokens": 2000, "evict_factor": 1.1, "pool_tokens": 10_000,
    }
    code, run_dir, load, _ = _run(tmp_path, engine, [4], {"s4": params})
    assert code == 0
    s4 = load("summary_s4.json")
    assert s4["skipped"] == [{"prompt_target": 300000, "reason": "passt nicht in den Kontext"}]
    short, long = s4["rows"]
    assert long["prompt_target"] == 3000 and long["n"] == 2 and long["errors"] == 0
    assert 0.5 < long["cold_ttft_s"] < 0.9 and long["followup_ttft_s"] < 0.1 and long["speedup"] > 5
    assert long["followup_cached_share"] > 0.95 and long["after_pause_cached_share"] > 0.95
    assert long["after_pause_ttft_s"] < 0.1
    assert long["followup_delayed_ttft_s"] < 0.1 and long["followup_delayed_cached_share"] > 0.95  # zweiter Lauf, verzögert
    assert short["cold_ttft_s"] < long["cold_ttft_s"]

    eviction = s4["eviction"]
    assert eviction["flood_tokens"] >= 11_000
    row = eviction["rows"][0]
    assert row["before_ttft_s"] < 0.1 and row["before_cached_share"] > 0.9
    assert row["after_ttft_s"] > 0.3 and row["after_cached_share"] < 0.1 and row["evicted_by_ttft"] is True

    whats = [r.get("what") for r in read_jsonl(run_dir / "requests.jsonl") if r["stage"] == 4]
    assert whats.count("cold") == 5 and whats.count("followup") == 4 and whats.count("after_pause") == 2
    assert whats.count("flood") >= 6 and whats.count("after_eviction") == 1


# -- Stufe 5 -------------------------------------------------------------------


def test_stage5_measures_what_a_big_request_does_to_running_chats(tmp_path):
    engine = MockEngine(
        max_running=8, tokens_per_s=300, prefill_tokens_per_s=4000, prefill_slowdown=3.0, max_context=20_000
    )
    params = {
        "chats": 3, "prompt_tokens": 40, "max_tokens": 60, "big_lengths": [6000, 50_000, 300_000], "big_max_tokens": 20,
        "reps": 1, "warmup_s": 0.5, "baseline_s": 1.5, "recovery_s": 1.0,
    }
    code, run_dir, load, _ = _run(tmp_path, engine, [5], {"s5": params})
    assert code == 0
    s5 = load("summary_s5.json")
    assert s5["skipped"] == [{"big_target": 300_000, "reason": "passt nicht in den Kontext"}]
    ok, failed = s5["rows"]
    # 6000 Tokens bei 4000 Tokens/s: die Engine liest 1,5 s lang ein und bremst die Chats dabei.
    assert ok["big_ok"] and 1.3 < ok["big_ttft_s"] < 2.0
    assert ok["prefill"]["seconds"] == pytest.approx(ok["big_ttft_s"], abs=0.1)
    assert ok["slowdown_prefill_pct"] > 40 and ok["prefill"]["tps"] < ok["baseline"]["tps"]
    assert ok["recovery"]["tps"] > ok["prefill"]["tps"] * 1.5
    assert ok["prefill"]["gap_p50_ms"] > ok["baseline"]["gap_p50_ms"] * 2
    # 50.000 Tokens lehnt die Attrappe ab: Der Fehlschlag steht im Ergebnis, die Chats laufen weiter.
    assert not failed["big_ok"] and "HTTP 400" in failed["big_error"]
    assert s5["background_errors"] == 0 and s5["background_requests"] > 20

    requests = [r for r in read_jsonl(run_dir / "requests.jsonl") if r["stage"] == 5]
    assert sum(r.get("what") == "big" for r in requests) == 2
    kinds = [e["kind"] for e in read_jsonl(run_dir / "events.jsonl")]
    assert kinds.count("injection_start") == 2 and kinds.count("injection_end") == 2


# -- Stufen 6 und 7 --------------------------------------------------------------

SESSIONS = {
    "kurzer_chat": {"first_tokens": [100, 300], "turns": 2, "followup_tokens": [20, 60], "max_tokens": 32, "think_s": 0.2},
    "langer_chat": {"first_tokens": [500, 1500], "turns": 2, "followup_tokens": [20, 100], "max_tokens": 32, "think_s": 0.2},
    "dokumentarbeit": {"first_tokens": [2000, 4000], "turns": 2, "followup_tokens": [20, 60], "max_tokens": 32, "think_s": 0.2},
    "agentenlauf": {"first_tokens": [300, 600], "turns": 3, "followup_tokens": [50, 200], "max_tokens": 24, "think_s": 0},
}


def test_stage6_finds_sustained_rate_and_stage7_runs_at_a_share_of_it(tmp_path):
    # Die Attrappe schafft mit 2 gleichzeitigen Anfragen gut 10 Aufrufe je Sekunde; 60 je Sekunde sind Überlast.
    engine = MockEngine(max_running=2, tokens_per_s=200, prefill_tokens_per_s=50_000, max_context=20_000)
    overrides = {
        "s6": {
            "rates": [4.0, 60.0, 120.0], "duration_s": 6, "warmup_s": 1.5, "drain_s": 2, "pause_s": 0.2,
            "extra_levels_after_overload": 0, "sessions": SESSIONS,
        },
        "s7": {"duration_s": 6, "warmup_s": 1, "bucket_s": 1, "drain_s": 5},
    }
    code, run_dir, load, lines = _run(tmp_path, engine, [6], overrides)
    assert code == 0
    s6 = load("summary_s6.json")
    low, high = s6["rows"]  # die dritte Rate wird nach der Überlast nicht mehr gefahren
    assert low["overload"] == [] and low["errors"] == 0 and low["calls"] >= 1
    assert high["overload"] and high["inflight_last_quarter"] > high["inflight_first_quarter"]
    assert s6["overloaded_at"] == 60.0 and s6["max_sustained_rate"] == 4.0
    assert set(s6["users_supported"]) == {"zurueckhaltend", "mittel", "intensiv"}
    assert s6["users_supported"]["zurueckhaltend"] > s6["users_supported"]["intensiv"] > 0
    assert s6["calls_per_session"] == pytest.approx(0.6 * 2 + 0.25 * 2 + 0.1 * 2 + 0.05 * 3)
    assert set(low["by_kind"]) == set(SESSIONS) and low["by_turn"]["first"]["n"] >= 1
    assert any("getragen" in line for line in lines) and any("Überlast" in line for line in lines)

    requests = [r for r in read_jsonl(run_dir / "requests.jsonl") if r["stage"] == 6 and r.get("what") != "warmup"]
    assert {r["rate"] for r in requests} == {4.0, 60.0} and {r["kind"] for r in requests} <= set(SESSIONS)
    followups = [r for r in requests if r["turn"] > 0 and r["ok"] and r["rate"] == 4.0]
    assert followups and all(r["cached_tokens"] > 0 for r in followups)  # Folgefragen treffen den Cache
    assert any(r["error"] == "abgebrochen" for r in requests if r["rate"] == 60.0)
    assert all("_text" not in r for r in requests)  # Antworten bleiben nicht in den Rohdaten
    load_samples = [s for s in read_jsonl(run_dir / "samples.jsonl") if s["kind"] == "load"]
    assert {s["rate"] for s in load_samples} == {4.0, 60.0}

    # Stufe 7 im selben Laufordner: 70 % der getragenen Rate. Wie im echten Lauf beginnt die Stufe mit einer
    # frisch gestarteten Engine; die alte arbeitet noch den Rückstau der Überlast ab.
    engine = MockEngine(max_running=2, tokens_per_s=200, prefill_tokens_per_s=50_000, max_context=20_000)
    code, _, load, _ = _run(tmp_path, engine, [6, 7], overrides, resume=str(run_dir))
    assert code == 0
    s7 = load("summary_s7.json")
    assert s7["rate"] == pytest.approx(0.7 * s6["max_sustained_achieved_rate"], abs=0.001)
    assert "Stufe 6" in s7["rate_source"] and len(s7["buckets"]) == 5
    assert s7["overall"]["errors"] == 0, s7["overall"]["error_examples"]
    assert s7["overall"]["calls"] == sum(b["calls"] for b in s7["buckets"])
    assert "host_mem_available_first_gib" in s7["resources"]
    assert load("state.json")["stages_done"] == [6, 7]


def test_stage7_needs_a_rate(tmp_path):
    engine = MockEngine(max_context=20_000)
    code, _, load, _ = _run(tmp_path, engine, [7], {"s6": {"sessions": SESSIONS}})
    assert code == 0 and "Stufe 6" in load("summary_s7.json")["skipped"]

    # Mit fester Rate läuft sie auch allein.
    over = {"s6": {"sessions": SESSIONS}, "s7": {"rate": 3.0, "duration_s": 4, "warmup_s": 0.5, "bucket_s": 1, "drain_s": 5}}
    code, _, load, _ = _run(tmp_path, engine, [7], over)
    s7 = load("summary_s7.json")
    assert code == 0 and s7["rate"] == 3.0 and s7["rate_source"] == "stages.s7.rate" and s7["overall"]["calls"] > 0


# -- Stufe 8 -------------------------------------------------------------------


def test_quality_checks():
    from styrke.quality import check, extract_json, foreign_chars, load_questions

    questions = {q["id"]: q for q in load_questions(REPO / "quality" / "fragen.yaml")["fragen"]}
    assert len(questions) == 26 and {q["kategorie"] for q in questions.values()} == {
        "firmenwissen", "ehrlichkeit", "deutsch", "denken", "langer_kontext", "werkzeuge", "struktur",
    }
    assert all(q.get("referenz") for q in questions.values())

    good = check(questions["firma-01"], "Ein Chemiewerk in Wittenberg, das Ammoniak und Düngemittel herstellt.", [])
    assert good["ok"] is True
    bad = check(questions["firma-01"], "Ein Stahlwerk im Ruhrgebiet.", [])
    assert bad["ok"] is False and len(bad["fehlt"]) == 3
    assert check(questions["ehrlich-02"], "Das Produkt kenne ich nicht.", [])["ok"] is None  # nur von Hand
    assert check(questions["denken-02"], "Nein. Bernd arbeitet am Mittwoch.", [])["ok"] is True
    assert check(questions["denken-02"], "Ja, das geht.", [])["ok"] is False
    assert check(questions["denken-05"], "Die Reihenfolge ist E → C → A → B → D.", [])["ok"] is True
    assert check(questions["denken-04"], "Das sind 1.150 kg Stickstoff.", [])["ok"] is True

    answer = '```json\n{"lieferung": 4711, "datum": "03.11.2026", "menge_t": 24, "produkt": "Harnstoff", "kunde": "Agrarhandel Nord", "preis_eur_je_t": 412}\n```'
    assert extract_json(answer)["menge_t"] == 24 and check(questions["struktur-01"], answer, [])["ok"] is True
    assert check(questions["struktur-01"], answer.replace("412", '"412 Euro"'), [])["ok"] is False

    call = {"function": {"name": "termin_anlegen", "arguments": '{"titel": "Sicherheitsunterweisung", "datum": "2026-12-14", "uhrzeit": "09:30"}'}}
    assert check(questions["werkzeug-02"], "", [call])["ok"] is True
    wrong_day = {"function": {"name": "termin_anlegen", "arguments": '{"titel": "Sicherheitsunterweisung", "datum": "2026-12-15", "uhrzeit": "09:30"}'}}
    assert check(questions["werkzeug-02"], "", [wrong_day])["ok"] is False
    assert check(questions["werkzeug-04"], "Paris.", [])["ok"] is True
    assert check(questions["werkzeug-04"], "Paris.", [call])["ok"] is False  # Werkzeug ohne Anlass
    assert foreign_chars("Das ist обычный Text mit 中文.") == 9 and foreign_chars("Größe, Maß, café") == 0


def test_stage8_runs_questions_and_blind_rating_roundtrip(tmp_path):
    from styrke.quality import export_blind, import_blind

    engine = MockEngine(max_context=20_000)
    engine.canned = {"Freigabenummer": "Die Nummer lautet 7Q4-KX-2291.", "Hauptstadt": "Paris.", "Wem gehört": "Der Agrofert-Gruppe."}
    # Ohne Fragensatz neben der Testmatrix wird die Stufe ausgelassen, mit klarem Grund.
    code, _, load, _ = _run(tmp_path, engine, [8], {})
    assert code == 0 and "Fragensatz nicht gefunden" in load("summary_s8.json")["skipped"]

    over = {"s8": {"questions": str(REPO / "quality" / "fragen.yaml"), "thinking": ["off"],
                   "max_tokens": {"off": 60, "on": 60}, "context_cap": 3000}}
    code, run_dir, load, lines = _run(tmp_path, engine, [8], over)
    assert code == 0
    answers = {r["id"]: r for r in read_jsonl(run_dir / "answers" / "s8.jsonl")}
    assert len(answers) == 26
    assert answers["firma-02"]["check"]["ok"] is True and answers["firma-01"]["check"]["ok"] is False
    assert answers["kontext-01"]["check"]["ok"] is True and answers["kontext-01"]["prompt_tokens"] > 2500  # Nadel im langen Text
    assert answers["werkzeug-01"]["check"]["ok"] is True and answers["werkzeug-01"]["tool_calls"]
    assert answers["werkzeug-04"]["check"]["ok"] is True and answers["ehrlich-01"]["check"]["ok"] is None
    s8 = load("summary_s8.json")
    assert s8["totals"]["off"]["auto_ok"] == 4 and s8["totals"]["off"]["auto_checked"] == 20
    assert any("von Hand zu bewerten" in line for line in lines)

    # Blindbewertung: Bogen ohne Herkunft, Schlüssel getrennt, Punkte zurück je Lauf und Kategorie.
    out = tmp_path / "bewertung"
    info = export_blind([run_dir], out, seed=1)
    sheet = (out / "bogen.csv").read_text(encoding="utf-8-sig")
    assert info == {"antworten": 19, "laeufe": 1} and run_dir.name not in sheet and "flashnext" not in sheet
    assert "werkzeug-01" not in (out / "schluessel.json").read_text(encoding="utf-8")  # prüft styrke selbst
    filled = "\n".join(
        line if i == 0 else line.rsplit(";", 2)[0] + ";2;" for i, line in enumerate(sheet.splitlines()) if line.strip()
    )
    (out / "bogen.csv").write_text(filled, encoding="utf-8-sig")
    result = import_blind(out / "bogen.csv", out / "schluessel.json")
    assert result["laeufe"][run_dir.name]["mittel"] == 2.0 and result["laeufe"][run_dir.name]["n"] == 19
    assert set(result["laeufe"][run_dir.name]["kategorien"]) == {"firmenwissen", "ehrlichkeit", "deutsch", "denken"}


def test_saturation_is_withdrawn_when_throughput_rises_again():
    from styrke.stages.s2_concurrency import saturation

    # Messwerte des ersten Laufs an der echten Engine: Bei 20 parallelen Anfragen stieg der Durchsatz wieder.
    measured = [(1, 57.0), (2, 73.3), (4, 123.2), (8, 195.9), (10, 194.5), (15, 195.8), (20, 248.1)]
    assert saturation(measured[:5], 0.05, 2) == (10, False)
    assert saturation(measured[:6], 0.05, 2) == (10, False)
    assert saturation(measured, 0.05, 2) == (None, False)  # früher endete die Stufe hier
    assert saturation(measured + [(25, 250.0)], 0.05, 2) == (25, False)
    assert saturation(measured + [(25, 250.0), (30, 251.0), (35, 249.0)], 0.05, 2) == (25, True)
    assert saturation([(1, 100.0), (2, 101.0), (4, 102.0), (8, 103.0)], 0.05, 2) == (2, True)


def test_stage5_falls_back_to_largest_feasible_prompt(tmp_path):
    """Passt keine der großen Anfragen in den Kontext, misst die Stufe mit der größten lauffähigen."""
    engine = MockEngine(max_running=8, prefill_tokens_per_s=20_000, max_context=20_000)
    code, run_dir, _, _ = _run(tmp_path, engine, [0], {"s0": {"context_probes": [9000, 50_000]}})
    params = {"chats": 2, "prompt_tokens": 40, "max_tokens": 30, "big_lengths": [128_000], "big_max_tokens": 10, "reps": 1,
              "warmup_s": 0.3, "baseline_s": 0.5, "recovery_s": 0.3, "fallback_min_tokens": 8000}
    code, _, load, lines = _run(tmp_path, engine, [5], {"s5": params}, resume=str(run_dir))
    s5 = load("summary_s5.json")
    assert code == 0 and [r["big_target"] for r in s5["rows"]] == [9000] and s5["rows"][0]["big_ok"]
    assert s5["skipped"][0]["big_target"] == 128_000 and any("ersatzweise 9000" in line for line in lines)
