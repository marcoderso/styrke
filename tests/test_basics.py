"""Tests ohne Netzwerk: Lastmodell, Statistik, Testmatrix, Korpus, Prompts, FreeToken-Kommando."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

import pytest
import yaml

from styrke import loadmodel
from styrke.config import ConfigError, load_matrix
from styrke.corpus import BUILTIN_PARAGRAPHS, Corpus, fetch_wikipedia
from styrke.engines.base import EngineError
from styrke.engines.docker import render_args
from styrke.engines.freetoken import FreeTokenEngine, resolve_flags
from styrke.prompts import HFCounter, PromptBuilder, RatioCounter, make_counter
from styrke.stats import describe, percentile

REPO = Path(__file__).resolve().parents[1]


# -- Lastmodell ---------------------------------------------------------------


def test_lastmodell_reproduces_the_plan():
    matrix = load_matrix(REPO / "matrix.yaml", local="/nonexistent")
    rows = {r["profile"]: r for r in loadmodel.table(matrix.load_model)}
    assert rows["zurueckhaltend"]["peak"] == 5
    assert rows["mittel"]["peak"] == 21
    assert rows["intensiv"]["peak"] == 71
    assert rows["mittel"]["mean"] == 12.5
    assert rows["mittel"]["lambda"] == pytest.approx(0.417, abs=0.001)


def test_supported_users_is_the_inverse_of_arrival_rate():
    lam = loadmodel.arrival_rate(400, 0.25, 10, 1.5)
    assert loadmodel.supported_users(lam, 0.25, 10, 1.5) == pytest.approx(400)


# -- Statistik ----------------------------------------------------------------


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 50) is None
    assert percentile([None, 2, None, 4], 50) == 3


def test_describe_ignores_missing_values():
    d = describe([1.0, None, 3.0])
    assert d["n"] == 2 and d["min"] == 1.0 and d["max"] == 3.0 and d["mean"] == 2.0


# -- Testmatrix ---------------------------------------------------------------


def test_matrix_has_seven_cells():
    matrix = load_matrix(REPO / "matrix.yaml", local="/nonexistent")
    assert len(matrix.cells) == 7
    cell = matrix.cell("flashnext-freetoken")
    assert [v.id for v in cell.variants] == ["basis", "cache-gross", "kv-gross", "parallel-32"]
    assert matrix.models["qwen38-flash-next"].dir == "Qwen3.8-Flash-Next-FP8"
    for model in matrix.models.values():
        assert "FP8" in model.repo  # alle Modelle im Gewichtsformat FP8
    # Jede Zelle außer Flash-Next läuft auf vLLM oder SGLang.
    assert {c.engine for c in matrix.cells if c.id != "flashnext-freetoken"} == {"vllm", "sglang"}


def test_local_file_overrides_host(tmp_path):
    local = tmp_path / "local.yaml"
    local.write_text("host:\n  models_dir: /data/modelle\n", encoding="utf-8")
    matrix = load_matrix(REPO / "matrix.yaml", local=local)
    assert matrix.host.models_dir == "/data/modelle"
    assert matrix.host.port == 18000  # nicht überschriebene Werte bleiben


def test_smoke_parameters_are_smaller():
    matrix = load_matrix(REPO / "matrix.yaml", local="/nonexistent")
    full, smoke = matrix.stage_params(), matrix.stage_params(smoke=True)
    assert len(smoke["s2"]["levels"]) < len(full["s2"]["levels"])
    assert smoke["s2"]["saturation_gain"] == full["s2"]["saturation_gain"]


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"cells": [{"id": "x", "model": "gibtsnicht", "engine": "freetoken"}]}, "unbekanntes Modell"),
        ({"cells": [{"id": "x", "model": "qwen38-27b", "engine": "gibtsnicht"}]}, "unbekannte Engine"),
        ({"stages": {"s9": {}}}, "Unbekannte Stufe"),
        ({"host": {"tippfehler": 1}}, "Unbekannte Angabe"),
    ],
)
def test_matrix_errors_are_reported(tmp_path, patch, message):
    raw = yaml.safe_load((REPO / "matrix.yaml").read_text(encoding="utf-8"))
    raw.update(patch)
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_matrix(path, local="/nonexistent")


# -- Korpus -------------------------------------------------------------------


def test_corpus_falls_back_to_builtin(tmp_path):
    corpus = Corpus.load(tmp_path / "fehlt")
    assert corpus.builtin and len(corpus.paragraphs) == len(BUILTIN_PARAGRAPHS)
    text = corpus.text(random.Random(1), 50_000)  # länger als der Korpus: läuft im Kreis
    assert len(text) >= 50_000


def test_corpus_reads_text_files_and_skips_headings(tmp_path):
    (tmp_path / "a.txt").write_text("# Titel\n\n" + "Wort " * 100 + "\n\nkurz\n", encoding="utf-8")
    corpus = Corpus.load(tmp_path)
    assert not corpus.builtin and len(corpus.paragraphs) == 1


def _wiki_page(n_articles: int = 20) -> dict:
    rows = [{"row": {"title": f"Artikel {i}", "text": "Satz mit Inhalt. " * 400}} for i in range(n_articles)]
    rows.append({"row": {"title": "Stummel", "text": "zu kurz"}})
    return {"rows": rows}


def _is_first(url: str) -> bool:
    return "length=1" in url.split("&") and "offset=0" in url.split("&")


def test_fetch_wikipedia_writes_files_until_target(tmp_path):
    calls = []

    def fake(url: str) -> dict:
        calls.append(url)
        return {"num_rows_total": 2_800_000, "rows": []} if _is_first(url) else _wiki_page()

    result = fetch_wikipedia(tmp_path, target_mb=0.3, fetch_json=fake, log=lambda _: None, pause_s=0)
    assert result["chars"] >= 300_000 and result["files"] >= 2 and result["new_files"] == result["files"]
    assert "dataset=wikimedia%2Fwikipedia" in calls[0] and "config=20231101.de" in calls[0]
    corpus = Corpus.load(tmp_path)
    assert not corpus.builtin
    assert all("zu kurz" not in p for p in corpus.paragraphs)


def test_fetch_wikipedia_resumes_and_survives_outages(tmp_path):
    import urllib.error

    state = {"down": 0, "page_calls": 0}
    waits: list[float] = []

    def fake(url: str) -> dict:
        if state["down"] > 0:
            state["down"] -= 1
            raise urllib.error.HTTPError(url, 502, "Bad Gateway", None, None)
        if _is_first(url):
            return {"num_rows_total": 2_800_000, "rows": []}
        state["page_calls"] += 1
        return _wiki_page()

    quiet = dict(fetch_json=fake, log=lambda _: None, pause_s=0, sleep=waits.append)

    # Erster Abruf bis 0,3 MB.
    first = fetch_wikipedia(tmp_path, target_mb=0.3, **quiet)
    calls_first = state["page_calls"]

    # Schon erreicht: keine einzige Anfrage.
    again = fetch_wikipedia(tmp_path, target_mb=0.3, fetch_json=lambda url: 1 / 0, log=lambda _: None)
    assert again["new_files"] == 0 and again["chars"] == first["chars"]

    # Größeres Ziel, dazwischen sieben Ausfälle in Folge: der Abruf wartet mit
    # wachsenden Pausen, lädt nichts doppelt und zählt das Vorhandene mit.
    state["down"] = 7
    second = fetch_wikipedia(tmp_path, target_mb=0.6, **quiet)
    assert waits == [5, 10, 20, 40, 80, 120, 120]
    assert second["files"] == first["files"] + second["new_files"]
    assert state["page_calls"] - calls_first == second["new_files"]
    assert 600_000 <= second["chars"] < 600_000 + 200_000


def test_fetch_wikipedia_gives_up_with_usable_message(tmp_path):
    import urllib.error

    from styrke.corpus import CorpusError

    served = {"n": 0}

    def flaky(url: str) -> dict:
        if _is_first(url):
            return {"num_rows_total": 2_800_000, "rows": []}
        served["n"] += 1
        if served["n"] > 1:
            raise urllib.error.HTTPError(url, 502, "Bad Gateway", None, None)
        return _wiki_page()

    with pytest.raises(CorpusError) as err:
        fetch_wikipedia(tmp_path, target_mb=5, fetch_json=flaky, log=lambda _: None, pause_s=0, sleep=lambda _: None)
    text = str(err.value)
    assert "seit 10 Versuchen" in text and "(1 Datei)" in text and "setzt später dort fort" in text

    # Abgelehnte Anfragen (falsche Adresse, kein Zugriff) enden sofort.
    def denied(url: str) -> dict:
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    waits: list[float] = []
    with pytest.raises(CorpusError, match="lehnt die Anfrage ab"):
        fetch_wikipedia(tmp_path / "leer", fetch_json=denied, log=lambda _: None, sleep=waits.append)
    assert waits == []


# -- Prompts ------------------------------------------------------------------


@pytest.mark.parametrize("target", [100, 400, 1000, 20_000])
def test_ratio_builder_hits_target(target):
    builder = PromptBuilder(Corpus.load(None), RatioCounter(3.6), overhead_tokens=10)
    prompt = builder.build(target, seed=7)
    assert abs(prompt.local_tokens - target) <= max(3, target * 0.01)
    assert prompt.messages[-1]["role"] == "user"


def test_different_seeds_give_different_prefixes_and_same_seed_repeats():
    builder = PromptBuilder(Corpus.load(None), RatioCounter(3.6))
    a, b, a2 = builder.build(1000, seed=1), builder.build(1000, seed=2), builder.build(1000, seed=1)
    first_line = lambda p: p.messages[0]["content"].split("\n")[0]  # noqa: E731
    assert first_line(a) != first_line(b)  # kein Treffer im Präfix-Cache
    assert a.messages == a2.messages  # derselbe Vorgang ergibt gezielt einen Treffer


def test_system_prefix_becomes_a_system_message():
    builder = PromptBuilder(Corpus.load(None), RatioCounter(3.6))
    prompt = builder.build(500, seed=3, system_prefix="<|think|>")
    assert prompt.messages[0] == {"role": "system", "content": "<|think|>"}


def _tiny_tokenizer(tmp_path: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]"])
    tk.train_from_iterator(BUILTIN_PARAGRAPHS, trainer)
    path = tmp_path / "tokenizer.json"
    tk.save(str(path))
    return path


def test_tokenizer_counter_is_exact(tmp_path):
    counter = make_counter(_tiny_tokenizer(tmp_path))
    assert isinstance(counter, HFCounter)
    text = " ".join(BUILTIN_PARAGRAPHS)
    cut = counter.truncate(text, 50)
    assert counter.count(cut) == 50 and text.startswith(cut)
    builder = PromptBuilder(Corpus.load(None), counter, overhead_tokens=12)
    for target in (300, 5000):
        assert abs(builder.build(target, seed=11).local_tokens - target) <= 3


def test_make_counter_without_file_uses_ratio(tmp_path):
    assert isinstance(make_counter(tmp_path / "fehlt.json"), RatioCounter)
    assert isinstance(make_counter(None), RatioCounter)


# -- FreeToken-Kommando ---------------------------------------------------------


def test_render_args_skips_none_and_false():
    args = {"a": 1, "flag": True, "weg": False, "leer": None, "text": "x"}
    assert render_args(args) == ["--a", "1", "--flag", "--text", "x"]


def test_resolve_flags_swaps_renamed_option_and_rejects_unknown():
    help_text = "usage: ft serve --model M --moe-backend {auto,offload} --memory-ratio R"
    resolved = resolve_flags({"model": "/m", "moe-strategy": "offload", "memory-ratio": 0.9}, help_text, "img")
    assert resolved == {"model": "/m", "moe-backend": "offload", "memory-ratio": 0.9}
    with pytest.raises(EngineError, match="--gibt-es-nicht"):
        resolve_flags({"gibt-es-nicht": 1}, help_text, "img")


def test_freetoken_command_contains_model_ports_and_variant_args():
    matrix = load_matrix(REPO / "matrix.yaml", local="/nonexistent")
    cell = matrix.cell("flashnext-freetoken")
    model = matrix.models[cell.model]
    engine = FreeTokenEngine(matrix.host, matrix.engines["freetoken"], model, cell.variant("kv-gross"))
    cmd = engine.build_command()
    joined = " ".join(cmd)
    assert cmd[:5] == ["docker", "run", "-d", "--name", "styrke-engine"]
    assert "-p 127.0.0.1:18000:1919" in joined
    assert "--entrypoint ft" in joined and "--ulimit memlock=-1" in joined and "--memory 230g" in joined
    assert "serve --model /models/Qwen3.8-Flash-Next-FP8" in joined
    assert "--kv-reserve-tokens 2097152" in joined and "--max-running-requests 16" in joined
    assert "--enable-cache-report" in joined and "--enable-cache-report True" not in joined
    assert engine.base_url == "http://127.0.0.1:18000"


# -- Vorprüfung ---------------------------------------------------------------


def _fake_docker(monkeypatch, *, image_present: bool, busy: list[str], containers: str = "", stale: bool = False):
    """Ersetzt Docker und nvidia-smi durch feste Antworten; liefert die Liste der Aufrufe."""
    import styrke.engines.docker as d

    calls: list[tuple[str, ...]] = []

    async def run_cmd(*cmd: str, timeout: float = 120):
        calls.append(cmd)
        if cmd[:2] == ("docker", "version"):
            return 0, "27.0\n"
        if cmd[:3] == ("docker", "image", "inspect"):
            return (0, "sha256:abc\n") if image_present else (1, "No such image")
        if cmd[:3] == ("docker", "ps", "-aq"):
            return 0, "deadbeef\n" if stale else ""
        if cmd[:3] == ("docker", "ps", "--no-trunc"):
            return 0, containers
        return 0, ""

    monkeypatch.setattr(d, "run_cmd", run_cmd)
    monkeypatch.setattr(d, "gpu_processes", lambda: list(busy))
    return calls


def _freetoken_engine(tmp_path, with_model: bool = True, **kwargs):
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"host": {"models_dir": str(tmp_path)}}), encoding="utf-8")
    matrix = load_matrix(REPO / "matrix.yaml", local=local)
    cell = matrix.cell("flashnext-freetoken")
    model = matrix.models[cell.model]
    if with_model:
        (tmp_path / model.dir).mkdir(parents=True, exist_ok=True)
        (tmp_path / model.dir / "config.json").write_text("{}", encoding="utf-8")
    return FreeTokenEngine(matrix.host, matrix.engines["freetoken"], model, cell.variant("basis"), **kwargs)


def test_check_setup_names_all_gaps_at_once(tmp_path, monkeypatch):
    import styrke.engines.docker as d
    from styrke.engines import SetupError

    cid = "a" * 64
    monkeypatch.setattr(d, "container_of_pid", lambda pid, containers: containers.get(cid) if pid == "4711" else None)
    _fake_docker(
        monkeypatch, image_present=False, busy=["4711, /usr/bin/python3, 140000 MiB"], containers=f"{cid} dienst-xy\n"
    )
    engine = _freetoken_engine(tmp_path, with_model=False)
    with pytest.raises(SetupError) as err:
        asyncio.run(engine.check_setup())
    problems = err.value.problems
    assert len(problems) == 3
    assert "config.json nicht gefunden" in problems[0] and "host.models_dir" in problems[0]
    assert "docker build -t styrke/freetoken:0.2.1 engines/freetoken" in problems[1]
    assert "engines.freetoken.image" in problems[1]
    assert "vom Container dienst-xy belegt" in problems[2] and "docker stop dienst-xy" in problems[2]


def test_check_setup_passes_and_busy_gpu_can_be_allowed(tmp_path, monkeypatch):
    from styrke.engines import SetupError

    _fake_docker(monkeypatch, image_present=True, busy=[])
    engine = _freetoken_engine(tmp_path)
    asyncio.run(engine.check_setup())
    assert engine.image_id == "sha256:abc"

    _fake_docker(monkeypatch, image_present=True, busy=["1, x, 5 MiB"])
    with pytest.raises(SetupError, match="von einem anderen Prozess belegt"):
        asyncio.run(_freetoken_engine(tmp_path).check_setup())
    asyncio.run(_freetoken_engine(tmp_path, allow_busy_gpu=True).check_setup())


def test_check_setup_waits_for_gpu_after_own_container(tmp_path, monkeypatch):
    """Nach dem Entfernen eines eigenen Containers darf die GPU noch kurz belegt sein."""
    import styrke.engines.docker as d

    calls = _fake_docker(monkeypatch, image_present=True, busy=[], stale=True)
    answers = [["9, ft, 100 MiB"], ["9, ft, 100 MiB"], []]
    monkeypatch.setattr(d, "gpu_processes", lambda: answers.pop(0) if answers else [])

    async def no_sleep(_):
        return None

    monkeypatch.setattr(d.asyncio, "sleep", no_sleep)
    asyncio.run(_freetoken_engine(tmp_path).check_setup())
    assert ("docker", "rm", "-f", "styrke-engine") in calls and answers == []


def test_container_of_pid_reads_cgroup(monkeypatch):
    import styrke.engines.docker as d

    cid = "0123456789abcdef" * 4
    real = d.Path.read_text

    def read_text(self, **kw):
        return f"0::/system.slice/docker-{cid}.scope\n" if str(self) == "/proc/1/cgroup" else real(self, **kw)

    monkeypatch.setattr(d.Path, "read_text", read_text)
    assert d.container_of_pid("1", {cid: "dienst"}) == "dienst"
    assert d.container_of_pid("1", {}) is None


# -- vLLM und SGLang ------------------------------------------------------------


def _engine_for(cell_id: str, variant: str):
    from styrke.engines import make_adapter

    matrix = load_matrix(REPO / "matrix.yaml", local="/nonexistent")
    cell = matrix.cell(cell_id)
    return make_adapter(matrix.host, matrix.engines[cell.engine], matrix.models[cell.model], cell.variant(variant))


def test_vllm_command_and_flag_check():
    from styrke.engines.flags import resolve_flags as resolve

    engine = _engine_for("qwen38-27b-vllm", "parallel-128")
    joined = " ".join(engine.build_command())
    assert "--entrypoint vllm" in joined and "-p 127.0.0.1:18000:8000" in joined and "-e HF_HUB_OFFLINE=1" in joined
    assert "vllm/vllm-openai:latest serve /models/Qwen3.8-27B-FP8 " in joined
    assert "--max-model-len 262144" in joined and "--max-num-seqs 128" in joined and "--max-num-seqs 64" not in joined
    assert "--reasoning-parser qwen3" in joined and "--enable-prompt-tokens-details" in joined and "?" not in joined

    # Abgleich mit der Hilfe: Wünsche, die das Image nicht kennt, entfallen; Pflichtangaben führen zum Fehler.
    help_text = "--served-model-name --host --port --max-model-len --max-num-seqs --gpu-memory-utilization"
    dropped: list[str] = []
    resolved = resolve(engine.merged_args(), help_text, "img", dropped=dropped)
    assert "reasoning-parser" not in resolved and sorted(dropped) == [
        "enable-auto-tool-choice", "enable-prompt-tokens-details", "reasoning-parser", "tool-call-parser",
    ]
    with pytest.raises(EngineError, match="--max-num-seqs"):
        resolve(engine.merged_args(), "--served-model-name --host --port --max-model-len --gpu-memory-utilization", "img")


def test_sglang_command_and_gemma_without_reasoning_parser():
    engine = _engine_for("gemma4-sglang", "prefill-klein")
    joined = " ".join(engine.build_command())
    assert "--entrypoint python3" in joined and "lmsysorg/sglang:latest -m sglang.launch_server " in joined
    assert "--model-path /models/gemma-4-31B-it-FP8-block" in joined and "--context-length 262144" in joined
    assert "--chunked-prefill-size 2048" in joined and "--max-running-requests 64" in joined
    assert "--enable-metrics" in joined and "reasoning-parser" not in joined  # für SGLang ist kein Parser belegt

    vllm = " ".join(_engine_for("gemma4-vllm", "basis").build_command())
    assert "--reasoning-parser gemma4" in vllm and "--enable-auto-tool-choice" in vllm and "--tool-call-parser gemma4" in vllm
    matrix = load_matrix(REPO / "matrix.yaml", local="/nonexistent")
    assert matrix.models["gemma4-31b"].thinking_mode("on") == {"body": {"chat_template_kwargs": {"enable_thinking": True}}}


def test_all_cells_are_enabled_and_have_adapters():
    matrix = load_matrix(REPO / "matrix.yaml", local="/nonexistent")
    assert len(matrix.cells) == 7 and all(c.enabled and c.variants for c in matrix.cells)
    for cell in matrix.cells:
        assert _engine_for(cell.id, cell.variants[0].id).build_command()[0] == "docker"


def test_engine_metrics_parsers():
    from styrke.engines.probes import _find_key, parse_prometheus, vllm_kv_tokens

    text = (
        "# HELP vllm:num_requests_running x\n"
        'vllm:num_requests_running{model_name="styrke"} 7.0\n'
        'vllm:kv_cache_usage_perc{model_name="styrke"} 0.42\n'
        'vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="styrke"} 3.0\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="65536",enable_prefix_caching="True"} 1.0\n'
        "python_gc_objects_collected_total 5\n"
    )
    parsed = parse_prometheus(text, "vllm:")
    assert parsed['num_requests_running{model_name="styrke"}'] == 7.0 and len(parsed) == 3  # ohne Histogramm-Klassen
    assert vllm_kv_tokens(text) == 65536 * 16 and vllm_kv_tokens("nichts") is None
    assert _find_key({"a": [{"internal_states": [{"max_total_num_tokens": 900000}]}]}, "max_total_num_tokens") == 900000


# -- Bericht --------------------------------------------------------------------


def _fake_run(root: Path, name: str, cell: str, rates: list[tuple[float, float, list[str]]], loaded: bool = True) -> Path:
    import json

    run = root / name
    run.mkdir(parents=True)

    def dump(file: str, data: dict) -> None:
        (run / file).write_text(json.dumps(data), encoding="utf-8")

    dump("manifest.json", {"cell": cell, "variant": {"id": "basis"}, "model": {"repo": "x/y"}, "engine": {"image": "img"},
                           "gpu": [{"name": "Testkarte", "mem_total_mib": 143771}]})
    if not loaded:
        dump("feasibility.json", {"loaded": False, "error": "CUDA out of memory"})
        return run
    dump("feasibility.json", {"loaded": True, "load_time_s": 312.0, "max_context_ok": 250011})
    dump("summary_s1.json", {"rows": [
        {"thinking": "off", "prompt_target": 1000, "n": 3, "n_ok": 3, "ttft_s": 0.21, "ttfc_s": 0.21, "decode_tps": 68.2, "prefill_tps": 4800.0},
        {"thinking": "off", "prompt_target": 128000, "n": 3, "n_ok": 3, "ttft_s": 26.5, "ttfc_s": 26.5, "decode_tps": 61.0, "prefill_tps": 4830.0},
    ]})
    dump("summary_s2.json", {"rows": [
        {"level": n, "throughput_tps": t, "decode_tps_p50": t / n, "ttft_p50_s": 0.3, "ttft_p95_s": 0.5, "errors": 0}
        for n, t in ((1, 68.0), (4, 171.0), (8, 293.0))
    ], "max_throughput_tps": 293.0, "max_throughput_level": 8, "saturated_at": None, "stop_reason": "alle Laststufen gefahren"})
    rows = [
        {"rate": rate, "achieved_rate": rate, "calls": 100, "errors": 0, "inflight_mean": 3.0, "ttft_p50_s": ttft / 3,
         "ttft_p95_s": ttft, "total_p50_s": 9.0, "total_p95_s": 30.0, "throughput_tps": 120.0, "overload": overload,
         "users_supported": {"mittel": int(rate * 960)}}
        for rate, ttft, overload in rates
    ]
    carried = [r for r in rows if not r["overload"]]
    dump("summary_s6.json", {"thinking": "off", "rows": rows, "max_sustained_rate": carried[-1]["rate"] if carried else None,
                             "users_supported": {"mittel": int(carried[-1]["rate"] * 960)} if carried else None})
    return run


def test_report_compares_runs_and_applies_targets_afterwards(tmp_path):
    from styrke.report import build_report, de, discover, key_figures, load_run, log_ticks, nice_ticks

    assert de(1234567.891, 1) == "1.234.567,9" and de(None) == "–" and de(0.5, 2, " s") == "0,50 s"
    assert nice_ticks(0, 100) == [0, 20, 40, 60, 80, 100] and log_ticks(0.05, 30) == [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]

    fast = _fake_run(tmp_path, "20261101-000000_a", "zelle-a", [(0.25, 1.0, []), (0.42, 2.5, []), (0.7, 9.0, ["Rückstau wächst"])])
    _fake_run(tmp_path, "20261101-000000_b", "zelle-b", [(0.25, 4.0, []), (0.42, 20.0, ["Rückstau wächst"])])
    _fake_run(tmp_path, "20261101-000000_c", "zelle-c", [], loaded=False)
    newer = _fake_run(tmp_path, "20261102-000000_a", "zelle-a", [(0.25, 1.0, []), (0.42, 2.5, [])])
    found = discover(tmp_path)
    assert len(found) == 3 and newer in found and fast not in found  # je Zelle und Variante der neueste Lauf

    opts = {"ttft_p95": 2.0, "total_p95": None, "cost_month": 9000.0, "profile": "mittel"}
    figures = key_figures(load_run(fast), opts)
    assert figures["sustained_rate"] == 0.42 and figures["users"] == 403
    # Mit dem Zielwert von 2 s gilt nur noch die Rate 0,25; die Kosten je Berechtigtem beziehen sich darauf.
    assert figures["target_rate"] == 0.25 and figures["users_target"] == 240
    assert figures["cost_per_user"] == pytest.approx(9000 / 240)
    assert figures["cost_per_mtok"] == pytest.approx(9000 / (293.0 * 30 * 24 * 3600 / 1e6))
    assert key_figures(load_run(fast), {**opts, "ttft_p95": 0.5})["target_rate"] is None  # kein Abschnitt hält 0,5 s ein

    page = build_report(found, opts)
    for expected in ("zelle-a / basis", "zelle-b / basis", "lädt nicht", "CUDA out of memory", "Testkarte", "37,50 €",
                     "Zielwert 2,0 s", "<svg", "Überlast: Rückstau wächst", "Rate mit Zielwert"):
        assert expected in page, expected
    assert page.count("<svg") == 3 and "http" not in page.replace("http-equiv", "")  # eine Datei ohne äußere Abhängigkeiten
