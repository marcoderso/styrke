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


def test_matrix_has_seven_cells_and_one_enabled():
    matrix = load_matrix(REPO / "matrix.yaml", local="/nonexistent")
    assert len(matrix.cells) == 7
    assert [c.id for c in matrix.cells if c.enabled] == ["flashnext-freetoken"]
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
