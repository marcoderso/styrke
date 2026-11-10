"""Stufe 7, Dauerlauf: Stabilität über längere Zeit.

Der gemischte Betrieb der Stufe 6 läuft eine Stunde lang mit einem festen
Anteil der dort getragenen Rate. Ausgewertet wird in Zeitscheiben: Bleiben
Antwortzeiten und Durchsatz gleich, oder verschlechtern sie sich mit der Zeit?
Dazu kommt der Verlauf von GPU-Speicher und freiem Arbeitsspeicher, an dem ein
langsam volllaufender Speicher auffällt.

Die Rate stammt aus der Kurzauswertung der Stufe 6 desselben Laufs. Läuft
Stufe 7 ohne sie, muss ``stages.s7.rate`` in der Testmatrix stehen.
"""

from __future__ import annotations

from typing import Any

from ..store import read_jsonl
from .common import StageContext, ensure_calibrated, fmt
from .load import latency_stats, throughput_between
from .scenario import Scenario, overload_reasons, session_shapes, shapes_as_dicts, summarize


def buckets(scenario: Scenario, warmup_s: float, duration_s: float, bucket_s: float) -> list[dict[str, Any]]:
    """Kennzahlen je Zeitscheibe nach der Anlaufzeit."""
    rows: list[dict[str, Any]] = []
    t = scenario.t_start + warmup_s
    end = scenario.t_start + duration_s
    while t + bucket_s <= end + 1e-6:
        calls = [r for r in scenario.records if t <= r["ts_submit"] < t + bucket_s]
        inflight = [n for ts, n, _ in scenario.timeline if t <= ts < t + bucket_s]
        rows.append(
            {
                "minute": round((t - scenario.t_start) / 60, 1),
                "calls": len(calls),
                "errors": sum(1 for r in calls if not r["ok"] and r["error"] != "abgebrochen"),
                "throughput_tps": round(throughput_between(scenario.records, t, t + bucket_s) or 0.0, 1),
                "inflight_mean": round(sum(inflight) / len(inflight), 2) if inflight else None,
                **latency_stats([r for r in calls if r["ok"]]),
            }
        )
        t += bucket_s
    return rows


def _drift(rows: list[dict[str, Any]], key: str) -> float | None:
    """Veränderung vom ersten zum letzten Drittel der Zeitscheiben in Prozent."""
    third = max(1, len(rows) // 3)
    first = [r[key] for r in rows[:third] if r.get(key) is not None]
    last = [r[key] for r in rows[-third:] if r.get(key) is not None]
    if len(rows) < 3 or not first or not last or not sum(first):
        return None
    a, b = sum(first) / len(first), sum(last) / len(last)
    return round(100 * (b - a) / a, 1)


def resource_trend(ctx: StageContext, t0: float, t1: float) -> dict[str, Any]:
    """Verlauf von GPU-Speicher und freiem Arbeitsspeicher aus den mitgeschriebenen Messreihen."""
    gpu: list[float] = []
    ram: list[float] = []
    for sample in read_jsonl(ctx.store.dir / "samples.jsonl"):
        if not t0 <= sample.get("ts", 0) <= t1:
            continue
        if sample.get("kind") == "gpu" and sample.get("mem_used_mib") is not None:
            gpu.append(sample["mem_used_mib"])
        elif sample.get("kind") == "host" and sample.get("mem_available_gib") is not None:
            ram.append(sample["mem_available_gib"])

    def edges(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        n = max(1, len(values) // 10)
        return sum(values[:n]) / n, sum(values[-n:]) / n

    gpu_first, gpu_last = edges(gpu)
    ram_first, ram_last = edges(ram)
    return {
        "gpu_mem_first_mib": None if gpu_first is None else round(gpu_first),
        "gpu_mem_last_mib": None if gpu_last is None else round(gpu_last),
        "gpu_mem_max_mib": max(gpu) if gpu else None,
        "host_mem_available_first_gib": None if ram_first is None else round(ram_first, 1),
        "host_mem_available_last_gib": None if ram_last is None else round(ram_last, 1),
        "host_mem_available_min_gib": min(ram) if ram else None,
    }


async def run(ctx: StageContext) -> dict[str, Any]:
    params = ctx.params["s7"]
    scenario_params = ctx.params["s6"]
    await ensure_calibrated(ctx)

    rate = params.get("rate")
    source = "stages.s7.rate"
    if not rate:
        s6 = ctx.store.read_json("summary_s6.json") or {}
        base = s6.get("max_sustained_achieved_rate")
        if base:
            rate = round(params["share_of_sustained"] * base, 4)
            source = f"{params['share_of_sustained']:.0%} der in Stufe 6 getragenen Rate ({base:g} Aufrufe/s)"
    shapes, dropped = session_shapes(ctx, scenario_params)
    if not rate or not shapes:
        reason = (
            "keine Sitzungsart passt in den Kontext" if not shapes else
            "Es gibt keine getragene Rate aus Stufe 6. Erst Stufe 6 fahren oder stages.s7.rate in der Testmatrix angeben."
        )
        summary = {"skipped": reason}
        ctx.store.write_json("summary_s7.json", summary)
        ctx.log(f"  ausgelassen: {reason}")
        return summary

    duration, warmup = params["duration_s"], params["warmup_s"]
    ctx.log(f"  {duration / 60:g} Minuten mit {rate:g} Aufrufen/s ({source})")
    metrics_start = await ctx.adapter.metrics(ctx.session)
    scenario = Scenario(ctx, stage=7, rate=rate, shapes=shapes, thinking=scenario_params["thinking"])
    await scenario.run(duration, params["drain_s"])

    overall = summarize(scenario, warmup, duration, ctx.load_model)
    overall["overload"] = overload_reasons(overall, scenario_params)
    rows = buckets(scenario, warmup, duration, params["bucket_s"])
    summary = {
        "rate": rate,
        "rate_source": source,
        "duration_s": duration,
        "thinking": scenario_params["thinking"],
        "sessions": shapes_as_dicts(shapes),
        "sessions_dropped": dropped,
        "overall": overall,
        "buckets": rows,
        "ttft_p50_drift_pct": _drift(rows, "ttft_p50_s"),
        "ttft_p95_drift_pct": _drift(rows, "ttft_p95_s"),
        "throughput_drift_pct": _drift(rows, "throughput_tps"),
        "resources": resource_trend(ctx, scenario.t_start, scenario.t_start + duration),
        "engine_metrics_start": metrics_start,
        "engine_metrics_end": await ctx.adapter.metrics(ctx.session),
    }
    ctx.store.write_json("summary_s7.json", summary)

    ctx.log(f"  {'Minute':>7} {'Aufrufe':>8} {'laufend':>8} {'TTFT p50':>9} {'TTFT p95':>9} {'tok/s':>7} {'Fehler':>7}")
    for row in rows:
        ctx.log(
            f"  {row['minute']:>7g} {row['calls']:>8} {fmt(row['inflight_mean'], 1):>8} "
            f"{fmt(row['ttft_p50_s'], 2, ' s'):>9} {fmt(row['ttft_p95_s'], 2, ' s'):>9} "
            f"{row['throughput_tps']:>7.0f} {row['errors']:>7}"
        )
    res = summary["resources"]
    ctx.log(
        f"  Veränderung vom ersten zum letzten Drittel: TTFT p95 {fmt(summary['ttft_p95_drift_pct'], 0, ' %')}, "
        f"Durchsatz {fmt(summary['throughput_drift_pct'], 0, ' %')}; Fehler gesamt: {overall['errors']}"
    )
    ctx.log(
        f"  GPU-Speicher {fmt(res['gpu_mem_first_mib'], 0)} -> {fmt(res['gpu_mem_last_mib'], 0)} MiB, "
        f"freier Arbeitsspeicher {fmt(res['host_mem_available_first_gib'], 1)} -> "
        f"{fmt(res['host_mem_available_last_gib'], 1)} GiB"
    )
    if overall["overload"]:
        ctx.log("  Hinweis: Überlast im Dauerlauf: " + "; ".join(overall["overload"]))
    return summary
