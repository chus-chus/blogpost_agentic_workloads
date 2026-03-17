"""case study 2 driver."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import requests
import yaml
from vidhi import create_class_from_dict, load_yaml_config

from veeksha.benchmark import manage_benchmark_run
from veeksha.capacity_search import patch_traffic_knob
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TraceBundleConfig:
    output_dir: str
    generator_script: str
    generate_if_missing: bool = True
    linear_trace_name: str = "workload_a_linear.jsonl"
    dag_trace_name: str = "workload_b_dag.jsonl"
    linear_sessions: int = 30
    linear_requests: int = 5
    dag_sessions: int = 10
    fresh_input_tokens: int = 500
    output_tokens: int = 300
    wait_after_ready_s: float = 0.0
    dag_final_history_parent: int = 13

    @property
    def linear_trace_path(self) -> str:
        return str(Path(self.output_dir) / self.linear_trace_name)

    @property
    def dag_trace_path(self) -> str:
        return str(Path(self.output_dir) / self.dag_trace_name)


@dataclass(frozen=True)
class RateSearchParams:
    min_value: float = 0.1
    start_value: float = 1.0
    max_value: float = 64.0
    expansion_factor: float = 2.0
    precision: int = 2
    refinement_rounds: int = 4
    stop_after_consecutive_unhealthy: int = 2
    alternate_order: bool = True

    def round_value(self, value: float) -> float:
        scale = 10**self.precision
        return round(value * scale) / scale


@dataclass(frozen=True)
class Guardrails:
    min_completed_requests: int = 150
    min_completion_ratio: float = 0.97
    max_error_rate: float = 0.02
    max_ttfc_p99_s: float = 3.0
    max_e2e_p95_s: float = 20.0
    require_all_slos_met: bool = True


@dataclass(frozen=True)
class ObjectiveWeights:
    ttfc_p99_weight: float = 0.45
    e2e_p95_weight: float = 0.25
    throughput_weight: float = 0.15
    cache_reuse_weight: float = 0.15
    max_relative_gap: float = 3.0


@dataclass(frozen=True)
class VllmMetricsConfig:
    enabled: bool = True
    require_metrics: bool = True
    scrape_timeout_s: float = 10.0


@dataclass(frozen=True)
class WorkloadShapeSearchConfig:
    output_dir: str
    linear_benchmark_config: str
    dag_benchmark_config: str
    trace_bundle: TraceBundleConfig
    rate_search: RateSearchParams
    guardrails: Guardrails
    objective: ObjectiveWeights
    vllm_metrics: VllmMetricsConfig


@dataclass(frozen=True)
class BenchmarkRunSummary:
    workload: str
    rate: float
    run_dir: str
    total_requests: int
    completed_requests: int
    errored_requests: int
    error_rate: float
    completion_ratio: float
    all_slos_met: Optional[bool]
    observed_session_dispatch_rate: Optional[float]
    ttfc_p50_s: Optional[float]
    ttfc_p95_s: Optional[float]
    ttfc_p99_s: Optional[float]
    e2e_p50_s: Optional[float]
    e2e_p95_s: Optional[float]
    e2e_p99_s: Optional[float]
    tpot_mean_s: Optional[float]
    tpot_based_throughput: Optional[float]
    tbc_based_throughput: Optional[float]
    mean_total_prompt_tokens: Optional[float]
    mean_delta_prompt_tokens: Optional[float]
    mean_cacheable_prompt_tokens: Optional[float]
    mean_prompt_reuse_ratio: Optional[float]
    decode_window_tbc_p95_s: Optional[float]
    decode_window_tbc_p99_s: Optional[float]
    decode_window_duration_s: Optional[float]
    vllm_metrics_scraped: Optional[bool] = None
    vllm_metrics_url: Optional[str] = None
    vllm_metrics_scraped_at_utc: Optional[str] = None
    vllm_metrics_scrape_error: Optional[str] = None
    vllm_kv_cache_usage_perc: Optional[float] = None
    vllm_prompt_tokens_cached: Optional[float] = None
    vllm_prompt_tokens_recomputed: Optional[float] = None
    vllm_prompt_cache_token_ratio: Optional[float] = None
    vllm_prefix_cache_hits: Optional[float] = None
    vllm_prefix_cache_queries: Optional[float] = None
    vllm_prefix_cache_hit_rate: Optional[float] = None
    vllm_num_preemptions: Optional[float] = None

    def to_prefixed_dict(self, prefix: str) -> dict[str, Any]:
        flattened = asdict(self)
        return {f"{prefix}_{key}": value for key, value in flattened.items()}


@dataclass(frozen=True)
class PairedRateResult:
    rate: float
    phase: str
    run_order: list[str]
    linear: BenchmarkRunSummary
    dag: BenchmarkRunSummary
    healthy: bool
    status: str
    divergence_score: float
    load_factor: float
    overall_score: float
    notes: list[str]

    def to_flat_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "rate": self.rate,
            "phase": self.phase,
            "run_order": " -> ".join(self.run_order),
            "healthy": self.healthy,
            "status": self.status,
            "divergence_score": self.divergence_score,
            "load_factor": self.load_factor,
            "overall_score": self.overall_score,
            "notes": "; ".join(self.notes),
        }
        row.update(self.linear.to_prefixed_dict("linear"))
        row.update(self.dag.to_prefixed_dict("dag"))
        return row


def _resolve_input_path(raw_path: str, *, base_dir: Path) -> str:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return str(base_candidate)

    return str(cwd_candidate)


def _resolve_output_path(raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(data).__name__}.")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}, "
                    f"got {type(row).__name__}."
                )
            rows.append(row)
    return rows


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.quantile(arr, q))


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _slo_observed_value(slo_results: dict[str, Any], slo_metric_key: str) -> Optional[float]:
    results = slo_results.get("results")
    if not isinstance(results, list):
        return None
    for row in results:
        if not isinstance(row, dict):
            continue
        if row.get("slo_metric_key") != slo_metric_key:
            continue
        observed = row.get("observed_value")
        return _optional_float(observed)
    return None


def _relative_gap(lhs: Optional[float], rhs: Optional[float], *, cap: float) -> float:
    if lhs is None or rhs is None or lhs <= 0 or rhs <= 0:
        return 0.0
    return min(abs(lhs / rhs - 1.0), cap)


def _load_search_config(config_path: str) -> WorkloadShapeSearchConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    raw = load_yaml_config(str(resolved_config_path))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Search config {resolved_config_path} must resolve to a YAML mapping."
        )

    base_dir = resolved_config_path.parent

    trace_raw = dict(raw.get("trace_bundle") or {})
    rate_raw = dict(raw.get("rate_search") or {})
    guard_raw = dict(raw.get("guardrails") or {})
    objective_raw = dict(raw.get("objective") or {})
    vllm_metrics_raw = dict(raw.get("vllm_metrics") or {})

    trace_cfg = TraceBundleConfig(
        output_dir=_resolve_output_path(
            str(trace_raw.get("output_dir", "traces/workload_shape"))
        ),
        generator_script=_resolve_input_path(
            str(
                trace_raw.get(
                    "generator_script", "scripts/generate_workload_shape_traces.py"
                )
            ),
            base_dir=base_dir,
        ),
        generate_if_missing=bool(trace_raw.get("generate_if_missing", True)),
        linear_trace_name=str(
            trace_raw.get("linear_trace_name", "workload_a_linear.jsonl")
        ),
        dag_trace_name=str(trace_raw.get("dag_trace_name", "workload_b_dag.jsonl")),
        linear_sessions=int(trace_raw.get("linear_sessions", 30)),
        linear_requests=int(trace_raw.get("linear_requests", 5)),
        dag_sessions=int(trace_raw.get("dag_sessions", 10)),
        fresh_input_tokens=int(trace_raw.get("fresh_input_tokens", 500)),
        output_tokens=int(trace_raw.get("output_tokens", 300)),
        wait_after_ready_s=float(trace_raw.get("wait_after_ready_s", 0.0)),
        dag_final_history_parent=int(trace_raw.get("dag_final_history_parent", 13)),
    )

    return WorkloadShapeSearchConfig(
        output_dir=_resolve_output_path(str(raw.get("output_dir", "benchmark_output"))),
        linear_benchmark_config=_resolve_input_path(
            str(raw["linear_benchmark_config"]),
            base_dir=base_dir,
        ),
        dag_benchmark_config=_resolve_input_path(
            str(raw["dag_benchmark_config"]),
            base_dir=base_dir,
        ),
        trace_bundle=trace_cfg,
        rate_search=RateSearchParams(
            min_value=float(rate_raw.get("min_value", 0.1)),
            start_value=float(rate_raw.get("start_value", 1.0)),
            max_value=float(rate_raw.get("max_value", 64.0)),
            expansion_factor=float(rate_raw.get("expansion_factor", 2.0)),
            precision=int(rate_raw.get("precision", 2)),
            refinement_rounds=int(rate_raw.get("refinement_rounds", 4)),
            stop_after_consecutive_unhealthy=int(
                rate_raw.get("stop_after_consecutive_unhealthy", 2)
            ),
            alternate_order=bool(rate_raw.get("alternate_order", True)),
        ),
        guardrails=Guardrails(
            min_completed_requests=int(guard_raw.get("min_completed_requests", 150)),
            min_completion_ratio=float(guard_raw.get("min_completion_ratio", 0.97)),
            max_error_rate=float(guard_raw.get("max_error_rate", 0.02)),
            max_ttfc_p99_s=float(guard_raw.get("max_ttfc_p99_s", 3.0)),
            max_e2e_p95_s=float(guard_raw.get("max_e2e_p95_s", 20.0)),
            require_all_slos_met=bool(guard_raw.get("require_all_slos_met", True)),
        ),
        objective=ObjectiveWeights(
            ttfc_p99_weight=float(objective_raw.get("ttfc_p99_weight", 0.45)),
            e2e_p95_weight=float(objective_raw.get("e2e_p95_weight", 0.25)),
            throughput_weight=float(objective_raw.get("throughput_weight", 0.15)),
            cache_reuse_weight=float(objective_raw.get("cache_reuse_weight", 0.15)),
            max_relative_gap=float(objective_raw.get("max_relative_gap", 3.0)),
        ),
        vllm_metrics=VllmMetricsConfig(
            enabled=bool(vllm_metrics_raw.get("enabled", True)),
            require_metrics=bool(vllm_metrics_raw.get("require_metrics", True)),
            scrape_timeout_s=float(vllm_metrics_raw.get("scrape_timeout_s", 10.0)),
        ),
    )


def _resolve_benchmark_paths(
    config_dict: dict[str, Any], *, base_dir: Path
) -> dict[str, Any]:
    config_dict = json.loads(json.dumps(config_dict))

    if "output_dir" in config_dict and isinstance(config_dict["output_dir"], str):
        config_dict["output_dir"] = _resolve_output_path(config_dict["output_dir"])

    server = config_dict.get("server")
    if isinstance(server, dict) and isinstance(server.get("env_path"), str):
        server["env_path"] = _resolve_input_path(server["env_path"], base_dir=base_dir)

    session_generator = config_dict.get("session_generator")
    if isinstance(session_generator, dict):
        if isinstance(session_generator.get("trace_file"), str):
            session_generator["trace_file"] = _resolve_input_path(
                session_generator["trace_file"],
                base_dir=base_dir,
            )
        flavor = session_generator.get("flavor")
        if isinstance(flavor, dict) and isinstance(flavor.get("corpus_file"), str):
            flavor["corpus_file"] = _resolve_input_path(
                flavor["corpus_file"],
                base_dir=base_dir,
            )

    return config_dict


def _load_benchmark_config(config_path: str) -> BenchmarkConfig:
    resolved_path = Path(config_path).expanduser().resolve()
    raw = load_yaml_config(str(resolved_path))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Benchmark config {resolved_path} must resolve to a YAML mapping."
        )
    resolved_raw = _resolve_benchmark_paths(raw, base_dir=resolved_path.parent)
    return create_class_from_dict(BenchmarkConfig, resolved_raw)


def _ensure_trace_bundle(config: WorkloadShapeSearchConfig) -> None:
    linear_trace = Path(config.trace_bundle.linear_trace_path)
    dag_trace = Path(config.trace_bundle.dag_trace_path)
    if linear_trace.exists() and dag_trace.exists():
        return

    if not config.trace_bundle.generate_if_missing:
        raise FileNotFoundError(
            "Trace bundle is missing and trace_bundle.generate_if_missing is false."
        )

    output_dir = Path(config.trace_bundle.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        config.trace_bundle.generator_script,
        "--output-dir",
        config.trace_bundle.output_dir,
        "--linear-sessions",
        str(config.trace_bundle.linear_sessions),
        "--linear-requests",
        str(config.trace_bundle.linear_requests),
        "--dag-sessions",
        str(config.trace_bundle.dag_sessions),
        "--fresh-input-tokens",
        str(config.trace_bundle.fresh_input_tokens),
        "--output-tokens",
        str(config.trace_bundle.output_tokens),
        "--wait-after-ready-s",
        str(config.trace_bundle.wait_after_ready_s),
        "--dag-final-history-parent",
        str(config.trace_bundle.dag_final_history_parent),
    ]
    logger.info("Generating trace bundle: %s", " ".join(command))
    subprocess.run(command, check=True)


def summarize_run(
    *,
    workload: str,
    rate: float,
    run_dir: str,
) -> BenchmarkRunSummary:
    run_path = Path(run_dir)
    metrics_dir = run_path / "metrics"

    summary_stats = _read_json(metrics_dir / "summary_stats.json")
    throughput = _read_json(metrics_dir / "throughput_metrics.json")
    slo_results = _read_json(metrics_dir / "slo_results.json")
    request_rows = _read_jsonl(metrics_dir / "request_level_metrics.jsonl")

    decode_window_path = metrics_dir / "decode_window_metrics.json"
    decode_window = _read_json(decode_window_path) if decode_window_path.exists() else {}
    vllm_metrics_path = metrics_dir / "vllm_metrics_summary.json"
    vllm_metrics = _read_json(vllm_metrics_path) if vllm_metrics_path.exists() else {}

    ttfc_values = [
        float(row["ttfc"]) for row in request_rows if row.get("ttfc") is not None
    ]
    e2e_values = [
        float(row["end_to_end_latency"])
        for row in request_rows
        if row.get("end_to_end_latency") is not None
    ]
    tpot_values = [
        float(row["tpot"]) for row in request_rows if row.get("tpot") is not None
    ]
    total_prompt_values = [
        float(row["num_total_prompt_tokens"])
        for row in request_rows
        if row.get("num_total_prompt_tokens") is not None
    ]
    delta_prompt_values = [
        float(row["num_delta_prompt_tokens"])
        for row in request_rows
        if row.get("num_delta_prompt_tokens") is not None
    ]

    cacheable_prompt_values: list[float] = []
    prompt_reuse_values: list[float] = []
    for row in request_rows:
        total_prompt = row.get("num_total_prompt_tokens")
        delta_prompt = row.get("num_delta_prompt_tokens")
        if total_prompt is None or delta_prompt is None:
            continue
        total_prompt_f = float(total_prompt)
        delta_prompt_f = float(delta_prompt)
        cacheable_prompt_values.append(max(0.0, total_prompt_f - delta_prompt_f))
        if total_prompt_f > 0:
            prompt_reuse_values.append(
                max(0.0, min(1.0, 1.0 - (delta_prompt_f / total_prompt_f)))
            )

    decode_tbc_stats = decode_window.get("tbc_in_window_stats")
    decode_windows = decode_window.get("windows")

    total_requests = int(summary_stats.get("Number of Requests", 0))
    completed_requests = int(summary_stats.get("Number of Completed Requests", 0))
    errored_requests = int(summary_stats.get("Number of Errored Requests", 0))
    completion_ratio = (
        completed_requests / total_requests if total_requests > 0 else 0.0
    )

    return BenchmarkRunSummary(
        workload=workload,
        rate=rate,
        run_dir=run_dir,
        total_requests=total_requests,
        completed_requests=completed_requests,
        errored_requests=errored_requests,
        error_rate=float(summary_stats.get("Error Rate", 1.0)),
        completion_ratio=completion_ratio,
        all_slos_met=(
            bool(slo_results.get("all_slos_met"))
            if "all_slos_met" in slo_results
            else None
        ),
        observed_session_dispatch_rate=summary_stats.get("Observed Session Dispatch Rate"),
        ttfc_p50_s=_quantile(ttfc_values, 0.50),
        ttfc_p95_s=_quantile(ttfc_values, 0.95),
        ttfc_p99_s=_quantile(ttfc_values, 0.99),
        e2e_p50_s=_quantile(e2e_values, 0.50),
        e2e_p95_s=_quantile(e2e_values, 0.95),
        e2e_p99_s=_quantile(e2e_values, 0.99),
        tpot_mean_s=_mean(tpot_values),
        tpot_based_throughput=throughput.get("tpot_based_throughput"),
        tbc_based_throughput=throughput.get("tbc_based_throughput"),
        mean_total_prompt_tokens=_mean(total_prompt_values),
        mean_delta_prompt_tokens=_mean(delta_prompt_values),
        mean_cacheable_prompt_tokens=_mean(cacheable_prompt_values),
        mean_prompt_reuse_ratio=_mean(prompt_reuse_values),
        decode_window_tbc_p95_s=(
            float(decode_tbc_stats["p95"])
            if isinstance(decode_tbc_stats, dict)
            and decode_tbc_stats.get("p95") is not None
            else _slo_observed_value(slo_results, "tbc_p95")
        ),
        decode_window_tbc_p99_s=(
            float(decode_tbc_stats["p99"])
            if isinstance(decode_tbc_stats, dict)
            and decode_tbc_stats.get("p99") is not None
            else None
        ),
        decode_window_duration_s=(
            float(decode_windows["total_duration_s"])
            if isinstance(decode_windows, dict)
            and decode_windows.get("total_duration_s") is not None
            else None
        ),
        vllm_metrics_scraped=(
            bool(vllm_metrics.get("metrics_scraped"))
            if "metrics_scraped" in vllm_metrics
            else None
        ),
        vllm_metrics_url=vllm_metrics.get("metrics_url"),
        vllm_metrics_scraped_at_utc=vllm_metrics.get("scraped_at_utc"),
        vllm_metrics_scrape_error=vllm_metrics.get("scrape_error"),
        vllm_kv_cache_usage_perc=_optional_float(
            vllm_metrics.get("kv_cache_usage_perc")
        ),
        vllm_prompt_tokens_cached=_optional_float(
            vllm_metrics.get("prompt_tokens_cached")
        ),
        vllm_prompt_tokens_recomputed=_optional_float(
            vllm_metrics.get("prompt_tokens_recomputed")
        ),
        vllm_prompt_cache_token_ratio=_optional_float(
            vllm_metrics.get("prompt_cache_token_ratio")
        ),
        vllm_prefix_cache_hits=_optional_float(vllm_metrics.get("prefix_cache_hits")),
        vllm_prefix_cache_queries=_optional_float(
            vllm_metrics.get("prefix_cache_queries")
        ),
        vllm_prefix_cache_hit_rate=_optional_float(
            vllm_metrics.get("prefix_cache_hit_rate")
        ),
        vllm_num_preemptions=_optional_float(vllm_metrics.get("num_preemptions")),
    )


def _cache_divergence(
    *,
    linear: BenchmarkRunSummary,
    dag: BenchmarkRunSummary,
) -> float:
    vllm_components: list[float] = []

    if (
        linear.vllm_prefix_cache_hit_rate is not None
        and dag.vllm_prefix_cache_hit_rate is not None
    ):
        vllm_components.append(
            abs(dag.vllm_prefix_cache_hit_rate - linear.vllm_prefix_cache_hit_rate)
        )

    if (
        linear.vllm_prompt_cache_token_ratio is not None
        and dag.vllm_prompt_cache_token_ratio is not None
    ):
        vllm_components.append(
            abs(
                dag.vllm_prompt_cache_token_ratio
                - linear.vllm_prompt_cache_token_ratio
            )
        )

    if (
        linear.vllm_kv_cache_usage_perc is not None
        and dag.vllm_kv_cache_usage_perc is not None
    ):
        vllm_components.append(
            abs(dag.vllm_kv_cache_usage_perc - linear.vllm_kv_cache_usage_perc)
        )

    if vllm_components:
        return float(np.mean(np.asarray(vllm_components, dtype=float)))

    return abs((dag.mean_prompt_reuse_ratio or 0.0) - (linear.mean_prompt_reuse_ratio or 0.0))


def score_paired_candidate(
    *,
    linear: BenchmarkRunSummary,
    dag: BenchmarkRunSummary,
    guardrails: Guardrails,
    objective: ObjectiveWeights,
) -> PairedRateResult:
    notes: list[str] = []

    def workload_is_healthy(summary: BenchmarkRunSummary) -> bool:
        healthy = True
        if summary.completed_requests < guardrails.min_completed_requests:
            notes.append(
                f"{summary.workload}: completed_requests<{guardrails.min_completed_requests}"
            )
            healthy = False
        if summary.completion_ratio < guardrails.min_completion_ratio:
            notes.append(
                f"{summary.workload}: completion_ratio<{guardrails.min_completion_ratio:.3f}"
            )
            healthy = False
        if summary.error_rate > guardrails.max_error_rate:
            notes.append(
                f"{summary.workload}: error_rate>{guardrails.max_error_rate:.3f}"
            )
            healthy = False
        if summary.ttfc_p99_s is None or summary.ttfc_p99_s > guardrails.max_ttfc_p99_s:
            notes.append(
                f"{summary.workload}: ttfc_p99>{guardrails.max_ttfc_p99_s:.3f}s"
            )
            healthy = False
        if summary.e2e_p95_s is None or summary.e2e_p95_s > guardrails.max_e2e_p95_s:
            notes.append(
                f"{summary.workload}: e2e_p95>{guardrails.max_e2e_p95_s:.3f}s"
            )
            healthy = False
        if guardrails.require_all_slos_met and summary.all_slos_met is not True:
            notes.append(f"{summary.workload}: benchmark SLOs not met")
            healthy = False
        return healthy

    linear_healthy = workload_is_healthy(linear)
    dag_healthy = workload_is_healthy(dag)
    healthy = linear_healthy and dag_healthy

    latency_gap = (
        objective.ttfc_p99_weight
        * _relative_gap(
            dag.ttfc_p99_s,
            linear.ttfc_p99_s,
            cap=objective.max_relative_gap,
        )
        + objective.e2e_p95_weight
        * _relative_gap(
            dag.e2e_p95_s,
            linear.e2e_p95_s,
            cap=objective.max_relative_gap,
        )
    )
    throughput_gap = objective.throughput_weight * _relative_gap(
        linear.tpot_based_throughput,
        dag.tpot_based_throughput,
        cap=objective.max_relative_gap,
    )
    cache_gap = objective.cache_reuse_weight * _cache_divergence(
        linear=linear,
        dag=dag,
    )
    divergence_score = latency_gap + throughput_gap + cache_gap

    ttfc_utilization = max(
        (linear.ttfc_p99_s or 0.0) / guardrails.max_ttfc_p99_s,
        (dag.ttfc_p99_s or 0.0) / guardrails.max_ttfc_p99_s,
    )
    e2e_utilization = max(
        (linear.e2e_p95_s or 0.0) / guardrails.max_e2e_p95_s,
        (dag.e2e_p95_s or 0.0) / guardrails.max_e2e_p95_s,
    )
    load_factor = min(1.0, max(ttfc_utilization, e2e_utilization))
    overall_score = divergence_score * load_factor if healthy else 0.0

    status = "healthy" if healthy else "guardrail_failed"

    return PairedRateResult(
        rate=linear.rate,
        phase="",
        run_order=[],
        linear=linear,
        dag=dag,
        healthy=healthy,
        status=status,
        divergence_score=round(divergence_score, 6),
        load_factor=round(load_factor, 6),
        overall_score=round(overall_score, 6),
        notes=notes,
    )


def _best_candidate(results: Iterable[PairedRateResult]) -> Optional[PairedRateResult]:
    healthy = [result for result in results if result.healthy]
    if not healthy:
        return None
    return max(healthy, key=lambda result: (result.overall_score, result.rate))


def _geometric_midpoint(lhs: float, rhs: float, *, precision: int) -> Optional[float]:
    if lhs <= 0 or rhs <= 0 or lhs >= rhs:
        return None
    midpoint = math.sqrt(lhs * rhs)
    rounded = round(midpoint, precision)
    if rounded in {round(lhs, precision), round(rhs, precision)}:
        return None
    return rounded


def _persist_search_state(
    *,
    config: WorkloadShapeSearchConfig,
    results: Sequence[PairedRateResult],
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best = _best_candidate(results)
    best_rate = best.rate if best is not None else None

    results_json = {
        "search_config": asdict(config),
        "best_rate": best_rate,
        "num_evaluated_rates": len(results),
        "results": [
            {
                **result.to_flat_dict(),
                "notes": result.notes,
                "run_order": result.run_order,
            }
            for result in sorted(results, key=lambda item: item.rate)
        ],
    }
    with (output_dir / "workload_shape_search_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results_json, handle, indent=2, sort_keys=True)
        handle.write("\n")

    fieldnames = list(sorted(results[0].to_flat_dict().keys())) if results else []
    if fieldnames:
        with (output_dir / "workload_shape_search_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in sorted(results, key=lambda item: item.rate):
                writer.writerow(result.to_flat_dict())

    with (output_dir / "resolved_search_config.yml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False, default_flow_style=False)


def _normalize_run_order(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split("->") if part.strip()]
    return []


def _load_existing_paired_runs(
    source_output_dir: str,
) -> list[dict[str, Any]]:
    results_path = Path(source_output_dir) / "workload_shape_search_results.json"
    payload = _read_json(results_path)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(
            f"Expected a results list in {results_path}, found {type(rows).__name__}"
        )

    paired_runs: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rate = row.get("rate")
        linear_run_dir = row.get("linear_run_dir")
        dag_run_dir = row.get("dag_run_dir")
        if rate is None or not linear_run_dir or not dag_run_dir:
            continue
        paired_runs.append(
            {
                "rate": float(rate),
                "phase": str(row.get("phase", "")),
                "run_order": _normalize_run_order(row.get("run_order")),
                "linear_run_dir": str(linear_run_dir),
                "dag_run_dir": str(dag_run_dir),
            }
        )

    if not paired_runs:
        raise ValueError(
            f"No paired run directories were found in {results_path}. "
            "Expected linear_run_dir and dag_run_dir entries."
        )

    paired_runs.sort(key=lambda item: item["rate"])
    return paired_runs


def _benchmark_output_base(
    search_output_dir: str,
    *,
    rate: float,
    workload: str,
) -> str:
    safe_rate = str(rate).replace(".", "_")
    return str(Path(search_output_dir) / "runs" / f"rate_{safe_rate}" / workload)


def _parse_prometheus_labels(raw_labels: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not raw_labels:
        return labels

    items: list[str] = []
    current: list[str] = []
    in_quotes = False
    escape = False
    for char in raw_labels:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\":
            current.append(char)
            escape = True
            continue
        if char == '"':
            current.append(char)
            in_quotes = not in_quotes
            continue
        if char == "," and not in_quotes:
            items.append("".join(current))
            current = []
            continue
        current.append(char)

    if current:
        items.append("".join(current))

    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        labels[key.strip()] = value.strip().strip('"')
    return labels


def _split_metric_and_labels(metric_with_labels: str) -> tuple[str, dict[str, str]]:
    if "{" not in metric_with_labels:
        return metric_with_labels, {}
    metric_name, remainder = metric_with_labels.split("{", 1)
    return metric_name, _parse_prometheus_labels(remainder.rstrip("}"))


def _parse_prometheus_samples(
    text: str,
) -> tuple[dict[str, list[float]], dict[str, list[tuple[dict[str, str], float]]]]:
    samples: dict[str, list[float]] = {}
    labeled_samples: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            metric_with_labels, value_text = line.rsplit(None, 1)
            value = float(value_text)
        except ValueError:
            continue

        if not math.isfinite(value):
            continue

        metric_name, labels = _split_metric_and_labels(metric_with_labels)
        samples.setdefault(metric_name, []).append(value)
        labeled_samples.setdefault(metric_name, []).append((labels, value))

    return samples, labeled_samples


def _metric_sum(
    samples: dict[str, list[float]],
    metric_name: str,
) -> Optional[float]:
    values = samples.get(metric_name)
    if not values:
        return None
    return float(np.sum(np.asarray(values, dtype=float)))


def _metric_max(
    samples: dict[str, list[float]],
    metric_name: str,
) -> Optional[float]:
    values = samples.get(metric_name)
    if not values:
        return None
    return float(np.max(np.asarray(values, dtype=float)))


def _metric_sum_any(
    samples: dict[str, list[float]],
    *metric_names: str,
) -> Optional[float]:
    for metric_name in metric_names:
        value = _metric_sum(samples, metric_name)
        if value is not None:
            return value
    return None


def _metric_max_any(
    samples: dict[str, list[float]],
    *metric_names: str,
) -> Optional[float]:
    for metric_name in metric_names:
        value = _metric_max(samples, metric_name)
        if value is not None:
            return value
    return None


def _metric_sum_for_label(
    labeled_samples: dict[str, list[tuple[dict[str, str], float]]],
    metric_name: str,
    *,
    label_name: str,
    label_value: str,
) -> Optional[float]:
    samples = labeled_samples.get(metric_name)
    if not samples:
        return None

    values = [
        value
        for labels, value in samples
        if labels.get(label_name) == label_value and math.isfinite(value)
    ]
    if not values:
        return None
    return float(np.sum(np.asarray(values, dtype=float)))


def _metric_sum_for_label_any(
    labeled_samples: dict[str, list[tuple[dict[str, str], float]]],
    metric_names: Sequence[str],
    *,
    label_name: str,
    label_value: str,
) -> Optional[float]:
    for metric_name in metric_names:
        value = _metric_sum_for_label(
            labeled_samples,
            metric_name,
            label_name=label_name,
            label_value=label_value,
        )
        if value is not None:
            return value
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _scrape_vllm_metrics(
    *,
    run_dir: str,
    metrics_url: str,
    scrape_timeout_s: float,
) -> dict[str, Any]:
    metrics_dir = Path(run_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scraped_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    response = requests.get(metrics_url, timeout=scrape_timeout_s)
    response.raise_for_status()
    raw_metrics = response.text
    (metrics_dir / "vllm_metrics.prom").write_text(raw_metrics, encoding="utf-8")

    samples, labeled_samples = _parse_prometheus_samples(raw_metrics)
    prefix_cache_hits = _metric_sum_any(
        samples,
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_hits",
        "vllm:gpu_prefix_cache_hits_total",
        "vllm:gpu_prefix_cache_hits",
    )
    prefix_cache_queries = _metric_sum_any(
        samples,
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_queries",
        "vllm:gpu_prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_queries",
    )
    prompt_tokens_cached = _metric_sum_any(
        samples,
        "vllm:prompt_tokens_cached_total",
        "vllm:prompt_tokens_cached",
    )
    if prompt_tokens_cached is None:
        prompt_tokens_cached = _metric_sum_for_label_any(
            labeled_samples,
            ("vllm:prompt_tokens_by_source_total", "vllm:prompt_tokens_by_source"),
            label_name="source",
            label_value="cached",
        )
    prompt_tokens_recomputed = _metric_sum_any(
        samples,
        "vllm:prompt_tokens_recomputed_total",
        "vllm:prompt_tokens_recomputed",
    )
    if prompt_tokens_recomputed is None:
        prompt_tokens_recomputed = _metric_sum_for_label_any(
            labeled_samples,
            ("vllm:prompt_tokens_by_source_total", "vllm:prompt_tokens_by_source"),
            label_name="source",
            label_value="recomputed",
        )
    prompt_cache_denominator = (prompt_tokens_cached or 0.0) + (
        prompt_tokens_recomputed or 0.0
    )

    summary = {
        "metrics_scraped": True,
        "metrics_url": metrics_url,
        "scraped_at_utc": scraped_at,
        "available_metrics": sorted(samples.keys()),
        "kv_cache_usage_perc": _metric_max_any(
            samples,
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
        ),
        "prompt_tokens_cached": prompt_tokens_cached,
        "prompt_tokens_recomputed": prompt_tokens_recomputed,
        "prompt_cache_token_ratio": (
            (prompt_tokens_cached or 0.0) / prompt_cache_denominator
            if prompt_cache_denominator > 0
            else None
        ),
        "prefix_cache_hits": prefix_cache_hits,
        "prefix_cache_queries": prefix_cache_queries,
        "prefix_cache_hit_rate": (
            (prefix_cache_hits or 0.0) / prefix_cache_queries
            if prefix_cache_queries and prefix_cache_queries > 0
            else None
        ),
        "num_preemptions": _metric_sum_any(
            samples,
            "vllm:num_preemptions_total",
            "vllm:num_preemptions",
        ),
        "prompt_tokens_total": _metric_sum_any(
            samples,
            "vllm:prompt_tokens_total",
            "vllm:prompt_tokens",
        ),
        "generation_tokens_total": _metric_sum_any(
            samples,
            "vllm:generation_tokens_total",
            "vllm:generation_tokens",
        ),
    }
    _write_json(metrics_dir / "vllm_metrics_summary.json", summary)
    return summary


def _persist_vllm_metrics_error(
    *,
    run_dir: str,
    metrics_url: str,
    error: str,
) -> None:
    _write_json(
        Path(run_dir) / "metrics" / "vllm_metrics_summary.json",
        {
            "metrics_scraped": False,
            "metrics_url": metrics_url,
            "scrape_error": error,
            "scraped_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
        },
    )


def _make_vllm_metrics_hook(
    metrics_config: VllmMetricsConfig,
):
    def _hook(
        benchmark_config: BenchmarkConfig,
        server_info: dict[str, Any],
        _result: Optional[Any],
    ) -> None:
        resolved_server_config = server_info.get("config")
        server_engine = getattr(resolved_server_config, "engine", None)
        metrics_url = str(server_info.get("metrics_url") or "")

        if server_engine != "vllm":
            if metrics_config.require_metrics:
                raise RuntimeError(
                    "vLLM metrics collection is enabled, but the managed server is "
                    f"{server_engine!r} instead of 'vllm'."
                )
            return

        if not metrics_url:
            raise RuntimeError("Managed server did not provide a metrics_url.")

        try:
            _scrape_vllm_metrics(
                run_dir=benchmark_config.output_dir,
                metrics_url=metrics_url,
                scrape_timeout_s=metrics_config.scrape_timeout_s,
            )
        except Exception as exc:
            _persist_vllm_metrics_error(
                run_dir=benchmark_config.output_dir,
                metrics_url=metrics_url,
                error=str(exc),
            )
            if metrics_config.require_metrics:
                raise

    return _hook


def _run_one_workload(
    *,
    benchmark_config: BenchmarkConfig,
    workload: str,
    rate: float,
    search_output_dir: str,
    vllm_metrics: VllmMetricsConfig,
) -> BenchmarkRunSummary:
    run_cfg = replace(
        benchmark_config,
        output_dir=_benchmark_output_base(
            search_output_dir,
            rate=rate,
            workload=workload,
        ),
    )
    run_cfg = patch_traffic_knob(run_cfg, value=rate)

    logger.info("Running %s workload at rate=%s", workload, rate)
    manage_benchmark_run(
        run_cfg,
        server_post_run_hook=(
            _make_vllm_metrics_hook(vllm_metrics) if vllm_metrics.enabled else None
        ),
    )
    logger.info(
        "%s workload complete at rate=%s -> %s",
        workload,
        rate,
        run_cfg.output_dir,
    )

    return summarize_run(
        workload=workload,
        rate=rate,
        run_dir=run_cfg.output_dir,
    )


def _run_paired_rate(
    *,
    rate: float,
    phase: str,
    search_index: int,
    config: WorkloadShapeSearchConfig,
    linear_cfg: BenchmarkConfig,
    dag_cfg: BenchmarkConfig,
) -> PairedRateResult:
    if config.rate_search.alternate_order and search_index % 2 == 1:
        run_order = ["dag", "linear"]
    else:
        run_order = ["linear", "dag"]

    summaries: dict[str, BenchmarkRunSummary] = {}
    for workload in run_order:
        benchmark_cfg = linear_cfg if workload == "linear" else dag_cfg
        summaries[workload] = _run_one_workload(
            benchmark_config=benchmark_cfg,
            workload=workload,
            rate=rate,
            search_output_dir=config.output_dir,
            vllm_metrics=config.vllm_metrics,
        )

    paired = score_paired_candidate(
        linear=summaries["linear"],
        dag=summaries["dag"],
        guardrails=config.guardrails,
        objective=config.objective,
    )
    return replace(paired, phase=phase, run_order=run_order)


def _initial_rates(params: RateSearchParams) -> list[float]:
    rates: list[float] = []
    rate = params.round_value(params.start_value)
    while rate <= params.max_value:
        rates.append(rate)
        next_rate = params.round_value(rate * params.expansion_factor)
        if next_rate <= rate:
            next_rate = params.round_value(rate + (1 / (10**params.precision)))
        rate = next_rate
    return rates


def _lower_rates(params: RateSearchParams) -> list[float]:
    rates: list[float] = []
    floor = params.round_value(params.min_value)
    rate = params.round_value(params.start_value / params.expansion_factor)
    while rate >= floor:
        rates.append(rate)
        next_rate = params.round_value(rate / params.expansion_factor)
        if next_rate >= rate:
            next_rate = params.round_value(rate - (1 / (10**params.precision)))
        rate = next_rate
    return rates


def run_workload_shape_search(config: WorkloadShapeSearchConfig) -> dict[str, Any]:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    _ensure_trace_bundle(config)

    linear_cfg = _load_benchmark_config(config.linear_benchmark_config)
    dag_cfg = _load_benchmark_config(config.dag_benchmark_config)

    results_by_rate: dict[float, PairedRateResult] = {}
    evaluation_index = 0
    seen_healthy = False
    consecutive_unhealthy = 0

    for rate in _initial_rates(config.rate_search):
        if rate in results_by_rate:
            continue
        result = _run_paired_rate(
            rate=rate,
            phase="coarse",
            search_index=evaluation_index,
            config=config,
            linear_cfg=linear_cfg,
            dag_cfg=dag_cfg,
        )
        results_by_rate[rate] = result
        evaluation_index += 1
        _persist_search_state(config=config, results=list(results_by_rate.values()))

        if result.healthy:
            seen_healthy = True
            consecutive_unhealthy = 0
        elif seen_healthy:
            consecutive_unhealthy += 1
            if (
                consecutive_unhealthy
                >= config.rate_search.stop_after_consecutive_unhealthy
            ):
                break

    if not seen_healthy:
        for rate in _lower_rates(config.rate_search):
            if rate in results_by_rate:
                continue
            result = _run_paired_rate(
                rate=rate,
                phase="coarse",
                search_index=evaluation_index,
                config=config,
                linear_cfg=linear_cfg,
                dag_cfg=dag_cfg,
            )
            results_by_rate[rate] = result
            evaluation_index += 1
            _persist_search_state(
                config=config, results=list(results_by_rate.values())
            )

            if result.healthy:
                seen_healthy = True
                logger.info(
                    "Recovered a healthy starting point by backing off to rate=%s",
                    rate,
                )
                break

    for _ in range(config.rate_search.refinement_rounds):
        best = _best_candidate(results_by_rate.values())
        if best is None:
            break

        sorted_rates = sorted(results_by_rate)
        best_index = sorted_rates.index(best.rate)

        candidate_rates: list[float] = []
        if best_index > 0:
            lower = sorted_rates[best_index - 1]
            midpoint = _geometric_midpoint(
                lower,
                best.rate,
                precision=config.rate_search.precision,
            )
            if midpoint is not None:
                candidate_rates.append(midpoint)
        if best_index < len(sorted_rates) - 1:
            upper = sorted_rates[best_index + 1]
            midpoint = _geometric_midpoint(
                best.rate,
                upper,
                precision=config.rate_search.precision,
            )
            if midpoint is not None:
                candidate_rates.append(midpoint)

        candidate_rates = [
            rate
            for rate in candidate_rates
            if rate not in results_by_rate and rate <= config.rate_search.max_value
        ]
        if not candidate_rates:
            break

        for rate in candidate_rates:
            result = _run_paired_rate(
                rate=rate,
                phase="refine",
                search_index=evaluation_index,
                config=config,
                linear_cfg=linear_cfg,
                dag_cfg=dag_cfg,
            )
            results_by_rate[rate] = result
            evaluation_index += 1
            _persist_search_state(config=config, results=list(results_by_rate.values()))

    final_results = list(results_by_rate.values())
    final_results.sort(key=lambda item: item.rate)
    best = _best_candidate(final_results)
    _persist_search_state(config=config, results=final_results)

    logger.info(
        "Search complete. Best healthy rate: %s",
        best.rate if best is not None else "none",
    )
    return {
        "best_rate": best.rate if best is not None else None,
        "best_result": best.to_flat_dict() if best is not None else None,
        "num_results": len(final_results),
        "results": [result.to_flat_dict() for result in final_results],
    }


def run_single_workload_shape_rate(
    config: WorkloadShapeSearchConfig,
    *,
    rate: float,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    effective_config = replace(
        config,
        output_dir=_resolve_output_path(output_dir) if output_dir else config.output_dir,
    )
    Path(effective_config.output_dir).mkdir(parents=True, exist_ok=True)
    _ensure_trace_bundle(effective_config)

    linear_cfg = _load_benchmark_config(effective_config.linear_benchmark_config)
    dag_cfg = _load_benchmark_config(effective_config.dag_benchmark_config)

    result = _run_paired_rate(
        rate=effective_config.rate_search.round_value(rate),
        phase="single",
        search_index=0,
        config=effective_config,
        linear_cfg=linear_cfg,
        dag_cfg=dag_cfg,
    )
    _persist_search_state(config=effective_config, results=[result])

    logger.info(
        "Single paired run complete at rate=%s. Healthy=%s, status=%s",
        result.rate,
        result.healthy,
        result.status,
    )
    return {
        "rate": result.rate,
        "result": result.to_flat_dict(),
        "output_dir": effective_config.output_dir,
    }


def rescore_existing_workload_shape_search(
    config: WorkloadShapeSearchConfig,
    *,
    source_output_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    source_dir = source_output_dir or config.output_dir
    rescored_output_dir = output_dir or str(Path(source_dir) / "rescored")
    paired_runs = _load_existing_paired_runs(source_dir)

    rescored_results: list[PairedRateResult] = []
    for paired_run in paired_runs:
        rate = float(paired_run["rate"])
        linear_summary = summarize_run(
            workload="linear",
            rate=rate,
            run_dir=str(paired_run["linear_run_dir"]),
        )
        dag_summary = summarize_run(
            workload="dag",
            rate=rate,
            run_dir=str(paired_run["dag_run_dir"]),
        )
        rescored = score_paired_candidate(
            linear=linear_summary,
            dag=dag_summary,
            guardrails=config.guardrails,
            objective=config.objective,
        )
        rescored_results.append(
            replace(
                rescored,
                phase=str(paired_run.get("phase", "")),
                run_order=list(paired_run.get("run_order", [])),
            )
        )

    rescored_results.sort(key=lambda item: item.rate)
    persist_config = replace(config, output_dir=rescored_output_dir)
    _persist_search_state(config=persist_config, results=rescored_results)
    best = _best_candidate(rescored_results)

    logger.info(
        "Rescored %s existing rates from %s. Best healthy rate: %s",
        len(rescored_results),
        source_dir,
        best.rate if best is not None else "none",
    )
    return {
        "source_output_dir": source_dir,
        "rescored_output_dir": rescored_output_dir,
        "best_rate": best.rate if best is not None else None,
        "best_result": best.to_flat_dict() if best is not None else None,
        "num_results": len(rescored_results),
        "results": [result.to_flat_dict() for result in rescored_results],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the workload-shape case-study paired search against managed "
            "Veeksha benchmark configs."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the paired search YAML config.",
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Re-score existing benchmark runs without launching new servers.",
    )
    parser.add_argument(
        "--source-output-dir",
        help=(
            "Existing search output directory to rescore. "
            "Defaults to the config output_dir."
        ),
    )
    parser.add_argument(
        "--rescore-output-dir",
        help=(
            "Directory where rescored summary files should be written. "
            "Defaults to <source-output-dir>/rescored."
        ),
    )
    parser.add_argument(
        "--single-rate",
        type=float,
        help=(
            "Run exactly one paired rate with fresh managed servers, instead of "
            "performing a search."
        ),
    )
    parser.add_argument(
        "--single-rate-output-dir",
        help=(
            "Output directory for --single-rate runs. Defaults to the config "
            "output_dir."
        ),
    )
    args = parser.parse_args(argv)

    config = _load_search_config(args.config)
    if args.rescore_only and args.single_rate is not None:
        raise ValueError("--rescore-only and --single-rate cannot be used together.")
    if args.rescore_only:
        rescore_existing_workload_shape_search(
            config,
            source_output_dir=args.source_output_dir,
            output_dir=args.rescore_output_dir,
        )
    elif args.single_rate is not None:
        run_single_workload_shape_rate(
            config,
            rate=args.single_rate,
            output_dir=args.single_rate_output_dir,
        )
    else:
        run_workload_shape_search(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
