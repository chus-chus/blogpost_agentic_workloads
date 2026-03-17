"""case study 3 driver."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml
from vidhi import load_yaml_config

from veeksha.benchmark import manage_benchmark_run
from veeksha.capacity_search import _adaptive_capacity_search, patch_traffic_knob
from veeksha.case_studies.workload_shape_search import (
    BenchmarkRunSummary,
    VllmMetricsConfig,
    _load_benchmark_config,
    _make_vllm_metrics_hook,
    _resolve_input_path,
    _resolve_output_path,
    summarize_run,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)

RESULTS_JSON_NAME = "workload_deployment_claim_results.json"
RESULTS_CSV_NAME = "workload_deployment_claim_results.csv"
WORKLOAD_CHOICES = ("linear", "dag")


@dataclass(frozen=True)
class Guardrails:
    min_completed_requests: int = 150
    min_completion_ratio: float = 0.97
    max_error_rate: float = 0.02
    max_ttfc_p95_s: float = 0.75
    max_e2e_p95_s: Optional[float] = None
    max_tbc_p95_s: float = 0.05
    require_all_slos_met: bool = True


@dataclass(frozen=True)
class RateModelWorkload:
    requests_per_session: int
    fresh_input_tokens_per_session: Optional[float] = None
    output_tokens_per_session: Optional[float] = None


@dataclass(frozen=True)
class RateModel:
    rate_basis: str
    fresh_input_tokens_per_request: float
    output_tokens_per_request: float
    workloads: dict[str, RateModelWorkload]


@dataclass(frozen=True)
class WorkloadDeploymentClaimConfig:
    output_dir: str
    linear_benchmark_config: str
    dag_benchmark_config: str
    trace_metadata_path: str
    rate_basis: str = "request_rate"
    start_value: float = 1.0
    max_value: float = 4.0
    expansion_factor: float = 2.0
    max_iterations: int = 10
    precision: int = 2
    guardrails: Guardrails = Guardrails()
    vllm_metrics: VllmMetricsConfig = VllmMetricsConfig()
    gpu_hour_price_usd: Optional[float] = None

    def round_value(self, value: float) -> float:
        scale = 10**self.precision
        return round(value * scale) / scale


@dataclass(frozen=True)
class WorkloadSearchResult:
    workload: str
    normalized_request_rate: float
    derived_session_rate: float
    fresh_input_tokens_per_s: float
    requested_output_tokens_per_s: float
    run: BenchmarkRunSummary
    healthy: bool
    status: str
    notes: list[str]

    def to_flat_dict(self) -> dict[str, Any]:
        row = {
            "workload": self.workload,
            "normalized_request_rate": self.normalized_request_rate,
            "derived_session_rate": self.derived_session_rate,
            "fresh_input_tokens_per_s": self.fresh_input_tokens_per_s,
            "requested_output_tokens_per_s": self.requested_output_tokens_per_s,
            "healthy": self.healthy,
            "status": self.status,
            "notes": "; ".join(self.notes),
        }
        row.update(self.run.to_prefixed_dict("run"))
        return row


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _load_config(config_path: str) -> WorkloadDeploymentClaimConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    raw = load_yaml_config(str(resolved_config_path))
    if not isinstance(raw, dict):
        raise ValueError(f"Search config {resolved_config_path} must resolve to a YAML mapping.")

    base_dir = resolved_config_path.parent
    guardrails_raw = dict(raw.get("guardrails") or {})
    vllm_metrics_raw = dict(raw.get("vllm_metrics") or {})

    return WorkloadDeploymentClaimConfig(
        output_dir=_resolve_output_path(str(raw.get("output_dir", "benchmark_output"))),
        linear_benchmark_config=_resolve_input_path(str(raw["linear_benchmark_config"]), base_dir=base_dir),
        dag_benchmark_config=_resolve_input_path(str(raw["dag_benchmark_config"]), base_dir=base_dir),
        trace_metadata_path=_resolve_input_path(str(raw["trace_metadata_path"]), base_dir=base_dir),
        rate_basis=str(raw.get("rate_basis", "request_rate")),
        start_value=float(raw.get("start_value", 1.0)),
        max_value=float(raw.get("max_value", 4.0)),
        expansion_factor=float(raw.get("expansion_factor", 2.0)),
        max_iterations=int(raw.get("max_iterations", 10)),
        precision=int(raw.get("precision", 2)),
        guardrails=Guardrails(
            min_completed_requests=int(guardrails_raw.get("min_completed_requests", 150)),
            min_completion_ratio=float(guardrails_raw.get("min_completion_ratio", 0.97)),
            max_error_rate=float(guardrails_raw.get("max_error_rate", 0.02)),
            max_ttfc_p95_s=float(
                guardrails_raw.get("max_ttfc_p95_s", guardrails_raw.get("max_ttfc_p99_s", 0.75))
            ),
            max_e2e_p95_s=_optional_float(guardrails_raw.get("max_e2e_p95_s")),
            max_tbc_p95_s=float(
                guardrails_raw.get("max_tbc_p95_s", guardrails_raw.get("max_tbc_p99_s", 0.05))
            ),
            require_all_slos_met=bool(guardrails_raw.get("require_all_slos_met", True)),
        ),
        vllm_metrics=VllmMetricsConfig(
            enabled=bool(vllm_metrics_raw.get("enabled", True)),
            require_metrics=bool(vllm_metrics_raw.get("require_metrics", True)),
            scrape_timeout_s=float(vllm_metrics_raw.get("scrape_timeout_s", 10.0)),
        ),
        gpu_hour_price_usd=(
            float(raw["gpu_hour_price_usd"])
            if raw.get("gpu_hour_price_usd") is not None
            else None
        ),
    )


def _load_rate_model(metadata_path: str) -> RateModel:
    payload = _read_json(Path(metadata_path))
    raw_rate_model = payload.get("rate_model")
    if not isinstance(raw_rate_model, dict):
        raise ValueError("Trace metadata is missing a rate_model object.")

    workloads_raw = raw_rate_model.get("workloads")
    if not isinstance(workloads_raw, dict):
        raise ValueError("Trace metadata is missing rate_model.workloads.")

    workloads: dict[str, RateModelWorkload] = {}
    for workload in WORKLOAD_CHOICES:
        workload_raw = workloads_raw.get(workload)
        if not isinstance(workload_raw, Mapping):
            raise ValueError(f"Trace metadata is missing rate_model.workloads.{workload}.")
        workloads[workload] = RateModelWorkload(
            requests_per_session=int(workload_raw["requests_per_session"]),
            fresh_input_tokens_per_session=(
                float(workload_raw["fresh_input_tokens_per_session"])
                if workload_raw.get("fresh_input_tokens_per_session") is not None
                else None
            ),
            output_tokens_per_session=(
                float(workload_raw["output_tokens_per_session"])
                if workload_raw.get("output_tokens_per_session") is not None
                else None
            ),
        )

    rate_model = RateModel(
        rate_basis=str(raw_rate_model.get("rate_basis", "request_rate")),
        fresh_input_tokens_per_request=float(raw_rate_model["fresh_input_tokens_per_request"]),
        output_tokens_per_request=float(raw_rate_model["output_tokens_per_request"]),
        workloads=workloads,
    )
    if rate_model.rate_basis != "request_rate":
        raise ValueError(
            f"Unsupported rate basis {rate_model.rate_basis!r}. Only 'request_rate' is supported."
        )
    return rate_model


def _rate_summary_for_workload(
    rate_model: RateModel,
    *,
    workload: str,
    normalized_request_rate: float,
) -> dict[str, float]:
    workload_spec = rate_model.workloads[workload]
    derived_session_rate = normalized_request_rate / workload_spec.requests_per_session
    return {
        "derived_session_rate": derived_session_rate,
        "fresh_input_tokens_per_s": rate_model.fresh_input_tokens_per_request * normalized_request_rate,
        "requested_output_tokens_per_s": rate_model.output_tokens_per_request * normalized_request_rate,
    }


def _evaluate_guardrails(
    summary: BenchmarkRunSummary,
    guardrails: Guardrails,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    healthy = True

    if summary.completed_requests < guardrails.min_completed_requests:
        notes.append(f"completed_requests<{guardrails.min_completed_requests}")
        healthy = False
    if summary.completion_ratio < guardrails.min_completion_ratio:
        notes.append(f"completion_ratio<{guardrails.min_completion_ratio:.3f}")
        healthy = False
    if summary.error_rate > guardrails.max_error_rate:
        notes.append(f"error_rate>{guardrails.max_error_rate:.3f}")
        healthy = False
    if summary.ttfc_p95_s is None or summary.ttfc_p95_s > guardrails.max_ttfc_p95_s:
        notes.append(f"ttfc_p95>{guardrails.max_ttfc_p95_s:.3f}s")
        healthy = False
    if guardrails.max_e2e_p95_s is not None:
        if summary.e2e_p95_s is None or summary.e2e_p95_s > guardrails.max_e2e_p95_s:
            notes.append(f"e2e_p95>{guardrails.max_e2e_p95_s:.3f}s")
            healthy = False
    if (
        summary.decode_window_tbc_p95_s is None
        or summary.decode_window_tbc_p95_s > guardrails.max_tbc_p95_s
    ):
        notes.append(f"tbc_p95>{guardrails.max_tbc_p95_s:.3f}s")
        healthy = False
    if guardrails.require_all_slos_met and summary.all_slos_met is not True:
        notes.append("benchmark_slos_not_met")
        healthy = False

    return healthy, notes


def _resolve_mode_output_dir(
    base_output_dir: str,
    *,
    mode: str,
    workload: str,
    rate: Optional[float] = None,
) -> str:
    base = Path(base_output_dir)
    if mode == "search":
        return str((base / f"search_{workload}").resolve())
    if mode == "single_rate":
        if rate is None:
            raise ValueError("single_rate output dir resolution requires a rate.")
        safe_rate = str(rate).replace(".", "_")
        return str((base / f"single_rate_{workload}" / f"rate_{safe_rate}").resolve())
    raise ValueError(f"Unsupported mode {mode!r}.")


def _benchmark_output_base(
    search_output_dir: str,
    *,
    normalized_request_rate: float,
    workload: str,
) -> str:
    safe_rate = str(normalized_request_rate).replace(".", "_")
    return str(Path(search_output_dir) / "runs" / f"rate_{safe_rate}" / workload)


def _run_single_rate(
    *,
    config: WorkloadDeploymentClaimConfig,
    rate_model: RateModel,
    benchmark_config_path: str,
    workload: str,
    normalized_request_rate: float,
    output_dir: str,
) -> WorkloadSearchResult:
    benchmark_config = _load_benchmark_config(benchmark_config_path)
    rate_summary = _rate_summary_for_workload(
        rate_model,
        workload=workload,
        normalized_request_rate=normalized_request_rate,
    )

    run_cfg = replace(
        benchmark_config,
        output_dir=_benchmark_output_base(
            output_dir,
            normalized_request_rate=normalized_request_rate,
            workload=workload,
        ),
    )
    run_cfg = patch_traffic_knob(
        run_cfg,
        value=rate_summary["derived_session_rate"],
    )

    logger.info(
        "Running %s workload at rho=%s req/s (session_rate=%s)",
        workload,
        normalized_request_rate,
        rate_summary["derived_session_rate"],
    )
    manage_benchmark_run(
        run_cfg,
        server_post_run_hook=(
            _make_vllm_metrics_hook(config.vllm_metrics)
            if config.vllm_metrics.enabled
            else None
        ),
    )
    summary = summarize_run(
        workload=workload,
        rate=rate_summary["derived_session_rate"],
        run_dir=run_cfg.output_dir,
    )
    healthy, notes = _evaluate_guardrails(summary, config.guardrails)

    return WorkloadSearchResult(
        workload=workload,
        normalized_request_rate=normalized_request_rate,
        derived_session_rate=rate_summary["derived_session_rate"],
        fresh_input_tokens_per_s=rate_summary["fresh_input_tokens_per_s"],
        requested_output_tokens_per_s=rate_summary["requested_output_tokens_per_s"],
        run=summary,
        healthy=healthy,
        status="healthy" if healthy else "guardrail_failed",
        notes=notes,
    )


def _best_healthy_result(
    results: Sequence[WorkloadSearchResult],
) -> Optional[WorkloadSearchResult]:
    healthy = [result for result in results if result.healthy]
    if not healthy:
        return None
    return max(healthy, key=lambda result: result.normalized_request_rate)


def _persist_results(
    *,
    config: WorkloadDeploymentClaimConfig,
    rate_model: RateModel,
    workload: str,
    mode: str,
    output_dir: str,
    results: Sequence[WorkloadSearchResult],
    requested_normalized_request_rate: Optional[float],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(results, key=lambda item: item.normalized_request_rate)
    best = _best_healthy_result(sorted_results)

    payload = {
        "mode": mode,
        "workload": workload,
        "search_config": asdict(config),
        "rate_model": asdict(rate_model),
        "gpu_hour_price_usd": config.gpu_hour_price_usd,
        "requested_normalized_request_rate": requested_normalized_request_rate,
        "best_normalized_request_rate": (
            best.normalized_request_rate if best is not None else None
        ),
        "best_result": (
            {**best.to_flat_dict(), "notes": best.notes}
            if best is not None
            else None
        ),
        "num_evaluated_rates": len(sorted_results),
        "results": [
            {**result.to_flat_dict(), "notes": result.notes}
            for result in sorted_results
        ],
    }
    _write_json(output_path / RESULTS_JSON_NAME, payload)

    if sorted_results:
        fieldnames = list(sorted(sorted_results[0].to_flat_dict().keys()))
        with (output_path / RESULTS_CSV_NAME).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in sorted_results:
                writer.writerow(result.to_flat_dict())

    with (output_path / "resolved_search_config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False, default_flow_style=False)


def run_workload_search(
    config: WorkloadDeploymentClaimConfig,
    *,
    workload: str,
) -> dict[str, Any]:
    if workload not in WORKLOAD_CHOICES:
        raise ValueError(f"Unsupported workload {workload!r}.")

    rate_model = _load_rate_model(config.trace_metadata_path)
    if config.rate_basis != rate_model.rate_basis:
        raise ValueError(
            f"Config rate_basis={config.rate_basis!r} does not match trace metadata "
            f"rate_basis={rate_model.rate_basis!r}."
        )

    output_dir = _resolve_mode_output_dir(config.output_dir, mode="search", workload=workload)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    benchmark_config_path = (
        config.linear_benchmark_config if workload == "linear" else config.dag_benchmark_config
    )

    results_by_rate: dict[float, WorkloadSearchResult] = {}

    def is_passing(normalized_request_rate: float) -> bool:
        rounded_rate = config.round_value(normalized_request_rate)
        if rounded_rate not in results_by_rate:
            results_by_rate[rounded_rate] = _run_single_rate(
                config=config,
                rate_model=rate_model,
                benchmark_config_path=benchmark_config_path,
                workload=workload,
                normalized_request_rate=rounded_rate,
                output_dir=output_dir,
            )
            _persist_results(
                config=config,
                rate_model=rate_model,
                workload=workload,
                mode="search",
                output_dir=output_dir,
                results=list(results_by_rate.values()),
                requested_normalized_request_rate=None,
            )
        return results_by_rate[rounded_rate].healthy

    best_rate, iterations_used = _adaptive_capacity_search(
        start_value=config.start_value,
        max_value=config.max_value,
        expansion_factor=config.expansion_factor,
        is_passing=is_passing,
        max_iterations=config.max_iterations,
        precision=config.precision,
        integer_domain=False,
    )
    _persist_results(
        config=config,
        rate_model=rate_model,
        workload=workload,
        mode="search",
        output_dir=output_dir,
        results=list(results_by_rate.values()),
        requested_normalized_request_rate=None,
    )
    best_result = (
        {**results_by_rate[best_rate].to_flat_dict(), "notes": results_by_rate[best_rate].notes}
        if best_rate is not None and best_rate in results_by_rate
        else None
    )
    return {
        "mode": "search",
        "workload": workload,
        "output_dir": output_dir,
        "best_normalized_request_rate": best_rate,
        "best_result": best_result,
        "iterations_used": iterations_used,
        "num_results": len(results_by_rate),
        "results": [
            {**result.to_flat_dict(), "notes": result.notes}
            for result in sorted(results_by_rate.values(), key=lambda item: item.normalized_request_rate)
        ],
    }


def run_single_workload_rate(
    config: WorkloadDeploymentClaimConfig,
    *,
    workload: str,
    normalized_request_rate: float,
) -> dict[str, Any]:
    if workload not in WORKLOAD_CHOICES:
        raise ValueError(f"Unsupported workload {workload!r}.")

    rounded_rate = config.round_value(normalized_request_rate)
    rate_model = _load_rate_model(config.trace_metadata_path)
    if config.rate_basis != rate_model.rate_basis:
        raise ValueError(
            f"Config rate_basis={config.rate_basis!r} does not match trace metadata "
            f"rate_basis={rate_model.rate_basis!r}."
        )

    output_dir = _resolve_mode_output_dir(
        config.output_dir,
        mode="single_rate",
        workload=workload,
        rate=rounded_rate,
    )
    benchmark_config_path = (
        config.linear_benchmark_config if workload == "linear" else config.dag_benchmark_config
    )
    result = _run_single_rate(
        config=config,
        rate_model=rate_model,
        benchmark_config_path=benchmark_config_path,
        workload=workload,
        normalized_request_rate=rounded_rate,
        output_dir=output_dir,
    )
    _persist_results(
        config=config,
        rate_model=rate_model,
        workload=workload,
        mode="single_rate",
        output_dir=output_dir,
        results=[result],
        requested_normalized_request_rate=rounded_rate,
    )
    return {
        "mode": "single_rate",
        "workload": workload,
        "output_dir": output_dir,
        "requested_normalized_request_rate": rounded_rate,
        "result": {**result.to_flat_dict(), "notes": result.notes},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the rate-normalized deployment-claim workload study."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the deployment-claim search YAML config.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument(
        "--workload",
        required=True,
        choices=WORKLOAD_CHOICES,
    )

    single_rate_parser = subparsers.add_parser("single-rate")
    single_rate_parser.add_argument(
        "--workload",
        required=True,
        choices=WORKLOAD_CHOICES,
    )
    single_rate_parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Normalized request rate in requests per second.",
    )

    args = parser.parse_args(argv)
    config = _load_config(args.config)

    if args.command == "search":
        run_workload_search(config, workload=args.workload)
    elif args.command == "single-rate":
        run_single_workload_rate(
            config,
            workload=args.workload,
            normalized_request_rate=args.rate,
        )
    else:
        raise ValueError(f"Unsupported command {args.command!r}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
