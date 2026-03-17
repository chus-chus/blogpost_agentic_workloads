#!/usr/bin/env -S uv run
"""Generate case-study-3 frontier comparison figures in the house Altair style."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "altair>=5.0.0",
#   "numpy>=2.0.0",
#   "pillow>=11.0.0",
#   "vl-convert-python>=1.7.0",
# ]
# ///

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINEAR_SEARCH_DIR = ROOT / "workload_deployment_claim_case_study" / "qwen" / "search_linear"
DEFAULT_DAG_SEARCH_DIR = ROOT / "workload_deployment_claim_case_study" / "qwen" / "search_dag"
DEFAULT_OUTPUT_DIR = ROOT / "workload_deployment_claim_case_study" / "qwen" / "analysis"

PAPER_HEX = "#FBF7EF"
PAPER_RGB = np.array([251, 247, 239], dtype=np.int16)
TEXT_COLOR = "#5E5A55"
TITLE_COLOR = "#3F3A36"
GRID_COLOR = "#E9E1D5"
TICK_COLOR = "#D9D0C3"
LINEAR_COLOR = "#355C7D"
DAG_COLOR = "#E7865A"
LEGEND_LABEL_COLOR = "#4A4540"
PNG_SCALE_FACTOR = 2


@dataclass(frozen=True)
class FigurePaths:
    png: Path
    svg: Path
    transparent_png: Path
    transparent_svg: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a three-panel frontier comparison chart for the deployment-claim "
            "case study using the best linear and DAG search results."
        )
    )
    parser.add_argument("--linear-search-dir", type=Path, default=DEFAULT_LINEAR_SEARCH_DIR)
    parser.add_argument("--dag-search-dir", type=Path, default=DEFAULT_DAG_SEARCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="frontier_metric_facets")
    return parser.parse_args()


def make_figure_paths(output_dir: Path, stem: str) -> FigurePaths:
    return FigurePaths(
        png=output_dir / f"{stem}.png",
        svg=output_dir / f"{stem}.svg",
        transparent_png=output_dir / f"{stem}_transparent.png",
        transparent_svg=output_dir / f"{stem}_transparent.svg",
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def extract_best_result(search_dir: Path) -> dict[str, Any]:
    payload = read_json(search_dir / "workload_deployment_claim_results.json")
    best = payload.get("best_result")
    if not isinstance(best, dict):
        raise ValueError(f"Missing best_result in {search_dir}.")
    return best


def rate_dir_fragment(rate: float) -> str:
    return str(rate).replace(".", "_")


def find_run_dir(search_dir: Path, workload: str, normalized_rate: float) -> Path:
    rate_dir = search_dir / "runs" / f"rate_{rate_dir_fragment(normalized_rate)}" / workload
    candidates = sorted(path for path in rate_dir.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one run directory under {rate_dir}, found {len(candidates)}.")
    return candidates[0]


def round_up(value: float, step: float) -> float:
    return float(np.ceil(value / step) * step)


def workload_color(show_legend: bool) -> alt.Color:
    return alt.Color(
        "workload:N",
        scale=alt.Scale(
            domain=["Linear workload", "DAG workload"],
            range=[LINEAR_COLOR, DAG_COLOR],
        ),
        legend=alt.Legend(
            title=None,
            orient="top",
            direction="horizontal",
            columns=2,
            labelColor=LEGEND_LABEL_COLOR,
            labelFontSize=15,
            symbolType="stroke",
            symbolStrokeWidth=3,
            symbolSize=180,
            offset=8,
        )
        if show_legend
        else None,
    )


def base_cdf_axis() -> alt.Axis:
    return alt.Axis(
        values=[0.0, 0.25, 0.5, 0.75, 1.0],
        format=".0%",
        labelColor=TEXT_COLOR,
        titleColor=TITLE_COLOR,
        tickColor=TICK_COLOR,
        gridColor=GRID_COLOR,
        domain=False,
        labelFontSize=15,
        titleFontSize=17,
        titleFontWeight="bold",
        labelPadding=6,
        titlePadding=10,
    )


def cdf_axis(show_axis: bool) -> alt.Axis:
    if show_axis:
        return base_cdf_axis()
    return alt.Axis(
        values=[0.0, 0.25, 0.5, 0.75, 1.0],
        labels=False,
        ticks=False,
        domain=False,
        grid=True,
        gridColor=GRID_COLOR,
        title=None,
    )


def build_ecdf_rows(csv_path: Path, workload: str, value_column: str, scale: float = 1.0) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cdf = row.get("cdf")
            value = row.get(value_column)
            if cdf is None or value is None:
                continue
            rows.append(
                {
                    "cdf": float(cdf),
                    "value": float(value) * scale,
                    "workload": workload,
                }
            )
    if not rows:
        raise ValueError(f"No ECDF rows found in {csv_path}.")
    return rows


def build_ecdf_panel(
    *,
    rows: list[dict[str, float | str]],
    title: str,
    x_title: str,
    x_domain: list[float],
    x_ticks: list[Any],
    x_format: str | None,
    x_label_expr: str | None = None,
    show_legend: bool,
    show_y_axis: bool,
) -> alt.Chart:
    axis_kwargs: dict[str, Any] = {
        "values": x_ticks,
        "format": x_format,
        "labelColor": TEXT_COLOR,
        "titleColor": TITLE_COLOR,
        "tickColor": TICK_COLOR,
        "grid": False,
        "labelFlush": False,
        "labelFontSize": 15,
        "titleFontSize": 17,
        "titleFontWeight": "bold",
        "labelPadding": 6,
        "titlePadding": 10,
    }
    if x_label_expr is not None:
        axis_kwargs["labelExpr"] = x_label_expr

    return (
        alt.Chart(alt.InlineData(values=rows))
        .mark_line(strokeWidth=2.8, clip=True)
        .encode(
            x=alt.X(
                "value:Q",
                title=x_title,
                axis=alt.Axis(**axis_kwargs),
                scale=alt.Scale(domain=x_domain, nice=False),
            ),
            y=alt.Y(
                "cdf:Q",
                title="Share of requests" if show_y_axis else None,
                axis=cdf_axis(show_y_axis),
                scale=alt.Scale(domain=[0.0, 1.0], nice=False),
            ),
            color=workload_color(show_legend),
        )
        .properties(width=255, height=245, title=alt.TitleParams(title, color=TITLE_COLOR, fontSize=18))
    )


def trim_rows_to_domain(
    rows: list[dict[str, float | str]],
    *,
    max_value: float,
) -> list[dict[str, float | str]]:
    trimmed = [row for row in rows if float(row["value"]) <= max_value]
    if not trimmed:
        raise ValueError(f"No rows fall within the requested domain <= {max_value}.")

    last = trimmed[-1]
    if float(last["value"]) < max_value:
        trimmed.append(
            {
                "value": max_value,
                "cdf": float(last["cdf"]),
                "workload": str(last["workload"]),
            }
        )
    return trimmed


def build_reuse_panel(rows: list[dict[str, float | str]]) -> alt.Chart:
    return (
        alt.Chart(alt.InlineData(values=rows))
        .mark_bar(size=34)
        .encode(
            x=alt.X(
                "metric_label:N",
                title=None,
                sort=["Prompt reuse", "Prefix-cache hits"],
                axis=alt.Axis(
                    labelColor=TEXT_COLOR,
                    titleColor=TITLE_COLOR,
                    tickColor=TICK_COLOR,
                    grid=False,
                    labelFontSize=15,
                    labelPadding=10,
                    labelAngle=0,
                ),
            ),
            xOffset="workload:N",
            y=alt.Y(
                "value:Q",
                title="Share",
                axis=alt.Axis(
                    values=[0.0, 0.25, 0.5, 0.75, 1.0],
                    format=".0%",
                    labelColor=TEXT_COLOR,
                    titleColor=TITLE_COLOR,
                    tickColor=TICK_COLOR,
                    gridColor=GRID_COLOR,
                    domain=False,
                    labelFontSize=15,
                    titleFontSize=17,
                    titleFontWeight="bold",
                    labelPadding=6,
                    titlePadding=10,
                ),
                scale=alt.Scale(domain=[0.0, 1.0], nice=False),
            ),
            color=workload_color(False),
            tooltip=[
                alt.Tooltip("workload:N", title="Workload"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=".3f"),
            ],
        )
        .properties(width=255, height=245, title=alt.TitleParams("Reuse", color=TITLE_COLOR, fontSize=18))
    )


def build_chart(
    *,
    linear_ttfc_rows: list[dict[str, float | str]],
    dag_ttfc_rows: list[dict[str, float | str]],
    linear_tbc_rows: list[dict[str, float | str]],
    dag_tbc_rows: list[dict[str, float | str]],
    reuse_rows: list[dict[str, float | str]],
) -> alt.TopLevelMixin:
    ttfc_rows = linear_ttfc_rows + dag_ttfc_rows
    raw_tbc_rows = linear_tbc_rows + dag_tbc_rows

    ttfc_max = round_up(max(float(row["value"]) for row in ttfc_rows), 0.1)
    tbc_max = round_up(
        max(float(row["value"]) for row in raw_tbc_rows if float(row["cdf"]) <= 0.99),
        10.0,
    )
    tbc_rows = trim_rows_to_domain(linear_tbc_rows, max_value=tbc_max) + trim_rows_to_domain(
        dag_tbc_rows,
        max_value=tbc_max,
    )

    ttfc_panel = build_ecdf_panel(
        rows=ttfc_rows,
        title="TTFC",
        x_title="Seconds",
        x_domain=[0.0, ttfc_max],
        x_ticks=[round(tick, 1) for tick in np.arange(0.0, ttfc_max + 0.001, 0.1)],
        x_format=".1f",
        x_label_expr="datum.value === 0 ? '0' : replace(format(datum.value, '.1f'), /^0\\./, '.')",
        show_legend=True,
        show_y_axis=True,
    )
    tbc_panel = build_ecdf_panel(
        rows=tbc_rows,
        title="TBC",
        x_title="Milliseconds",
        x_domain=[0.0, tbc_max],
        x_ticks=list(range(0, int(tbc_max) + 1, 20)),
        x_format=".0f",
        show_legend=False,
        show_y_axis=False,
    )

    chart = alt.hconcat(
        ttfc_panel,
        tbc_panel,
        build_reuse_panel(reuse_rows),
        spacing=16,
    ).resolve_scale(y="shared")

    return (
        chart.configure_view(stroke=None, fill=PAPER_HEX)
        .configure(background=PAPER_HEX, padding={"left": 8, "right": 8, "top": 8, "bottom": 8})
        .configure_axis(labelFont="Helvetica Neue", titleFont="Helvetica Neue")
        .configure_legend(labelFont="Helvetica Neue")
        .configure_title(font="Helvetica Neue")
    )


def save_chart(chart: alt.TopLevelMixin, paths: FigurePaths) -> None:
    chart.save(paths.png, scale_factor=PNG_SCALE_FACTOR)
    chart.save(paths.svg)
    transparent_chart = chart.configure(
        background="transparent",
        view=alt.ViewConfig(stroke=None, fill="transparent"),
    )
    transparent_chart.save(paths.transparent_svg)

    image = Image.open(paths.png).convert("RGBA")
    pixels = np.array(image)
    rgb = pixels[..., :3].astype(np.int16)
    distance = np.abs(rgb - PAPER_RGB).sum(axis=-1)
    pixels[distance <= 10, 3] = 0
    Image.fromarray(pixels, mode="RGBA").save(paths.transparent_png)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = make_figure_paths(output_dir, args.stem)

    linear_best = extract_best_result(args.linear_search_dir.resolve())
    dag_best = extract_best_result(args.dag_search_dir.resolve())

    linear_run_dir = find_run_dir(
        args.linear_search_dir.resolve(),
        "linear",
        float(linear_best["normalized_request_rate"]),
    )
    dag_run_dir = find_run_dir(
        args.dag_search_dir.resolve(),
        "dag",
        float(dag_best["normalized_request_rate"]),
    )

    linear_ttfc_rows = build_ecdf_rows(
        linear_run_dir / "metrics" / "ttfc.csv",
        "Linear workload",
        "Time to First Chunk",
    )
    dag_ttfc_rows = build_ecdf_rows(
        dag_run_dir / "metrics" / "ttfc.csv",
        "DAG workload",
        "Time to First Chunk",
    )
    linear_tbc_rows = build_ecdf_rows(
        linear_run_dir / "metrics" / "tbc.csv",
        "Linear workload",
        "Time Between Chunks",
        scale=1000.0,
    )
    dag_tbc_rows = build_ecdf_rows(
        dag_run_dir / "metrics" / "tbc.csv",
        "DAG workload",
        "Time Between Chunks",
        scale=1000.0,
    )

    reuse_rows = [
        {
            "metric": "Prompt reuse ratio",
            "metric_label": "Prompt reuse",
            "value": float(linear_best["run_mean_prompt_reuse_ratio"]),
            "workload": "Linear workload",
        },
        {
            "metric": "Prompt reuse ratio",
            "metric_label": "Prompt reuse",
            "value": float(dag_best["run_mean_prompt_reuse_ratio"]),
            "workload": "DAG workload",
        },
        {
            "metric": "Prefix-cache hit rate",
            "metric_label": "Prefix-cache hits",
            "value": float(linear_best["run_vllm_prefix_cache_hit_rate"]),
            "workload": "Linear workload",
        },
        {
            "metric": "Prefix-cache hit rate",
            "metric_label": "Prefix-cache hits",
            "value": float(dag_best["run_vllm_prefix_cache_hit_rate"]),
            "workload": "DAG workload",
        },
    ]

    chart = build_chart(
        linear_ttfc_rows=linear_ttfc_rows,
        dag_ttfc_rows=dag_ttfc_rows,
        linear_tbc_rows=linear_tbc_rows,
        dag_tbc_rows=dag_tbc_rows,
        reuse_rows=reuse_rows,
    )
    save_chart(chart, figure_paths)

    for path in (
        figure_paths.png,
        figure_paths.svg,
        figure_paths.transparent_png,
        figure_paths.transparent_svg,
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
