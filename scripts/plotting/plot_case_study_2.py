#!/usr/bin/env python3
"""Generate case-study-2 workload-shape figures in the house Altair style."""

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

import json
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT.parent / "blog_inference_eval" / "veeksha-results" / "case-study-2" / "runs"
OUTDIR = ROOT / "static" / "2026" / "agentic_workloads"

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


def make_figure_paths(stem: str) -> FigurePaths:
    return FigurePaths(
        png=OUTDIR / f"{stem}.png",
        svg=OUTDIR / f"{stem}.svg",
        transparent_png=OUTDIR / f"{stem}_transparent.png",
        transparent_svg=OUTDIR / f"{stem}_transparent.svg",
    )


ECDF_PATHS = make_figure_paths("case_study_2_shape_ecdfs")
OVERLAP_PATHS = make_figure_paths("case_study_2_decode_overlap")


def load_request_rows(workload: str) -> list[dict]:
    path = RESULTS_DIR / workload / "metrics" / "request_level_metrics.jsonl"
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("client_completed_at") is None:
                continue
            rows.append(row)
    return rows


def round_up(value: float, step: float) -> float:
    return float(np.ceil(value / step) * step)


def build_ecdf_rows(values: np.ndarray, workload: str) -> list[dict]:
    sorted_values = np.sort(values.astype(float))
    if len(sorted_values) == 1:
        cdf = np.array([1.0], dtype=float)
    else:
        cdf = np.linspace(0.0, 1.0, len(sorted_values), dtype=float)
    return [
        {"value": float(value), "cdf": float(probability), "workload": workload}
        for value, probability in zip(sorted_values, cdf, strict=True)
    ]


def build_decode_overlap_rows(rows: list[dict], workload: str) -> list[dict]:
    events: list[tuple[float, int]] = []
    for row in rows:
        decode_start = float(row["client_picked_up_at"]) + float(row["ttfc"])
        decode_end = float(row["client_completed_at"])
        if decode_end <= decode_start:
            continue
        events.append((decode_start, 1))
        events.append((decode_end, -1))

    events.sort(key=lambda item: (item[0], item[1]))

    active = 0
    previous_time: float | None = None
    durations: list[tuple[int, float]] = []
    for time, delta in events:
        if previous_time is not None and time > previous_time and active > 0:
            durations.append((active, time - previous_time))
        active += delta
        previous_time = time

    total_decode_time = sum(duration for _, duration in durations)
    max_active = max((count for count, _ in durations), default=0)

    rows_out: list[dict] = []
    for threshold in range(1, max_active + 1):
        duration_at_or_above = sum(
            duration for count, duration in durations if count >= threshold
        )
        rows_out.append(
            {
                "active_requests": threshold,
                "share": float(duration_at_or_above / total_decode_time) if total_decode_time else 0.0,
                "workload": workload,
            }
        )
    return rows_out


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


def build_ecdf_panel(
    rows: list[dict],
    title: str,
    x_title: str,
    x_domain: list[float],
    x_ticks: list[float],
    x_format: str,
    show_legend: bool,
    show_y_axis: bool,
) -> alt.Chart:
    return (
        alt.Chart(alt.InlineData(values=rows))
        .mark_line(strokeWidth=2.8)
        .encode(
            x=alt.X(
                "value:Q",
                title=x_title,
                axis=alt.Axis(
                    values=x_ticks,
                    format=x_format,
                    labelColor=TEXT_COLOR,
                    titleColor=TITLE_COLOR,
                    tickColor=TICK_COLOR,
                    grid=False,
                    labelFontSize=15,
                    titleFontSize=17,
                    titleFontWeight="bold",
                    labelPadding=6,
                    titlePadding=10,
                ),
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
        .properties(width=285, height=245, title=alt.TitleParams(title, color=TITLE_COLOR, fontSize=18))
    )


def build_ecdf_chart(linear_rows: list[dict], dag_rows: list[dict]) -> alt.TopLevelMixin:
    prompt_values = np.array([row["num_total_prompt_tokens"] for row in linear_rows + dag_rows], dtype=float)
    ttfc_values = np.array([row["ttfc"] for row in linear_rows + dag_rows], dtype=float)
    e2e_values = np.array([row["end_to_end_latency"] for row in linear_rows + dag_rows], dtype=float)

    prompt_panel_rows = build_ecdf_rows(
        np.array([row["num_total_prompt_tokens"] for row in linear_rows], dtype=float),
        "Linear workload",
    ) + build_ecdf_rows(
        np.array([row["num_total_prompt_tokens"] for row in dag_rows], dtype=float),
        "DAG workload",
    )
    ttfc_panel_rows = build_ecdf_rows(
        np.array([row["ttfc"] for row in linear_rows], dtype=float),
        "Linear workload",
    ) + build_ecdf_rows(
        np.array([row["ttfc"] for row in dag_rows], dtype=float),
        "DAG workload",
    )
    e2e_panel_rows = build_ecdf_rows(
        np.array([row["end_to_end_latency"] for row in linear_rows], dtype=float),
        "Linear workload",
    ) + build_ecdf_rows(
        np.array([row["end_to_end_latency"] for row in dag_rows], dtype=float),
        "DAG workload",
    )

    prompt_max = round_up(float(prompt_values.max()), 500.0)
    ttfc_max = round_up(float(ttfc_values.max()), 0.1)
    e2e_max = round_up(float(e2e_values.max()), 1.0)

    chart = alt.hconcat(
        build_ecdf_panel(
            rows=prompt_panel_rows,
            title="Effective context length",
            x_title="Total prompt tokens",
            x_domain=[0.0, prompt_max],
            x_ticks=list(range(0, int(prompt_max) + 1, 1000)),
            x_format="~s",
            show_legend=True,
            show_y_axis=True,
        ),
        build_ecdf_panel(
            rows=ttfc_panel_rows,
            title="TTFC",
            x_title="Seconds",
            x_domain=[0.0, ttfc_max],
            x_ticks=[round(tick, 1) for tick in np.arange(0.0, ttfc_max + 0.001, 0.1)],
            x_format=".1f",
            show_legend=False,
            show_y_axis=False,
        ),
        build_ecdf_panel(
            rows=e2e_panel_rows,
            title="End-to-end latency",
            x_title="Seconds",
            x_domain=[0.0, e2e_max],
            x_ticks=list(range(0, int(e2e_max) + 1, 1)),
            x_format=".0f",
            show_legend=False,
            show_y_axis=False,
        ),
        spacing=16,
    ).resolve_scale(y="shared")

    return (
        chart.configure_view(stroke=None, fill=PAPER_HEX)
        .configure(background=PAPER_HEX, padding={"left": 8, "right": 8, "top": 8, "bottom": 8})
        .configure_axis(labelFont="Helvetica Neue", titleFont="Helvetica Neue")
        .configure_legend(labelFont="Helvetica Neue")
        .configure_title(font="Helvetica Neue")
    )


def build_decode_overlap_chart(linear_rows: list[dict], dag_rows: list[dict]) -> alt.TopLevelMixin:
    overlap_rows = build_decode_overlap_rows(linear_rows, "Linear workload") + build_decode_overlap_rows(
        dag_rows, "DAG workload"
    )
    max_active = max(row["active_requests"] for row in overlap_rows)
    x_ticks = [1]
    x_ticks.extend(list(range(5, int(round_up(float(max_active), 5.0)) + 1, 5)))

    chart = (
        alt.Chart(alt.InlineData(values=overlap_rows))
        .mark_line(strokeWidth=2.9)
        .encode(
            x=alt.X(
                "active_requests:Q",
                title="Simultaneous active decode requests",
                axis=alt.Axis(
                    values=x_ticks,
                    labelColor=TEXT_COLOR,
                    titleColor=TITLE_COLOR,
                    tickColor=TICK_COLOR,
                    grid=False,
                    labelFontSize=16,
                    titleFontSize=18,
                    titleFontWeight="bold",
                    labelPadding=6,
                    titlePadding=10,
                ),
                scale=alt.Scale(domain=[1.0, float(max_active)], nice=False),
            ),
            y=alt.Y(
                "share:Q",
                title="Share of decode-active time",
                axis=alt.Axis(
                    values=[0.0, 0.25, 0.5, 0.75, 1.0],
                    format=".0%",
                    labelColor=TEXT_COLOR,
                    titleColor=TITLE_COLOR,
                    tickColor=TICK_COLOR,
                    gridColor=GRID_COLOR,
                    domain=False,
                    labelFontSize=16,
                    titleFontSize=18,
                    titleFontWeight="bold",
                    labelPadding=6,
                    titlePadding=10,
                ),
                scale=alt.Scale(domain=[0.0, 1.0], nice=False),
            ),
            color=workload_color(show_legend=True),
        )
        .properties(width=760, height=290)
        .configure_view(stroke=None, fill=PAPER_HEX)
        .configure(background=PAPER_HEX, padding={"left": 8, "right": 8, "top": 8, "bottom": 8})
        .configure_axis(labelFont="Helvetica Neue", titleFont="Helvetica Neue")
        .configure_legend(labelFont="Helvetica Neue")
    )
    return chart


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    linear_rows = load_request_rows("linear")
    dag_rows = load_request_rows("dag")

    ecdf_chart = build_ecdf_chart(linear_rows, dag_rows)
    overlap_chart = build_decode_overlap_chart(linear_rows, dag_rows)

    save_chart(ecdf_chart, ECDF_PATHS)
    save_chart(overlap_chart, OVERLAP_PATHS)

    for path in (
        ECDF_PATHS.png,
        ECDF_PATHS.svg,
        ECDF_PATHS.transparent_png,
        ECDF_PATHS.transparent_svg,
        OVERLAP_PATHS.png,
        OVERLAP_PATHS.svg,
        OVERLAP_PATHS.transparent_png,
        OVERLAP_PATHS.transparent_svg,
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
