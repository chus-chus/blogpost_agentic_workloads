#!/usr/bin/env python3
"""Generate the case-study-1 TTFC scaling figure in the house Altair style."""

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
RESULTS_DIR = ROOT.parent / "blog_inference_eval" / "veeksha-results" / "case-study-1"
OUTDIR = ROOT / "static" / "2026" / "agentic_workloads"

PAPER_HEX = "#FBF7EF"
PAPER_RGB = np.array([251, 247, 239], dtype=np.int16)
TEXT_COLOR = "#5E5A55"
TITLE_COLOR = "#3F3A36"
GRID_COLOR = "#E9E1D5"
TICK_COLOR = "#D9D0C3"
LEGEND_LABEL_COLOR = "#4A4540"
CACHE_COLOR = "#355C7D"
NOCACHE_COLOR = "#E7865A"
PNG_SCALE_FACTOR = 2
BIN_WIDTH = 5_000


@dataclass(frozen=True)
class FigurePaths:
    png: Path
    svg: Path
    transparent_png: Path
    transparent_svg: Path


FIGURE_PATHS = FigurePaths(
    png=OUTDIR / "case_study_1_ttfc_scaling.png",
    svg=OUTDIR / "case_study_1_ttfc_scaling.svg",
    transparent_png=OUTDIR / "case_study_1_ttfc_scaling_transparent.png",
    transparent_svg=OUTDIR / "case_study_1_ttfc_scaling_transparent.svg",
)

SERIES_SPECS = (
    ("cache", "Default prefix cache"),
    ("nocache", "Prefix cache disabled"),
)

SERIES_ORDER = [label for _, label in SERIES_SPECS]


def round_up(value: float, step: float) -> float:
    return float(np.ceil(value / step) * step)


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def load_request_rows(workload: str, label: str) -> list[dict]:
    path = RESULTS_DIR / workload / "metrics" / "request_level_metrics.jsonl"
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("client_completed_at") is None:
                continue
            rows.append(
                {
                    "prompt_tokens": float(row["num_total_prompt_tokens"]),
                    "ttfc": float(row["ttfc"]),
                    "workload": label,
                }
            )
    return rows


def build_summary_rows(rows: list[dict], label: str, x_max: float) -> list[dict]:
    summary_rows: list[dict] = []
    bin_edges = np.arange(0.0, x_max + BIN_WIDTH, BIN_WIDTH, dtype=float)
    for left, right in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        bucket = [row for row in rows if left <= row["prompt_tokens"] < right]
        if not bucket:
            continue
        prompt_values = np.array([row["prompt_tokens"] for row in bucket], dtype=float)
        ttfc_values = np.array([row["ttfc"] for row in bucket], dtype=float)
        summary_rows.append(
            {
                "prompt_tokens": float(np.median(prompt_values)),
                "ttfc": quantile(ttfc_values, 0.5),
                "workload": label,
            }
        )
    return summary_rows


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


def workload_scale() -> alt.Scale:
    return alt.Scale(
        domain=SERIES_ORDER,
        range=[CACHE_COLOR, NOCACHE_COLOR],
    )


def workload_color(show_legend: bool) -> alt.Color:
    return alt.Color(
        "workload:N",
        sort=SERIES_ORDER,
        scale=workload_scale(),
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


def build_plot_chart(summary_rows: list[dict], x_max: float) -> alt.TopLevelMixin:
    x_ticks = [0, 20_000, 40_000, 60_000, 80_000, 100_000]
    y_ticks = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    x_encoding = alt.X(
        "prompt_tokens:Q",
        title="Total prompt tokens",
        axis=alt.Axis(
            values=x_ticks,
            format="~s",
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
        scale=alt.Scale(domain=[0.0, x_max], nice=False),
    )
    y_encoding = alt.Y(
        "ttfc:Q",
        title="log TTFC (s)",
        axis=alt.Axis(
            values=y_ticks,
            format="~g",
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
        scale=alt.Scale(domain=[0.04, 12.0], type="log", nice=False),
    )

    return (
        alt.Chart(alt.InlineData(values=summary_rows))
        .mark_line(strokeWidth=3.0)
        .encode(
            x=x_encoding,
            y=y_encoding,
            color=workload_color(show_legend=True),
        )
        .properties(width=760, height=360)
    )


def build_chart(summary_rows: list[dict], x_max: float) -> alt.TopLevelMixin:
    return (
        build_plot_chart(summary_rows, x_max)
        .configure_view(stroke=None, fill=PAPER_HEX)
        .configure(background=PAPER_HEX, padding={"left": 8, "right": 12, "top": 4, "bottom": 8})
        .configure_axis(labelFont="Helvetica Neue", titleFont="Helvetica Neue")
        .configure_legend(labelFont="Helvetica Neue")
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    per_series_rows: list[tuple[list[dict], str]] = []
    for workload, label in SERIES_SPECS:
        rows = load_request_rows(workload, label)
        raw_rows.extend(rows)
        per_series_rows.append((rows, label))

    max_prompt = max(row["prompt_tokens"] for row in raw_rows)
    x_max = round_up(max_prompt, 5_000.0)

    summary_rows: list[dict] = []
    for rows, label in per_series_rows:
        summary_rows.extend(build_summary_rows(rows, label, x_max))

    chart = build_chart(summary_rows, x_max)
    save_chart(chart, FIGURE_PATHS)

    print(f"Wrote {FIGURE_PATHS.png}")
    print(f"Wrote {FIGURE_PATHS.svg}")
    print(f"Wrote {FIGURE_PATHS.transparent_png}")
    print(f"Wrote {FIGURE_PATHS.transparent_svg}")


if __name__ == "__main__":
    main()
