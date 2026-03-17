#!/usr/bin/env python3
"""Generate the case-study-1 wait-after-ready CCDF in the house Altair style."""

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
DAG_PATH = ROOT.parent / "blog_inference_eval" / "blog-sessions" / "openclaw-case-study-1-interp" / "dag.json"
OUTDIR = ROOT / "static" / "2026" / "agentic_workloads"

PAPER_HEX = "#FBF7EF"
PAPER_RGB = np.array([251, 247, 239], dtype=np.int16)
TEXT_COLOR = "#5E5A55"
TITLE_COLOR = "#3F3A36"
GRID_COLOR = "#E9E1D5"
TICK_COLOR = "#D9D0C3"
LEGEND_LABEL_COLOR = "#4A4540"
EMPIRICAL_COLOR = "#355C7D"
FIT_COLOR = "#E7865A"
PNG_SCALE_FACTOR = 2
TAIL_THRESHOLD_MS = 100.0


@dataclass(frozen=True)
class FigurePaths:
    png: Path
    svg: Path
    transparent_png: Path
    transparent_svg: Path


FIGURE_PATHS = FigurePaths(
    png=OUTDIR / "wait_after_ready_pareto_tail_ccdf.png",
    svg=OUTDIR / "wait_after_ready_pareto_tail_ccdf.svg",
    transparent_png=OUTDIR / "wait_after_ready_pareto_tail_ccdf_transparent.png",
    transparent_svg=OUTDIR / "wait_after_ready_pareto_tail_ccdf_transparent.svg",
)

SERIES_ORDER = ["Empirical CCDF", "Pareto tail fit", "Tail threshold"]


def round_up(value: float, step: float) -> float:
    return float(np.ceil(value / step) * step)


def load_wait_values() -> np.ndarray:
    dag = json.loads(DAG_PATH.read_text())
    values = [
        float(node["wait_after_ready_ms"])
        for node in dag["nodes"]
        if node.get("wait_after_ready_ms") is not None
    ]
    return np.sort(np.array(values, dtype=float))


def build_empirical_rows(wait_values: np.ndarray) -> list[dict]:
    unique_values, counts = np.unique(wait_values, return_counts=True)
    surviving = counts[::-1].cumsum()[::-1]
    total = float(wait_values.size)
    return [
        {
            "wait_ms": float(value),
            "probability": float(count / total),
            "series": "Empirical CCDF",
        }
        for value, count in zip(unique_values, surviving, strict=True)
    ]


def fit_pareto_tail(wait_values: np.ndarray, x_max: float) -> tuple[list[dict], float]:
    tail_values = wait_values[wait_values >= TAIL_THRESHOLD_MS]
    if tail_values.size < 2:
        raise ValueError("Need at least two tail values to fit Pareto tail")

    alpha = 1.0 + tail_values.size / float(np.log(tail_values / TAIL_THRESHOLD_MS).sum())
    ccdf_exponent = alpha - 1.0
    tail_fraction = float(tail_values.size / wait_values.size)
    x_values = np.geomspace(TAIL_THRESHOLD_MS, x_max, num=240)
    y_values = tail_fraction * np.power(x_values / TAIL_THRESHOLD_MS, -ccdf_exponent)
    rows = [
        {
            "wait_ms": float(wait_ms),
            "probability": float(probability),
            "series": "Pareto tail fit",
        }
        for wait_ms, probability in zip(x_values, y_values, strict=True)
    ]
    return rows, alpha


def build_threshold_rows(y_min: float, y_max: float) -> list[dict]:
    return [
        {"wait_ms": TAIL_THRESHOLD_MS, "probability": y_min, "series": "Tail threshold"},
        {"wait_ms": TAIL_THRESHOLD_MS, "probability": y_max, "series": "Tail threshold"},
    ]


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


def series_color() -> alt.Color:
    return alt.Color(
        "series:N",
        sort=SERIES_ORDER,
        scale=alt.Scale(
            domain=SERIES_ORDER,
            range=[EMPIRICAL_COLOR, FIT_COLOR, FIT_COLOR],
        ),
        legend=alt.Legend(
            title=None,
            orient="top",
            direction="horizontal",
            columns=3,
            labelColor=LEGEND_LABEL_COLOR,
            labelFontSize=15,
            symbolType="stroke",
            symbolStrokeWidth=3,
            symbolSize=180,
            offset=8,
        ),
    )


def series_stroke_dash() -> alt.StrokeDash:
    return alt.StrokeDash(
        "series:N",
        sort=SERIES_ORDER,
        scale=alt.Scale(
            domain=SERIES_ORDER,
            range=[[1, 0], [1, 0], [9, 6]],
        ),
        legend=None,
    )


def build_chart(wait_values: np.ndarray) -> tuple[alt.TopLevelMixin, float]:
    x_max = round_up(float(wait_values.max()), 10_000.0)
    y_min = 1.0 / (2.0 * wait_values.size)
    y_max = 1.15

    empirical_rows = build_empirical_rows(wait_values)
    fit_rows, alpha = fit_pareto_tail(wait_values, x_max)
    threshold_rows = build_threshold_rows(y_min, y_max)

    x_encoding = alt.X(
        "wait_ms:Q",
        title="Wait after ready (ms)",
        axis=alt.Axis(
            values=[10, 100, 1_000, 10_000, 100_000],
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
        scale=alt.Scale(domain=[float(wait_values.min()), x_max], type="log", nice=False),
    )
    y_encoding = alt.Y(
        "probability:Q",
        title="Share of waits >= x",
        axis=alt.Axis(
            values=[1.0, 0.3, 0.1, 0.03, 0.01, 0.003],
            format="~%",
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
        scale=alt.Scale(domain=[y_min, y_max], type="log", nice=False),
    )

    empirical_chart = (
        alt.Chart(alt.InlineData(values=empirical_rows))
        .mark_line(strokeWidth=3.0, interpolate="step-after")
        .encode(
            x=x_encoding,
            y=y_encoding,
            color=series_color(),
            strokeDash=series_stroke_dash(),
        )
    )
    fit_chart = (
        alt.Chart(alt.InlineData(values=fit_rows))
        .mark_line(strokeWidth=3.0)
        .encode(
            x=x_encoding,
            y=y_encoding,
            color=series_color(),
            strokeDash=series_stroke_dash(),
        )
    )
    threshold_chart = (
        alt.Chart(alt.InlineData(values=threshold_rows))
        .mark_line(strokeWidth=2.6)
        .encode(
            x=x_encoding,
            y=y_encoding,
            color=series_color(),
            strokeDash=series_stroke_dash(),
        )
    )

    chart = (
        alt.layer(empirical_chart, fit_chart, threshold_chart)
        .properties(width=760, height=426)
        .configure_view(stroke=None, fill=PAPER_HEX)
        .configure(background=PAPER_HEX, padding={"left": 8, "right": 12, "top": 4, "bottom": 8})
        .configure_axis(labelFont="Helvetica Neue", titleFont="Helvetica Neue")
        .configure_legend(labelFont="Helvetica Neue")
    )
    return chart, alpha


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    wait_values = load_wait_values()
    chart, alpha = build_chart(wait_values)
    save_chart(chart, FIGURE_PATHS)

    tail_count = int(np.sum(wait_values >= TAIL_THRESHOLD_MS))
    print(f"Wrote {FIGURE_PATHS.png}")
    print(f"Wrote {FIGURE_PATHS.svg}")
    print(f"Wrote {FIGURE_PATHS.transparent_png}")
    print(f"Wrote {FIGURE_PATHS.transparent_svg}")
    print(
        "Tail fit: "
        f"threshold={TAIL_THRESHOLD_MS:.0f} ms, tail_count={tail_count}, "
        f"alpha={alpha:.3f}, ccdf_exponent={alpha - 1.0:.3f}"
    )


if __name__ == "__main__":
    main()
