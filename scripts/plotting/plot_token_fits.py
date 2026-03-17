#!/usr/bin/env python3
"""Regenerate case-study-1 token-fit plots in the context-growth Altair style."""

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


ROOT = Path(__file__).resolve().parent
SESSION_DIR = ROOT / "blog-sessions" / "openclaw-case-study-1-interp"
DAG_PATH = SESSION_DIR / "dag.json"
PAPER_HEX = "#FBF7EF"
PAPER_RGB = np.array([251, 247, 239], dtype=np.int16)
TEXT_COLOR = "#5E5A55"
TITLE_COLOR = "#3F3A36"
GRID_COLOR = "#E9E1D5"
TICK_COLOR = "#D9D0C3"
HIST_COLOR = "#E8D9B5"
INV_GAUSS_COLOR = "#355C7D"
LOGNORMAL_COLOR = "#E7865A"
WIDTH = 720
HEIGHT = 360
PNG_SCALE_FACTOR = 2


@dataclass(frozen=True)
class PlotSpec:
    key: str
    xlabel: str
    output_name: str


PLOT_SPECS = (
    PlotSpec(
        key="new_tokens",
        xlabel="New input tokens per turn",
        output_name="new_tokens_fit_p95_linear.png",
    ),
    PlotSpec(
        key="output_tokens",
        xlabel="Output tokens per turn",
        output_name="output_tokens_fit_p95_linear.png",
    ),
)


def load_nodes() -> list[dict]:
    return json.loads(DAG_PATH.read_text())["nodes"]


def extract_values(nodes: list[dict], key: str) -> np.ndarray:
    if key == "new_tokens":
        values = [float(node["new_tokens"]["total"]) for node in nodes]
    elif key == "output_tokens":
        values = [float(node["usage"]["output"]) for node in nodes]
    else:
        raise ValueError(f"Unsupported series: {key}")
    return np.array(values, dtype=float)


def fit_lognormal(values: np.ndarray) -> tuple[float, float]:
    log_values = np.log(values)
    return float(log_values.mean()), float(log_values.std(ddof=0))


def fit_inverse_gaussian(values: np.ndarray) -> tuple[float, float]:
    mu = float(values.mean())
    denominator = np.sum((1.0 / values) - (1.0 / mu))
    lam = float(len(values) / denominator)
    return mu, lam


def lognormal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(x)
    return np.exp(-((np.log(x) - mu) ** 2) / (2.0 * sigma**2)) / (
        x * sigma * np.sqrt(2.0 * np.pi)
    )


def inverse_gaussian_pdf(x: np.ndarray, mu: float, lam: float) -> np.ndarray:
    if mu <= 0 or lam <= 0:
        return np.zeros_like(x)
    coefficient = np.sqrt(lam / (2.0 * np.pi * x**3))
    exponent = -lam * (x - mu) ** 2 / (2.0 * mu**2 * x)
    return coefficient * np.exp(exponent)


def round_up(value: float, step: float) -> float:
    return float(np.ceil(value / step) * step)


def choose_x_tick_step(cutoff: float) -> int:
    if cutoff >= 2500:
        return 1000
    if cutoff >= 1200:
        return 500
    return 250


def choose_y_tick_step(peak_density: float) -> float:
    if peak_density >= 0.0025:
        return 0.001
    if peak_density >= 0.0012:
        return 0.0005
    return 0.00025


def choose_bin_edges(values: np.ndarray, cutoff: float) -> np.ndarray:
    q25, q75 = np.percentile(values, [25, 75], method="linear")
    iqr = float(q75 - q25)
    if iqr > 0:
        width = 2.0 * iqr / np.cbrt(len(values))
        bins = int(np.ceil(cutoff / width)) if width > 0 else 0
    else:
        bins = 0
    bins = int(np.clip(bins, 16, 28))
    return np.linspace(0.0, cutoff, bins + 1)


def make_background_transparent(path: Path) -> None:
    image = Image.open(path).convert("RGBA")
    pixels = np.array(image)
    rgb = pixels[..., :3].astype(np.int16)
    distance = np.abs(rgb - PAPER_RGB).sum(axis=-1)
    pixels[distance <= 10, 3] = 0
    Image.fromarray(pixels, mode="RGBA").save(path)


def build_histogram_rows(values: np.ndarray, bin_edges: np.ndarray) -> list[dict]:
    density, _ = np.histogram(values, bins=bin_edges, density=True)
    rows = []
    for left, right, height in zip(bin_edges[:-1], bin_edges[1:], density, strict=True):
        if height <= 0:
            continue
        rows.append(
            {
                "x": float(left),
                "x2": float(right),
                "baseline": 0.0,
                "density": float(height),
                "series": "Empirical",
            }
        )
    return rows


def build_line_rows(x: np.ndarray, inverse_gaussian: np.ndarray, lognormal: np.ndarray) -> list[dict]:
    rows = []
    for label, density in (
        ("Inverse Gaussian", inverse_gaussian),
        ("Lognormal", lognormal),
    ):
        rows.extend(
            {
                "x": float(x_value),
                "density": float(y_value),
                "series": label,
            }
            for x_value, y_value in zip(x, density, strict=True)
        )
    return rows


def build_chart(values: np.ndarray, spec: PlotSpec) -> alt.Chart:
    cutoff = float(np.percentile(values, 95, method="linear"))
    trimmed = values[values <= cutoff]
    bin_edges = choose_bin_edges(trimmed, cutoff)

    x = np.linspace(1e-3, cutoff, 2400)
    inv_mu, inv_lambda = fit_inverse_gaussian(trimmed)
    log_mu, log_sigma = fit_lognormal(trimmed)
    inverse_gaussian = inverse_gaussian_pdf(x, inv_mu, inv_lambda)
    lognormal = lognormal_pdf(x, log_mu, log_sigma)

    histogram_rows = build_histogram_rows(trimmed, bin_edges)
    line_rows = build_line_rows(x, inverse_gaussian, lognormal)

    hist_peak = max((row["density"] for row in histogram_rows), default=0.0)
    peak_density = max(hist_peak, float(inverse_gaussian.max()), float(lognormal.max()))
    y_step = choose_y_tick_step(peak_density)
    y_max = round_up(peak_density * 1.02, y_step / 2.0)
    y_ticks = [round(tick, 4) for tick in np.arange(0.0, y_max + y_step * 0.5, y_step)]
    x_tick_step = choose_x_tick_step(cutoff)
    x_ticks = list(range(0, int(round_up(cutoff, float(x_tick_step))) + 1, x_tick_step))

    color = alt.Color(
        "series:N",
        scale=alt.Scale(
            domain=["Empirical", "Inverse Gaussian", "Lognormal"],
            range=[HIST_COLOR, INV_GAUSS_COLOR, LOGNORMAL_COLOR],
        ),
        legend=alt.Legend(
            title=None,
            orient="top",
            direction="horizontal",
            columns=3,
            labelColor="#4A4540",
            labelFontSize=15,
            symbolType="square",
            symbolStrokeWidth=0,
            symbolSize=180,
            offset=8,
        ),
    )

    x_encoding = alt.X(
        "x:Q",
        title=spec.xlabel,
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
        scale=alt.Scale(domain=[0, cutoff], nice=False),
    )
    y_encoding = alt.Y(
        "density:Q",
        title="Density",
        axis=alt.Axis(
            values=y_ticks,
            format=".3f",
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
        scale=alt.Scale(domain=[0, y_max], nice=False),
    )

    histogram = (
        alt.Chart(alt.InlineData(values=histogram_rows))
        .mark_bar(opacity=0.9, stroke=PAPER_HEX, strokeWidth=0.8)
        .encode(
            x=x_encoding,
            x2="x2:Q",
            y=y_encoding,
            y2="baseline:Q",
            color=color,
        )
    )

    lines = (
        alt.Chart(alt.InlineData(values=line_rows))
        .mark_line(strokeWidth=2.9)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, cutoff], nice=False)),
            y=alt.Y("density:Q", scale=alt.Scale(domain=[0, y_max], nice=False)),
            color=color,
        )
    )

    chart = (
        alt.layer(histogram, lines)
        .properties(width=WIDTH, height=HEIGHT)
        .configure_view(stroke=None, fill=PAPER_HEX)
        .configure(background=PAPER_HEX, padding={"left": 6, "right": 6, "top": 4, "bottom": 4})
        .configure_axis(labelFont="Helvetica Neue", titleFont="Helvetica Neue")
        .configure_legend(labelFont="Helvetica Neue")
    )
    return chart


def main() -> None:
    nodes = load_nodes()
    for spec in PLOT_SPECS:
        values = extract_values(nodes, spec.key)
        output_path = SESSION_DIR / spec.output_name
        build_chart(values, spec).save(output_path, scale_factor=PNG_SCALE_FACTOR)
        make_background_transparent(output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
