#!/usr/bin/env python3
"""Generate a polished context-growth chart for the agentic workloads post."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "static" / "2026" / "agentic_workloads"
PNG_PATH = OUTDIR / "context_growth_over_turns.png"
SVG_PATH = OUTDIR / "context_growth_over_turns.svg"
TRANSPARENT_PNG_PATH = OUTDIR / "context_growth_over_turns_transparent.png"
TRANSPARENT_SVG_PATH = OUTDIR / "context_growth_over_turns_transparent.svg"
PAPER_RGB = np.array([251, 247, 239], dtype=np.int16)


def build_source() -> pd.DataFrame:
    turns = np.arange(1, 121)

    system_prompt = np.full_like(turns, 12_000, dtype=float)
    tool_definitions = np.full_like(turns, 5_500, dtype=float)
    fresh_user_input = np.full_like(turns, 900, dtype=float)
    # Low-frequency variation: broad changes in how much each turn adds to history.
    history_additions = (
        900
        + 18 * turns
        + 460 * np.sin((turns - 6) / 8.4)
        + 240 * np.sin((turns + 3) / 15.5)
        + 340 * np.exp(-((turns - 32) / 9.5) ** 2)
        - 260 * np.exp(-((turns - 63) / 8.5) ** 2)
        + 390 * np.exp(-((turns - 95) / 10.0) ** 2)
    )
    history_additions = np.clip(history_additions, 650, None)
    history = np.concatenate(([0.0], np.cumsum(history_additions[:-1])))

    base = pd.DataFrame(
        {
            "turn": turns,
            "Earlier history": history,
            "System prompt": system_prompt,
            "Tool definitions": tool_definitions,
            "Fresh user input": fresh_user_input,
        }
    )
    layers = []
    order = ["Earlier history", "System prompt", "Tool definitions", "Fresh user input"]
    running = np.zeros_like(turns, dtype=float)
    for component in order:
        values = base[component].to_numpy(dtype=float)
        layers.append(
            pd.DataFrame(
                {
                    "turn": turns,
                    "component": component,
                    "y0": running,
                    "y1": running + values,
                }
            )
        )
        running = running + values

    return pd.concat(layers, ignore_index=True)


def build_chart() -> alt.Chart:
    long_df = build_source()
    max_total = float(long_df["y1"].max())
    y_max = int(np.ceil(max_total / 25_000.0) * 25_000)
    y_ticks = list(range(0, y_max + 1, 25_000))

    palette = {
        "System prompt": "#E8D9B5",
        "Tool definitions": "#85B8B0",
        "Earlier history": "#E7865A",
        "Fresh input": "#355C7D",
    }

    base = (
        alt.Chart(long_df)
        .mark_area(line={"color": "#FBF7EF", "opacity": 0.9, "width": 1.2}, opacity=0.96)
        .encode(
            x=alt.X(
                "turn:Q",
                title="Agent turn",
                axis=alt.Axis(
                    values=[1, 25, 50, 75, 100, 120],
                    labelColor="#5E5A55",
                    titleColor="#3F3A36",
                    tickColor="#D9D0C3",
                    grid=False,
                    labelFontSize=18,
                    titleFontSize=22,
                    labelPadding=8,
                    titlePadding=14,
                ),
                scale=alt.Scale(domain=[1, 120], nice=False),
            ),
            y=alt.Y(
                "y1:Q",
                title="Context length",
                axis=alt.Axis(
                    format="~s",
                    values=y_ticks,
                    labelColor="#5E5A55",
                    titleColor="#3F3A36",
                    tickColor="#D9D0C3",
                    gridColor="#E9E1D5",
                    domain=False,
                    labelFontSize=18,
                    titleFontSize=22,
                    labelPadding=8,
                    titlePadding=14,
                ),
                scale=alt.Scale(domain=[0, y_max]),
            ),
            y2="y0:Q",
            color=alt.Color(
                "component:N",
                sort=["System prompt", "Tool definitions", "Earlier history", "Fresh user input"],
                scale=alt.Scale(
                    domain=["System prompt", "Tool definitions", "Earlier history", "Fresh user input"],
                    range=[
                        palette["System prompt"],
                        palette["Tool definitions"],
                        palette["Earlier history"],
                        palette["Fresh input"],
                    ],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    direction="horizontal",
                    columns=4,
                    labelColor="#4A4540",
                    labelFontSize=17,
                    symbolType="square",
                    symbolStrokeWidth=0,
                    symbolSize=220,
                    offset=14,
                ),
            ),
        )
    )

    chart = (
        base
        .properties(
            width=980,
            height=520,
        )
        .configure_view(stroke=None, fill="#FBF7EF")
        .configure(background="#FBF7EF", padding={"left": 8, "right": 18, "top": 8, "bottom": 8})
        .configure_axis(labelFont="Helvetica Neue", titleFont="Helvetica Neue")
        .configure_legend(labelFont="Helvetica Neue")
    )
    return chart


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    chart = build_chart()
    chart.save(PNG_PATH, scale_factor=2)
    chart.save(SVG_PATH)
    chart.configure(
        background="transparent",
        view=alt.ViewConfig(stroke=None, fill="transparent"),
    ).save(
        TRANSPARENT_SVG_PATH
    )

    image = Image.open(PNG_PATH).convert("RGBA")
    pixels = np.array(image)
    rgb = pixels[..., :3].astype(np.int16)
    distance = np.abs(rgb - PAPER_RGB).sum(axis=-1)
    pixels[distance <= 10, 3] = 0
    Image.fromarray(pixels, mode="RGBA").save(TRANSPARENT_PNG_PATH)

    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {TRANSPARENT_PNG_PATH}")
    print(f"Wrote {TRANSPARENT_SVG_PATH}")


if __name__ == "__main__":
    main()
