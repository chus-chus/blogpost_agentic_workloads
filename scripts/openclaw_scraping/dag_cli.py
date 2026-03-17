#!/usr/bin/env python3
"""CLI for extracting and visualizing LLM request DAGs from OpenClaw sessions.

Prints a summary table to stdout and writes dag.json + dag.dot next to the input.

Usage:
    python3 dag_cli.py <sessions.json or .jsonl>
    python3 dag_cli.py <path> --no-dot          # skip DOT file
    python3 dag_cli.py <path> --json out.json   # custom JSON path
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from build_dag import build_dag, dag_summary_table, dag_to_dot


def resolve_sessions_json(input_path: str) -> Path:
    """Given either a sessions.json or a .jsonl file, find the sessions.json."""
    p = Path(input_path)
    if p.name == "sessions.json":
        return p
    if p.suffix == ".jsonl":
        candidate = p.parent / "sessions.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Could not find sessions.json next to {p}. "
            "Please pass the sessions.json path directly."
        )
    raise ValueError(f"Expected a sessions.json or .jsonl file, got: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and visualize LLM request DAGs from OpenClaw sessions.",
        epilog="By default prints a human-readable summary table. "
        "Use --json for machine-readable output or --dot for Graphviz.",
    )
    parser.add_argument(
        "input", help="Path to sessions.json or any session .jsonl file"
    )
    parser.add_argument(
        "--json", default=None, metavar="FILE",
        help="Write full DAG JSON to FILE (default: <input_dir>/dag.json)",
    )
    parser.add_argument(
        "--dot", default=None, metavar="FILE",
        help="Write Graphviz DOT to FILE (default: <input_dir>/dag.dot)",
    )
    parser.add_argument(
        "--no-json", action="store_true", help="Skip JSON output",
    )
    parser.add_argument(
        "--no-dot", action="store_true", help="Skip DOT output",
    )

    args = parser.parse_args()
    sessions_json = resolve_sessions_json(args.input)
    dag = build_dag(sessions_json)

    # Use shortened root session ID as folder/file name
    root_sid = None
    for sid, meta in dag["sessions"].items():
        if meta.get("spawned_by") is None:
            root_sid = sid
            break
    folder_name = f"dag-{root_sid[:8]}" if root_sid else "dag"
    out_dir = Path.cwd() / folder_name
    out_dir.mkdir(exist_ok=True)

    # Summary table always goes to stdout
    print(dag_summary_table(dag))

    # JSON
    if not args.no_json:
        json_path = Path(args.json) if args.json else out_dir / "dag.json"
        with open(json_path, "w") as f:
            json.dump(dag, f, indent=2)
            f.write("\n")
        print(f"Wrote {json_path}")

    # DOT + PNG
    if not args.no_dot:
        dot_path = Path(args.dot) if args.dot else out_dir / "dag.dot"
        dot_content = dag_to_dot(dag)
        with open(dot_path, "w") as f:
            f.write(dot_content)
            f.write("\n")
        print(f"Wrote {dot_path}")

        if shutil.which("dot"):
            png_path = dot_path.with_suffix(".png")
            subprocess.run(
                ["dot", "-Tpng", "-o", str(png_path), str(dot_path)],
                check=True,
            )
            print(f"Wrote {png_path}")
        else:
            print("graphviz not found, skipping PNG render")


if __name__ == "__main__":
    main()
