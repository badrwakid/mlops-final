"""Render DVC pipeline DAG to docs/screenshots/dvc_dag.png (no Graphviz binary required)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docs/screenshots/dvc_dag.png"),
        help="Output PNG path (relative to repo root)",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    g = nx.DiGraph()
    raw = "data/raw/hour.csv.dvc"
    g.add_edges_from(
        [
            (raw, "prepare"),
            ("prepare", "preprocess"),
            ("preprocess", "featurize"),
            ("preprocess", "train"),
            ("featurize", "train"),
        ]
    )
    pos = {
        raw: (0.0, 4.0),
        "prepare": (0.0, 3.0),
        "preprocess": (0.0, 2.0),
        "featurize": (-1.2, 1.0),
        "train": (1.2, 1.0),
    }

    plt.figure(figsize=(9, 7))
    nx.draw_networkx(
        g,
        pos,
        with_labels=True,
        labels={
            raw: "hour.csv.dvc",
            "prepare": "prepare",
            "preprocess": "preprocess",
            "featurize": "featurize",
            "train": "train",
        },
        node_color="#e8f4f8",
        edge_color="#333333",
        node_size=5200,
        font_size=10,
        font_family="sans-serif",
        arrows=True,
        arrowsize=18,
        width=1.4,
        edgecolors="#2c5f7c",
        linewidths=1.2,
    )
    plt.title("DVC pipeline DAG", fontsize=12, pad=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
