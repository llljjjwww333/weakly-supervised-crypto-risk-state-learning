from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.utils.io import ensure_dir

FRAMEWORK_LABELS = {
    "classification_only": "C only",
    "ct_weighted_mcdm": "C+T weighted",
    "ct_topsis_mcdm": "C+T TOPSIS",
    "cts_weighted_oracle": "C+T+S weighted",
}

POOL_LABELS = {
    "cross_asset_common": "Five-asset\ncommon pool",
    "btc_extended": "BTC\nextended pool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot semantic regret of conventional evaluation frameworks.")
    parser.add_argument(
        "--input_path",
        default="experiments/summary/evaluation_framework_increment/framework_increment_summary.csv",
    )
    parser.add_argument("--output_dir", default="figures/main")
    return parser.parse_args()


def _annotate_heatmap(ax: plt.Axes, data: pd.DataFrame, suffix: str = "") -> None:
    for y, row_name in enumerate(data.index):
        for x, col_name in enumerate(data.columns):
            value = float(data.loc[row_name, col_name])
            label = f"{value:.3f}{suffix}" if suffix == "" else f"{value:.2f}{suffix}"
            ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=8.8, color="#222222")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    df = pd.read_csv(args.input_path)
    df = df.loc[df["framework"].isin(FRAMEWORK_LABELS)].copy()
    df["Framework"] = df["framework"].map(FRAMEWORK_LABELS)
    df["Pool"] = df["pool"].map(POOL_LABELS)

    framework_order = ["C only", "C+T weighted", "C+T TOPSIS", "C+T+S weighted"]
    pool_order = ["Five-asset\ncommon pool", "BTC\nextended pool"]
    regret = (
        df.pivot(index="Framework", columns="Pool", values="mean_semantic_regret_vs_protocol")
        .reindex(index=framework_order, columns=pool_order)
    )
    disagreement = (
        df.pivot(index="Framework", columns="Pool", values="selection_disagreement_rate")
        .reindex(index=framework_order, columns=pool_order)
    )

    sns.set_theme(
        style="white",
        context="paper",
        font="DejaVu Serif",
        rc={
            "figure.titlesize": 12.5,
            "axes.titlesize": 10.8,
            "axes.labelsize": 10.2,
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "legend.fontsize": 9.5,
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
        },
    )
    cmap = LinearSegmentedColormap.from_list("regret", ["#f7fbff", "#c6dbef", "#6baed6", "#e45756"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), gridspec_kw={"width_ratios": [1.0, 1.0]})
    fig.subplots_adjust(left=0.17, right=0.99, top=0.78, bottom=0.22, wspace=0.34)

    sns.heatmap(
        regret,
        ax=axes[0],
        cmap=cmap,
        vmin=0,
        vmax=max(0.30, float(regret.max().max())),
        cbar=False,
        linewidths=1.2,
        linecolor="white",
        annot=False,
    )
    _annotate_heatmap(axes[0], regret)
    axes[0].set_title("Semantic regret", fontsize=10.5)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].tick_params(axis="y", rotation=0, length=0)

    sns.heatmap(
        disagreement,
        ax=axes[1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=1.2,
        linecolor="white",
        annot=False,
    )
    _annotate_heatmap(axes[1], disagreement, suffix="")
    axes[1].set_title("Disagreement with protocol", fontsize=10.5)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].tick_params(axis="y", rotation=0, length=0)
    axes[1].set_yticklabels([])

    fig.suptitle("Increment over conventional multi-criteria evaluation", y=0.96, fontsize=12.4)
    fig.text(
        0.17,
        0.08,
        "Semantic regret measures how much nominal significant-risk layering is lost by each conventional selector.",
        fontsize=8.8,
        color="#4a4a4a",
    )
    fig.savefig(output_dir / "framework_increment_semantic_regret.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
