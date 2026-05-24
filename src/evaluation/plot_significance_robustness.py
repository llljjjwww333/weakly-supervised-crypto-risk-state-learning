from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

from src.utils.io import ensure_dir

DISPLAY_NAMES = {
    "proxy_label": "Proxy label",
    "logreg_postproc_match": "Logreg + post-proc.",
    "histgb_postproc": "HistGB + post-proc.",
    "main_gru": "Main GRU",
    "tcn_96x4": "TCN-96x4",
}
ORDER = ["proxy_label", "logreg_postproc_match", "histgb_postproc", "main_gru", "tcn_96x4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot nominal-vs-bootstrap significance robustness.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    df = pd.read_csv(args.input_path)
    df = df.loc[df["source"].isin(ORDER)].copy()
    df["source"] = pd.Categorical(df["source"], categories=ORDER, ordered=True)
    df = df.sort_values("source").reset_index(drop=True)
    df["method"] = df["source"].map(DISPLAY_NAMES)
    df["nominal"] = df["significant_risk_layering_rate"].astype(float)
    df["bootstrap"] = df["bootstrap_significant_risk_layering_rate"].astype(float)
    df["shrinkage"] = df["nominal"] - df["bootstrap"]

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
            "axes.facecolor": "#fcfcf8",
            "figure.facecolor": "white",
        },
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    fig.subplots_adjust(left=0.29, right=0.95, top=0.84, bottom=0.24)
    y_positions = list(range(len(df)))[::-1]

    for y, (_, row) in zip(y_positions, df.iterrows()):
        ax.plot([row["bootstrap"], row["nominal"]], [y, y], color="#c2cad1", linewidth=2.4, alpha=0.9, zorder=1)
        arrow = FancyArrowPatch(
            (row["nominal"], y + 0.16),
            (row["bootstrap"], y + 0.16),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color="#8a9299",
            alpha=0.95,
        )
        ax.add_patch(arrow)
        ax.scatter(row["bootstrap"], y, s=76, color="#c44e52", edgecolor="white", linewidth=0.9, zorder=3)
        ax.scatter(row["nominal"], y, s=76, color="#4c78a8", edgecolor="white", linewidth=0.9, zorder=3)
        ax.text(row["nominal"] + 0.018, y + 0.08, f"{row['nominal']:.2f}", color="#315f8a", fontsize=8.8)
        ax.text(row["bootstrap"] + 0.018, y - 0.18, f"{row['bootstrap']:.2f}", color="#9e3b40", fontsize=8.8)
        mid_x = (row["bootstrap"] + row["nominal"]) / 2
        ax.text(
            mid_x,
            y + 0.28,
            f"drop {row['shrinkage']:.2f}",
            color="#767676",
            fontsize=8.2,
            va="bottom",
            ha="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.12},
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["method"].tolist())
    ax.set_xlim(0.0, max(df["nominal"].max() + 0.12, 0.86))
    ax.set_xlabel("Significant risk-layering rate")
    ax.set_title("BTCUSDT: nominal significance contracts under block bootstrap")
    ax.grid(axis="x", linewidth=0.6, color="#d8dde2", alpha=0.9)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#4c78a8", markeredgecolor="white", markersize=8, label="Nominal"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#c44e52", markeredgecolor="white", markersize=8, label="Block bootstrap"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.13))
    fig.text(
        0.50,
        0.055,
        "Arrow direction shows contraction from nominal to block-bootstrap significance.",
        ha="center",
        fontsize=8.2,
        color="#777777",
    )

    fig.savefig(output_dir / "btc_significance_nominal_vs_block_bootstrap.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "btc_significance_nominal_vs_block_bootstrap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
