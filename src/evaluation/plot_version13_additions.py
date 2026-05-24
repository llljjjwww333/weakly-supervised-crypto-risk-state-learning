from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir


MODEL_COLORS = {
    "Proxy label": "#6A8CAF",
    "LogReg + post-proc.": "#D4A72C",
    "HistGB + post-proc.": "#C44E52",
    "Main GRU": "#2F5DA8",
    "TCN-96x4": "#59A14F",
}

MODEL_ORDER = ["Proxy label", "LogReg + post-proc.", "HistGB + post-proc.", "Main GRU", "TCN-96x4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot added Version13 robustness figures.")
    parser.add_argument("--summary_dir", default="experiments/summary")
    parser.add_argument("--output_dir", default="figures/main")
    return parser.parse_args()


def _style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="DejaVu Serif",
        rc={
            "figure.titlesize": 12.5,
            "axes.titlesize": 10.8,
            "axes.labelsize": 10.2,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.1,
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.facecolor": "#fbfbf8",
            "figure.facecolor": "white",
        },
    )


def plot_block_size_sensitivity(summary_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_dir / "bootstrap_block_sensitivity.csv").copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["model", "bootstrap_block_size"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.20)

    for model, group in df.groupby("model", observed=True):
        ax.plot(
            group["bootstrap_block_size"],
            group["bootstrap_sig_risk_rate"],
            color=MODEL_COLORS[str(model)],
            linewidth=2.0,
            marker="o",
            markersize=5.5,
            label=str(model),
        )
        nominal_rate = float(group["nominal_sig_risk_rate"].iloc[0])
        ax.scatter(
            [0],
            [nominal_rate],
            color=MODEL_COLORS[str(model)],
            marker="D",
            s=42,
            zorder=4,
        )
        ax.plot([0, group["bootstrap_block_size"].iloc[0]], [nominal_rate, group["bootstrap_sig_risk_rate"].iloc[0]], color=MODEL_COLORS[str(model)], linewidth=1.1, alpha=0.45)

    ax.set_xlim(-2, 76)
    ax.set_ylim(0.0, 0.82)
    ax.set_xticks([0, 12, 24, 48, 72])
    ax.set_xticklabels(["nominal", "12h", "24h", "48h", "72h"])
    ax.set_xlabel("Significance criterion")
    ax.set_ylabel("Significant risk-layering rate")
    ax.set_title("BTCUSDT: significance contracts as block length grows")
    ax.grid(axis="y", linewidth=0.6, alpha=0.32)
    ax.grid(axis="x", linewidth=0.5, alpha=0.12)
    ax.legend(ncol=2, frameon=False, loc="upper right")

    fig.savefig(output_dir / "btc_block_size_sensitivity.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "btc_block_size_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_4h_gap(summary_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_dir / "frequency_4h_robustness.csv").copy()
    df = df.loc[df["asset"].isin(["BTC", "ETH"])].copy()
    df["risk_order_rate"] = pd.to_numeric(df["risk_order_rate"], errors="coerce")
    df["return_order_rate"] = pd.to_numeric(df["return_order_rate"], errors="coerce")
    df["model"] = pd.Categorical(
        df["model"],
        categories=["Proxy label", "LogReg", "LogReg + post-proc.", "HistGB + post-proc."],
        ordered=True,
    )
    df = df.sort_values(["asset", "model"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.22, wspace=0.12)

    for ax, asset in zip(axes, ["BTC", "ETH"]):
        asset_df = df.loc[df["asset"] == asset].copy()
        positions = list(range(len(asset_df)))
        ax.barh(
            [pos + 0.16 for pos in positions],
            asset_df["risk_order_rate"].fillna(0.0),
            height=0.28,
            color="#4C78A8",
            label="Risk ordering" if asset == "BTC" else None,
        )
        ax.barh(
            [pos - 0.16 for pos in positions],
            asset_df["return_order_rate"].fillna(0.0),
            height=0.28,
            color="#C44E52",
            label="Return ordering" if asset == "BTC" else None,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(asset_df["model"].astype(str).tolist())
        ax.set_xlim(0.0, 0.82)
        ax.set_title(f"{asset} 4h")
        ax.grid(axis="x", linewidth=0.6, alpha=0.28)
        for pos, risk_rate, ret_rate in zip(positions, asset_df["risk_order_rate"].fillna(0.0), asset_df["return_order_rate"].fillna(0.0)):
            ax.text(float(risk_rate) + 0.015, pos + 0.16, f"{float(risk_rate):.2f}", va="center", fontsize=8.5, color="#355A87")
            ax.text(float(ret_rate) + 0.015, pos - 0.16, f"{float(ret_rate):.2f}", va="center", fontsize=8.5, color="#973C40")

    axes[0].set_ylabel("Model")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("4h robustness: risk semantics remain stronger than return semantics", y=0.97)

    fig.savefig(output_dir / "frequency_4h_risk_return_gap.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "frequency_4h_risk_return_gap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    output_dir = ensure_dir(args.output_dir)
    _style()
    plot_block_size_sensitivity(summary_dir, output_dir)
    plot_frequency_4h_gap(summary_dir, output_dir)


if __name__ == "__main__":
    main()
