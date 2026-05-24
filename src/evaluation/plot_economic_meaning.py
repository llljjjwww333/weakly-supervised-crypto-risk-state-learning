from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir

STATE_ORDER = ["bear", "neutral", "bull"]
SOURCE_ORDER = ["proxy_label", "logreg", "main_gru"]
SOURCE_LABELS = {
    "proxy_label": "Proxy label",
    "logreg": "Logistic regression",
    "main_gru": "Main GRU",
}
STATE_LABELS = {
    "bear": "Bear",
    "neutral": "Neutral",
    "bull": "Bull",
}
PALETTE = {"bear": "#b03a2e", "neutral": "#7f8c8d", "bull": "#117a65"}
METRIC_CONFIG = [
    ("future_return_24_mean", "Mean Future\nReturn (24h)"),
    ("future_return_72_mean", "Mean Future\nReturn (72h)"),
    ("future_vol_24_mean", "Mean Future\nVolatility (24h)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot economic meaning evaluation summaries.")
    parser.add_argument("--summary_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def plot_overall(overall: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(METRIC_CONFIG), figsize=(14.5, 5.1))
    fig.subplots_adjust(top=0.80, bottom=0.24, left=0.07, right=0.985, wspace=0.28)

    for ax, (metric, title) in zip(axes, METRIC_CONFIG):
        sns.barplot(
            data=overall,
            x="source",
            y=metric,
            hue="state_label",
            order=SOURCE_ORDER,
            hue_order=STATE_ORDER,
            palette=PALETTE,
            ax=ax,
        )
        ax.set_title(title, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(range(len(SOURCE_ORDER)))
        ax.set_xticklabels([SOURCE_LABELS[label] for label in SOURCE_ORDER], rotation=0)
        ax.tick_params(axis="x", pad=6)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.grid(axis="y", linewidth=0.6, alpha=0.35)
        ax.grid(axis="x", visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    for ax in axes[1:]:
        if ax.legend_ is not None:
            ax.legend_.remove()
    fig.legend(
        handles,
        [STATE_LABELS.get(label.lower(), label) for label in labels],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
    )
    fig.suptitle("BTC Test Period: Overall Economic Meaning by State Source", y=0.95)
    fig.savefig(output_dir / "btc_economic_meaning_overall.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_period(by_period: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        len(METRIC_CONFIG),
        len(SOURCE_ORDER),
        figsize=(16, 10),
        sharex=True,
    )
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.09, right=0.99, hspace=0.34, wspace=0.20)

    for row_idx, (metric, row_title) in enumerate(METRIC_CONFIG):
        row_values = by_period[metric].dropna()
        ymin = float(row_values.min()) if not row_values.empty else -0.01
        ymax = float(row_values.max()) if not row_values.empty else 0.01
        pad = max((ymax - ymin) * 0.15, 1e-4)

        for col_idx, source in enumerate(SOURCE_ORDER):
            ax = axes[row_idx, col_idx]
            subset = by_period.loc[by_period["source"] == source].copy()
            sns.lineplot(
                data=subset,
                x="period",
                y=metric,
                hue="state_label",
                hue_order=STATE_ORDER,
                style="state_label",
                markers=True,
                dashes=False,
                palette=PALETTE,
                linewidth=2.0,
                markersize=7.5,
                ax=ax,
            )
            if ax.legend_ is not None:
                ax.legend_.remove()
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlabel("" if row_idx < len(METRIC_CONFIG) - 1 else "Period")
            ax.set_ylabel(row_title if col_idx == 0 else "")
            ax.tick_params(axis="x", rotation=0)
            ax.grid(axis="y", linewidth=0.6, alpha=0.35)
            ax.grid(axis="x", linewidth=0.4, alpha=0.15)
            if row_idx == 0:
                ax.set_title(SOURCE_LABELS[source], pad=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        [STATE_LABELS.get(label.lower(), label) for label in labels],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.8,
        columnspacing=1.8,
    )
    fig.suptitle(
        "BTC Test Period: Return Semantics Shift Across Periods, Risk Semantics Stay Stable",
        y=0.93,
    )
    fig.savefig(output_dir / "btc_economic_meaning_by_period.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    output_dir = ensure_dir(args.output_dir)

    overall = pd.read_csv(summary_dir / "overall_summary.csv")
    by_period = pd.read_csv(summary_dir / "by_period_summary.csv")
    overall["state_label"] = pd.Categorical(overall["state_label"], categories=STATE_ORDER, ordered=True)
    by_period["state_label"] = pd.Categorical(by_period["state_label"], categories=STATE_ORDER, ordered=True)

    sns.set_theme(
        style="whitegrid",
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
    plot_overall(overall, output_dir)
    plot_by_period(by_period, output_dir)


if __name__ == "__main__":
    main()
