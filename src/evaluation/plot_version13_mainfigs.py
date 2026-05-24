from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir


EXTENDED_MODELS = [
    "LogReg + post-proc.",
    "HistGB + post-proc.",
    "Main GRU",
    "TCN-96x4",
    "Transformer-2L",
]

MODEL_COLORS = {
    "Proxy label": "#6A8CAF",
    "LogReg + post-proc.": "#C99700",
    "HistGB + post-proc.": "#C44E52",
    "Main GRU": "#2F5DA8",
    "TCN-96x4": "#59A14F",
    "Transformer-2L": "#8C6BB1",
}

SOURCE_ORDER = ["proxy_label", "logreg_postproc", "main_gru"]
SOURCE_LABELS = {
    "proxy_label": "Proxy label",
    "logreg_postproc": "LogReg + post-proc.",
    "main_gru": "Main GRU",
}
STATE_ORDER = ["bear", "neutral", "bull"]
STATE_LABELS = ["Bear", "Neutral", "Bull"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Version13 main manuscript figures.")
    parser.add_argument("--summary_dir", default="experiments/summary")
    parser.add_argument("--output_dir", default="figures/main")
    return parser.parse_args()


def _style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="DejaVu Serif",
        rc={
            "figure.titlesize": 13.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.2,
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.facecolor": "#FBFBF8",
            "figure.facecolor": "white",
            "axes.edgecolor": "#D0D0CC",
            "grid.color": "#DADAD4",
        },
    )


def _load_btc_extended(summary_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(summary_dir / "results_manifest.csv")
    df = manifest.loc[
        (manifest["experiment_group"] == "benchmark_main")
        & (manifest["asset"] == "BTC")
        & (manifest["model"].isin(EXTENDED_MODELS))
    , ["model", "macro_f1", "switch_day", "risk_order_rate", "sig_risk_rate", "bootstrap_rate"]].copy()
    df["model"] = pd.Categorical(df["model"], categories=EXTENDED_MODELS, ordered=True)
    return df.sort_values("model").reset_index(drop=True)


def plot_ranking_reversal_heatmap(summary_dir: Path, output_dir: Path) -> None:
    df = _load_btc_extended(summary_dir)
    metric_map = {
        "Proxy-label fit": ("macro_f1", False),
        "Path smoothness": ("switch_day", True),
        "Risk ordering": ("risk_order_rate", False),
        "Nominal sig-risk": ("sig_risk_rate", False),
        "Bootstrap sig-risk": ("bootstrap_rate", False),
    }

    rank_matrix = []
    annot_matrix: list[list[str]] = []
    for _, row in df.iterrows():
        annot_row = []
        rank_row = []
        for _, (column, lower_is_better) in metric_map.items():
            series = df[column].astype(float)
            ascending = lower_is_better
            ranks = series.rank(method="min", ascending=ascending)
            rank_value = int(ranks.loc[row.name])
            score_value = float(row[column])
            rank_row.append(rank_value)
            annot_row.append(f"#{rank_value}\n{score_value:.2f}")
        rank_matrix.append(rank_row)
        annot_matrix.append(annot_row)

    heatmap_df = pd.DataFrame(rank_matrix, index=df["model"].astype(str), columns=list(metric_map.keys()))
    annot_df = pd.DataFrame(annot_matrix, index=heatmap_df.index, columns=heatmap_df.columns)

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.84, bottom=0.16)
    sns.heatmap(
        heatmap_df,
        ax=ax,
        cmap=sns.color_palette("YlGnBu_r", as_cmap=True),
        vmin=1,
        vmax=len(heatmap_df.index),
        cbar_kws={"label": "Rank (1 = best)"},
        linewidths=1.2,
        linecolor="white",
        annot=annot_df,
        fmt="",
        annot_kws={"fontsize": 8.6, "ha": "center", "va": "center"},
    )
    ax.set_title("BTC extended pool: ranking reverses across evaluation axes")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.savefig(output_dir / "fig3_extended_pool_ranking_reversal_heatmap.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig3_extended_pool_ranking_reversal_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_semantics_multipanel(summary_dir: Path, output_dir: Path) -> None:
    return_df = pd.read_csv(summary_dir / "btc_economic_meaning_extended" / "by_period_summary.csv")
    risk_df = pd.read_csv(summary_dir / "btc_risk_state_semantics_extended" / "by_period_summary.csv")
    for frame in [return_df, risk_df]:
        frame = frame.loc[frame["source"].isin(SOURCE_ORDER)].copy()
    return_df = return_df.loc[return_df["source"].isin(SOURCE_ORDER)].copy()
    risk_df = risk_df.loc[risk_df["source"].isin(SOURCE_ORDER)].copy()
    for frame in [return_df, risk_df]:
        frame["source"] = pd.Categorical(frame["source"], categories=SOURCE_ORDER, ordered=True)
        frame["state_label"] = pd.Categorical(frame["state_label"], categories=STATE_ORDER, ordered=True)
    periods = sorted(return_df["period"].unique().tolist())
    period_labels = ["2025-H1", "2025-H2", "2026-YTD"]
    metric_specs = [
        (return_df, "future_return_72_mean", "72h return semantics", "RdBu_r", True),
        (risk_df, "future_vol_24_mean", "24h volatility semantics", "YlOrRd", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.8))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.88, bottom=0.12, wspace=0.10, hspace=0.22)

    for row_idx, (source_df, metric, row_title, cmap, centered) in enumerate(metric_specs):
        values = source_df[metric].astype(float)
        if centered:
            vmax = float(values.abs().max())
            vmin = -vmax
            center = 0.0
        else:
            vmin = float(values.min())
            vmax = float(values.max())
            center = None

        for col_idx, source in enumerate(SOURCE_ORDER):
            ax = axes[row_idx, col_idx]
            subset = source_df.loc[source_df["source"] == source]
            matrix = subset.pivot(index="period", columns="state_label", values=metric).reindex(index=periods, columns=STATE_ORDER)
            annot = matrix.map(lambda x: f"{float(x)*100:.1f}%" if pd.notna(x) else "")
            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                linewidths=1.0,
                linecolor="white",
                annot=annot,
                fmt="",
                annot_kws={"fontsize": 8.0},
            )
            if row_idx == 0:
                ax.set_title(SOURCE_LABELS[source], pad=8)
            if col_idx == 0:
                ax.set_ylabel(row_title)
                ax.set_yticklabels(period_labels, rotation=0)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if row_idx == 1:
                ax.set_xticklabels(STATE_LABELS, rotation=0)
            else:
                ax.set_xticklabels([])
            ax.set_xlabel("")
            ax.tick_params(length=0)

    fig.suptitle("Return semantics drift, but risk semantics remain ordered", y=0.96)
    fig.savefig(output_dir / "fig4_return_unstable_vs_risk_stable.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig4_return_unstable_vs_risk_stable.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_block_size_slopegraph(summary_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_dir / "bootstrap_block_sensitivity.csv")
    df = df.loc[df["model"].isin(["Proxy label", "LogReg + post-proc.", "HistGB + post-proc.", "Main GRU", "TCN-96x4"])].copy()
    order = ["Proxy label", "LogReg + post-proc.", "HistGB + post-proc.", "Main GRU", "TCN-96x4"]
    x_labels = ["nominal", "12h", "24h", "48h", "72h"]
    x_positions = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.20)

    for model in order:
        group = df.loc[df["model"] == model].sort_values("bootstrap_block_size")
        y_values = [float(group["nominal_sig_risk_rate"].iloc[0]), *group["bootstrap_sig_risk_rate"].astype(float).tolist()]
        ax.plot(
            x_positions,
            y_values,
            marker="o",
            markersize=5.8,
            linewidth=2.0,
            color=MODEL_COLORS[model],
            alpha=0.96,
            label=model,
        )
        ax.scatter([x_positions[0]], [y_values[0]], marker="D", s=42, color=MODEL_COLORS[model], zorder=4)

    ax.set_xlim(-0.15, 4.15)
    ax.set_ylim(0.0, 0.82)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Significant risk-layering rate")
    ax.set_xlabel("Significance criterion")
    ax.set_title("BTCUSDT: significance shrinks under longer bootstrap blocks")
    ax.grid(axis="y", linewidth=0.6, alpha=0.30)
    ax.grid(axis="x", linewidth=0.4, alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)

    fig.savefig(output_dir / "fig5_block_bootstrap_slopegraph.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig5_block_bootstrap_slopegraph.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selector_matrix(summary_dir: Path, output_dir: Path) -> None:
    selection = pd.read_csv(summary_dir / "evaluation_framework_increment" / "framework_selection_comparison.csv")
    selection = selection.loc[selection["framework"] != "semantic_protocol"].copy()

    framework_labels = {
        "classification_only": "C only",
        "ct_topsis_mcdm": "C+T TOPSIS",
        "ct_weighted_mcdm": "C+T weighted",
        "cts_weighted_oracle": "C+T+S weighted",
    }
    pool_labels = {
        "btc_extended": "BTC extended",
        "cross_asset_common": "Five-asset common",
    }
    method_order = ["logreg", "logreg_postproc", "histgb_postproc", "main_gru", "main_lstm", "tcn_96x4"]
    method_labels = {
        "logreg": "LogReg",
        "logreg_postproc": "LogReg+post",
        "histgb_postproc": "HistGB+post",
        "main_gru": "GRU",
        "main_lstm": "LSTM",
        "tcn_96x4": "TCN",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.82, bottom=0.20, wspace=0.18)

    for ax, pool in zip(axes, ["btc_extended", "cross_asset_common"]):
        pool_df = selection.loc[selection["pool"] == pool].copy()
        matrix = pd.DataFrame(0.0, index=list(framework_labels.values()), columns=[method_labels[m] for m in method_order])
        annotations = pd.DataFrame("", index=matrix.index, columns=matrix.columns)

        for framework_key, framework_label in framework_labels.items():
            rows = pool_df.loc[pool_df["framework"] == framework_key].copy()
            for method_key, sub in rows.groupby("selected_method"):
                if method_key not in method_order:
                    continue
                col = method_labels[method_key]
                count = len(sub)
                regret = float(sub["semantic_regret_vs_protocol"].mean())
                matrix.loc[framework_label, col] = regret
                if pool == "btc_extended":
                    annotations.loc[framework_label, col] = f"pick\n{regret:.3f}"
                else:
                    annotations.loc[framework_label, col] = f"{count}/5\n{regret:.3f}"

        sns.heatmap(
            matrix,
            ax=ax,
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            vmin=0.0,
            vmax=float(selection["semantic_regret_vs_protocol"].max()),
            linewidths=1.1,
            linecolor="white",
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 8.1},
            cbar=ax is axes[-1],
            cbar_kws={"label": "Semantic regret"} if ax is axes[-1] else None,
        )
        ax.set_title(pool_labels[pool])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0, length=0)

    axes[0].set_ylabel("Selector")
    fig.suptitle("Conventional selectors often collapse to the same picked model", y=0.95)
    fig.savefig(output_dir / "fig6_selector_pick_matrix.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig6_selector_pick_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown_dumbbell(summary_dir: Path, output_dir: Path) -> None:
    manifest = pd.read_csv(summary_dir / "results_manifest.csv")
    df = manifest.loc[manifest["experiment_group"] == "drawdown_deconfounding", ["experiment_id", "model", "macro_f1", "sig_risk_rate"]].copy()

    rows = [
        ("LogReg feature-side", df.loc[df["experiment_id"] == "btc_logreg_no_drawdown_feature", "macro_f1"].iloc[0], df.loc[df["experiment_id"] == "btc_logreg_no_drawdown_feature", "sig_risk_rate"].iloc[0], 0.769738, 15 / 21),
        ("Main GRU feature-side", df.loc[df["experiment_id"] == "btc_gru_no_drawdown_feature", "macro_f1"].iloc[0], df.loc[df["experiment_id"] == "btc_gru_no_drawdown_feature", "sig_risk_rate"].iloc[0], 0.665132, 16 / 21),
        ("LogReg label-side", df.loc[df["experiment_id"] == "btc_logreg_no_drawdown_rule", "macro_f1"].iloc[0], df.loc[df["experiment_id"] == "btc_logreg_no_drawdown_rule", "sig_risk_rate"].iloc[0], 0.769738, 15 / 21),
        ("Main GRU label-side", df.loc[df["experiment_id"] == "btc_gru_no_drawdown_rule", "macro_f1"].iloc[0], df.loc[df["experiment_id"] == "btc_gru_no_drawdown_rule", "sig_risk_rate"].iloc[0], 0.665132, 16 / 21),
    ]
    plot_df = pd.DataFrame(rows, columns=["case", "perturbed_macro_f1", "perturbed_sig_risk", "default_macro_f1", "default_sig_risk"])
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    display_labels = [
        "GRU - no rule",
        "LogReg - no rule",
        "GRU - no feature",
        "LogReg - no feature",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.6, 5.0),
        gridspec_kw={"width_ratios": [1.15, 2.45, 2.45]},
        sharey=False,
    )
    label_ax, metric_ax1, metric_ax2 = axes
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.18, wspace=0.06)

    metric_specs = [
        ("default_macro_f1", "perturbed_macro_f1", "Macro-F1", metric_ax1, (0.55, 0.92)),
        ("default_sig_risk", "perturbed_sig_risk", "Significant risk-layering rate", metric_ax2, (0.10, 0.82)),
    ]

    label_ax.set_xlim(0.38, 1.0)
    label_ax.set_ylim(-0.5, len(plot_df) - 0.5)
    label_ax.axis("off")
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        label_ax.text(0.98, idx, display_labels[idx], fontsize=9.3, color="#333333", va="center", ha="right")

    for left_col, right_col, title, ax, xlim in metric_specs:
        for idx, row in plot_df.iterrows():
            color = "#C44E52" if "label-side" in row["case"] else "#4C78A8"
            ax.plot([row[left_col], row[right_col]], [idx, idx], color="#B8B8B0", linewidth=2.2, zorder=1)
            ax.scatter(row[left_col], idx, s=58, color="#1F1F1F", edgecolor="white", linewidth=0.8, zorder=3)
            ax.scatter(row[right_col], idx, s=58, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.5, len(plot_df) - 0.5)
        ax.grid(axis="x", linewidth=0.6, alpha=0.28)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0, labelleft=False)

    metric_ax1.set_ylabel("")
    metric_ax2.tick_params(axis="y", left=False, labelleft=False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1F1F1F", markeredgecolor="white", markersize=7, label="Default rule"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#4C78A8", markeredgecolor="white", markersize=7, label="No drawdown feature"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#C44E52", markeredgecolor="white", markersize=7, label="No drawdown filter"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Drawdown deconfounding: label fit can rise while semantic validity collapses", y=0.97)

    fig.savefig(output_dir / "fig7_drawdown_deconfounding_dumbbell.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig7_drawdown_deconfounding_dumbbell.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    output_dir = ensure_dir(args.output_dir)
    _style()
    plot_ranking_reversal_heatmap(summary_dir, output_dir)
    plot_semantics_multipanel(summary_dir, output_dir)
    plot_block_size_slopegraph(summary_dir, output_dir)
    plot_selector_matrix(summary_dir, output_dir)
    plot_drawdown_dumbbell(summary_dir, output_dir)


if __name__ == "__main__":
    main()
