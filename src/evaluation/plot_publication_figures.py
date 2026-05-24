from __future__ import annotations

import argparse
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir


MODEL_PALETTE = {
    "GRU": "#2F5DA8",
    "LSTM": "#5FA8A9",
    "TCN-64x3": "#F28E2B",
    "TCN-128x3": "#E15759",
    "TCN-96x4": "#59A14F",
    "Transformer-2L": "#9C6ADE",
    "Logreg + post-proc.": "#D4A72C",
    "HistGB + post-proc.": "#C44E52",
}

TRADEOFF_ORDER = [
    "Logreg + post-proc.",
    "HistGB + post-proc.",
    "GRU",
    "TCN-96x4",
    "Transformer-2L",
]

TRADEOFF_LINESTYLES = {
    "Logreg + post-proc.": (0, (4.0, 1.5)),
    "HistGB + post-proc.": "-",
    "GRU": "-",
    "TCN-96x4": (0, (2.2, 1.6)),
    "Transformer-2L": (0, (1.0, 1.4)),
}

TRADEOFF_MARKERS = {
    "Logreg + post-proc.": "o",
    "HistGB + post-proc.": "s",
    "GRU": "D",
    "TCN-96x4": "^",
    "Transformer-2L": "P",
}

TEMPORAL_MARKERS = {
    "GRU": "D",
    "LSTM": "o",
    "TCN-64x3": "^",
    "TCN-128x3": "^",
    "TCN-96x4": "^",
    "Transformer-2L": "P",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot publication-ready main-result figures.")
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
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "legend.fontsize": 9.5,
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.facecolor": "#fbfbf8",
            "figure.facecolor": "white",
            "axes.edgecolor": "#b5b5b5",
        },
    )


def _read_single_row(path: Path) -> pd.Series:
    return pd.read_csv(path).iloc[0]


def load_temporal_backbone_df(summary_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_dir / "btc_temporal_backbone_comparison_extended.csv").copy()
    transformer_cls_path = summary_dir / "btc_classification_transformer" / "classification_metrics.csv"
    transformer_stability_path = summary_dir / "btc_transformer_stability.csv"
    if transformer_cls_path.exists() and transformer_stability_path.exists():
        cls = _read_single_row(transformer_cls_path)
        stability = _read_single_row(transformer_stability_path)
        transformer_row = {
            "model": "Transformer-2L",
            "macro_f1": float(cls["macro_f1"]),
            "balanced_accuracy": float(cls["balanced_accuracy"]),
            "bull_f1": float(cls["bull_f1"]),
            "bear_f1": float(cls["bear_f1"]),
            "transitions": int(stability["transitions"]),
            "daily_switch_rate": float(stability["daily_switch_rate"]),
            "avg_state_duration_bars": float(stability["avg_state_duration_bars"]),
        }
        df = pd.concat([df, pd.DataFrame([transformer_row])], ignore_index=True)
    df.to_csv(summary_dir / "btc_temporal_backbone_comparison_with_transformer.csv", index=False)
    return df


def load_extended_tradeoff_df(summary_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    postproc_gru = pd.read_csv(summary_dir / "btc_postprocess_vs_gru_summary.csv")
    risk_ext = pd.read_csv(summary_dir / "btc_risk_state_significance_extended" / "risk_significance_summary.csv")
    tcn_risk = pd.read_csv(summary_dir / "btc_risk_state_significance_gru_tcn96_compare" / "risk_significance_summary.csv")
    histgb_risk = pd.read_csv(summary_dir / "btc_risk_state_significance_histgb_postproc" / "risk_significance_summary.csv")
    transformer_risk = pd.read_csv(summary_dir / "btc_risk_state_significance_transformer" / "risk_significance_summary.csv")

    def add_row(
        model: str,
        macro_f1: float,
        daily_switch_rate: float,
        avg_state_duration_bars: float,
        risk_order_rate: float,
        significant_risk_layering_rate: float,
    ) -> None:
        rows.append(
            {
                "model": model,
                "macro_f1": macro_f1,
                "daily_switch_rate": daily_switch_rate,
                "avg_state_duration_bars": avg_state_duration_bars,
                "risk_order_rate": risk_order_rate,
                "significant_risk_layering_rate": significant_risk_layering_rate,
            }
        )

    logreg_row = postproc_gru.loc[postproc_gru["method"] == "logreg_postproc_match"].iloc[0]
    logreg_risk = risk_ext.loc[risk_ext["source"] == "logreg_postproc"].iloc[0]
    add_row(
        "Logreg + post-proc.",
        float(logreg_row["macro_f1"]),
        float(logreg_row["daily_switch_rate"]),
        float(logreg_row["avg_state_duration_bars"]),
        float(logreg_risk["order_match_rate"]),
        float(logreg_risk["significant_risk_layering_rate"]),
    )

    histgb_cls = _read_single_row(Path("experiments/revision/histgb_btc_postproc/eval/classification_metrics.csv"))
    histgb_stability = _read_single_row(Path("experiments/revision/histgb_btc_postproc/stability.csv"))
    histgb_sig = histgb_risk.loc[histgb_risk["source"] == "histgb_postproc"].iloc[0]
    add_row(
        "HistGB + post-proc.",
        float(histgb_cls["macro_f1"]),
        float(histgb_stability["daily_switch_rate"]),
        float(histgb_stability["avg_state_duration_bars"]),
        float(histgb_sig["order_match_rate"]),
        float(histgb_sig["significant_risk_layering_rate"]),
    )

    gru_row = postproc_gru.loc[postproc_gru["method"] == "main_gru"].iloc[0]
    gru_risk = risk_ext.loc[risk_ext["source"] == "main_gru"].iloc[0]
    add_row(
        "GRU",
        float(gru_row["macro_f1"]),
        float(gru_row["daily_switch_rate"]),
        float(gru_row["avg_state_duration_bars"]),
        float(gru_risk["order_match_rate"]),
        float(gru_risk["significant_risk_layering_rate"]),
    )

    tcn_row = pd.read_csv(summary_dir / "btc_temporal_backbone_comparison_extended.csv")
    tcn_row = tcn_row.loc[tcn_row["model"] == "TCN-96x4"].iloc[0]
    tcn_sig_row = tcn_risk.loc[tcn_risk["source"] == "main_tcn_96x4"].iloc[0]
    add_row(
        "TCN-96x4",
        float(tcn_row["macro_f1"]),
        float(tcn_row["daily_switch_rate"]),
        float(tcn_row["avg_state_duration_bars"]),
        float(tcn_sig_row["order_match_rate"]),
        float(tcn_sig_row["significant_risk_layering_rate"]),
    )

    transformer_cls = _read_single_row(summary_dir / "btc_classification_transformer" / "classification_metrics.csv")
    transformer_stability = _read_single_row(summary_dir / "btc_transformer_stability.csv")
    transformer_sig = transformer_risk.loc[transformer_risk["source"] == "transformer_btc"].iloc[0]
    add_row(
        "Transformer-2L",
        float(transformer_cls["macro_f1"]),
        float(transformer_stability["daily_switch_rate"]),
        float(transformer_stability["avg_state_duration_bars"]),
        float(transformer_sig["order_match_rate"]),
        float(transformer_sig["significant_risk_layering_rate"]),
    )

    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=TRADEOFF_ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    df.to_csv(summary_dir / "btc_extended_candidate_tradeoff.csv", index=False)
    return df


def plot_temporal_backbone(summary_dir: Path, output_dir: Path) -> None:
    df = load_temporal_backbone_df(summary_dir)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.84, bottom=0.18)
    ax.set_facecolor("#fcfcf7")

    label_offsets = {
        "GRU": (0.004, 0.0014),
        "LSTM": (0.004, 0.0012),
        "TCN-96x4": (0.007, 0.0020),
        "Transformer-2L": (0.006, -0.0022),
    }
    size_scale = 5.2

    xmin = df["daily_switch_rate"].min() - 0.025
    xmax = df["daily_switch_rate"].max() + 0.035
    ymin = df["macro_f1"].min() - 0.020
    ymax = df["macro_f1"].max() + 0.020

    # Emphasize the desirable corner: higher proxy-label fit and lower switching.
    ax.axvspan(xmin, df["daily_switch_rate"].quantile(0.35), color="#EAF4E2", alpha=0.45, zorder=0)
    ax.axhspan(df["macro_f1"].quantile(0.65), ymax, color="#EEF5FF", alpha=0.35, zorder=0)

    tcn_df = df.loc[df["model"].str.startswith("TCN-")].sort_values("daily_switch_rate")
    ax.plot(
        tcn_df["daily_switch_rate"],
        tcn_df["macro_f1"],
        color="#D4831F",
        linewidth=1.4,
        alpha=0.55,
        linestyle=(0, (3.2, 2.2)),
        zorder=1,
    )

    for _, row in df.iterrows():
        model = row["model"]
        ax.scatter(
            row["daily_switch_rate"],
            row["macro_f1"],
            s=70 + row["avg_state_duration_bars"] * size_scale,
            color=MODEL_PALETTE.get(model, "#6b6b6b"),
            marker=TEMPORAL_MARKERS.get(model, "o"),
            edgecolor="white",
            linewidth=1.2,
            alpha=0.96,
            zorder=3,
        )
        if model in {"TCN-64x3", "TCN-128x3"}:
            continue
        x_offset, y_offset = label_offsets.get(model, (0.005, 0.001))
        label_text = f"{model}\n{row['avg_state_duration_bars']:.0f} h"
        if model == "TCN-96x4":
            ax.annotate(
                label_text,
                xy=(row["daily_switch_rate"], row["macro_f1"]),
                xytext=(row["daily_switch_rate"] + x_offset, row["macro_f1"] + y_offset),
                textcoords="data",
                fontsize=9.0,
                va="center",
                ha="left",
                color="#1f1f1f",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.15},
                arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "#6a6a6a"},
            )
        else:
            ax.text(
                row["daily_switch_rate"] + x_offset,
                row["macro_f1"] + y_offset,
                label_text,
                fontsize=9.2,
                va="center",
                color="#1f1f1f",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.15},
            )

    ax.axvline(df.loc[df["model"] == "GRU", "daily_switch_rate"].iloc[0], color="#4c78a8", lw=0.8, alpha=0.25)
    ax.axhline(df.loc[df["model"] == "GRU", "macro_f1"].iloc[0], color="#4c78a8", lw=0.8, alpha=0.25)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Daily switching rate (lower is smoother)")
    ax.set_ylabel("Macro-F1 against proxy labels")
    ax.set_title("BTCUSDT temporal-backbone tradeoff")
    ax.grid(axis="both", linewidth=0.6, alpha=0.28, color="#cfcfcf")

    ax.annotate(
        "better proxy-label fit",
        xy=(xmin + 0.005, ymax - 0.006),
        xytext=(xmin + 0.005, ymax - 0.020),
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#5B6C8E"},
        fontsize=8.6,
        color="#4A5870",
        ha="left",
    )
    ax.annotate(
        "smoother path",
        xy=(xmin + 0.010, ymin + 0.006),
        xytext=(xmin + 0.080, ymin + 0.006),
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#5B6C8E"},
        fontsize=8.6,
        color="#4A5870",
        va="center",
    )

    family_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=MODEL_PALETTE["GRU"], markeredgecolor="white", markersize=8, label="RNN"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=MODEL_PALETTE["TCN-96x4"], markeredgecolor="white", markersize=8, label="TCN"),
        Line2D([0], [0], marker="P", color="none", markerfacecolor=MODEL_PALETTE["Transformer-2L"], markeredgecolor="white", markersize=8, label="Transformer"),
    ]
    legend = ax.legend(
        handles=family_handles,
        title="Backbone family",
        loc="lower right",
        frameon=True,
        fancybox=False,
        borderpad=0.65,
        labelspacing=0.7,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d6d6d6")
    legend.get_frame().set_alpha(0.94)
    tcn_rows = (
        df.loc[df["model"].isin(["TCN-64x3", "TCN-128x3", "TCN-96x4"]), ["model", "macro_f1", "daily_switch_rate", "avg_state_duration_bars"]]
        .copy()
        .set_index("model")
        .loc[["TCN-64x3", "TCN-128x3", "TCN-96x4"]]
    )
    inset_lines = ["TCN variants", "F1 / sw.day / dur."]
    for model, row in tcn_rows.iterrows():
        inset_lines.append(f"{model}: {row['macro_f1']:.3f} / {row['daily_switch_rate']:.3f} / {row['avg_state_duration_bars']:.0f} h")
    ax.text(
        0.02,
        0.18,
        "\n".join(inset_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        color="#2a2a2a",
        bbox={"facecolor": "white", "edgecolor": "#d6d6d6", "boxstyle": "round,pad=0.28", "alpha": 0.96},
    )
    fig.savefig(output_dir / "btc_temporal_backbone_comparison_extended.pdf", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / "btc_temporal_backbone_comparison_extended.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _benefit_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    score = (series - series.min()) / span
    return score if higher_is_better else 1 - score


def plot_btc_extended_tradeoff(summary_dir: Path, output_dir: Path) -> None:
    df = load_extended_tradeoff_df(summary_dir)
    metrics = [
        ("macro_f1", "Fit", True),
        ("daily_switch_rate", "Smooth", False),
        ("risk_order_rate", "Risk ordering", True),
        ("significant_risk_layering_rate", "Sig. risk", True),
    ]
    angles = [idx / len(metrics) * 2 * pi for idx in range(len(metrics))]
    angles += angles[:1]

    fig = plt.figure(figsize=(9.2, 7.4))
    ax = fig.add_subplot(111, projection="polar")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.82, bottom=0.22)
    ax.set_facecolor("#fcfcf8")
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_axisbelow(True)

    for model in TRADEOFF_ORDER:
        row = df.loc[df["model"] == model].iloc[0]
        color = MODEL_PALETTE[model]
        normalized = []
        for metric, _, higher in metrics:
            normalized.append(float(_benefit_score(df[metric], higher).loc[row.name]))
        values = normalized + normalized[:1]
        linewidth = 2.7 if model == "GRU" else 2.0
        alpha = 1.0 if model == "GRU" else 0.95
        ax.plot(
            angles,
            values,
            color=color,
            linewidth=linewidth,
            linestyle=TRADEOFF_LINESTYLES[model],
            marker=TRADEOFF_MARKERS[model],
            markersize=6.2,
            alpha=alpha,
            label=model,
        )
        ax.fill(angles, values, color=color, alpha=0.055 if model != "GRU" else 0.09)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _, label, _ in metrics], fontsize=9.6)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8.5, color="#6a6a6a")
    ax.grid(linewidth=0.82, alpha=0.34, color="#c9c9c9")
    ax.spines["polar"].set_color("#c3c3c3")
    fig.suptitle("BTCUSDT extended-pool tradeoff across fit, smoothness, and semantics", y=0.965, fontsize=11.8)

    leader_annotations = [
        (0.50, 0.86, "HistGB + post-proc.\nfit + smoothness leader", MODEL_PALETTE["HistGB + post-proc."]),
        (0.82, 0.15, "Logreg + post-proc.\nrisk-order leader", MODEL_PALETTE["Logreg + post-proc."]),
        (0.16, 0.15, "GRU\nsig-risk leader", MODEL_PALETTE["GRU"]),
    ]
    for x, y, text, color in leader_annotations:
        fig.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8.8,
            color="#1f1f1f",
            bbox={"facecolor": "white", "edgecolor": color, "boxstyle": "round,pad=0.28", "alpha": 0.94},
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_PALETTE[model],
            linewidth=2.5 if model == "GRU" else 2.0,
            linestyle=TRADEOFF_LINESTYLES[model],
            marker=TRADEOFF_MARKERS[model],
            markersize=6.2,
            label=model,
        )
        for model in TRADEOFF_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.27),
        columnspacing=1.4,
        handletextpad=0.6,
    )
    fig.text(
        0.5,
        0.06,
        "Each spoke is min-max normalized within the BTC extended candidate pool; larger is better on every spoke.",
        ha="center",
        fontsize=8.5,
        color="#4a4a4a",
    )
    fig.savefig(output_dir / "btc_extended_tradeoff_radar.pdf", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / "btc_gru_vs_tcn96_tradeoff.pdf", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / "btc_extended_tradeoff_radar.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "btc_gru_vs_tcn96_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown_deconfounding(summary_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_dir / "btc_drawdown_deconfounding_comparison.csv")
    df["family"] = df["method"].str.extract(r"^(LogReg|GRU)")
    df["variant"] = df["method"].str.contains("no drawdown").map({False: "Default", True: "No drawdown"})
    metrics = [
        ("macro_f1", "Macro-F1", "{:.3f}"),
        ("balanced_accuracy", "Balanced accuracy", "{:.3f}"),
        ("risk_layering_rate", "Sig. risk-layering rate", "{:.1%}"),
    ]
    colors = {"LogReg": "#f58518", "GRU": "#4c78a8"}
    markers = {"Default": "o", "No drawdown": "D"}

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.73, bottom=0.20, wspace=0.52)

    for ax, (metric, title, fmt) in zip(axes, metrics):
        values = df[metric]
        pad = max((values.max() - values.min()) * 0.22, 0.015)
        for x_pos, family in enumerate(["LogReg", "GRU"]):
            subset = df.loc[df["family"] == family].set_index("variant")
            y0 = float(subset.loc["Default", metric])
            y1 = float(subset.loc["No drawdown", metric])
            ax.plot([x_pos, x_pos], [y0, y1], color=colors[family], linewidth=2.0, alpha=0.75)
            for variant, y in [("Default", y0), ("No drawdown", y1)]:
                ax.scatter(
                    x_pos,
                    y,
                    s=74,
                    marker=markers[variant],
                    color=colors[family],
                    edgecolor="white",
                    linewidth=0.9,
                    zorder=3,
                )
                x_offset = 0.08 if family == "LogReg" else -0.08
                ha = "left" if family == "LogReg" else "right"
                ax.text(
                    x_pos + x_offset,
                    y,
                    fmt.format(y),
                    ha=ha,
                    va="center",
                    fontsize=8.4,
                    color="#333333",
                )
        ax.set_title(title, pad=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["LogReg", "GRU"])
        ax.set_xlim(-0.36, 1.36)
        ax.set_ylim(values.min() - pad, values.max() + pad)
        ax.grid(axis="y", linewidth=0.6, alpha=0.35)
        ax.grid(axis="x", visible=False)

    handles = [
        plt.Line2D([0], [0], color="#555555", marker=markers["Default"], linestyle="", label="Default"),
        plt.Line2D([0], [0], color="#555555", marker=markers["No drawdown"], linestyle="", label="No drawdown feature"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("BTCUSDT drawdown-feature deconfounding check", y=0.86)
    fig.savefig(output_dir / "btc_drawdown_deconfounding_comparison.pdf", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / "btc_drawdown_deconfounding_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    output_dir = ensure_dir(args.output_dir)

    _style()
    plot_temporal_backbone(summary_dir, output_dir)
    plot_btc_extended_tradeoff(summary_dir, output_dir)
    plot_drawdown_deconfounding(summary_dir, output_dir)


if __name__ == "__main__":
    main()
