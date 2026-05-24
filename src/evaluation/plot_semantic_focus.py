from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

from src.utils.io import ensure_dir

STATE_ORDER = ["bear", "neutral", "bull"]
STATE_LABELS = ["Bear\nhigh-risk", "Neutral\ntransition", "Bull\nlow-risk"]
SOURCE_ORDER = ["proxy_label", "logreg_postproc_match", "main_gru"]
SOURCE_LABELS = {
    "proxy_label": "Proxy label",
    "logreg_postproc_match": "Logreg + post-proc.",
    "main_gru": "Main GRU",
}
OVERALL_ROWS = [
    ("future_return_24_mean", "Return 24h"),
    ("future_return_72_mean", "Return 72h"),
    ("future_vol_24_mean", "Volatility 24h"),
]
PERIOD_LABELS = {
    "2026-H1": "2026-YTD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot publication-focused semantic figures.")
    parser.add_argument("--summary_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _style() -> None:
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


def _heatmap_kwargs() -> dict[str, object]:
    return {
        "linewidths": 1.15,
        "linecolor": "#f3f3ef",
        "annot_kws": {"fontsize": 8.6, "color": "#202020"},
        "square": False,
    }


def _available(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[frame["source"].isin(SOURCE_ORDER)].copy()
    out["source"] = pd.Categorical(out["source"], categories=SOURCE_ORDER, ordered=True)
    out["state_label"] = pd.Categorical(out["state_label"], categories=STATE_ORDER, ordered=True)
    return out.sort_values(["source", "period" if "period" in out.columns else "state_label", "state_label"])


def plot_overall(overall: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    row_order: list[str] = []
    for source in SOURCE_ORDER:
        source_frame = overall.loc[overall["source"] == source]
        for metric, label in OVERALL_ROWS:
            row_name = f"{SOURCE_LABELS[source]}\n{label}"
            row_order.append(row_name)
            for state in STATE_ORDER:
                value = float(source_frame.loc[source_frame["state_label"] == state, metric].iloc[0])
                rows.append(
                    {
                        "row": row_name,
                        "state": state,
                        "value": value,
                        "annotation": f"{value * 100:.2f}%",
                    }
                )
    plot_df = pd.DataFrame(rows)
    matrix = plot_df.pivot(index="row", columns="state", values="value").reindex(row_order)
    annotations = plot_df.pivot(index="row", columns="state", values="annotation").reindex(row_order)
    matrix = matrix[STATE_ORDER]
    annotations = annotations[STATE_ORDER]

    vmax = float(matrix.abs().max().max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(9.1, 6.5))
    fig.subplots_adjust(left=0.33, right=0.88, top=0.87, bottom=0.13)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        norm=norm,
        annot=annotations,
        fmt="",
        cbar=False,
        **_heatmap_kwargs(),
    )
    cbar_ax = fig.add_axes([0.90, 0.19, 0.02, 0.56])
    scalar = ScalarMappable(norm=norm, cmap="RdBu_r")
    cbar = fig.colorbar(scalar, cax=cbar_ax)
    cbar.outline.set_edgecolor("#c9c9c9")
    cbar.ax.tick_params(labelsize=8.4, length=0)
    cbar.set_label("Mean future outcome", fontsize=8.8)
    ax.set_xticklabels(STATE_LABELS, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("BTCUSDT: ex-post state semantics by method and metric", pad=14)
    ax.tick_params(axis="y", rotation=0, length=0)
    for split in [3, 6]:
        ax.hlines(split, *ax.get_xlim(), colors="#d8d8d1", linewidth=1.2)
    fig.savefig(output_dir / "btc_semantics_overall_main.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "btc_semantics_overall_main.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _period_heatmap(
    by_period: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    cmap: str,
    center: float | None,
    value_suffix: str,
) -> None:
    periods = sorted(by_period["period"].dropna().unique().tolist())
    display_periods = [PERIOD_LABELS.get(period, period) for period in periods]
    fig, axes = plt.subplots(1, len(SOURCE_ORDER), figsize=(12.4, 4.0), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.80, bottom=0.21, wspace=0.08)
    values = by_period[metric].astype(float)
    norm = None
    vmin = None
    vmax = None
    if center is not None:
        vmax = float(values.abs().max())
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=center, vmax=vmax)
        vmin = -vmax
    else:
        vmin = float(values.min())
        vmax = float(values.max())

    for ax, source in zip(axes, SOURCE_ORDER):
        subset = by_period.loc[by_period["source"] == source]
        matrix = subset.pivot(index="period", columns="state_label", values=metric).reindex(index=periods, columns=STATE_ORDER)
        annot = matrix.map(lambda value: f"{float(value) * 100:.2f}{value_suffix}" if pd.notna(value) else "")
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            norm=norm,
            annot=annot,
            fmt="",
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            **_heatmap_kwargs(),
        )
        ax.set_title(SOURCE_LABELS[source], pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(STATE_LABELS, rotation=0)
        ax.set_yticklabels(display_periods, rotation=0)
        ax.tick_params(axis="y", rotation=0, length=0)

    cbar_ax = fig.add_axes([0.915, 0.24, 0.015, 0.48])
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(scalar, cax=cbar_ax)
    cbar.outline.set_edgecolor("#c9c9c9")
    cbar.ax.tick_params(labelsize=8.1, length=0)
    cbar.set_label("Mean future outcome", fontsize=8.5)

    fig.suptitle(title, y=0.96)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_return_semantics(by_period: pd.DataFrame, output_dir: Path) -> None:
    _period_heatmap(
        by_period,
        "future_return_72_mean",
        "BTCUSDT: 72h return semantics are unstable across subperiods",
        output_dir / "btc_return_semantics_by_period_main.pdf",
        cmap="RdBu_r",
        center=0.0,
        value_suffix="%",
    )


def plot_risk_semantics(by_period: pd.DataFrame, output_dir: Path) -> None:
    _period_heatmap(
        by_period,
        "future_vol_24_mean",
        "BTCUSDT: 24h volatility semantics remain ordered by risk",
        output_dir / "btc_risk_semantics_by_period_main.pdf",
        cmap="YlOrRd",
        center=None,
        value_suffix="%",
    )


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    output_dir = ensure_dir(args.output_dir)

    overall = _available(pd.read_csv(summary_dir / "overall_summary.csv"))
    by_period = _available(pd.read_csv(summary_dir / "by_period_summary.csv"))

    _style()
    plot_overall(overall, output_dir)
    plot_return_semantics(by_period, output_dir)
    plot_risk_semantics(by_period, output_dir)


if __name__ == "__main__":
    main()
