from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mutual_info_score
from matplotlib.lines import Line2D
import seaborn as sns

from src.utils.io import ensure_dir


LABEL_ROOT = Path("data/labels_threshold")
MANIFEST_PATH = Path("experiments/summary/results_manifest.csv")
SUMMARY_PATH = Path("experiments/summary/sor_analysis.csv")
DOWNSTREAM_PATH = Path("experiments/summary/sor_downstream_relationship.csv")
FIGURE_PATH = Path("figures/main/fig_sor_vs_sig_risk.pdf")
FIGURE_PNG_PATH = Path("figures/main/fig_sor_vs_sig_risk.png")

ASSETS = ["BTC", "BNB", "ETH", "SOL", "XRP"]
VARIANTS = ["default", "loose", "strict"]
MODEL_ORDER = ["Proxy label", "LogReg + post-proc."]
EPS = 1e-6

ASSET_COLORS = {
    "BTC": "#1f4e79",
    "BNB": "#b58900",
    "ETH": "#6c71c4",
    "SOL": "#2aa198",
    "XRP": "#cb4b16",
}

VARIANT_MARKERS = {
    "default": "o",
    "loose": "s",
    "strict": "^",
}


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
            "legend.fontsize": 9.0,
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.facecolor": "#FBFBF8",
            "figure.facecolor": "white",
            "axes.edgecolor": "#D0D0CC",
            "grid.color": "#DADAD4",
        },
    )


def discretize_quantiles(series: pd.Series, q: int = 5) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(index=series.index, dtype="float64")
    bins = min(q, clean.nunique())
    if bins < 2:
        out = pd.Series(index=series.index, dtype="float64")
        out.loc[clean.index] = 0
        return out
    codes = pd.qcut(clean, q=bins, labels=False, duplicates="drop")
    out = pd.Series(index=series.index, dtype="float64")
    out.loc[codes.index] = codes.astype(float)
    return out


def mutual_information(labels: pd.Series, target: pd.Series) -> float:
    frame = pd.DataFrame({"label": labels, "target": target}).dropna()
    if frame.empty or frame["label"].nunique() < 2 or frame["target"].nunique() < 2:
        return 0.0
    return float(mutual_info_score(frame["label"], frame["target"]))


def compute_sor_for_file(path: Path, asset: str, variant: str) -> dict[str, float | str | int]:
    df = pd.read_parquet(path, columns=["proxy_label_id", "future_return_24", "future_vol_24"])
    vol_bins = discretize_quantiles(df["future_vol_24"])
    ret_bins = discretize_quantiles(df["future_return_24"])
    valid = pd.DataFrame(
        {
            "proxy_label_id": df["proxy_label_id"],
            "future_vol_bin": vol_bins,
            "future_return_bin": ret_bins,
        }
    )
    valid_vol = valid[["proxy_label_id", "future_vol_bin"]].dropna()
    valid_ret = valid[["proxy_label_id", "future_return_bin"]].dropna()
    mi_vol = mutual_information(valid["proxy_label_id"], valid["future_vol_bin"])
    mi_ret = mutual_information(valid["proxy_label_id"], valid["future_return_bin"])
    sor = mi_vol / (mi_ret + EPS)
    return {
        "asset": asset,
        "label_variant": variant,
        "rows_total": int(len(df)),
        "rows_valid_vol": int(len(valid_vol)),
        "rows_valid_return": int(len(valid_ret)),
        "mi_vol_24": mi_vol,
        "mi_return_24": mi_ret,
        "sor": sor,
    }


def rank_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 2:
        return float("nan")
    return float(frame["x"].rank(method="average").corr(frame["y"].rank(method="average")))


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 2:
        return float("nan")
    return float(frame["x"].corr(frame["y"]))


def add_regression_line(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 2:
        return
    slope, intercept = pd.Series(frame["y"]).cov(frame["x"]) / frame["x"].var(), frame["y"].mean()
    intercept = frame["y"].mean() - slope * frame["x"].mean()
    xs = pd.Series(sorted(frame["x"].tolist()))
    ys = intercept + slope * xs
    ax.plot(xs, ys, color="#444444", linewidth=1.2, linestyle="--", zorder=1)


def build_relationship_frame(sor_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    relevant = manifest[
        (manifest["experiment_group"] == "threshold_sensitivity")
        & (manifest["model"].isin(MODEL_ORDER))
    ][["asset", "label_variant", "model", "sig_risk_rate", "risk_order_rate"]].copy()
    return relevant.merge(sor_df, on=["asset", "label_variant"], how="left")


def plot_relationship(df: pd.DataFrame) -> None:
    ensure_dir(FIGURE_PATH.parent)
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.25, wspace=0.12)
    for ax, model in zip(axes, MODEL_ORDER):
        panel = df[df["model"] == model].copy()
        for _, row in panel.iterrows():
            ax.scatter(
                row["sor"],
                row["sig_risk_rate"],
                s=82,
                color=ASSET_COLORS.get(row["asset"], "#444444"),
                marker=VARIANT_MARKERS.get(row["label_variant"], "o"),
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

        add_regression_line(ax, panel["sor"], panel["sig_risk_rate"])
        spearman = rank_corr(panel["sor"], panel["sig_risk_rate"])
        pearson = pearson_corr(panel["sor"], panel["sig_risk_rate"])
        ax.text(
            0.97,
            0.07,
            f"Spearman={spearman:.2f}\nPearson={pearson:.2f}",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.25"},
        )
        ax.set_title(model, fontsize=11)
        ax.set_xlabel("SOR = MI(proxy, future vol) / MI(proxy, future return)")
        ax.set_ylabel("Significant risk layering rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0.8, 6.7)
        ax.grid(axis="both", alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    asset_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=7, label=asset)
        for asset, color in ASSET_COLORS.items()
    ]
    variant_handles = [
        Line2D([0], [0], marker=marker, color="#666666", linestyle="none", markerfacecolor="#666666", markersize=7, label=variant.capitalize())
        for variant, marker in VARIANT_MARKERS.items()
    ]
    fig.legend(handles=asset_handles, title="Asset (color)", loc="lower center", bbox_to_anchor=(0.34, 0.02), ncol=5, frameon=False)
    fig.legend(handles=variant_handles, title="Threshold variant (shape)", loc="lower center", bbox_to_anchor=(0.80, 0.02), ncol=3, frameon=False)

    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    fig.savefig(FIGURE_PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows: list[dict[str, float | str | int]] = []
    for variant in VARIANTS:
        variant_root = LABEL_ROOT / variant
        for asset in ASSETS:
            path = variant_root / f"{asset}USDT_labels.parquet"
            rows.append(compute_sor_for_file(path, asset, variant))

    sor_df = pd.DataFrame(rows).sort_values(["asset", "label_variant"]).reset_index(drop=True)
    ensure_dir(SUMMARY_PATH.parent)
    sor_df.to_csv(SUMMARY_PATH, index=False)

    manifest = pd.read_csv(MANIFEST_PATH)
    relationship = build_relationship_frame(sor_df, manifest).sort_values(["model", "asset", "label_variant"]).reset_index(drop=True)
    relationship["spearman_sor_sig_risk"] = relationship.groupby("model")["sor"].transform(lambda s: math.nan)
    relationship.to_csv(DOWNSTREAM_PATH, index=False)

    plot_relationship(relationship)

    stats = []
    for model in MODEL_ORDER:
        panel = relationship[relationship["model"] == model]
        stats.append(
            {
                "model": model,
                "spearman_sor_sig_risk": rank_corr(panel["sor"], panel["sig_risk_rate"]),
                "pearson_sor_sig_risk": pearson_corr(panel["sor"], panel["sig_risk_rate"]),
                "points": int(len(panel)),
            }
        )
    stats_df = pd.DataFrame(stats)
    stats_path = SUMMARY_PATH.parent / "sor_correlation_summary.csv"
    stats_df.to_csv(stats_path, index=False)

    print(sor_df.to_string(index=False))
    print()
    print(relationship[["asset", "label_variant", "model", "sor", "sig_risk_rate", "risk_order_rate"]].to_string(index=False))
    print()
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
