from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir, ensure_parent


ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

RETURN_METRICS = {"future_return_24", "future_return_72"}
EXPECTED_RETURN_ORDER = "bull > neutral > bear"


@dataclass(frozen=True)
class ThresholdVariant:
    name: str
    args: tuple[str, ...]


VARIANTS = [
    ThresholdVariant(name="default", args=()),
    ThresholdVariant(
        name="loose",
        args=(
            "--bull_return_24_min",
            "0.005",
            "--bear_return_24_max",
            "-0.005",
            "--ema_gap_pos_min",
            "0.001",
            "--ema_gap_neg_max",
            "-0.001",
            "--max_drawdown_bull_max",
            "-0.040",
            "--max_drawdown_bear_max",
            "-0.040",
        ),
    ),
    ThresholdVariant(
        name="strict",
        args=(
            "--bull_return_24_min",
            "0.015",
            "--bear_return_24_max",
            "-0.015",
            "--ema_gap_pos_min",
            "0.003",
            "--ema_gap_neg_max",
            "-0.003",
            "--max_drawdown_bull_max",
            "-0.020",
            "--max_drawdown_bear_max",
            "-0.080",
        ),
    ),
]


def run_command(args: list[str]) -> None:
    print("[run]", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def split_name(open_time: pd.Series) -> pd.Series:
    time_col = pd.to_datetime(open_time, utc=True)
    train_cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    valid_cutoff = pd.Timestamp("2025-01-01", tz="UTC")
    return pd.Series(
        pd.cut(
            time_col,
            bins=[
                pd.Timestamp.min.tz_localize("UTC"),
                train_cutoff,
                valid_cutoff,
                pd.Timestamp.max.tz_localize("UTC"),
            ],
            labels=["train", "valid", "test"],
            right=False,
        ).astype(str),
        index=open_time.index,
    )


def build_label_distribution(variant: str, asset: str, label_path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(label_path)
    labels = labels.dropna(subset=["proxy_label_id"]).copy()
    labels["split"] = split_name(labels["open_time"])
    counts = (
        labels.groupby(["split", "proxy_label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ["bear", "neutral", "bull"]:
        if label not in counts:
            counts[label] = 0
    counts["rows"] = counts[["bear", "neutral", "bull"]].sum(axis=1)
    for label in ["bear", "neutral", "bull"]:
        counts[f"{label}_rate"] = counts[label] / counts["rows"]
    counts.insert(0, "asset", asset)
    counts.insert(0, "variant", variant)
    return counts[
        [
            "variant",
            "asset",
            "split",
            "rows",
            "bear",
            "neutral",
            "bull",
            "bear_rate",
            "neutral_rate",
            "bull_rate",
        ]
    ]


def load_return_summary(variant: str, asset: str, ordering_path: Path) -> pd.DataFrame:
    ordering = pd.read_csv(ordering_path)
    ordering = ordering.loc[ordering["metric"].isin(RETURN_METRICS)].copy()
    ordering["matches_expected"] = (
        ordering["descending_order"].astype(str).str.strip() == EXPECTED_RETURN_ORDER
    )
    summary = (
        ordering.groupby("source", as_index=False)
        .agg(return_checks=("matches_expected", "count"), return_matches=("matches_expected", "sum"))
        .assign(return_match_rate=lambda df: df["return_matches"] / df["return_checks"])
        .rename(columns={"source": "method"})
    )
    summary.insert(0, "asset", asset)
    summary.insert(0, "variant", variant)
    return summary


def load_risk_summary(variant: str, asset: str, summary_path: Path) -> pd.DataFrame:
    risk = pd.read_csv(summary_path).rename(columns={"source": "method"})
    risk = risk[
        [
            "method",
            "checks",
            "order_matches",
            "significant_risk_layering",
            "order_match_rate",
            "significant_risk_layering_rate",
        ]
    ].rename(
        columns={
            "checks": "risk_checks",
            "order_matches": "risk_order_matches",
            "order_match_rate": "risk_match_rate",
        }
    )
    risk.insert(0, "asset", asset)
    risk.insert(0, "variant", variant)
    return risk


def load_classification(variant: str, asset: str, method: str, eval_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(eval_path)
    metrics.insert(0, "method", method)
    metrics.insert(0, "asset", asset)
    metrics.insert(0, "variant", variant)
    return metrics


def main() -> None:
    label_root = Path("data/labels_threshold")
    experiment_root = Path("experiments/threshold_sensitivity")
    summary_root = Path("experiments/summary")

    label_distributions: list[pd.DataFrame] = []
    semantic_summaries: list[pd.DataFrame] = []
    classification_summaries: list[pd.DataFrame] = []

    for variant in VARIANTS:
        labels_dir = label_root / variant.name
        ensure_dir(labels_dir)
        run_command(
            [
                sys.executable,
                "-m",
                "src.features.build_labels",
                "--input_dir",
                "data/processed/features/1h",
                "--output_dir",
                str(labels_dir),
                *variant.args,
            ]
        )

        for asset, symbol in ASSETS.items():
            asset_lower = asset.lower()
            label_path = labels_dir / f"{symbol}_labels.parquet"
            asset_dir = ensure_dir(experiment_root / variant.name / asset_lower)
            logreg_dir = ensure_dir(asset_dir / "logreg")
            postproc_dir = ensure_dir(asset_dir / "logreg_postproc")

            logreg_pred_path = logreg_dir / "test_predictions.parquet"
            postproc_pred_path = postproc_dir / "test_predictions.parquet"

            label_distributions.append(build_label_distribution(variant.name, asset, label_path))

            run_command(
                [
                    sys.executable,
                    "-m",
                    "src.models.baselines.run_logreg",
                    "--input_path",
                    str(label_path),
                    "--output_dir",
                    str(logreg_dir),
                    "--train_end",
                    "2023-12-31",
                    "--valid_end",
                    "2024-12-31",
                ]
            )
            run_command(
                [
                    sys.executable,
                    "-m",
                    "src.models.baselines.postprocess_predictions",
                    "--input_path",
                    str(logreg_pred_path),
                    "--output_path",
                    str(postproc_pred_path),
                    "--window",
                    "6",
                    "--switch_margin",
                    "0.10",
                ]
            )

            for method, pred_path, method_dir in [
                ("logistic_regression", logreg_pred_path, logreg_dir),
                ("logreg_postproc", postproc_pred_path, postproc_dir),
            ]:
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.evaluation.evaluate_classification",
                        "--input_path",
                        str(pred_path),
                        "--output_dir",
                        str(method_dir / "eval"),
                    ]
                )
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.evaluation.evaluate_stability",
                        "--input_path",
                        str(pred_path),
                        "--output_path",
                        str(method_dir / "stability.csv"),
                    ]
                )
                classification_summaries.append(
                    load_classification(variant.name, asset, method, method_dir / "eval" / "classification_metrics.csv")
                )

            economic_dir = asset_dir / "economic_meaning"
            significance_dir = asset_dir / "risk_state_significance"
            run_command(
                [
                    sys.executable,
                    "-m",
                    "src.evaluation.evaluate_economic_meaning",
                    "--label_path",
                    str(label_path),
                    "--prediction_paths",
                    str(logreg_pred_path),
                    str(postproc_pred_path),
                    "--prediction_names",
                    "logistic_regression",
                    "logreg_postproc",
                    "--output_dir",
                    str(economic_dir),
                    "--test_start",
                    "2025-01-01",
                ]
            )
            run_command(
                [
                    sys.executable,
                    "-m",
                    "src.evaluation.evaluate_risk_state_significance",
                    "--label_path",
                    str(label_path),
                    "--prediction_paths",
                    str(logreg_pred_path),
                    str(postproc_pred_path),
                    "--prediction_names",
                    "logistic_regression",
                    "logreg_postproc",
                    "--output_dir",
                    str(significance_dir),
                    "--test_start",
                    "2025-01-01",
                    "--horizon",
                    "24",
                    "--alpha",
                    "0.05",
                ]
            )

            returns = load_return_summary(variant.name, asset, economic_dir / "ordering_summary.csv")
            risks = load_risk_summary(variant.name, asset, significance_dir / "risk_significance_summary.csv")
            semantic_summaries.append(risks.merge(returns, on=["variant", "asset", "method"], how="outer"))

    label_output = ensure_parent(summary_root / "threshold_sensitivity_label_distribution.csv")
    semantic_output = ensure_parent(summary_root / "threshold_sensitivity_semantics.csv")
    classification_output = ensure_parent(summary_root / "threshold_sensitivity_classification.csv")

    pd.concat(label_distributions, ignore_index=True).to_csv(label_output, index=False)
    pd.concat(semantic_summaries, ignore_index=True).to_csv(semantic_output, index=False)
    pd.concat(classification_summaries, ignore_index=True).to_csv(classification_output, index=False)

    print(f"[wrote] {label_output}")
    print(f"[wrote] {semantic_output}")
    print(f"[wrote] {classification_output}")


if __name__ == "__main__":
    main()
