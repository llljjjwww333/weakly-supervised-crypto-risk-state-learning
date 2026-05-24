from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_parent


RETURN_METRICS = {"future_return_24", "future_return_72"}
EXPECTED_RETURN_ORDER = "bull > neutral > bear"

METHOD_MAP = {
    "logistic_regression": "logreg",
    "logreg": "logreg",
    "logreg_postproc": "logreg_postproc",
    "logreg_postproc_match": "logreg_postproc",
    "main_gru": "main_gru",
    "main_lstm": "main_lstm",
    "main_gru_eth": "main_gru_eth",
    "btc_trained_gru": "main_gru_cross_asset",
    "main_gru_cross_asset": "main_gru_cross_asset",
    "proxy_label": "proxy_label",
}


@dataclass(frozen=True)
class AssetConfig:
    asset: str
    return_ordering_path: Path
    risk_summary_path: Path
    uncertainty_path: Path


def method_name(source: str) -> str:
    return METHOD_MAP.get(source, source)


def load_return_summary(config: AssetConfig) -> pd.DataFrame:
    ordering = pd.read_csv(config.return_ordering_path)
    ordering = ordering.loc[ordering["metric"].isin(RETURN_METRICS)].copy()
    ordering["method"] = ordering["source"].map(method_name)
    ordering["matches_expected"] = (
        ordering["descending_order"].astype(str).str.strip() == EXPECTED_RETURN_ORDER
    )

    summary = (
        ordering.groupby("method", as_index=False)
        .agg(
            return_checks=("matches_expected", "count"),
            return_matches=("matches_expected", "sum"),
        )
        .assign(return_match_rate=lambda df: df["return_matches"] / df["return_checks"])
    )
    summary.insert(0, "asset", config.asset)
    return summary


def load_risk_summary(config: AssetConfig) -> pd.DataFrame:
    risk = pd.read_csv(config.risk_summary_path)
    risk["method"] = risk["source"].map(method_name)

    summary = (
        risk.groupby("method", as_index=False)
        .agg(
            risk_checks=("checks", "sum"),
            risk_order_matches=("order_matches", "sum"),
            significant_risk_layering=("significant_risk_layering", "sum"),
        )
        .assign(
            risk_match_rate=lambda df: df["risk_order_matches"] / df["risk_checks"],
            significant_risk_layering_rate=lambda df: df["significant_risk_layering"]
            / df["risk_checks"],
        )
    )
    summary.insert(0, "asset", config.asset)
    return summary


def load_uncertainty(config: AssetConfig) -> pd.DataFrame:
    uncertainty = pd.read_csv(config.uncertainty_path)
    uncertainty["method"] = uncertainty["source"].map(method_name)
    uncertainty = uncertainty.drop(columns=["source"])
    columns = ["asset", "method"] + [
        column for column in uncertainty.columns if column not in {"asset", "method"}
    ]
    return uncertainty[columns]


def main() -> None:
    root = Path("experiments/summary")
    configs = [
        AssetConfig(
            asset="BTC",
            return_ordering_path=root / "btc_economic_meaning_extended" / "ordering_summary.csv",
            risk_summary_path=root / "btc_risk_state_significance_extended" / "risk_significance_summary.csv",
            uncertainty_path=root / "btc_semantic_uncertainty.csv",
        ),
        AssetConfig(
            asset="ETH",
            return_ordering_path=root / "eth_economic_meaning_extended" / "ordering_summary.csv",
            risk_summary_path=root / "eth_risk_state_significance_extended" / "risk_significance_summary.csv",
            uncertainty_path=root / "eth_semantic_uncertainty.csv",
        ),
        AssetConfig(
            asset="BNB",
            return_ordering_path=root / "bnb_economic_meaning_extended" / "ordering_summary.csv",
            risk_summary_path=root / "bnb_risk_state_significance_extended" / "risk_significance_summary.csv",
            uncertainty_path=root / "bnb_semantic_uncertainty.csv",
        ),
        AssetConfig(
            asset="SOL",
            return_ordering_path=root / "sol_economic_meaning_extended" / "ordering_summary.csv",
            risk_summary_path=root / "sol_risk_state_significance_extended" / "risk_significance_summary.csv",
            uncertainty_path=root / "sol_semantic_uncertainty.csv",
        ),
        AssetConfig(
            asset="XRP",
            return_ordering_path=root / "xrp_economic_meaning_extended" / "ordering_summary.csv",
            risk_summary_path=root / "xrp_risk_state_significance_extended" / "risk_significance_summary.csv",
            uncertainty_path=root / "xrp_semantic_uncertainty.csv",
        ),
    ]

    return_summary = pd.concat([load_return_summary(config) for config in configs], ignore_index=True)
    risk_summary = pd.concat([load_risk_summary(config) for config in configs], ignore_index=True)
    uncertainty = pd.concat([load_uncertainty(config) for config in configs], ignore_index=True)

    comparison = risk_summary.merge(return_summary, on=["asset", "method"], how="outer")
    comparison = comparison[
        [
            "asset",
            "method",
            "risk_match_rate",
            "return_match_rate",
            "risk_checks",
            "risk_order_matches",
            "return_checks",
            "return_matches",
            "significant_risk_layering",
            "significant_risk_layering_rate",
        ]
    ].sort_values(["asset", "method"])

    legacy_comparison = comparison[["asset", "method", "risk_match_rate", "return_match_rate"]]
    significance = comparison[
        [
            "asset",
            "method",
            "risk_checks",
            "risk_match_rate",
            "significant_risk_layering_rate",
        ]
    ].dropna(subset=["risk_checks"])

    outputs = {
        root / "risk_vs_return_consistency_comparison_extended.csv": legacy_comparison,
        root / "semantic_consistency_comparison_extended_detailed.csv": comparison,
        root / "risk_significance_comparison.csv": significance,
        root / "semantic_uncertainty_comparison.csv": uncertainty.sort_values(["asset", "method"]),
    }
    for path, df in outputs.items():
        output_path = ensure_parent(path)
        df.to_csv(output_path, index=False)
        print(f"[wrote] {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
