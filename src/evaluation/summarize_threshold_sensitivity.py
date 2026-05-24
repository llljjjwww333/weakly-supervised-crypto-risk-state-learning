from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import ensure_parent


def main() -> None:
    summary_root = Path("experiments/summary")
    semantics = pd.read_csv(summary_root / "threshold_sensitivity_semantics.csv")
    labels = pd.read_csv(summary_root / "threshold_sensitivity_label_distribution.csv")

    aggregate = (
        semantics.groupby(["variant", "method"], as_index=False)
        .agg(
            asset_method_rows=("asset", "count"),
            assets=("asset", "nunique"),
            mean_risk_checks=("risk_checks", "mean"),
            min_risk_checks=("risk_checks", "min"),
            mean_risk_match_rate=("risk_match_rate", "mean"),
            min_risk_match_rate=("risk_match_rate", "min"),
            mean_significant_risk_layering_rate=("significant_risk_layering_rate", "mean"),
            min_significant_risk_layering_rate=("significant_risk_layering_rate", "min"),
            mean_return_match_rate=("return_match_rate", "mean"),
            max_return_match_rate=("return_match_rate", "max"),
        )
        .sort_values(["variant", "method"])
    )

    coverage_flags = semantics.loc[
        (semantics["risk_checks"] < 21) | (semantics["return_checks"] < 6)
    ].copy()

    test_labels = labels.loc[labels["split"] == "test"].copy()
    sparse_test_states = test_labels.loc[
        (test_labels["bull"] < 50) | (test_labels["bear"] < 50)
    ].sort_values(["variant", "asset"])

    aggregate_path = ensure_parent(summary_root / "threshold_sensitivity_summary_by_variant_method.csv")
    coverage_path = ensure_parent(summary_root / "threshold_sensitivity_check_coverage_flags.csv")
    sparse_path = ensure_parent(summary_root / "threshold_sensitivity_sparse_test_states.csv")

    aggregate.to_csv(aggregate_path, index=False)
    coverage_flags.to_csv(coverage_path, index=False)
    sparse_test_states.to_csv(sparse_path, index=False)

    print("[aggregate]")
    print(aggregate.to_string(index=False))
    print(f"[wrote] {aggregate_path}")
    print(f"[wrote] {coverage_path}")
    print(f"[wrote] {sparse_path}")


if __name__ == "__main__":
    main()
