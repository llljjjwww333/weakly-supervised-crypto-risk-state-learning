from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate uncertainty intervals for semantic rate summaries by bootstrap resampling of checks."
    )
    parser.add_argument("--risk_detail_path", required=True)
    parser.add_argument("--return_ordering_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--asset", default=None, help="Optional asset label to add to the output table.")
    parser.add_argument("--return_expected_order", default="bull > neutral > bear")
    parser.add_argument("--return_metrics", nargs="*", default=["future_return_24", "future_return_72"])
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def bootstrap_rate(indicators: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    if indicators.size == 0:
        return float("nan"), float("nan"), float("nan")
    if indicators.size == 1:
        value = float(indicators[0])
        return value, value, value
    draws = rng.choice(indicators, size=(samples, indicators.size), replace=True).mean(axis=1)
    return (
        float(indicators.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    risk_detail = pd.read_csv(args.risk_detail_path)
    return_ordering = pd.read_csv(args.return_ordering_path)

    return_ordering = return_ordering.loc[return_ordering["metric"].isin(args.return_metrics)].copy()
    return_ordering["matches_expected"] = (
        return_ordering["descending_order"].astype(str).str.strip() == args.return_expected_order.strip()
    )

    risk_summary = {}
    for source, group in risk_detail.groupby("source", sort=True):
        risk_indicators = group["order_matches_expected"].astype(float).to_numpy()
        sig_indicators = group["significant_risk_layering"].astype(float).to_numpy()
        risk_summary[source] = {
            "risk_checks": int(len(risk_indicators)),
            "risk_match_rate": bootstrap_rate(risk_indicators, args.bootstrap_samples, rng),
            "significant_checks": int(len(sig_indicators)),
            "significant_risk_layering_rate": bootstrap_rate(sig_indicators, args.bootstrap_samples, rng),
        }

    rows: list[dict[str, object]] = []
    all_sources = sorted(set(risk_detail["source"]).union(return_ordering["source"]))
    for source in all_sources:
        return_group = return_ordering.loc[return_ordering["source"] == source]
        return_indicators = return_group["matches_expected"].astype(float).to_numpy()
        return_rate, return_low, return_high = bootstrap_rate(return_indicators, args.bootstrap_samples, rng)

        risk_info = risk_summary.get(source, {})
        risk_rate, risk_low, risk_high = risk_info.get("risk_match_rate", (float("nan"),) * 3)
        sig_rate, sig_low, sig_high = risk_info.get("significant_risk_layering_rate", (float("nan"),) * 3)

        row = {
            "source": source,
            "return_checks": int(len(return_indicators)),
            "return_match_rate": return_rate,
            "return_match_rate_ci_low": return_low,
            "return_match_rate_ci_high": return_high,
            "risk_checks": int(risk_info.get("risk_checks", 0)),
            "risk_match_rate": risk_rate,
            "risk_match_rate_ci_low": risk_low,
            "risk_match_rate_ci_high": risk_high,
            "significant_checks": int(risk_info.get("significant_checks", 0)),
            "significant_risk_layering_rate": sig_rate,
            "significant_risk_layering_rate_ci_low": sig_low,
            "significant_risk_layering_rate_ci_high": sig_high,
            "bootstrap_samples": int(args.bootstrap_samples),
        }
        if args.asset is not None:
            row["asset"] = args.asset
        rows.append(row)

    summary = pd.DataFrame(rows)
    column_order = [
        "asset",
        "source",
        "return_checks",
        "return_match_rate",
        "return_match_rate_ci_low",
        "return_match_rate_ci_high",
        "risk_checks",
        "risk_match_rate",
        "risk_match_rate_ci_low",
        "risk_match_rate_ci_high",
        "significant_checks",
        "significant_risk_layering_rate",
        "significant_risk_layering_rate_ci_low",
        "significant_risk_layering_rate_ci_high",
        "bootstrap_samples",
    ]
    available_columns = [column for column in column_order if column in summary.columns]
    summary = summary[available_columns]

    output_path = ensure_parent(Path(args.output_path))
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
