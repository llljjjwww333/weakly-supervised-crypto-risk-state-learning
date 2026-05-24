from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate_risk_state_significance import (
    build_proxy_frame,
    build_test_rows,
    load_label_frame,
    load_prediction_frame,
    summarize_tests,
)


MODEL_SPECS = [
    ("proxy_label", "Proxy label", None),
    ("logreg_postproc", "LogReg + post-proc.", "experiments/baselines/logreg_btc_postproc_match/test_predictions.parquet"),
    ("histgb_postproc", "HistGB + post-proc.", "experiments/revision/histgb_btc_postproc/test_predictions.parquet"),
    ("gru_main", "Main GRU", "experiments/main/gru_btc/test_predictions.parquet"),
    ("tcn96", "TCN-96x4", "experiments/improved/main/tcn_btc_96x4/test_predictions.parquet"),
]

BLOCK_SIZES = [12, 24, 48, 72]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTC block-bootstrap sensitivity for risk significance.")
    parser.add_argument("--label_path", default="data/labels_improved/default/BTCUSDT_labels.parquet")
    parser.add_argument("--output_path", default="experiments/summary/bootstrap_block_sensitivity.csv")
    parser.add_argument("--test_start", default="2025-01-01")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap_samples", type=int, default=500)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    return parser.parse_args()


def _build_combined_frame(label_path: str, test_start: str, horizon: int) -> pd.DataFrame:
    label_frame = load_label_frame(label_path, test_start, horizon)
    frames = [build_proxy_frame(label_frame)]
    for source, _, prediction_path in MODEL_SPECS:
        if prediction_path is None:
            continue
        frames.append(load_prediction_frame(prediction_path, source, label_frame))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    combined = _build_combined_frame(args.label_path, args.test_start, args.horizon)
    nominal_detail = build_test_rows(
        combined,
        alpha=args.alpha,
        bootstrap_samples=0,
        bootstrap_block_size=24,
        bootstrap_seed=args.bootstrap_seed,
    )
    nominal, _ = summarize_tests(nominal_detail)
    nominal = nominal.rename(
        columns={
            "checks": "nominal_checks",
            "significant_risk_layering": "nominal_sig_risk_checks",
            "significant_risk_layering_rate": "nominal_sig_risk_rate",
        }
    )[["source", "nominal_checks", "nominal_sig_risk_checks", "nominal_sig_risk_rate"]]

    rows: list[dict[str, object]] = []
    for block_size in BLOCK_SIZES:
        detail = build_test_rows(
            combined,
            alpha=args.alpha,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_block_size=block_size,
            bootstrap_seed=args.bootstrap_seed,
        )
        summary, _ = summarize_tests(detail)
        merged = summary.merge(nominal, on="source", how="left")
        for _, row in merged.iterrows():
            rows.append(
                {
                    "asset": "BTC",
                    "source": row["source"],
                    "model": next(model for source, model, _ in MODEL_SPECS if source == row["source"]),
                    "bootstrap_samples": int(args.bootstrap_samples),
                    "bootstrap_block_size": int(block_size),
                    "checks": int(row["checks"]),
                    "risk_order_matches": int(row["order_matches"]),
                    "risk_order_rate": float(row["order_match_rate"]),
                    "nominal_sig_risk_checks": int(row["nominal_sig_risk_checks"]) if pd.notna(row["nominal_sig_risk_checks"]) else None,
                    "nominal_sig_risk_rate": float(row["nominal_sig_risk_rate"]) if pd.notna(row["nominal_sig_risk_rate"]) else None,
                    "bootstrap_sig_risk_checks": int(row["bootstrap_significant_risk_layering"]),
                    "bootstrap_sig_risk_rate": float(row["bootstrap_significant_risk_layering_rate"]),
                }
            )

    result = pd.DataFrame(rows).sort_values(["model", "bootstrap_block_size"]).reset_index(drop=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"[wrote] {output_path} rows={len(result)}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
