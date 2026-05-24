from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

from src.utils.io import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the direct rule-based proxy labels.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--predictions_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    clean = df.dropna(subset=["proxy_label_id"])
    y_true = clean["proxy_label_id"].astype(int)
    y_pred = clean["proxy_label_id"].astype(int)

    metrics = pd.DataFrame(
        [
            {
                "method": "rule_based",
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "rows": len(clean),
            }
        ]
    )
    output_path = ensure_parent(args.output_path)
    metrics.to_csv(output_path, index=False)
    if args.predictions_path:
        predictions_path = ensure_parent(args.predictions_path)
        pred_df = clean[["open_time", "symbol", "proxy_label", "proxy_label_id"]].copy()
        pred_df["pred_label_id"] = pred_df["proxy_label_id"]
        pred_df["pred_label"] = pred_df["proxy_label"]
        pred_df.to_parquet(predictions_path, index=False)
        print(f"[saved] {predictions_path}")
    print(metrics.to_string(index=False))
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
