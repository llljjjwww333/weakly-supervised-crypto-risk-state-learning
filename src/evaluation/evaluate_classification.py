from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classification predictions against proxy labels.")
    parser.add_argument("--input_path", required=True, help="Prediction parquet or csv with proxy_label_id and pred_label_id")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    path = Path(args.input_path)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    clean = df.dropna(subset=["proxy_label_id", "pred_label_id"]).copy()
    if clean.empty:
        raise ValueError("No rows available after dropping missing labels.")

    y_true = clean["proxy_label_id"].astype(int)
    y_pred = clean["pred_label_id"].astype(int)
    metrics_df = pd.DataFrame(
        [
            {
                "rows": int(len(clean)),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "bull_f1": f1_score(y_true, y_pred, average=None, labels=[2], zero_division=0)[0],
                "bear_f1": f1_score(y_true, y_pred, average=None, labels=[0], zero_division=0)[0],
            }
        ]
    )
    metrics_df.to_csv(Path(output_dir) / "classification_metrics.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    matrix_df = pd.DataFrame(
        matrix,
        index=["true_bear", "true_neutral", "true_bull"],
        columns=["pred_bear", "pred_neutral", "pred_bull"],
    )
    matrix_df.to_csv(Path(output_dir) / "confusion_matrix.csv")

    print(metrics_df.to_string(index=False))
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
