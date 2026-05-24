from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_schema import parse_feature_list, resolve_feature_columns
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a logistic regression baseline.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_end", default="2023-12-31")
    parser.add_argument("--valid_end", default="2024-12-31")
    parser.add_argument("--include_features", default=None, help="Comma-separated feature names to keep.")
    parser.add_argument("--exclude_features", default=None, help="Comma-separated feature names to drop.")
    return parser.parse_args()


def split_by_time(df: pd.DataFrame, train_end: str, valid_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_col = pd.to_datetime(df["open_time"], utc=True)
    train_cutoff = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
    valid_cutoff = pd.Timestamp(valid_end, tz="UTC") + pd.Timedelta(days=1)
    train = df[time_col < train_cutoff]
    valid = df[(time_col >= train_cutoff) & (time_col < valid_cutoff)]
    test = df[time_col >= valid_cutoff]
    return train, valid, test


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    df = pd.read_parquet(args.input_path)
    df = df.dropna(subset=["proxy_label_id"]).copy()
    include_features = parse_feature_list(args.include_features)
    exclude_features = parse_feature_list(args.exclude_features)
    feature_columns = resolve_feature_columns(include_features, exclude_features)

    train_df, valid_df, test_df = split_by_time(df, args.train_end, args.valid_end)
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Check the input date coverage.")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            # scikit-learn >= 1.8 removed the multi_class argument and infers
            # the multinomial setting automatically for multiclass problems.
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]
    )

    X_train = train_df[feature_columns]
    y_train = train_df["proxy_label_id"].astype(int)
    X_valid = valid_df[feature_columns]
    y_valid = valid_df["proxy_label_id"].astype(int)
    X_test = test_df[feature_columns]
    y_test = test_df["proxy_label_id"].astype(int)

    model.fit(X_train, y_train)

    result_rows = []
    for split_name, x_split, y_split in [
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        if len(y_split) == 0:
            continue
        pred = model.predict(x_split)
        result_rows.append(
            {
                "method": "logreg",
                "split": split_name,
                "macro_f1": f1_score(y_split, pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y_split, pred),
                "rows": int(len(y_split)),
            }
        )
        report = classification_report(y_split, pred, digits=4)
        print(f"[{split_name}]")
        print(report)

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(Path(output_dir) / "metrics.csv", index=False)

    pred_df = test_df[["open_time", "symbol", "proxy_label", "proxy_label_id"]].copy()
    pred_df["pred_label_id"] = model.predict(X_test)
    pred_df["pred_label"] = pred_df["pred_label_id"].map({0: "bear", 1: "neutral", 2: "bull"})
    if hasattr(model[-1], "predict_proba"):
        proba = model.predict_proba(X_test)
        pred_df["proba_bear"] = proba[:, 0]
        pred_df["proba_neutral"] = proba[:, 1]
        pred_df["proba_bull"] = proba[:, 2]
    pred_df.to_parquet(Path(output_dir) / "test_predictions.parquet", index=False)

    with open(Path(output_dir) / "run_args.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                **vars(args),
                "feature_columns": feature_columns,
            },
            file,
            indent=2,
        )

    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
