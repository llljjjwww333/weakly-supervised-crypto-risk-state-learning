from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_schema import parse_feature_list, resolve_feature_columns
from src.utils.io import ensure_dir

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:  # pragma: no cover
    raise ImportError("hmmlearn is required for run_hmm.py. Install it with pip.") from exc

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gaussian HMM baseline.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_states", type=int, default=3)
    parser.add_argument("--train_end", default="2023-12-31")
    parser.add_argument("--valid_end", default="2024-12-31")
    parser.add_argument("--include_features", default=None, help="Comma-separated feature names to keep.")
    parser.add_argument("--exclude_features", default=None, help="Comma-separated feature names to drop.")
    return parser.parse_args()


def split_by_time(df: pd.DataFrame, train_end: str, valid_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_col = pd.to_datetime(df["open_time"], utc=True)
    train_cutoff = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
    valid_cutoff = pd.Timestamp(valid_end, tz="UTC") + pd.Timedelta(days=1)
    train = df[time_col < train_cutoff]
    test = df[time_col >= valid_cutoff]
    return train, test


def map_states_to_labels(hidden_states: np.ndarray, proxy_labels: pd.Series, n_states: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for state in range(n_states):
        mask = hidden_states == state
        if mask.sum() == 0:
            mapping[state] = 1
            continue
        majority_label = proxy_labels[mask].mode(dropna=True)
        mapping[state] = int(majority_label.iloc[0]) if not majority_label.empty else 1
    return mapping


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    df = pd.read_parquet(args.input_path).dropna(subset=["proxy_label_id"]).copy()
    include_features = parse_feature_list(args.include_features)
    exclude_features = parse_feature_list(args.exclude_features)
    feature_columns = resolve_feature_columns(include_features, exclude_features)
    train_df, test_df = split_by_time(df, args.train_end, args.valid_end)
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Check the input date coverage.")

    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x_train = preprocessor.fit_transform(train_df[feature_columns])
    x_test = preprocessor.transform(test_df[feature_columns])
    y_train = train_df["proxy_label_id"].astype(int).reset_index(drop=True)
    y_test = test_df["proxy_label_id"].astype(int).reset_index(drop=True)

    model = GaussianHMM(
        n_components=args.n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
    )
    model.fit(x_train)

    train_states = model.predict(x_train)
    test_states = model.predict(x_test)
    mapping = map_states_to_labels(train_states, y_train, args.n_states)
    test_pred = np.array([mapping[state] for state in test_states], dtype=int)

    metrics = pd.DataFrame(
        [
            {
                "method": "gaussian_hmm",
                "n_states": args.n_states,
                "macro_f1": f1_score(y_test, test_pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
                "rows": int(len(y_test)),
            }
        ]
    )
    metrics.to_csv(Path(output_dir) / "metrics.csv", index=False)

    pred_df = test_df[["open_time", "symbol", "proxy_label", "proxy_label_id"]].copy()
    pred_df["hidden_state"] = test_states
    pred_df["pred_label_id"] = test_pred
    pred_df["pred_label"] = pred_df["pred_label_id"].map({0: "bear", 1: "neutral", 2: "bull"})
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

    print(metrics.to_string(index=False))
    print(classification_report(y_test, test_pred, digits=4))


if __name__ == "__main__":
    main()
