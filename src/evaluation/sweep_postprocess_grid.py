from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.utils.io import ensure_parent, read_table

PROBA_COLUMNS = ["proba_bear", "proba_neutral", "proba_bull"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep temporal post-processing settings on prediction probabilities.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--windows", nargs="+", type=int, required=True)
    parser.add_argument("--margins", nargs="+", type=float, required=True)
    return parser.parse_args()


def temporal_metrics(open_time: pd.Series, pred: np.ndarray) -> tuple[int, float]:
    ordered_time = pd.to_datetime(open_time, utc=True)
    pred_series = pd.Series(pred)
    transitions = int((pred_series != pred_series.shift(1)).sum() - 1)
    transitions = max(transitions, 0)
    day_count = max((ordered_time.max() - ordered_time.min()).total_seconds() / 86400.0, 1.0)
    return transitions, float(transitions / day_count)


def postprocess_group(group: pd.DataFrame, window: int, margin: float) -> np.ndarray:
    ordered = group.sort_values("open_time").reset_index(drop=True)
    smoothed = ordered[PROBA_COLUMNS].rolling(window=window, min_periods=1).mean().to_numpy()

    predictions: list[int] = []
    current_state = int(np.argmax(smoothed[0]))
    predictions.append(current_state)
    for probs in smoothed[1:]:
        proposed_state = int(np.argmax(probs))
        current_prob = float(probs[current_state])
        proposed_prob = float(probs[proposed_state])
        if proposed_state != current_state and proposed_prob < current_prob + margin:
            predictions.append(current_state)
            continue
        current_state = proposed_state
        predictions.append(current_state)
    return np.asarray(predictions, dtype=int)


def main() -> None:
    args = parse_args()
    df = read_table(args.input_path).copy()
    required = {"open_time", "symbol", "proxy_label_id", *PROBA_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)

    result_rows: list[dict[str, float | int]] = []
    for window in args.windows:
        for margin in args.margins:
            pred_chunks: list[np.ndarray] = []
            time_chunks: list[pd.Series] = []
            true_chunks: list[np.ndarray] = []
            for _, group in df.groupby("symbol", sort=True):
                pred = postprocess_group(group.copy(), window=window, margin=margin)
                pred_chunks.append(pred)
                true_chunks.append(group["proxy_label_id"].astype(int).to_numpy())
                time_chunks.append(group["open_time"])

            y_pred = np.concatenate(pred_chunks)
            y_true = np.concatenate(true_chunks)
            all_time = pd.concat(time_chunks, ignore_index=True)
            transitions, daily_switch_rate = temporal_metrics(all_time, y_pred)
            result_rows.append(
                {
                    "window": int(window),
                    "margin": float(margin),
                    "rows": int(len(y_true)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "transitions": int(transitions),
                    "daily_switch_rate": float(daily_switch_rate),
                }
            )

    summary = pd.DataFrame(result_rows).sort_values(
        ["daily_switch_rate", "macro_f1"],
        ascending=[True, False],
    )
    summary.to_csv(ensure_parent(args.output_path), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
