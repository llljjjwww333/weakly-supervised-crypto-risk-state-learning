from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal stability of bull/bear predictions.")
    parser.add_argument("--input_path", required=True, help="Prediction parquet or csv with open_time, symbol, pred_label_id")
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def segment_lengths(values: pd.Series) -> list[int]:
    lengths: list[int] = []
    current_length = 0
    previous = None
    for value in values.tolist():
        if previous is None or value == previous:
            current_length += 1
        else:
            lengths.append(current_length)
            current_length = 1
        previous = value
    if current_length > 0:
        lengths.append(current_length)
    return lengths


def summarize_symbol(df: pd.DataFrame) -> dict[str, float | int | str]:
    ordered = df.sort_values("open_time").reset_index(drop=True)
    preds = ordered["pred_label_id"].astype(int)
    transitions = int((preds != preds.shift(1)).sum() - 1)
    transitions = max(transitions, 0)
    segments = segment_lengths(preds)
    total_rows = len(ordered)
    day_count = max((ordered["open_time"].max() - ordered["open_time"].min()).total_seconds() / 86400.0, 1.0)
    return {
        "symbol": str(ordered["symbol"].iloc[0]),
        "rows": int(total_rows),
        "transitions": transitions,
        "avg_state_duration_bars": float(sum(segments) / len(segments)),
        "max_state_duration_bars": int(max(segments)),
        "daily_switch_rate": float(transitions / day_count),
    }


def main() -> None:
    args = parse_args()
    path = Path(args.input_path)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "open_time" not in df.columns or "symbol" not in df.columns or "pred_label_id" not in df.columns:
        raise ValueError("Input must contain open_time, symbol, and pred_label_id columns.")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    rows = [summarize_symbol(group.copy()) for _, group in df.groupby("symbol", sort=True)]
    summary_df = pd.DataFrame(rows)
    output_path = ensure_parent(args.output_path)
    summary_df.to_csv(output_path, index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
