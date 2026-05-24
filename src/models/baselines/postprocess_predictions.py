from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_parent, read_table, write_table

PROBA_COLUMNS = ["proba_bear", "proba_neutral", "proba_bull"]
LABELS = ["bear", "neutral", "bull"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process prediction probabilities with temporal smoothing.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--window", type=int, default=6, help="Rolling mean window size in bars.")
    parser.add_argument(
        "--switch_margin",
        type=float,
        default=0.0,
        help="Only switch if proposed state probability exceeds current state probability by this margin.",
    )
    return parser.parse_args()


def smooth_group(group: pd.DataFrame, window: int, switch_margin: float) -> pd.DataFrame:
    ordered = group.sort_values("open_time").reset_index(drop=True).copy()
    smoothed = ordered[PROBA_COLUMNS].rolling(window=window, min_periods=1).mean()
    smoothed_values = smoothed.to_numpy()

    predictions: list[int] = []
    current_state = int(np.argmax(smoothed_values[0]))
    predictions.append(current_state)
    for probs in smoothed_values[1:]:
        proposed_state = int(np.argmax(probs))
        current_prob = float(probs[current_state])
        proposed_prob = float(probs[proposed_state])
        if proposed_state != current_state and proposed_prob < current_prob + switch_margin:
            predictions.append(current_state)
            continue
        current_state = proposed_state
        predictions.append(current_state)

    pred_array = np.asarray(predictions, dtype=int)
    ordered["pred_label_id"] = pred_array
    ordered["pred_label"] = [LABELS[idx] for idx in pred_array]
    for idx, column in enumerate(PROBA_COLUMNS):
        ordered[column] = smoothed_values[:, idx]
    return ordered


def main() -> None:
    args = parse_args()
    df = read_table(args.input_path).copy()
    required = {"open_time", "symbol", *PROBA_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    processed = [
        smooth_group(group.copy(), window=args.window, switch_margin=args.switch_margin)
        for _, group in df.groupby("symbol", sort=True)
    ]
    out = pd.concat(processed, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)
    write_table(out, ensure_parent(Path(args.output_path)))
    print(
        f"[saved] {args.output_path} window={args.window} switch_margin={args.switch_margin} rows={len(out)}"
    )


if __name__ == "__main__":
    main()
