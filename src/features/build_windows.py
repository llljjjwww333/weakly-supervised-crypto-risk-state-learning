from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_schema import DEFAULT_FEATURE_COLUMNS, parse_feature_list, resolve_feature_columns
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rolling window datasets.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--include_features",
        default=None,
        help="Comma-separated feature names to keep. Default keeps the canonical feature list.",
    )
    parser.add_argument(
        "--exclude_features",
        default=None,
        help="Comma-separated feature names to drop from the canonical feature list.",
    )
    return parser.parse_args()


def make_windows(df: pd.DataFrame, window: int, stride: int, feature_columns: list[str]) -> pd.DataFrame:
    clean = df.dropna(subset=feature_columns + ["proxy_label_id"]).reset_index(drop=True)
    if len(clean) < window:
        return pd.DataFrame(
            columns=[
                "open_time",
                "symbol",
                "window_start",
                "window_end",
                "proxy_label",
                "proxy_label_id",
                "rolling_vol_24_last",
                "features",
            ]
        )

    values = clean[feature_columns].to_numpy(dtype=np.float32)
    end_positions = np.arange(window - 1, len(clean), stride)
    start_positions = end_positions - window + 1
    features = [values[start_idx : end_idx + 1].tolist() for start_idx, end_idx in zip(start_positions, end_positions)]

    return pd.DataFrame(
        {
            "open_time": clean.loc[end_positions, "open_time"].to_list(),
            "symbol": clean.loc[end_positions, "symbol"].to_list(),
            "window_start": clean.loc[start_positions, "open_time"].to_list(),
            "window_end": clean.loc[end_positions, "open_time"].to_list(),
            "proxy_label": clean.loc[end_positions, "proxy_label"].to_list(),
            "proxy_label_id": clean.loc[end_positions, "proxy_label_id"].astype(int).to_list(),
            "rolling_vol_24_last": clean.loc[end_positions, "rolling_vol_24"].astype(float).to_list(),
            "features": features,
        }
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)
    include_features = parse_feature_list(args.include_features)
    exclude_features = parse_feature_list(args.exclude_features)
    feature_columns = resolve_feature_columns(include_features, exclude_features)

    for path in sorted(input_dir.glob("*_labels.parquet")):
        df = pd.read_parquet(path)
        win_df = make_windows(df, args.window, args.stride, feature_columns)
        symbol = path.stem.replace("_labels", "")
        file_name = f"{symbol}_win{args.window}.parquet"
        if args.stride != 1:
            file_name = f"{symbol}_win{args.window}_s{args.stride}.parquet"
        out_path = Path(output_dir) / file_name
        win_df.to_parquet(out_path, index=False)
        print(f"[saved] {out_path} rows={len(win_df)}")

    config_path = Path(output_dir) / "window_config.json"
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "window": args.window,
                "stride": args.stride,
                "feature_columns": feature_columns,
                "default_feature_columns": DEFAULT_FEATURE_COLUMNS,
                "include_features": include_features,
                "exclude_features": exclude_features,
            },
            file,
            indent=2,
        )
    print(f"[saved] {config_path}")


if __name__ == "__main__":
    main()
