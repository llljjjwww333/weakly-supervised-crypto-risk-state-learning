from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir

RAW_NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw csv files into canonical parquet tables.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def clean_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="mixed")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, format="mixed")

    for column in RAW_NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.sort_values("open_time")
        .drop_duplicates(subset=["open_time"], keep="last")
        .dropna(subset=["open_time", "open", "high", "low", "close", "volume"])
        .reset_index(drop=True)
    )
    return df


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)

    for csv_path in sorted(input_dir.glob("*.csv")):
        df = clean_table(csv_path)
        out_path = Path(output_dir) / f"{csv_path.stem}_clean.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[saved] {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
