from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_parent

INTERVAL_TO_DELTA = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw Binance kline csv files.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--interval", required=True)
    parser.add_argument("--report_path", required=True)
    return parser.parse_args()


def validate_file(path: Path, interval: str) -> dict[str, object]:
    df = pd.read_csv(path)
    if df.empty:
        return {
            "symbol": path.stem,
            "rows": 0,
            "duplicate_open_time": 0,
            "missing_intervals": 0,
            "non_monotonic": True,
            "nan_rows": 0,
            "min_time": None,
            "max_time": None,
        }

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="mixed")
    df = df.sort_values("open_time")
    duplicate_open_time = int(df["open_time"].duplicated().sum())
    non_monotonic = bool(not df["open_time"].is_monotonic_increasing)
    expected_index = pd.date_range(
        start=df["open_time"].min(),
        end=df["open_time"].max(),
        freq=INTERVAL_TO_DELTA[interval],
        tz="UTC",
    )
    missing_intervals = int(len(expected_index.difference(df["open_time"])))
    nan_rows = int(df.isna().any(axis=1).sum())

    return {
        "symbol": path.stem,
        "rows": int(len(df)),
        "duplicate_open_time": duplicate_open_time,
        "missing_intervals": missing_intervals,
        "non_monotonic": non_monotonic,
        "nan_rows": nan_rows,
        "min_time": df["open_time"].min().isoformat(),
        "max_time": df["open_time"].max().isoformat(),
    }


def main() -> None:
    args = parse_args()
    if args.interval not in INTERVAL_TO_DELTA:
        raise ValueError(f"Unsupported interval: {args.interval}")

    input_dir = Path(args.input_dir)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No csv files found in {input_dir}")

    reports = [validate_file(path, args.interval) for path in csv_files]
    report_df = pd.DataFrame(reports)
    report_path = ensure_parent(args.report_path)
    report_df.to_csv(report_path, index=False)
    print(report_df.to_string(index=False))
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
