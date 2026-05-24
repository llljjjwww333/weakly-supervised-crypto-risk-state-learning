from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.data.download_binance_klines import fetch_klines, standardize_klines, to_millis
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally update local Binance klines.")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--end", help="Optional end date YYYY-MM-DD, defaults to tomorrow UTC")
    return parser.parse_args()


def default_end_date() -> str:
    tomorrow = datetime.now(tz=timezone.utc).date() + timedelta(days=1)
    return tomorrow.isoformat()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    end_text = args.end or default_end_date()
    end_ms = to_millis(end_text)

    for symbol in args.symbols:
        input_path = Path(args.input_dir) / f"{symbol}.csv"
        output_path = Path(args.out_dir) / f"{symbol}.csv"
        if input_path.exists():
            existing = pd.read_csv(input_path)
            existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True, format="mixed")
            existing["close_time"] = pd.to_datetime(existing["close_time"], utc=True, format="mixed")
            if existing.empty:
                raise ValueError(f"Existing file is empty: {input_path}")
            latest_open_time = pd.to_datetime(existing["open_time"], utc=True).max()
            start_ms = int(latest_open_time.timestamp() * 1000)
        else:
            raise FileNotFoundError(f"Missing source file for incremental update: {input_path}")

        raw_df = fetch_klines(symbol, args.interval, start_ms, end_ms, sleep_seconds=0.12)
        if raw_df.empty:
            merged = existing.copy()
        else:
            new_df = standardize_klines(raw_df, symbol, args.interval)
            merged = (
                pd.concat([existing, new_df], ignore_index=True)
                .sort_values("open_time")
                .drop_duplicates(subset=["open_time"], keep="last")
            )

        merged.to_csv(output_path, index=False)
        print(f"[updated] {output_path} rows={len(merged)}")


if __name__ == "__main__":
    main()
