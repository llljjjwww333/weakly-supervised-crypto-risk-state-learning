from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import urllib3

from src.utils.io import ensure_dir

API_URLS = [
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
]
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance spot klines with the public API.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Trading symbols, e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--interval", default="1h", help="Binance interval, default 1h")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--sleep", type=float, default=0.12, help="Sleep seconds between API calls")
    return parser.parse_args()


def to_millis(date_text: str) -> int:
    dt = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def request_klines(
    session: requests.Session,
    params: dict[str, object],
    timeout: int,
    verify_ssl: bool,
) -> list[list[object]]:
    last_error: Exception | None = None
    for api_url in API_URLS:
        try:
            response = session.get(api_url, params=params, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError("No Binance API endpoints were attempted.")
    raise last_error


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, sleep_seconds: float) -> pd.DataFrame:
    rows: list[list[object]] = []
    step_ms = INTERVAL_TO_MS[interval]
    current = start_ms
    session = requests.Session()
    session.headers.update({"User-Agent": "APIN/0.1"})
    verify_ssl = True
    insecure_warning_shown = False

    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            batch = request_klines(session, params=params, timeout=30, verify_ssl=verify_ssl)
        except requests.exceptions.SSLError:
            if not verify_ssl:
                raise
            verify_ssl = False
            if not insecure_warning_shown:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                print("[warn] SSL verification failed; retrying Binance requests with verify=False.")
                insecure_warning_shown = True
            continue
        if not batch:
            break
        rows.extend(batch)
        current = int(batch[-1][0]) + step_ms
        time.sleep(sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS[:-1])

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df = df.drop(columns=["ignore"])
    return df


def standardize_klines(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    numeric_columns = [
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
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    df["symbol"] = symbol
    df["interval"] = interval
    return df


def main() -> None:
    args = parse_args()
    if args.interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {args.interval}")

    out_dir = ensure_dir(args.out_dir)
    start_ms = to_millis(args.start)
    end_ms = to_millis(args.end)

    for symbol in args.symbols:
        print(f"[download] symbol={symbol} interval={args.interval} start={args.start} end={args.end}")
        raw_df = fetch_klines(symbol, args.interval, start_ms, end_ms, args.sleep)
        df = standardize_klines(raw_df, symbol, args.interval)
        out_path = Path(out_dir) / f"{symbol}.csv"
        df.to_csv(out_path, index=False)
        print(f"[saved] {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
