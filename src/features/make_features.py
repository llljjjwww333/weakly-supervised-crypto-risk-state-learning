from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build engineered features from cleaned kline tables.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def rolling_drawdown(close: pd.Series, window: int) -> pd.Series:
    peak = close.rolling(window, min_periods=1).max()
    return close / peak - 1.0


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window).corr(b)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("open_time").reset_index(drop=True)

    out["log_close"] = np.log(out["close"].clip(lower=1e-12))
    out["log_return_1"] = out["log_close"].diff(1)
    out["log_return_4"] = out["log_close"].diff(4)
    out["log_return_24"] = out["log_close"].diff(24)

    out["rolling_vol_24"] = out["log_return_1"].rolling(24).std()
    out["rolling_vol_72"] = out["log_return_1"].rolling(72).std()

    out["high_low_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["open_close_change"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)

    out["volume_zscore_24"] = rolling_zscore(out["volume"], 24)
    out["quote_volume_zscore_24"] = rolling_zscore(out["quote_asset_volume"], 24)
    out["trade_count_zscore_24"] = rolling_zscore(out["number_of_trades"], 24)

    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_24"] = out["close"].ewm(span=24, adjust=False).mean()
    out["ema_48"] = out["close"].ewm(span=48, adjust=False).mean()
    out["ema_72"] = out["close"].ewm(span=72, adjust=False).mean()
    out["ema_gap_12_48"] = out["ema_12"] / out["ema_48"] - 1.0
    out["ema_gap_24_72"] = out["ema_24"] / out["ema_72"] - 1.0

    out["rolling_skew_24"] = out["log_return_1"].rolling(24).skew()
    out["rolling_kurt_24"] = out["log_return_1"].rolling(24).kurt()
    out["up_bar_ratio_24"] = (out["log_return_1"] > 0).rolling(24).mean()
    out["down_bar_ratio_24"] = (out["log_return_1"] < 0).rolling(24).mean()

    out["drawdown_72"] = rolling_drawdown(out["close"], 72)
    out["max_drawdown_72"] = out["drawdown_72"].rolling(72).min()
    out["trend_strength_24"] = out["ema_gap_12_48"] / out["rolling_vol_24"].replace(0, np.nan)

    out["taker_buy_ratio"] = (
        out["taker_buy_base_asset_volume"] / out["volume"].replace(0, np.nan)
    )
    out["volume_price_corr_24"] = rolling_corr(out["log_return_1"], out["volume"], 24)

    out["future_return_24"] = out["close"].shift(-24) / out["close"] - 1.0
    out["future_return_72"] = out["close"].shift(-72) / out["close"] - 1.0
    out["future_vol_24"] = out["log_return_1"].shift(-23).rolling(24).std()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.drop(columns=["log_close", "ema_12", "ema_24", "ema_48", "ema_72", "drawdown_72"])
    return out


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)

    for path in sorted(input_dir.glob("*_clean.parquet")):
        df = pd.read_parquet(path)
        feature_df = build_features(df)
        symbol = path.stem.replace("_clean", "")
        out_path = Path(output_dir) / f"{symbol}_features.parquet"
        feature_df.to_parquet(out_path, index=False)
        print(f"[saved] {out_path} rows={len(feature_df)}")


if __name__ == "__main__":
    main()
