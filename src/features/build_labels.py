from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir

LABEL_MAP = {"bear": 0, "neutral": 1, "bull": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weakly supervised bull/bear/neutral labels.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bull_return_24_min", type=float, default=0.01)
    parser.add_argument("--bear_return_24_max", type=float, default=-0.01)
    parser.add_argument("--ema_gap_pos_min", type=float, default=0.002)
    parser.add_argument("--ema_gap_neg_max", type=float, default=-0.002)
    parser.add_argument("--max_drawdown_bull_max", type=float, default=-0.03)
    parser.add_argument("--max_drawdown_bear_max", type=float, default=-0.06)
    parser.add_argument(
        "--disable_drawdown_filter",
        action="store_true",
        help="Do not use drawdown conditions inside the weak-label rule.",
    )
    return parser.parse_args()


def assign_rule_label(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    bull_mask = (
        (df["log_return_24"] > args.bull_return_24_min)
        & (df["log_return_24"] > df["rolling_vol_24"].fillna(0))
        & (df["ema_gap_24_72"] > args.ema_gap_pos_min)
    )
    bear_mask = (
        (df["log_return_24"] < args.bear_return_24_max)
        & (df["log_return_24"] < -df["rolling_vol_24"].fillna(0))
        & (df["ema_gap_24_72"] < args.ema_gap_neg_max)
    )

    if not args.disable_drawdown_filter:
        bull_mask = bull_mask & (df["max_drawdown_72"] > args.max_drawdown_bull_max)
        bear_mask = bear_mask & (df["max_drawdown_72"] < args.max_drawdown_bear_max)

    label = np.full(len(df), "neutral", dtype=object)
    label[bull_mask] = "bull"
    label[bear_mask] = "bear"
    return pd.Series(label, index=df.index)


def build_label_frame(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["proxy_label"] = assign_rule_label(out, args)
    out["proxy_label_id"] = out["proxy_label"].map(LABEL_MAP)
    out["bull_score_rule"] = (
        out["log_return_24"].fillna(0)
        + out["ema_gap_24_72"].fillna(0)
        - out["rolling_vol_24"].fillna(0) * 0.5
        - out["max_drawdown_72"].abs().fillna(0) * 0.2
    )
    out["bear_score_rule"] = (
        -out["log_return_24"].fillna(0)
        - out["ema_gap_24_72"].fillna(0)
        + out["rolling_vol_24"].fillna(0) * 0.5
        + out["max_drawdown_72"].abs().fillna(0) * 0.2
    )
    return out


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)

    for path in sorted(input_dir.glob("*_features.parquet")):
        df = pd.read_parquet(path)
        labeled = build_label_frame(df, args)
        symbol = path.stem.replace("_features", "")
        out_path = Path(output_dir) / f"{symbol}_labels.parquet"
        labeled.to_parquet(out_path, index=False)
        print(f"[saved] {out_path} rows={len(labeled)}")

    config_path = Path(output_dir) / "label_config.json"
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "bull_return_24_min": args.bull_return_24_min,
                "bear_return_24_max": args.bear_return_24_max,
                "ema_gap_pos_min": args.ema_gap_pos_min,
                "ema_gap_neg_max": args.ema_gap_neg_max,
                "max_drawdown_bull_max": args.max_drawdown_bull_max,
                "max_drawdown_bear_max": args.max_drawdown_bear_max,
                "disable_drawdown_filter": args.disable_drawdown_filter,
            },
            file,
            indent=2,
        )
    print(f"[saved] {config_path}")


if __name__ == "__main__":
    main()
