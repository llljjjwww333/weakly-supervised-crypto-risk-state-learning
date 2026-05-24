from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DETAIL_COLUMNS = [
    "asset",
    "model_key",
    "model",
    "seed",
    "macro_f1",
    "balanced_accuracy",
    "bull_f1",
    "bear_f1",
    "switch_day",
    "avg_duration",
    "risk_order_checks",
    "risk_order_matches",
    "risk_order_rate",
    "return_order_checks",
    "return_order_matches",
    "return_order_rate",
    "sig_risk_checks",
    "sig_risk_rate",
    "run_dir",
]

SUMMARY_COLUMNS = [
    "asset",
    "model_key",
    "model",
    "seeds",
    "macro_f1_mean",
    "macro_f1_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "switch_day_mean",
    "switch_day_std",
    "avg_duration_mean",
    "avg_duration_std",
    "risk_order_rate_mean",
    "risk_order_rate_std",
    "sig_risk_rate_mean",
    "sig_risk_rate_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GRU, TCN, and Transformer 5-seed robustness results.")
    parser.add_argument("--gru_detail", default="experiments/summary/seed_robustness_5seed.csv")
    parser.add_argument("--tcn_detail", default="upload_for_colab/experiments/summary/TCN/seed_robustness_5seed.csv")
    parser.add_argument(
        "--transformer_detail",
        default="upload_for_colab/experiments/summary/Transformer/seed_robustness_5seed.csv",
    )
    parser.add_argument("--output_detail", default="experiments/summary/seed_robustness_5seed.csv")
    parser.add_argument("--output_summary", default="experiments/summary/seed_robustness_5seed_summary.csv")
    return parser.parse_args()


def _load_detail(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in DETAIL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df[DETAIL_COLUMNS].copy()


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["asset"] = out["asset"].astype(str)
    out["model_key"] = out["model_key"].astype(str)
    out["model"] = out["model"].astype(str)
    out["seed"] = out["seed"].astype(int)
    out["run_dir"] = out["run_dir"].astype(str).str.replace("/", "\\", regex=False)
    for column in DETAIL_COLUMNS:
        if column in {"asset", "model_key", "model", "seed", "run_dir"}:
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["asset", "model_key", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            switch_day_mean=("switch_day", "mean"),
            switch_day_std=("switch_day", "std"),
            avg_duration_mean=("avg_duration", "mean"),
            avg_duration_std=("avg_duration", "std"),
            risk_order_rate_mean=("risk_order_rate", "mean"),
            risk_order_rate_std=("risk_order_rate", "std"),
            sig_risk_rate_mean=("sig_risk_rate", "mean"),
            sig_risk_rate_std=("sig_risk_rate", "std"),
        )
        .sort_values(["asset", "model"])
        .reset_index(drop=True)
    )
    return summary[SUMMARY_COLUMNS]


def main() -> None:
    args = parse_args()
    merged = pd.concat(
        [
            _normalize_frame(_load_detail(Path(args.gru_detail))),
            _normalize_frame(_load_detail(Path(args.tcn_detail))),
            _normalize_frame(_load_detail(Path(args.transformer_detail))),
        ],
        ignore_index=True,
    )
    merged = merged.sort_values(["asset", "model", "seed"]).reset_index(drop=True)
    summary = _summarize(merged)

    output_detail = Path(args.output_detail)
    output_summary = Path(args.output_summary)
    output_detail.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_detail, index=False)
    summary.to_csv(output_summary, index=False)

    print(f"[wrote] {output_detail} rows={len(merged)}")
    print(f"[wrote] {output_summary} rows={len(summary)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
