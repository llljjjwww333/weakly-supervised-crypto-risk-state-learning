from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

from src.utils.io import ensure_dir, read_table

LABEL_ORDER = ["bear", "neutral", "bull"]
RISK_METRICS = [
    "future_vol_24",
    "future_abs_return_24",
    "future_path_loss_24",
    "future_path_gain_24",
    "future_range_24",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether weakly supervised states align with future risk-state semantics."
    )
    parser.add_argument("--label_path", required=True)
    parser.add_argument("--prediction_paths", nargs="*", default=[])
    parser.add_argument("--prediction_names", nargs="*", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_start", default="2025-01-01")
    parser.add_argument("--horizon", type=int, default=24)
    return parser.parse_args()


def read_columns(path: str | Path, columns: list[str]) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path, columns=columns)
    df = read_table(table_path)
    return df[columns].copy()


def assign_halfyear_period(series: pd.Series) -> pd.Series:
    return series.apply(lambda ts: f"{ts.year}-{'H1' if ts.month <= 6 else 'H2'}")


def future_window_low(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


def future_window_high(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]


def load_label_frame(path: str | Path, test_start: str, horizon: int) -> pd.DataFrame:
    keep_cols = ["open_time", "symbol", "proxy_label", "close", "low", "high", "future_return_24", "future_vol_24"]
    df = read_columns(path, keep_cols).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)

    future_low = future_window_low(df["low"], horizon)
    future_high = future_window_high(df["high"], horizon)
    close = df["close"].replace(0, pd.NA).astype(float)

    df["future_abs_return_24"] = df["future_return_24"].abs()
    df["future_path_loss_24"] = (1.0 - future_low / close).clip(lower=0.0)
    df["future_path_gain_24"] = (future_high / close - 1.0).clip(lower=0.0)
    df["future_range_24"] = df["future_path_loss_24"] + df["future_path_gain_24"]
    df["loss_hit_2pct_24"] = (df["future_path_loss_24"] >= 0.02).astype(float)
    df["loss_hit_5pct_24"] = (df["future_path_loss_24"] >= 0.05).astype(float)

    start = pd.Timestamp(test_start, tz="UTC")
    keep = [
        "open_time",
        "symbol",
        "proxy_label",
        "future_vol_24",
        "future_abs_return_24",
        "future_path_loss_24",
        "future_path_gain_24",
        "future_range_24",
        "loss_hit_2pct_24",
        "loss_hit_5pct_24",
    ]
    out = df.loc[df["open_time"] >= start, keep].copy()
    out["period"] = assign_halfyear_period(out["open_time"])
    return out


def build_proxy_frame(label_frame: pd.DataFrame) -> pd.DataFrame:
    out = label_frame.copy()
    out["source"] = "proxy_label"
    out["state_label"] = out["proxy_label"].astype(str)
    return out.drop(columns=["proxy_label"])


def load_prediction_frame(path: str | Path, source_name: str, label_frame: pd.DataFrame) -> pd.DataFrame:
    pred = read_columns(path, ["open_time", "symbol", "pred_label"]).copy()
    pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
    merged = pred.merge(
        label_frame.drop(columns=["proxy_label"], errors="ignore"),
        on=["open_time", "symbol"],
        how="inner",
    )
    merged["source"] = source_name
    merged["state_label"] = merged["pred_label"].astype(str)
    return merged.drop(columns=["pred_label"])


def summarize_groups(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["rows"] = int(len(group))
        for metric in RISK_METRICS:
            series = group[metric].dropna()
            row[f"{metric}_mean"] = float(series.mean()) if not series.empty else float("nan")
            row[f"{metric}_median"] = float(series.median()) if not series.empty else float("nan")
        row["loss_hit_2pct_24_rate"] = float(group["loss_hit_2pct_24"].mean())
        row["loss_hit_5pct_24_rate"] = float(group["loss_hit_5pct_24"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_ordering_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordering_metrics = RISK_METRICS + ["loss_hit_2pct_24_rate", "loss_hit_5pct_24_rate"]
    for (source, period), group in summary.groupby(["source", "period"], dropna=False, sort=True):
        for metric in ordering_metrics:
            value_col = metric if metric.endswith("_rate") else f"{metric}_mean"
            available = group[["state_label", value_col]].dropna().copy()
            if available.empty:
                continue
            ordered = available.sort_values(value_col, ascending=False)["state_label"].tolist()
            rows.append(
                {
                    "source": source,
                    "period": period,
                    "metric": metric,
                    "descending_order": " > ".join(ordered),
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = [("bear", "neutral"), ("bear", "bull"), ("neutral", "bull")]
    test_metrics = RISK_METRICS + ["loss_hit_2pct_24", "loss_hit_5pct_24"]
    for (source, period), group in frame.groupby(["source", "period"], dropna=False, sort=True):
        for left_label, right_label in pairs:
            left_group = group.loc[group["state_label"] == left_label]
            right_group = group.loc[group["state_label"] == right_label]
            for metric in test_metrics:
                left_values = left_group[metric].dropna()
                right_values = right_group[metric].dropna()
                if len(left_values) < 2 or len(right_values) < 2:
                    continue
                test = ttest_ind(left_values, right_values, equal_var=False, nan_policy="omit")
                rows.append(
                    {
                        "source": source,
                        "period": period,
                        "metric": metric,
                        "left_label": left_label,
                        "right_label": right_label,
                        "left_rows": int(len(left_values)),
                        "right_rows": int(len(right_values)),
                        "left_mean": float(left_values.mean()),
                        "right_mean": float(right_values.mean()),
                        "mean_diff": float(left_values.mean() - right_values.mean()),
                        "t_stat": float(test.statistic),
                        "p_value": float(test.pvalue),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if len(args.prediction_paths) != len(args.prediction_names):
        raise ValueError("prediction_paths and prediction_names must have the same length.")

    output_dir = ensure_dir(args.output_dir)
    label_frame = load_label_frame(args.label_path, args.test_start, args.horizon)

    frames = [build_proxy_frame(label_frame)]
    for path, name in zip(args.prediction_paths, args.prediction_names):
        frames.append(load_prediction_frame(path, name, label_frame))

    combined = pd.concat(frames, ignore_index=True)
    combined["state_label"] = pd.Categorical(combined["state_label"], categories=LABEL_ORDER, ordered=True)
    combined = combined.sort_values(["source", "open_time"]).reset_index(drop=True)

    overall = summarize_groups(combined, ["source", "state_label"]).sort_values(["source", "state_label"])
    by_period = summarize_groups(combined, ["source", "period", "state_label"]).sort_values(
        ["source", "period", "state_label"]
    )
    ordering = build_ordering_summary(by_period)
    pairwise = build_pairwise_tests(combined)

    overall.to_csv(Path(output_dir) / "overall_summary.csv", index=False)
    by_period.to_csv(Path(output_dir) / "by_period_summary.csv", index=False)
    ordering.to_csv(Path(output_dir) / "ordering_summary.csv", index=False)
    pairwise.to_csv(Path(output_dir) / "pairwise_tests.csv", index=False)

    print("[overall]")
    print(overall.to_string(index=False))
    print("\n[ordering]")
    print(ordering.to_string(index=False))


if __name__ == "__main__":
    main()
