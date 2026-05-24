from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

from src.utils.io import ensure_dir, read_table

RETURN_METRICS = ["future_return_24", "future_return_72"]
RISK_METRICS = ["future_vol_24"]
ALL_METRICS = RETURN_METRICS + RISK_METRICS
LABEL_ORDER = ["bear", "neutral", "bull"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the economic meaning of proxy labels or predicted market states."
    )
    parser.add_argument("--label_path", required=True, help="Parquet/CSV with proxy labels and future metrics.")
    parser.add_argument(
        "--prediction_paths",
        nargs="*",
        default=[],
        help="Prediction parquet/csv files with open_time and pred_label.",
    )
    parser.add_argument(
        "--prediction_names",
        nargs="*",
        default=[],
        help="Display names aligned with prediction_paths.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_start", default="2025-01-01")
    return parser.parse_args()


def read_columns(path: str | Path, columns: list[str]) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path, columns=columns)
    df = read_table(table_path)
    return df[columns].copy()


def assign_halfyear_period(series: pd.Series) -> pd.Series:
    def format_period(ts: pd.Timestamp) -> str:
        half = "H1" if ts.month <= 6 else "H2"
        return f"{ts.year}-{half}"

    return series.apply(format_period)


def load_label_frame(path: str | Path, test_start: str) -> pd.DataFrame:
    keep_cols = ["open_time", "symbol", "proxy_label"] + ALL_METRICS
    df = read_columns(path, keep_cols).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    start = pd.Timestamp(test_start, tz="UTC")
    clean = df.loc[df["open_time"] >= start, keep_cols].copy()
    clean["period"] = assign_halfyear_period(clean["open_time"])
    return clean


def build_source_frame(
    base: pd.DataFrame,
    source_name: str,
    label_col: str,
) -> pd.DataFrame:
    out = base[["open_time", "symbol", "period"] + ALL_METRICS].copy()
    out["source"] = source_name
    out["state_label"] = base[label_col].astype(str)
    return out.dropna(subset=["state_label"])


def load_prediction_frame(
    path: str | Path,
    source_name: str,
    label_frame: pd.DataFrame,
) -> pd.DataFrame:
    pred = read_columns(path, ["open_time", "symbol", "pred_label"]).copy()
    pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)

    merged = pred.merge(
        label_frame[["open_time", "symbol", "period"] + ALL_METRICS],
        on=["open_time", "symbol"],
        how="inner",
    )
    merged["source"] = source_name
    merged["state_label"] = merged["pred_label"].astype(str)
    return merged[["open_time", "symbol", "period", "source", "state_label"] + ALL_METRICS]


def summarize_groups(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["rows"] = int(len(group))
        for metric in ALL_METRICS:
            series = group[metric].dropna()
            row[f"{metric}_mean"] = float(series.mean()) if not series.empty else float("nan")
            row[f"{metric}_median"] = float(series.median()) if not series.empty else float("nan")
        if "future_return_24" in group.columns:
            r24 = group["future_return_24"].dropna()
            row["positive_rate_24"] = float((r24 > 0).mean()) if not r24.empty else float("nan")
            row["negative_rate_24"] = float((r24 < 0).mean()) if not r24.empty else float("nan")
        if "future_return_72" in group.columns:
            r72 = group["future_return_72"].dropna()
            row["positive_rate_72"] = float((r72 > 0).mean()) if not r72.empty else float("nan")
            row["negative_rate_72"] = float((r72 < 0).mean()) if not r72.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_ordering_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (source, period), group in summary.groupby(["source", "period"], dropna=False, sort=True):
        for metric in ALL_METRICS:
            metric_col = f"{metric}_mean"
            available = group[["state_label", metric_col]].dropna().copy()
            if available.empty:
                continue
            ordered = available.sort_values(metric_col, ascending=False)["state_label"].tolist()
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
    pairs = [("bull", "bear"), ("bull", "neutral"), ("bear", "neutral")]
    for (source, period), group in frame.groupby(["source", "period"], dropna=False, sort=True):
        for left_label, right_label in pairs:
            left_group = group.loc[group["state_label"] == left_label]
            right_group = group.loc[group["state_label"] == right_label]
            for metric in ALL_METRICS:
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
    label_frame = load_label_frame(args.label_path, args.test_start)

    frames = [build_source_frame(label_frame, "proxy_label", "proxy_label")]
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
