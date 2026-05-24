from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


def parse_variant_items(items: list[str]) -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --variant item: {item!r}. Expected name=path.")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Variant name cannot be empty: {item!r}")
        variants[name] = Path(raw_path.strip())
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether continuity regularization and the volatility gate reduce switching in low-volatility periods."
    )
    parser.add_argument("--label_path", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="Inclusive UTC start time, e.g. 2025-01-01")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant prediction path in name=path form. Can be passed multiple times.",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def summarize_variant(
    name: str,
    prediction_path: Path,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_parquet(prediction_path)[["open_time", "pred_label_id", "proxy_label_id"]].copy()
    pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
    pred = pred.merge(labels[["open_time", "rolling_vol_24"]], on="open_time", how="left")
    pred = pred.sort_values("open_time").reset_index(drop=True)

    pred["local_vol"] = (pred["rolling_vol_24"].shift(1) + pred["rolling_vol_24"]) * 0.5
    pred["switch"] = pred["pred_label_id"].ne(pred["pred_label_id"].shift(1)).fillna(False)
    pred["proxy_switch"] = pred["proxy_label_id"].ne(pred["proxy_label_id"].shift(1)).fillna(False)
    pred["unsupported_switch"] = pred["switch"] & (~pred["proxy_switch"])

    valid = pred[pred["local_vol"].notna()].copy()
    q1, q2 = valid["local_vol"].quantile([1 / 3, 2 / 3]).tolist()
    valid["vol_bin"] = pd.cut(
        valid["local_vol"],
        bins=[-float("inf"), q1, q2, float("inf")],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )

    rows: list[dict[str, object]] = []
    for vol_bin, group in valid.groupby("vol_bin", observed=False):
        rows.append(
            {
                "variant": name,
                "vol_bin": str(vol_bin),
                "rows": int(len(group)),
                "mean_local_vol": float(group["local_vol"].mean()),
                "switches": int(group["switch"].sum()),
                "switch_rate": float(group["switch"].mean()),
                "proxy_switches": int(group["proxy_switch"].sum()),
                "proxy_switch_rate": float(group["proxy_switch"].mean()),
                "unsupported_switches": int(group["unsupported_switch"].sum()),
                "unsupported_switch_rate": float(group["unsupported_switch"].mean()),
            }
        )

    rows.append(
        {
            "variant": name,
            "vol_bin": "all",
            "rows": int(len(valid)),
            "mean_local_vol": float(valid["local_vol"].mean()),
            "switches": int(valid["switch"].sum()),
            "switch_rate": float(valid["switch"].mean()),
            "proxy_switches": int(valid["proxy_switch"].sum()),
            "proxy_switch_rate": float(valid["proxy_switch"].mean()),
            "unsupported_switches": int(valid["unsupported_switch"].sum()),
            "unsupported_switch_rate": float(valid["unsupported_switch"].mean()),
        }
    )

    summary = pd.DataFrame(rows)
    derived = pd.DataFrame(
        [
            {
                "variant": name,
                "low_to_high_switch_ratio": float(
                    summary.loc[summary["vol_bin"] == "low", "switch_rate"].iloc[0]
                    / summary.loc[summary["vol_bin"] == "high", "switch_rate"].iloc[0]
                ),
                "low_to_high_unsupported_ratio": float(
                    summary.loc[summary["vol_bin"] == "low", "unsupported_switch_rate"].iloc[0]
                    / summary.loc[summary["vol_bin"] == "high", "unsupported_switch_rate"].iloc[0]
                ),
                "all_switch_rate": float(summary.loc[summary["vol_bin"] == "all", "switch_rate"].iloc[0]),
                "all_unsupported_switch_rate": float(
                    summary.loc[summary["vol_bin"] == "all", "unsupported_switch_rate"].iloc[0]
                ),
            }
        ]
    )
    return summary, derived


def main() -> None:
    args = parse_args()
    variants = parse_variant_items(args.variant)
    output_dir = ensure_dir(args.output_dir)

    labels = pd.read_parquet(args.label_path)[["open_time", "symbol", "rolling_vol_24", "proxy_label_id"]].copy()
    labels["open_time"] = pd.to_datetime(labels["open_time"], utc=True)
    labels = labels[(labels["symbol"] == args.symbol) & (labels["open_time"] >= pd.Timestamp(args.start, tz="UTC"))]
    labels = labels.sort_values("open_time").reset_index(drop=True)

    all_rows: list[pd.DataFrame] = []
    all_derived: list[pd.DataFrame] = []
    for name, path in variants.items():
        summary, derived = summarize_variant(name, path, labels)
        all_rows.append(summary)
        all_derived.append(derived)

    summary_df = pd.concat(all_rows, ignore_index=True)
    derived_df = pd.concat(all_derived, ignore_index=True)

    pivot = summary_df.pivot(index="variant", columns="vol_bin", values="switch_rate").reset_index()
    if {"low", "high"}.issubset(pivot.columns):
        pivot["delta_low_vs_high"] = pivot["low"] - pivot["high"]
    pivot_unsupported = summary_df.pivot(index="variant", columns="vol_bin", values="unsupported_switch_rate").reset_index()
    if {"low", "high"}.issubset(pivot_unsupported.columns):
        pivot_unsupported["delta_low_vs_high"] = pivot_unsupported["low"] - pivot_unsupported["high"]

    summary_df.to_csv(output_dir / "local_switch_behavior_by_volatility.csv", index=False)
    derived_df.to_csv(output_dir / "local_switch_behavior_summary.csv", index=False)
    pivot.to_csv(output_dir / "switch_rate_pivot.csv", index=False)
    pivot_unsupported.to_csv(output_dir / "unsupported_switch_rate_pivot.csv", index=False)

    print("[saved]", output_dir / "local_switch_behavior_by_volatility.csv")
    print("[saved]", output_dir / "local_switch_behavior_summary.csv")
    print("[saved]", output_dir / "switch_rate_pivot.csv")
    print("[saved]", output_dir / "unsupported_switch_rate_pivot.csv")
    print(summary_df.to_string(index=False))
    print("\nDerived summary:")
    print(derived_df.to_string(index=False))


if __name__ == "__main__":
    main()
