from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize how often an expected state ordering appears.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--expected_order", required=True, help="Expected order string, e.g. 'bear > neutral > bull'")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric names to keep before summarizing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_path)
    if args.metrics:
        df = df.loc[df["metric"].isin(args.metrics)].copy()
        if df.empty:
            raise ValueError("No rows left after applying metric filter.")
    expected = args.expected_order.strip()
    df["matches_expected"] = df["descending_order"].astype(str).str.strip() == expected

    summary = (
        df.groupby(["source", "metric"], as_index=False)
        .agg(
            periods=("period", "count"),
            matches=("matches_expected", "sum"),
        )
        .assign(match_rate=lambda x: x["matches"] / x["periods"])
        .sort_values(["source", "metric"])
    )

    overall = (
        df.groupby("source", as_index=False)
        .agg(
            checks=("matches_expected", "count"),
            matches=("matches_expected", "sum"),
        )
        .assign(match_rate=lambda x: x["matches"] / x["checks"])
        .sort_values("match_rate", ascending=False)
    )

    output_path = Path(args.output_path)
    ensure_parent(output_path)
    summary.to_csv(output_path, index=False)
    overall.to_csv(output_path.with_name(f"{output_path.stem}_overall.csv"), index=False)

    print("[metric_summary]")
    print(summary.to_string(index=False))
    print("\n[overall_summary]")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
