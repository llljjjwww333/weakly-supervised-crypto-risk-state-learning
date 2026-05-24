from __future__ import annotations

import argparse
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from src.evaluation.evaluate_risk_state_semantics import (
    LABEL_ORDER,
    build_proxy_frame,
    load_label_frame,
    load_prediction_frame,
)
from src.utils.io import ensure_dir

RISK_TEST_METRICS = [
    "future_vol_24",
    "future_abs_return_24",
    "future_path_loss_24",
    "future_path_gain_24",
    "future_range_24",
    "loss_hit_2pct_24",
    "loss_hit_5pct_24",
]
EXPECTED_ORDER = "bear > neutral > bull"
EXPECTED_PAIRS = [("bear", "neutral"), ("neutral", "bull"), ("bear", "bull")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nonparametric significance tests for risk-state separation."
    )
    parser.add_argument("--label_path", required=True)
    parser.add_argument("--prediction_paths", nargs="*", default=[])
    parser.add_argument("--prediction_names", nargs="*", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_start", default="2025-01-01")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap_samples", type=int, default=0)
    parser.add_argument("--bootstrap_block_size", type=int, default=24)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    return parser.parse_args()


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (original_idx, p_value) in enumerate(indexed):
        candidate = min(1.0, p_value * (total - rank))
        running_max = max(running_max, candidate)
        adjusted[original_idx] = running_max
    return adjusted


def descending_order(group: pd.DataFrame, metric: str, agg: str) -> str:
    if agg == "mean":
        values = group.groupby("state_label")[metric].mean()
    elif agg == "median":
        values = group.groupby("state_label")[metric].median()
    else:
        raise ValueError(f"Unsupported aggregate: {agg}")
    ordered = values.dropna().sort_values(ascending=False).index.tolist()
    return " > ".join(str(label) for label in ordered)


def _sample_block_bootstrap(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return values
    if block_size <= 1 or n == 1:
        indices = rng.integers(0, n, size=n)
        return values[indices]

    blocks: list[np.ndarray] = []
    while sum(len(block) for block in blocks) < n:
        start = int(rng.integers(0, n))
        block_idx = (np.arange(block_size) + start) % n
        blocks.append(values[block_idx])
    sampled = np.concatenate(blocks)[:n]
    return sampled


def _bootstrap_greater_p_value(
    left_values: np.ndarray,
    right_values: np.ndarray,
    samples: int,
    block_size: int,
    rng: np.random.Generator,
) -> float:
    if samples <= 0:
        return float("nan")
    diffs = np.empty(samples, dtype=float)
    for idx in range(samples):
        left_sample = _sample_block_bootstrap(left_values, block_size, rng)
        right_sample = _sample_block_bootstrap(right_values, block_size, rng)
        diffs[idx] = float(left_sample.mean() - right_sample.mean())
    return float(np.mean(diffs <= 0.0))


def build_test_rows(
    frame: pd.DataFrame,
    alpha: float,
    bootstrap_samples: int = 0,
    bootstrap_block_size: int = 24,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["source", "period"]
    for (source, period), group in frame.groupby(group_cols, dropna=False, sort=False):
        for metric in RISK_TEST_METRICS:
            state_values = {
                label: group.loc[group["state_label"] == label, metric].dropna().to_numpy()
                for label in LABEL_ORDER
            }
            if any(len(values) < 2 for values in state_values.values()):
                continue

            kruskal_result = kruskal(*(state_values[label] for label in LABEL_ORDER), nan_policy="omit")
            raw_pairwise_p: list[float] = []
            pairwise_rows: list[dict[str, object]] = []
            pairwise_bootstrap_p: list[float] = []
            bootstrap_key = f"{source}|{period}|{metric}".encode("utf-8")
            metric_rng = np.random.default_rng(bootstrap_seed + zlib.crc32(bootstrap_key))
            for left_label, right_label in EXPECTED_PAIRS:
                left_values = state_values[left_label]
                right_values = state_values[right_label]
                test = mannwhitneyu(left_values, right_values, alternative="greater")
                raw_pairwise_p.append(float(test.pvalue))
                bootstrap_p = _bootstrap_greater_p_value(
                    left_values,
                    right_values,
                    samples=bootstrap_samples,
                    block_size=bootstrap_block_size,
                    rng=metric_rng,
                )
                pairwise_bootstrap_p.append(bootstrap_p)
                pairwise_rows.append(
                    {
                        "left_label": left_label,
                        "right_label": right_label,
                        "u_stat": float(test.statistic),
                        "raw_p_value": float(test.pvalue),
                        "bootstrap_p_value": bootstrap_p,
                        "left_mean": float(left_values.mean()),
                        "right_mean": float(right_values.mean()),
                        "left_median": float(np.median(left_values)),
                        "right_median": float(np.median(right_values)),
                    }
                )

            adjusted_p = holm_adjust(raw_pairwise_p)
            pairwise_significant = []
            row: dict[str, object] = {
                "source": source,
                "period": period,
                "metric": metric,
                "checks_definition": "7 risk metrics x half-year periods",
                "mean_descending_order": descending_order(group, metric, "mean"),
                "median_descending_order": descending_order(group, metric, "median"),
                "order_matches_expected": descending_order(group, metric, "mean") == EXPECTED_ORDER,
                "kruskal_stat": float(kruskal_result.statistic),
                "kruskal_p_value": float(kruskal_result.pvalue),
                "kruskal_significant": bool(float(kruskal_result.pvalue) < alpha),
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_block_size": int(bootstrap_block_size),
                "n_bear": int(len(state_values["bear"])),
                "n_neutral": int(len(state_values["neutral"])),
                "n_bull": int(len(state_values["bull"])),
            }
            for idx, pair in enumerate(pairwise_rows):
                pair_name = f"{pair['left_label']}_gt_{pair['right_label']}"
                pair_sig = adjusted_p[idx] < alpha
                pairwise_significant.append(pair_sig)
                row[f"{pair_name}_u_stat"] = pair["u_stat"]
                row[f"{pair_name}_raw_p_value"] = pair["raw_p_value"]
                row[f"{pair_name}_holm_p_value"] = adjusted_p[idx]
                row[f"{pair_name}_significant"] = pair_sig
                row[f"{pair_name}_bootstrap_p_value"] = pair["bootstrap_p_value"]
                row[f"{pair_name}_bootstrap_significant"] = bool(
                    bootstrap_samples > 0 and float(pair["bootstrap_p_value"]) < alpha
                )
                row[f"{pair_name}_left_mean"] = pair["left_mean"]
                row[f"{pair_name}_right_mean"] = pair["right_mean"]
                row[f"{pair_name}_left_median"] = pair["left_median"]
                row[f"{pair_name}_right_median"] = pair["right_median"]

            row["all_pairwise_significant"] = all(pairwise_significant)
            if bootstrap_samples > 0:
                row["all_pairwise_bootstrap_significant"] = bool(
                    all(bool(row[f"{pair[0]}_gt_{pair[1]}_bootstrap_significant"]) for pair in EXPECTED_PAIRS)
                )
            else:
                row["all_pairwise_bootstrap_significant"] = False
            row["significant_risk_layering"] = bool(
                row["order_matches_expected"] and row["kruskal_significant"] and row["all_pairwise_significant"]
            )
            row["bootstrap_significant_risk_layering"] = bool(
                bootstrap_samples > 0
                and row["order_matches_expected"]
                and row["all_pairwise_bootstrap_significant"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_tests(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        detail.groupby("source", as_index=False)
        .agg(
            checks=("metric", "count"),
            order_matches=("order_matches_expected", "sum"),
            kruskal_significant=("kruskal_significant", "sum"),
            pairwise_significant=("all_pairwise_significant", "sum"),
            significant_risk_layering=("significant_risk_layering", "sum"),
            bootstrap_pairwise_significant=("all_pairwise_bootstrap_significant", "sum"),
            bootstrap_significant_risk_layering=("bootstrap_significant_risk_layering", "sum"),
        )
        .assign(
            order_match_rate=lambda x: x["order_matches"] / x["checks"],
            kruskal_significant_rate=lambda x: x["kruskal_significant"] / x["checks"],
            pairwise_significant_rate=lambda x: x["pairwise_significant"] / x["checks"],
            significant_risk_layering_rate=lambda x: x["significant_risk_layering"] / x["checks"],
            bootstrap_pairwise_significant_rate=lambda x: x["bootstrap_pairwise_significant"] / x["checks"],
            bootstrap_significant_risk_layering_rate=lambda x: x["bootstrap_significant_risk_layering"] / x["checks"],
        )
        .sort_values("significant_risk_layering_rate", ascending=False)
    )

    metric_summary = (
        detail.groupby(["source", "metric"], as_index=False)
        .agg(
            checks=("period", "count"),
            order_matches=("order_matches_expected", "sum"),
            significant_risk_layering=("significant_risk_layering", "sum"),
            bootstrap_significant_risk_layering=("bootstrap_significant_risk_layering", "sum"),
        )
        .assign(
            order_match_rate=lambda x: x["order_matches"] / x["checks"],
            significant_risk_layering_rate=lambda x: x["significant_risk_layering"] / x["checks"],
            bootstrap_significant_risk_layering_rate=lambda x: x["bootstrap_significant_risk_layering"] / x["checks"],
        )
        .sort_values(["source", "metric"])
    )
    return summary, metric_summary


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

    detail = build_test_rows(
        combined,
        alpha=args.alpha,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block_size=args.bootstrap_block_size,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary, metric_summary = summarize_tests(detail)

    detail.to_csv(Path(output_dir) / "risk_significance_detail.csv", index=False)
    summary.to_csv(Path(output_dir) / "risk_significance_summary.csv", index=False)
    metric_summary.to_csv(Path(output_dir) / "risk_significance_metric_summary.csv", index=False)

    print("[risk_significance_summary]")
    print(summary.to_string(index=False))
    print("\n[risk_significance_metric_summary]")
    print(metric_summary.to_string(index=False))


if __name__ == "__main__":
    main()
