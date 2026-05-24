from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.utils.io import ensure_dir


@dataclass(frozen=True)
class MethodSpec:
    asset: str
    pool: str
    method: str
    prediction_path: Path
    label_path: Path
    significance_path: Path
    significance_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify the incremental contribution of the semantic protocol over "
            "classification-only and conventional classification+stability MCDM baselines."
        )
    )
    parser.add_argument("--output_dir", default="experiments/summary/evaluation_framework_increment")
    parser.add_argument("--test_start", default="2025-01-01")
    return parser.parse_args()


def _path(text: str) -> Path:
    return Path(text)


def build_specs() -> list[MethodSpec]:
    common = [
        ("BTC", "logreg", "experiments/baselines/logreg_btc/test_predictions.parquet", "logistic_regression"),
        (
            "BTC",
            "logreg_postproc",
            "experiments/baselines/logreg_btc_postproc_match/test_predictions.parquet",
            "logreg_postproc",
        ),
        ("BTC", "main_gru", "experiments/main/gru_btc/test_predictions.parquet", "main_gru"),
        ("ETH", "logreg", "experiments/baselines/logreg_eth/test_predictions.parquet", "logistic_regression"),
        (
            "ETH",
            "logreg_postproc",
            "experiments/baselines/logreg_eth_postproc_match/test_predictions.parquet",
            "logreg_postproc",
        ),
        ("ETH", "main_gru", "experiments/main/gru_eth/test_predictions.parquet", "main_gru_eth"),
        ("BNB", "logreg", "experiments/baselines/logreg_bnb/test_predictions.parquet", "logistic_regression"),
        (
            "BNB",
            "logreg_postproc",
            "experiments/baselines/logreg_bnb_postproc_match/test_predictions.parquet",
            "logreg_postproc",
        ),
        ("BNB", "main_gru", "experiments/main/gru_btc_cross_bnb/test_predictions.parquet", "btc_trained_gru"),
        ("SOL", "logreg", "experiments/baselines/logreg_sol/test_predictions.parquet", "logistic_regression"),
        (
            "SOL",
            "logreg_postproc",
            "experiments/baselines/logreg_sol_postproc_match/test_predictions.parquet",
            "logreg_postproc",
        ),
        ("SOL", "main_gru", "experiments/main/gru_btc_cross_sol/test_predictions.parquet", "btc_trained_gru"),
        ("XRP", "logreg", "experiments/baselines/logreg_xrp/test_predictions.parquet", "logistic_regression"),
        (
            "XRP",
            "logreg_postproc",
            "experiments/baselines/logreg_xrp_postproc_match/test_predictions.parquet",
            "logreg_postproc",
        ),
        ("XRP", "main_gru", "experiments/main/gru_btc_cross_xrp/test_predictions.parquet", "btc_trained_gru"),
    ]
    btc_extra = [
        (
            "BTC",
            "histgb_postproc",
            "experiments/revision/histgb_btc_postproc/test_predictions.parquet",
            "histgb_postproc",
        ),
        ("BTC", "main_lstm", "experiments/main/lstm_btc/test_predictions.parquet", "main_lstm"),
        ("BTC", "tcn_96x4", "experiments/improved/main/tcn_btc_96x4/test_predictions.parquet", "tcn_96x4"),
    ]

    specs: list[MethodSpec] = []
    for asset, method, pred_path, source in common:
        specs.append(
            MethodSpec(
                asset=asset,
                pool="cross_asset_common",
                method=method,
                prediction_path=_path(pred_path),
                label_path=_path(f"data/labels_improved/default/{asset}USDT_labels.parquet"),
                significance_path=_path(
                    f"experiments/summary/{asset.lower()}_risk_state_significance_extended/risk_significance_summary.csv"
                ),
                significance_source=source,
            )
        )
    for asset, method, pred_path, source in [*common[:3], *btc_extra]:
        sig_path = (
            _path("experiments/revision/btc_risk_state_significance_with_histgb/risk_significance_summary.csv")
            if source in {"main_gru", "histgb_postproc", "tcn_96x4"}
            else _path("experiments/summary/btc_risk_state_significance_extended/risk_significance_summary.csv")
        )
        specs.append(
            MethodSpec(
                asset=asset,
                pool="btc_extended",
                method=method,
                prediction_path=_path(pred_path),
                label_path=_path("data/labels_improved/default/BTCUSDT_labels.parquet"),
                significance_path=sig_path,
                significance_source=source,
            )
        )
    return specs


def load_label_frame(path: Path, test_start: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["open_time", "symbol", "proxy_label_id"]).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    start = pd.Timestamp(test_start, tz="UTC")
    return df.loc[df["open_time"] >= start].copy()


def temporal_stability(pred: pd.DataFrame) -> tuple[int, float, float]:
    ordered = pred.sort_values("open_time").reset_index(drop=True)
    values = ordered["pred_label_id"].astype(int)
    transitions = int((values != values.shift(1)).sum() - 1)
    transitions = max(transitions, 0)
    day_count = max((ordered["open_time"].max() - ordered["open_time"].min()).total_seconds() / 86400.0, 1.0)

    segment_lengths: list[int] = []
    current = 0
    previous = None
    for value in values.tolist():
        if previous is None or value == previous:
            current += 1
        else:
            segment_lengths.append(current)
            current = 1
        previous = value
    if current:
        segment_lengths.append(current)

    return transitions, float(transitions / day_count), float(sum(segment_lengths) / len(segment_lengths))


def _benefit(values: pd.Series, higher_is_better: bool) -> pd.Series:
    span = values.max() - values.min()
    if span == 0:
        return pd.Series(np.ones(len(values)), index=values.index)
    scaled = (values - values.min()) / span
    return scaled if higher_is_better else 1.0 - scaled


def add_framework_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (pool, asset), group in candidates.groupby(["pool", "asset"], sort=True):
        out = group.copy()
        out["benefit_macro_f1"] = _benefit(out["macro_f1"], True)
        out["benefit_balanced_accuracy"] = _benefit(out["balanced_accuracy"], True)
        out["benefit_smoothness"] = _benefit(out["daily_switch_rate"], False)
        out["benefit_risk_significance"] = _benefit(out["significant_risk_layering_rate"], True)
        out["ct_weighted_score"] = (
            0.4 * out["benefit_macro_f1"]
            + 0.3 * out["benefit_balanced_accuracy"]
            + 0.3 * out["benefit_smoothness"]
        )
        out["cts_weighted_score"] = (
            0.25 * out["benefit_macro_f1"]
            + 0.20 * out["benefit_balanced_accuracy"]
            + 0.20 * out["benefit_smoothness"]
            + 0.35 * out["benefit_risk_significance"]
        )

        matrix = out[["benefit_macro_f1", "benefit_balanced_accuracy", "benefit_smoothness"]].to_numpy(float)
        denom = np.linalg.norm(matrix, axis=0)
        denom[denom == 0.0] = 1.0
        weighted = matrix / denom * np.asarray([0.4, 0.3, 0.3])
        ideal = weighted.max(axis=0)
        anti = weighted.min(axis=0)
        d_ideal = np.linalg.norm(weighted - ideal, axis=1)
        d_anti = np.linalg.norm(weighted - anti, axis=1)
        out["ct_topsis_score"] = d_anti / np.maximum(d_ideal + d_anti, 1e-12)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def choose_by_framework(group: pd.DataFrame, framework: str) -> pd.Series:
    sort_specs = {
        "classification_only": (["macro_f1", "balanced_accuracy", "benefit_smoothness"], [False, False, False]),
        "ct_weighted_mcdm": (["ct_weighted_score", "macro_f1"], [False, False]),
        "ct_topsis_mcdm": (["ct_topsis_score", "macro_f1"], [False, False]),
        "cts_weighted_oracle": (["cts_weighted_score", "macro_f1"], [False, False]),
        "semantic_protocol": (
            ["significant_risk_layering_rate", "daily_switch_rate", "macro_f1"],
            [False, True, False],
        ),
    }
    cols, ascending = sort_specs[framework]
    return group.sort_values(cols, ascending=ascending).iloc[0]


def build_selection_comparison(candidates: pd.DataFrame) -> pd.DataFrame:
    frameworks = [
        "classification_only",
        "ct_weighted_mcdm",
        "ct_topsis_mcdm",
        "cts_weighted_oracle",
        "semantic_protocol",
    ]
    rows: list[dict[str, object]] = []
    for (pool, asset), group in candidates.groupby(["pool", "asset"], sort=True):
        semantic_choice = choose_by_framework(group, "semantic_protocol")
        for framework in frameworks:
            choice = choose_by_framework(group, framework)
            rows.append(
                {
                    "pool": pool,
                    "asset": asset,
                    "framework": framework,
                    "selected_method": choice["method"],
                    "macro_f1": choice["macro_f1"],
                    "balanced_accuracy": choice["balanced_accuracy"],
                    "daily_switch_rate": choice["daily_switch_rate"],
                    "risk_match_rate": choice["risk_match_rate"],
                    "significant_risk_layering_rate": choice["significant_risk_layering_rate"],
                    "semantic_protocol_method": semantic_choice["method"],
                    "selection_differs_from_protocol": choice["method"] != semantic_choice["method"],
                    "semantic_regret_vs_protocol": semantic_choice["significant_risk_layering_rate"]
                    - choice["significant_risk_layering_rate"],
                    "macro_f1_cost_vs_protocol": choice["macro_f1"] - semantic_choice["macro_f1"],
                    "switch_cost_vs_protocol": choice["daily_switch_rate"] - semantic_choice["daily_switch_rate"],
                }
            )
    return pd.DataFrame(rows)


def build_increment_summary(selection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (pool, framework), group in selection.groupby(["pool", "framework"], sort=True):
        if framework == "semantic_protocol":
            continue
        rows.append(
            {
                "pool": pool,
                "framework": framework,
                "asset_count": int(group["asset"].nunique()),
                "selection_disagreement_rate": float(group["selection_differs_from_protocol"].mean()),
                "mean_semantic_regret_vs_protocol": float(group["semantic_regret_vs_protocol"].mean()),
                "max_semantic_regret_vs_protocol": float(group["semantic_regret_vs_protocol"].max()),
                "mean_macro_f1_cost_vs_protocol": float(group["macro_f1_cost_vs_protocol"].mean()),
                "mean_switch_cost_vs_protocol": float(group["switch_cost_vs_protocol"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["pool", "framework"])


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    rows: list[dict[str, object]] = []
    label_cache: dict[Path, pd.DataFrame] = {}
    sig_cache: dict[Path, pd.DataFrame] = {}

    for spec in build_specs():
        if not spec.prediction_path.exists():
            raise FileNotFoundError(spec.prediction_path)
        labels = label_cache.setdefault(spec.label_path, load_label_frame(spec.label_path, args.test_start))
        pred = pd.read_parquet(spec.prediction_path, columns=["open_time", "symbol", "pred_label_id"]).copy()
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        merged = labels.merge(pred, on=["open_time", "symbol"], how="inner")
        y_true = merged["proxy_label_id"].astype(int)
        y_pred = merged["pred_label_id"].astype(int)
        transitions, switch_rate, avg_duration = temporal_stability(pred)

        significance = sig_cache.setdefault(spec.significance_path, pd.read_csv(spec.significance_path).set_index("source"))
        sig_row = significance.loc[spec.significance_source]
        rows.append(
            {
                "pool": spec.pool,
                "asset": spec.asset,
                "method": spec.method,
                "rows": int(len(merged)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "transitions": transitions,
                "daily_switch_rate": switch_rate,
                "avg_state_duration_bars": avg_duration,
                "risk_match_rate": float(sig_row["order_match_rate"]),
                "significant_risk_layering_rate": float(sig_row["significant_risk_layering_rate"]),
                "bootstrap_significant_risk_layering_rate": float(
                    sig_row.get("bootstrap_significant_risk_layering_rate", np.nan)
                ),
            }
        )

    candidates = add_framework_scores(pd.DataFrame(rows))
    selection = build_selection_comparison(candidates)
    increment = build_increment_summary(selection)

    candidates.to_csv(output_dir / "framework_candidates.csv", index=False)
    selection.to_csv(output_dir / "framework_selection_comparison.csv", index=False)
    increment.to_csv(output_dir / "framework_increment_summary.csv", index=False)

    print("[framework_increment_summary]")
    print(increment.to_string(index=False))
    print("\n[framework_selection_comparison]")
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
