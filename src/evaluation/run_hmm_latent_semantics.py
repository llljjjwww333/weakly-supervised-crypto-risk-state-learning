from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.evaluation.evaluate_economic_meaning import (
    LABEL_ORDER as ECON_LABEL_ORDER,
    build_ordering_summary as build_return_ordering_summary,
    build_pairwise_tests as build_return_pairwise_tests,
    load_label_frame as load_return_label_frame,
    load_prediction_frame as load_return_prediction_frame,
    summarize_groups as summarize_return_groups,
)
from src.evaluation.evaluate_risk_state_semantics import (
    RISK_METRICS,
    build_ordering_summary as build_risk_ordering_summary,
    build_pairwise_tests as build_risk_pairwise_tests,
    load_label_frame as load_risk_label_frame,
    load_prediction_frame as load_risk_prediction_frame,
    summarize_groups as summarize_risk_groups,
)
from src.evaluation.evaluate_risk_state_significance import build_test_rows, summarize_tests
from src.evaluation.evaluate_stability import summarize_symbol
from src.utils.io import ensure_dir


LABEL_ID_TO_NAME = {0: "bear", 1: "neutral", 2: "bull"}
LABEL_NAME_TO_ID = {value: key for key, value in LABEL_ID_TO_NAME.items()}
RISK_SCORE_COLUMNS = RISK_METRICS + ["loss_hit_2pct_24", "loss_hit_5pct_24"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HMM latent states under two label mappings.")
    parser.add_argument("--label_path", default="data/labels_improved/default/BTCUSDT_labels.parquet")
    parser.add_argument("--prediction_path", default="experiments/baselines/hmm_btc/test_predictions.parquet")
    parser.add_argument("--output_dir", default="experiments/summary/hmm_latent_semantics_btc")
    parser.add_argument("--test_start", default="2025-01-01")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap_samples", type=int, default=0)
    parser.add_argument("--bootstrap_block_size", type=int, default=24)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    return parser.parse_args()


def load_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def build_semantic_order_predictions(
    predictions: pd.DataFrame,
    label_path: str | Path,
    test_start: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_frame = load_risk_label_frame(label_path, test_start, horizon)
    merged = predictions.merge(label_frame, on=["open_time", "symbol"], how="inner")

    state_summary = (
        merged.groupby("hidden_state", as_index=False)[RISK_SCORE_COLUMNS]
        .mean()
        .sort_values("hidden_state")
        .reset_index(drop=True)
    )
    rank_columns = []
    for metric in RISK_SCORE_COLUMNS:
        rank_col = f"{metric}_rank"
        state_summary[rank_col] = state_summary[metric].rank(method="first", ascending=False)
        rank_columns.append(rank_col)
    state_summary["mean_risk_rank"] = state_summary[rank_columns].mean(axis=1)
    state_summary = state_summary.sort_values(["mean_risk_rank", "hidden_state"]).reset_index(drop=True)

    semantic_labels = ["bear", "neutral", "bull"]
    if len(state_summary) != 3:
        raise ValueError(f"Expected 3 hidden states, found {len(state_summary)}.")
    state_summary["semantic_label"] = semantic_labels[: len(state_summary)]
    mapping = dict(zip(state_summary["hidden_state"], state_summary["semantic_label"]))

    semantic = predictions.copy()
    semantic["pred_label"] = semantic["hidden_state"].map(mapping)
    semantic["pred_label_id"] = semantic["pred_label"].map(LABEL_NAME_TO_ID)
    return semantic, state_summary


def classification_metrics(frame: pd.DataFrame, method: str) -> dict[str, object]:
    y_true = frame["proxy_label_id"].astype(int)
    y_pred = frame["pred_label_id"].astype(int)
    return {
        "method": method,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "bull_f1": float(f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
        "bear_f1": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "rows": int(len(frame)),
    }


def stability_metrics(frame: pd.DataFrame, method: str) -> dict[str, object]:
    summary = summarize_symbol(frame[["open_time", "symbol", "pred_label_id"]].copy())
    summary["method"] = method
    return summary


def build_return_semantics(
    label_path: str | Path,
    prediction_paths: list[Path],
    prediction_names: list[str],
    output_dir: Path,
    test_start: str,
) -> None:
    label_frame = load_return_label_frame(label_path, test_start)
    frames = []
    for path, name in zip(prediction_paths, prediction_names):
        frame = load_return_prediction_frame(path, name, label_frame)
        frame["state_label"] = pd.Categorical(frame["state_label"], categories=ECON_LABEL_ORDER, ordered=True)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(["source", "open_time"]).reset_index(drop=True)

    overall = summarize_return_groups(combined, ["source", "state_label"]).sort_values(["source", "state_label"])
    by_period = summarize_return_groups(combined, ["source", "period", "state_label"]).sort_values(
        ["source", "period", "state_label"]
    )
    ordering = build_return_ordering_summary(by_period)
    pairwise = build_return_pairwise_tests(combined)

    overall.to_csv(output_dir / "return_overall_summary.csv", index=False)
    by_period.to_csv(output_dir / "return_by_period_summary.csv", index=False)
    ordering.to_csv(output_dir / "return_ordering_summary.csv", index=False)
    pairwise.to_csv(output_dir / "return_pairwise_tests.csv", index=False)


def build_risk_semantics_and_significance(
    label_path: str | Path,
    prediction_paths: list[Path],
    prediction_names: list[str],
    output_dir: Path,
    test_start: str,
    horizon: int,
    alpha: float,
    bootstrap_samples: int,
    bootstrap_block_size: int,
    bootstrap_seed: int,
) -> None:
    label_frame = load_risk_label_frame(label_path, test_start, horizon)
    frames = []
    for path, name in zip(prediction_paths, prediction_names):
        frame = load_risk_prediction_frame(path, name, label_frame)
        frame["state_label"] = pd.Categorical(frame["state_label"], categories=ECON_LABEL_ORDER, ordered=True)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(["source", "open_time"]).reset_index(drop=True)

    overall = summarize_risk_groups(combined, ["source", "state_label"]).sort_values(["source", "state_label"])
    by_period = summarize_risk_groups(combined, ["source", "period", "state_label"]).sort_values(
        ["source", "period", "state_label"]
    )
    ordering = build_risk_ordering_summary(by_period)
    pairwise = build_risk_pairwise_tests(combined)
    detail = build_test_rows(
        combined,
        alpha=alpha,
        bootstrap_samples=bootstrap_samples,
        bootstrap_block_size=bootstrap_block_size,
        bootstrap_seed=bootstrap_seed,
    )
    summary, metric_summary = summarize_tests(detail)

    overall.to_csv(output_dir / "risk_overall_summary.csv", index=False)
    by_period.to_csv(output_dir / "risk_by_period_summary.csv", index=False)
    ordering.to_csv(output_dir / "risk_ordering_summary.csv", index=False)
    pairwise.to_csv(output_dir / "risk_pairwise_tests.csv", index=False)
    detail.to_csv(output_dir / "risk_significance_detail.csv", index=False)
    summary.to_csv(output_dir / "risk_significance_summary.csv", index=False)
    metric_summary.to_csv(output_dir / "risk_significance_metric_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    proxy_predictions = load_predictions(args.prediction_path)
    semantic_predictions, state_mapping = build_semantic_order_predictions(
        proxy_predictions, args.label_path, args.test_start, args.horizon
    )

    proxy_path = output_dir / "hmm_proxy_mapping_predictions.parquet"
    semantic_path = output_dir / "hmm_semantic_ordering_predictions.parquet"
    proxy_predictions.to_parquet(proxy_path, index=False)
    semantic_predictions.to_parquet(semantic_path, index=False)
    state_mapping.to_csv(output_dir / "hmm_semantic_state_mapping.csv", index=False)

    proxy_metrics = classification_metrics(proxy_predictions, "hmm_proxy_mapping")
    semantic_metrics = classification_metrics(semantic_predictions, "hmm_semantic_ordering")
    pd.DataFrame([proxy_metrics, semantic_metrics]).to_csv(output_dir / "classification_metrics.csv", index=False)

    proxy_stability = stability_metrics(proxy_predictions, "hmm_proxy_mapping")
    semantic_stability = stability_metrics(semantic_predictions, "hmm_semantic_ordering")
    pd.DataFrame([proxy_stability, semantic_stability]).to_csv(output_dir / "stability_metrics.csv", index=False)

    prediction_paths = [proxy_path, semantic_path]
    prediction_names = ["hmm_proxy_mapping", "hmm_semantic_ordering"]
    build_return_semantics(args.label_path, prediction_paths, prediction_names, output_dir, args.test_start)
    build_risk_semantics_and_significance(
        args.label_path,
        prediction_paths,
        prediction_names,
        output_dir,
        args.test_start,
        args.horizon,
        args.alpha,
        args.bootstrap_samples,
        args.bootstrap_block_size,
        args.bootstrap_seed,
    )

    print(f"[wrote] {output_dir / 'classification_metrics.csv'}")
    print(f"[wrote] {output_dir / 'stability_metrics.csv'}")
    print(f"[wrote] {output_dir / 'return_ordering_summary.csv'}")
    print(f"[wrote] {output_dir / 'risk_ordering_summary.csv'}")
    print(f"[wrote] {output_dir / 'risk_significance_summary.csv'}")


if __name__ == "__main__":
    main()
