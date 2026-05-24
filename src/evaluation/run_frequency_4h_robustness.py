from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.evaluation.evaluate_risk_state_semantics import (
    build_ordering_summary,
    build_proxy_frame,
    summarize_groups,
)
from src.evaluation.evaluate_risk_state_significance import build_test_rows, summarize_tests
from src.features.build_labels import build_label_frame
from src.features.make_features import build_features
from src.models.baselines.postprocess_predictions import smooth_group
from src.models.baselines.run_histgb import split_by_time as histgb_split_by_time
from src.utils.io import ensure_dir

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler


ASSETS = ["BTCUSDT", "ETHUSDT"]
MODEL_ORDER = ["Proxy label", "LogReg", "LogReg + post-proc.", "HistGB + post-proc."]
TRAIN_END = "2023-12-31"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"

FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_4",
    "log_return_24",
    "rolling_vol_24",
    "rolling_vol_72",
    "high_low_range",
    "open_close_change",
    "volume_zscore_24",
    "quote_volume_zscore_24",
    "trade_count_zscore_24",
    "ema_gap_12_48",
    "ema_gap_24_72",
    "rolling_skew_24",
    "rolling_kurt_24",
    "up_bar_ratio_24",
    "down_bar_ratio_24",
    "max_drawdown_72",
    "trend_strength_24",
    "taker_buy_ratio",
    "volume_price_corr_24",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4h robustness baselines for BTC and ETH.")
    parser.add_argument("--raw_dir", default="data/raw/spot/1h")
    parser.add_argument("--output_root", default="experiments/frequency_4h")
    parser.add_argument("--summary_path", default="experiments/summary/frequency_4h_robustness.csv")
    parser.add_argument("--bootstrap_samples", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--post_window", type=int, default=6)
    parser.add_argument("--post_margin", type=float, default=0.0)
    return parser.parse_args()


def resample_4h(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="mixed")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, format="mixed")
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    df = df.set_index("open_time")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_asset_volume": "sum",
        "number_of_trades": "sum",
        "taker_buy_base_asset_volume": "sum",
        "taker_buy_quote_asset_volume": "sum",
        "close_time": "last",
    }
    out = df.resample("4h", label="left", closed="left").agg(agg).dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index()
    out["symbol"] = raw_path.stem
    return out


def default_label_args() -> SimpleNamespace:
    return SimpleNamespace(
        bull_return_24_min=0.01,
        bear_return_24_max=-0.01,
        ema_gap_pos_min=0.002,
        ema_gap_neg_max=-0.002,
        max_drawdown_bull_max=-0.03,
        max_drawdown_bear_max=-0.06,
        disable_drawdown_filter=False,
    )


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return histgb_split_by_time(df, TRAIN_END, VALID_END)


def add_prediction_columns(df: pd.DataFrame, pred_ids: pd.Series, pred_labels: pd.Series) -> pd.DataFrame:
    out = df[["open_time", "symbol", "proxy_label", "proxy_label_id"]].copy()
    pred_ids_series = pd.Series(pred_ids, index=df.index)
    pred_labels_series = pd.Series(pred_labels, index=df.index)
    out["pred_label_id"] = pred_ids_series.astype(int)
    out["pred_label"] = pred_labels_series.astype(str)
    return out


def summarize_stability(pred_df: pd.DataFrame) -> tuple[float, float]:
    ordered = pred_df.sort_values("open_time").reset_index(drop=True)
    preds = ordered["pred_label_id"].astype(int)
    transitions = int((preds != preds.shift(1)).sum() - 1)
    transitions = max(transitions, 0)
    segment_lengths: list[int] = []
    current_length = 0
    previous = None
    for value in preds.tolist():
        if previous is None or value == previous:
            current_length += 1
        else:
            segment_lengths.append(current_length)
            current_length = 1
        previous = value
    if current_length > 0:
        segment_lengths.append(current_length)
    day_count = max((ordered["open_time"].max() - ordered["open_time"].min()).total_seconds() / 86400.0, 1.0)
    return float(transitions / day_count), float(sum(segment_lengths) / len(segment_lengths))


def risk_and_return_summary(label_df: pd.DataFrame, prediction_frames: list[tuple[str, str, pd.DataFrame]], alpha: float, bootstrap_samples: int) -> pd.DataFrame:
    label_frame = label_df[["open_time", "symbol", "proxy_label", "close", "low", "high", "future_return_24", "future_vol_24"]].copy()
    future_low = label_frame["low"].shift(-1).iloc[::-1].rolling(24, min_periods=24).min().iloc[::-1]
    future_high = label_frame["high"].shift(-1).iloc[::-1].rolling(24, min_periods=24).max().iloc[::-1]
    close = label_frame["close"].replace(0, pd.NA).astype(float)
    label_frame["future_abs_return_24"] = label_frame["future_return_24"].abs()
    label_frame["future_path_loss_24"] = (1.0 - future_low / close).clip(lower=0.0)
    label_frame["future_path_gain_24"] = (future_high / close - 1.0).clip(lower=0.0)
    label_frame["future_range_24"] = label_frame["future_path_loss_24"] + label_frame["future_path_gain_24"]
    label_frame["loss_hit_2pct_24"] = (label_frame["future_path_loss_24"] >= 0.02).astype(float)
    label_frame["loss_hit_5pct_24"] = (label_frame["future_path_loss_24"] >= 0.05).astype(float)
    label_frame = label_frame[
        [
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
    ].copy()
    label_frame["period"] = label_frame["open_time"].apply(lambda ts: f"{ts.year}-{'H1' if ts.month <= 6 else 'H2'}")
    combined = [build_proxy_frame(label_frame)]
    for source_name, _, pred_df in prediction_frames:
        combined.append(load_prediction_frame(pred_df, source_name, label_frame))
    combined_df = pd.concat(combined, ignore_index=True).sort_values(["source", "open_time"]).reset_index(drop=True)

    by_period = summarize_groups(combined_df, ["source", "period", "state_label"]).sort_values(
        ["source", "period", "state_label"]
    )
    ordering = build_ordering_summary(by_period)
    return_summary = (
        ordering.loc[ordering["metric"].isin(["future_vol_24", "future_abs_return_24"])]
        .groupby("source", as_index=False)
        .agg(
            return_order_checks=("metric", "count"),
            return_order_matches=("descending_order", lambda s: int((s == "bull > neutral > bear").sum())),
        )
    )
    return_summary["return_order_rate"] = return_summary["return_order_matches"] / return_summary["return_order_checks"]

    risk_detail = build_test_rows(
        combined_df,
        alpha=alpha,
        bootstrap_samples=bootstrap_samples,
        bootstrap_block_size=24,
        bootstrap_seed=42,
    )
    risk_summary, _ = summarize_tests(risk_detail)
    risk_summary = risk_summary.rename(
        columns={
            "checks": "risk_order_checks",
            "order_matches": "risk_order_matches",
            "order_match_rate": "risk_order_rate",
            "significant_risk_layering": "sig_risk_checks",
            "significant_risk_layering_rate": "sig_risk_rate",
        }
    )
    keep = ["source", "risk_order_checks", "risk_order_matches", "risk_order_rate", "sig_risk_checks", "sig_risk_rate"]
    merged = risk_summary[keep].merge(return_summary, on="source", how="left")
    return merged


def load_prediction_frame(pred_df: pd.DataFrame, source_name: str, label_frame: pd.DataFrame) -> pd.DataFrame:
    merged = pred_df[["open_time", "symbol", "pred_label"]].copy().merge(
        label_frame.drop(columns=["proxy_label"], errors="ignore"),
        on=["open_time", "symbol"],
        how="inner",
    )
    merged["source"] = source_name
    merged["state_label"] = merged["pred_label"].astype(str)
    return merged.drop(columns=["pred_label"])


def build_rows_for_asset(asset: str, args: argparse.Namespace) -> list[dict[str, object]]:
    raw_path = Path(args.raw_dir) / f"{asset}.csv"
    out_root = Path(args.output_root) / asset
    ensure_dir(out_root)

    raw_4h = resample_4h(raw_path)
    features = build_features(raw_4h)
    labeled = build_label_frame(features, default_label_args()).dropna(subset=["proxy_label_id"]).reset_index(drop=True)
    train_df, valid_df, test_df = split_by_time(labeled)
    if train_df.empty or test_df.empty:
        raise ValueError(f"{asset} 4h split is empty.")

    feature_train = train_df[FEATURE_COLUMNS]
    feature_test = test_df[FEATURE_COLUMNS]
    y_train = train_df["proxy_label_id"].astype(int)
    y_test = test_df["proxy_label_id"].astype(int)

    label_map = {0: "bear", 1: "neutral", 2: "bull"}
    prediction_frames: list[tuple[str, str, pd.DataFrame]] = []

    proxy_pred = add_prediction_columns(test_df, test_df["proxy_label_id"], test_df["proxy_label"])
    prediction_frames.append(("proxy_label", "Proxy label", proxy_pred))

    logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]
    )
    logreg.fit(feature_train, y_train)
    logreg_pred_ids = pd.Series(logreg.predict(feature_test), index=test_df.index)
    logreg_pred_labels = logreg_pred_ids.map(label_map)
    logreg_pred = add_prediction_columns(test_df, logreg_pred_ids, logreg_pred_labels)
    proba = logreg.predict_proba(feature_test)
    for idx, name in enumerate(["proba_bear", "proba_neutral", "proba_bull"]):
        logreg_pred[name] = proba[:, idx]
    prediction_frames.append(("logreg_raw", "LogReg", logreg_pred))

    logreg_post = smooth_group(logreg_pred.copy(), window=args.post_window, switch_margin=args.post_margin)
    prediction_frames.append(("logreg_postproc", "LogReg + post-proc.", logreg_post))

    imputer = SimpleImputer(strategy="median")
    x_train_hist = imputer.fit_transform(feature_train)
    x_test_hist = imputer.transform(feature_test)
    histgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        max_iter=300,
        random_state=42,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    histgb.fit(x_train_hist, y_train, sample_weight=sample_weight)
    histgb_pred_ids = pd.Series(histgb.predict(x_test_hist), index=test_df.index)
    histgb_pred_labels = histgb_pred_ids.map(label_map)
    histgb_pred = add_prediction_columns(test_df, histgb_pred_ids, histgb_pred_labels)
    histgb_proba = histgb.predict_proba(x_test_hist)
    for idx, name in enumerate(["proba_bear", "proba_neutral", "proba_bull"]):
        histgb_pred[name] = histgb_proba[:, idx]
    histgb_post = smooth_group(histgb_pred.copy(), window=args.post_window, switch_margin=args.post_margin)
    prediction_frames.append(("histgb_postproc", "HistGB + post-proc.", histgb_post))

    semantic = risk_and_return_summary(test_df, prediction_frames, args.alpha, args.bootstrap_samples)
    semantic_map = semantic.set_index("source").to_dict(orient="index")

    rows: list[dict[str, object]] = []
    for source_name, model_name, pred_df in prediction_frames:
        y_pred = pred_df["pred_label_id"].astype(int)
        switch_day, avg_duration = summarize_stability(pred_df)
        metrics = {
            "asset": asset.replace("USDT", ""),
            "frequency": "4h",
            "model": model_name,
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "bull_f1": float(f1_score(y_test, y_pred, average=None, labels=[2], zero_division=0)[0]),
            "bear_f1": float(f1_score(y_test, y_pred, average=None, labels=[0], zero_division=0)[0]),
            "switch_day": switch_day,
            "avg_duration": avg_duration,
        }
        metrics.update(semantic_map.get(source_name, {}))
        rows.append(metrics)
        pred_df.to_parquet(out_root / f"{source_name}_test_predictions.parquet", index=False)

    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for asset in ASSETS:
        rows.extend(build_rows_for_asset(asset, args))
    result = pd.DataFrame(rows)
    result["model"] = pd.Categorical(result["model"], categories=MODEL_ORDER, ordered=True)
    result = result.sort_values(["asset", "model"]).reset_index(drop=True)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(summary_path, index=False)
    print(f"[wrote] {summary_path} rows={len(result)}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
