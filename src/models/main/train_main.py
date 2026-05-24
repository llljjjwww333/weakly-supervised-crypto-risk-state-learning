from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.main.bull_bear_model import ModelConfig, build_model
from src.models.main.losses import class_balance_penalty, continuity_loss, semantic_ordering_loss
from src.utils.io import ensure_dir
from src.utils.seed import seed_everything


LABEL_ID_TO_NAME = {0: "bear", 1: "neutral", 2: "bull"}
RISK_TEST_METRICS = [
    "future_vol_24",
    "future_abs_return_24",
    "future_path_loss_24",
    "future_path_gain_24",
    "future_range_24",
    "loss_hit_2pct_24",
    "loss_hit_5pct_24",
]
EXPECTED_ORDER = ["bear", "neutral", "bull"]
EXPECTED_PAIRS = [("bear", "neutral"), ("neutral", "bull"), ("bear", "bull")]


def stack_batch_features(values: list[list[list[float]]]) -> np.ndarray:
    return np.stack([np.asarray(sequence, dtype=np.float32) for sequence in values], axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight bull/bear/neutral temporal model.")
    parser.add_argument("--input_path", required=True, help="Window parquet built by build_windows.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_end", default="2023-12-31")
    parser.add_argument("--valid_end", default="2024-12-31")
    parser.add_argument("--test_end", default=None, help="Optional inclusive end date for the test split.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--load_batch_size", type=int, default=1024)
    parser.add_argument("--model_type", choices=["gru", "lstm", "tcn", "transformer", "direct"], default="gru")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_ff_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--continuity_weight", type=float, default=0.15)
    parser.add_argument("--balance_weight", type=float, default=0.05)
    parser.add_argument("--balance_mode", choices=["uniform", "empirical", "off"], default="uniform")
    parser.add_argument("--ce_class_weight_mode", choices=["none", "inverse_freq"], default="none")
    parser.add_argument("--volatility_gate_strength", type=float, default=3.0)
    parser.add_argument(
        "--checkpoint_selection",
        choices=["valid_macro_f1", "semantic_audit", "semantic_signal_aware"],
        default="valid_macro_f1",
        help="How to choose the best validation checkpoint.",
    )
    parser.add_argument(
        "--label_path",
        default=None,
        help="Original label parquet with future-return/risk columns. Required for semantic_audit checkpointing.",
    )
    parser.add_argument("--semantic_horizon", type=int, default=24)
    parser.add_argument("--semantic_alpha", type=float, default=0.05)
    parser.add_argument("--semantic_order_weight", type=float, default=0.0)
    parser.add_argument("--semantic_margin", type=float, default=0.001)
    parser.add_argument("--semantic_min_state_mass", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class WindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        volatility: np.ndarray,
        semantic_target: np.ndarray | None = None,
    ) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.labels = torch.tensor(labels.astype(np.int64, copy=False), dtype=torch.long)
        self.volatility = torch.tensor(volatility.astype(np.float32, copy=False), dtype=torch.float32)
        if semantic_target is None:
            semantic_target = np.full(len(labels), np.nan, dtype=np.float32)
        self.semantic_target = torch.tensor(semantic_target.astype(np.float32, copy=False), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.volatility[idx], self.semantic_target[idx]


def split_by_time(
    frame: pd.DataFrame,
    features: np.ndarray,
    train_end: str,
    valid_end: str,
    test_end: str | None = None,
) -> tuple[tuple[pd.DataFrame, np.ndarray], tuple[pd.DataFrame, np.ndarray], tuple[pd.DataFrame, np.ndarray]]:
    time_index = pd.DatetimeIndex(pd.to_datetime(frame["open_time"], utc=True))
    train_cutoff = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
    valid_cutoff = pd.Timestamp(valid_end, tz="UTC") + pd.Timedelta(days=1)
    train_end_idx = int(time_index.searchsorted(train_cutoff, side="left"))
    valid_end_idx = int(time_index.searchsorted(valid_cutoff, side="left"))
    test_end_idx = len(frame)
    if test_end is not None:
        test_cutoff = pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)
        test_end_idx = int(time_index.searchsorted(test_cutoff, side="left"))

    train = (frame.iloc[:train_end_idx].reset_index(drop=True).copy(), features[:train_end_idx])
    valid = (
        frame.iloc[train_end_idx:valid_end_idx].reset_index(drop=True).copy(),
        features[train_end_idx:valid_end_idx],
    )
    test = (
        frame.iloc[valid_end_idx:test_end_idx].reset_index(drop=True).copy(),
        features[valid_end_idx:test_end_idx],
    )
    return train, valid, test


def load_window_dataset(
    path: str | Path,
    batch_size: int = 1024,
    feature_store_path: str | Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    parquet_file = pq.ParquetFile(path)
    schema_names = set(parquet_file.schema.names)
    meta_columns = ["open_time", "symbol", "proxy_label", "proxy_label_id"]
    if "rolling_vol_24_last" in schema_names:
        meta_columns.append("rolling_vol_24_last")

    frame = pd.read_parquet(path, columns=meta_columns).copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)

    total_rows = parquet_file.metadata.num_rows
    features_array: np.ndarray | None = None
    offset = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["features"]):
        batch_features = stack_batch_features(batch.to_pydict()["features"])
        if features_array is None:
            feature_shape = (total_rows, batch_features.shape[1], batch_features.shape[2])
            try:
                features_array = np.empty(feature_shape, dtype=np.float32)
            except MemoryError:
                if feature_store_path is None:
                    raise
                feature_store = Path(feature_store_path)
                if feature_store.exists():
                    feature_store.unlink()
                feature_store.parent.mkdir(parents=True, exist_ok=True)
                features_array = np.memmap(feature_store, mode="w+", dtype=np.float32, shape=feature_shape)
        next_offset = offset + len(batch_features)
        features_array[offset:next_offset] = batch_features
        offset = next_offset

    if features_array is None:
        raise ValueError(f"No features were loaded from {path}.")

    if "rolling_vol_24_last" not in frame.columns:
        frame["rolling_vol_24_last"] = features_array[:, -1, 3]

    if not frame["open_time"].is_monotonic_increasing:
        order = np.argsort(frame["open_time"].to_numpy(dtype="datetime64[ns]"))
        frame = frame.iloc[order].reset_index(drop=True)
        features_array = features_array[order]
    else:
        frame = frame.reset_index(drop=True)

    return frame, features_array


def make_loader(frame: pd.DataFrame, features: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    labels = frame["proxy_label_id"].to_numpy(dtype=np.int64)
    volatility = frame["rolling_vol_24_last"].to_numpy(dtype=np.float32)
    semantic_target = None
    if "future_vol_24" in frame.columns:
        semantic_target = frame["future_vol_24"].to_numpy(dtype=np.float32)
    dataset = WindowDataset(features, labels, volatility, semantic_target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_class_prior(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64, copy=False), minlength=num_classes).astype(np.float32)
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("Cannot compute class prior from an empty label array.")
    return torch.tensor(counts / total, dtype=torch.float32)


def compute_inverse_frequency_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64, copy=False), minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_balance_target(mode: str, labels: np.ndarray, num_classes: int) -> torch.Tensor | None:
    if mode == "off":
        return None
    if mode == "uniform":
        return torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32)
    if mode == "empirical":
        return compute_class_prior(labels, num_classes)
    raise ValueError(f"Unsupported balance_mode: {mode}")


def build_ce_class_weight(mode: str, labels: np.ndarray, num_classes: int) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "inverse_freq":
        return compute_inverse_frequency_weights(labels, num_classes)
    raise ValueError(f"Unsupported ce_class_weight_mode: {mode}")


def select_device() -> torch.device:
    if os.environ.get("FORCE_CPU", "").strip() == "1":
        return torch.device("cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_model_to_device(model: nn.Module, device: torch.device) -> tuple[nn.Module, torch.device]:
    try:
        return model.to(device), device
    except Exception as exc:
        message = str(exc).lower()
        if device.type == "cuda" and "out of memory" in message:
            print("CUDA OOM while moving model to device; retrying on CPU.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpu_device = torch.device("cpu")
            return model.to(cpu_device), cpu_device
        raise


def semantic_gap_stat(
    logits: torch.Tensor,
    future_risk: torch.Tensor,
    min_state_mass: float = 0.1,
    eps: float = 1e-6,
) -> float | None:
    if logits.shape[0] == 0:
        return None
    if future_risk.ndim > 1:
        future_risk = future_risk.squeeze(-1)
    valid_mask = torch.isfinite(future_risk)
    if int(valid_mask.sum().item()) < 2:
        return None
    probs = F.softmax(logits[valid_mask], dim=-1)
    target = future_risk[valid_mask].to(device=logits.device, dtype=probs.dtype)
    state_mass = probs.sum(dim=0)
    if state_mass[0] < min_state_mass or state_mass[2] < min_state_mass:
        return None
    weighted_mean = (probs * target.unsqueeze(-1)).sum(dim=0) / (state_mass + eps)
    return float((weighted_mean[0] - weighted_mean[2]).detach().cpu().item())


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    continuity_weight: float,
    balance_weight: float,
    gate_strength: float,
    balance_target: torch.Tensor | None,
    ce_class_weight: torch.Tensor | None,
    semantic_order_weight: float = 0.0,
    semantic_margin: float = 0.001,
    semantic_min_state_mass: float = 0.1,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    balance_target_device = balance_target.to(device) if balance_target is not None else None
    ce_class_weight_device = ce_class_weight.to(device) if ce_class_weight is not None else None

    with torch.no_grad():
        for features, labels, volatility, semantic_target in loader:
            features = features.to(device)
            labels = labels.to(device)
            volatility = volatility.to(device)
            semantic_target = semantic_target.to(device)
            logits = model(features)
            ce = F.cross_entropy(logits, labels, weight=ce_class_weight_device)
            cont = continuity_loss(logits, volatility, gate_strength)
            balance = logits.new_tensor(0.0)
            if balance_target_device is not None and balance_weight > 0.0:
                balance = class_balance_penalty(logits, balance_target_device)
            sem = logits.new_tensor(0.0)
            if semantic_order_weight > 0.0:
                sem = semantic_ordering_loss(
                    logits,
                    semantic_target,
                    margin=semantic_margin,
                    min_state_mass=semantic_min_state_mass,
                )
            loss = ce + continuity_weight * cont + balance_weight * balance + semantic_order_weight * sem
            losses.append(float(loss.item()))
            y_true.append(labels.cpu().numpy())
            y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())

    y_true_array = np.concatenate(y_true)
    y_pred_array = np.concatenate(y_pred)
    return {
        "loss": float(np.mean(losses)),
        "macro_f1": float(f1_score(y_true_array, y_pred_array, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
    }


def predict_with_model(
    model: nn.Module,
    frame: pd.DataFrame,
    features: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    labels = frame["proxy_label_id"].to_numpy(dtype=np.int64)
    volatility = frame["rolling_vol_24_last"].to_numpy(dtype=np.float32)
    dataset = WindowDataset(features, labels, volatility)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    pred_batches: list[np.ndarray] = []
    proba_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch_features, _, _, _ in loader:
            batch_features = batch_features.to(device)
            logits = model(batch_features)
            proba = F.softmax(logits, dim=-1).cpu().numpy()
            proba_batches.append(proba)
            pred_batches.append(np.argmax(proba, axis=1))

    proba_array = np.concatenate(proba_batches)
    pred_array = np.concatenate(pred_batches)
    out = frame[["open_time", "symbol", "proxy_label", "proxy_label_id"]].copy()
    out["pred_label_id"] = pred_array
    out["pred_label"] = out["pred_label_id"].map(LABEL_ID_TO_NAME)
    out["proba_bear"] = proba_array[:, 0]
    out["proba_neutral"] = proba_array[:, 1]
    out["proba_bull"] = proba_array[:, 2]
    return out


def segment_lengths(values: pd.Series) -> list[int]:
    lengths: list[int] = []
    current_length = 0
    previous = None
    for value in values.tolist():
        if previous is None or value == previous:
            current_length += 1
        else:
            lengths.append(current_length)
            current_length = 1
        previous = value
    if current_length > 0:
        lengths.append(current_length)
    return lengths


def summarize_stability(pred_df: pd.DataFrame) -> dict[str, float]:
    ordered = pred_df.sort_values("open_time").reset_index(drop=True)
    preds = ordered["pred_label_id"].astype(int)
    transitions = int((preds != preds.shift(1)).sum() - 1)
    transitions = max(transitions, 0)
    segments = segment_lengths(preds)
    day_count = max((ordered["open_time"].max() - ordered["open_time"].min()).total_seconds() / 86400.0, 1.0)
    return {
        "transitions": float(transitions),
        "avg_state_duration_bars": float(sum(segments) / len(segments)),
        "daily_switch_rate": float(transitions / day_count),
    }


def future_window_low(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


def future_window_high(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]


def assign_halfyear_period(series: pd.Series) -> pd.Series:
    return series.apply(lambda ts: f"{ts.year}-{'H1' if ts.month <= 6 else 'H2'}")


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


def load_semantic_label_frame(
    label_path: str | Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    keep_cols = ["open_time", "symbol", "proxy_label", "close", "low", "high", "future_return_24", "future_vol_24"]
    df = pd.read_parquet(label_path, columns=keep_cols).copy()
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

    keep_mask = (df["open_time"] >= start_time) & (df["open_time"] <= end_time)
    out = df.loc[
        keep_mask,
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
        ],
    ].copy()
    out["period"] = assign_halfyear_period(out["open_time"])
    return out


def load_semantic_training_targets(
    label_path: str | Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    df = pd.read_parquet(label_path, columns=["open_time", "symbol", "future_vol_24"]).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    mask = (df["open_time"] >= start_time) & (df["open_time"] <= end_time)
    return df.loc[mask, ["open_time", "symbol", "future_vol_24"]].copy()


def attach_semantic_targets(frame: pd.DataFrame, semantic_target_frame: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(semantic_target_frame, on=["open_time", "symbol"], how="left")


def evaluate_semantic_audit(
    pred_df: pd.DataFrame,
    semantic_label_frame: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    merged = pred_df[["open_time", "symbol", "pred_label", "pred_label_id"]].merge(
        semantic_label_frame,
        on=["open_time", "symbol"],
        how="inner",
    )
    merged["state_label"] = merged["pred_label"].astype(str)
    checks = 0
    order_matches = 0
    significant = 0

    for (_, period_group) in merged.groupby(["period"], dropna=False, sort=True):
        for metric in RISK_TEST_METRICS:
            state_values = {
                label: period_group.loc[period_group["state_label"] == label, metric].dropna().to_numpy()
                for label in EXPECTED_ORDER
            }
            if any(len(values) < 2 for values in state_values.values()):
                continue

            checks += 1
            means = {label: float(state_values[label].mean()) for label in EXPECTED_ORDER}
            order_match = means["bear"] > means["neutral"] > means["bull"]
            if order_match:
                order_matches += 1

            kw = kruskal(*(state_values[label] for label in EXPECTED_ORDER), nan_policy="omit")
            raw_p_values: list[float] = []
            for left_label, right_label in EXPECTED_PAIRS:
                test = mannwhitneyu(state_values[left_label], state_values[right_label], alternative="greater")
                raw_p_values.append(float(test.pvalue))
            adjusted = holm_adjust(raw_p_values)
            all_pairwise_significant = all(p_value < alpha for p_value in adjusted)
            if order_match and float(kw.pvalue) < alpha and all_pairwise_significant:
                significant += 1

    stability = summarize_stability(pred_df)
    risk_order_rate = float(order_matches / checks) if checks > 0 else 0.0
    significant_rate = float(significant / checks) if checks > 0 else 0.0
    return {
        "semantic_checks": float(checks),
        "risk_order_rate": risk_order_rate,
        "significant_risk_layering_rate": significant_rate,
        "transitions": stability["transitions"],
        "avg_state_duration_bars": stability["avg_state_duration_bars"],
        "daily_switch_rate": stability["daily_switch_rate"],
    }


def export_predictions(
    model: nn.Module,
    frame: pd.DataFrame,
    features: np.ndarray,
    device: torch.device,
    output_path: Path,
) -> None:
    out = predict_with_model(model, frame, features, device)
    out.to_parquet(output_path, index=False)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    if args.checkpoint_selection in {"semantic_audit", "semantic_signal_aware"} and not args.label_path:
        raise ValueError("--label_path is required when semantic-aware checkpoint selection is used.")
    if args.semantic_order_weight > 0.0 and not args.label_path:
        raise ValueError("--label_path is required when --semantic_order_weight > 0.")
    if args.semantic_order_weight > 0.0 and args.semantic_horizon != 24:
        raise ValueError("SAWS currently expects --semantic_horizon 24 to align with future_vol_24.")

    feature_cache_path = Path(output_dir) / "_feature_cache.dat"
    frame, feature_array = load_window_dataset(
        args.input_path,
        batch_size=args.load_batch_size,
        feature_store_path=feature_cache_path,
    )
    (train_df, train_features), (valid_df, valid_features), (test_df, test_features) = split_by_time(
        frame,
        feature_array,
        args.train_end,
        args.valid_end,
        args.test_end,
    )
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Train, valid, and test splits must all be non-empty.")

    semantic_label_frame: pd.DataFrame | None = None
    semantic_target_frame: pd.DataFrame | None = None
    if args.checkpoint_selection in {"semantic_audit", "semantic_signal_aware"}:
        semantic_label_frame = load_semantic_label_frame(
            args.label_path,
            start_time=pd.Timestamp(valid_df["open_time"].min()),
            end_time=pd.Timestamp(valid_df["open_time"].max()),
            horizon=args.semantic_horizon,
        )
    if args.semantic_order_weight > 0.0:
        semantic_target_frame = load_semantic_training_targets(
            args.label_path,
            start_time=pd.Timestamp(train_df["open_time"].min()),
            end_time=pd.Timestamp(test_df["open_time"].max()),
        )
        train_df = attach_semantic_targets(train_df, semantic_target_frame)
        valid_df = attach_semantic_targets(valid_df, semantic_target_frame)
        test_df = attach_semantic_targets(test_df, semantic_target_frame)

    train_loader = make_loader(train_df, train_features, batch_size=args.batch_size, shuffle=False)
    valid_loader = make_loader(valid_df, valid_features, batch_size=args.batch_size, shuffle=False)

    sample = train_features[0]
    num_classes = 3
    train_labels = train_df["proxy_label_id"].to_numpy(dtype=np.int64)
    balance_target = build_balance_target(args.balance_mode, train_labels, num_classes=num_classes)
    ce_class_weight = build_ce_class_weight(args.ce_class_weight_mode, train_labels, num_classes=num_classes)
    config = ModelConfig(
        input_dim=int(sample.shape[-1]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type,
        transformer_heads=args.transformer_heads,
        transformer_ff_dim=args.transformer_ff_dim,
    )
    device = select_device()
    print(f"Using device: {device}")
    model, device = move_model_to_device(build_model(config), device)
    print(f"Model device: {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_selection_key: tuple[float, ...] | None = None
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_semantic_gaps: list[float] = []
        for features, labels, volatility, semantic_target in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            volatility = volatility.to(device)
            semantic_target = semantic_target.to(device)

            optimizer.zero_grad()
            logits = model(features)
            ce = F.cross_entropy(
                logits,
                labels,
                weight=ce_class_weight.to(device) if ce_class_weight is not None else None,
            )
            cont = continuity_loss(logits, volatility, args.volatility_gate_strength)
            balance = logits.new_tensor(0.0)
            if balance_target is not None and args.balance_weight > 0.0:
                balance = class_balance_penalty(logits, balance_target)
            sem = logits.new_tensor(0.0)
            if args.semantic_order_weight > 0.0:
                sem = semantic_ordering_loss(
                    logits,
                    semantic_target,
                    margin=args.semantic_margin,
                    min_state_mass=args.semantic_min_state_mass,
                )
                gap = semantic_gap_stat(
                    logits,
                    semantic_target,
                    min_state_mass=args.semantic_min_state_mass,
                )
                if gap is not None:
                    epoch_semantic_gaps.append(gap)
            loss = ce + args.continuity_weight * cont + args.balance_weight * balance + args.semantic_order_weight * sem
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_metrics = evaluate_model(
            model,
            train_loader,
            device,
            args.continuity_weight,
            args.balance_weight,
            args.volatility_gate_strength,
            balance_target,
            ce_class_weight,
            args.semantic_order_weight,
            args.semantic_margin,
            args.semantic_min_state_mass,
        )
        valid_metrics = evaluate_model(
            model,
            valid_loader,
            device,
            args.continuity_weight,
            args.balance_weight,
            args.volatility_gate_strength,
            balance_target,
            ce_class_weight,
            args.semantic_order_weight,
            args.semantic_margin,
            args.semantic_min_state_mass,
        )
        semantic_metrics: dict[str, float] = {}
        if semantic_label_frame is not None:
            valid_predictions = predict_with_model(model, valid_df, valid_features, device)
            semantic_metrics = evaluate_semantic_audit(
                valid_predictions,
                semantic_label_frame,
                alpha=args.semantic_alpha,
            )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "train_macro_f1": train_metrics["macro_f1"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
        }
        if args.semantic_order_weight > 0.0:
            row["train_semantic_gap_mean"] = float(np.mean(epoch_semantic_gaps)) if epoch_semantic_gaps else float("nan")
        row.update({f"valid_{key}": value for key, value in semantic_metrics.items()})
        history.append(row)
        print(pd.DataFrame([row]).to_string(index=False))

        if args.checkpoint_selection == "semantic_audit":
            selection_key = (
                float(semantic_metrics.get("significant_risk_layering_rate", 0.0)),
                float(semantic_metrics.get("risk_order_rate", 0.0)),
                float(valid_metrics["macro_f1"]),
                -float(semantic_metrics.get("daily_switch_rate", float("inf"))),
            )
        elif args.checkpoint_selection == "semantic_signal_aware":
            semantic_signal = float(semantic_metrics.get("significant_risk_layering_rate", 0.0))
            if semantic_signal > 0.0:
                selection_key = (
                    1.0,
                    semantic_signal,
                    float(semantic_metrics.get("risk_order_rate", 0.0)),
                    float(valid_metrics["macro_f1"]),
                    -float(semantic_metrics.get("daily_switch_rate", float("inf"))),
                )
            else:
                selection_key = (
                    0.0,
                    float(valid_metrics["macro_f1"]),
                    float(semantic_metrics.get("risk_order_rate", 0.0)),
                    -float(semantic_metrics.get("daily_switch_rate", float("inf"))),
                )
        else:
            selection_key = (float(valid_metrics["macro_f1"]),)

        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_epoch = epoch

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(output_dir) / "history.csv", index=False)

    test_loader = make_loader(test_df, test_features, batch_size=args.batch_size, shuffle=False)
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        args.continuity_weight,
        args.balance_weight,
        args.volatility_gate_strength,
        balance_target,
        ce_class_weight,
        args.semantic_order_weight,
        args.semantic_margin,
        args.semantic_min_state_mass,
    )
    metrics_df = pd.DataFrame(
        [
            {
                "method": f"main_{args.model_type}",
                "model_type": args.model_type,
                "split": "test",
                "macro_f1": test_metrics["macro_f1"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "loss": test_metrics["loss"],
                "rows": int(len(test_df)),
            }
        ]
    )
    metrics_df.to_csv(Path(output_dir) / "metrics.csv", index=False)
    export_predictions(model, valid_df, valid_features, device, Path(output_dir) / "valid_predictions.parquet")
    export_predictions(model, test_df, test_features, device, Path(output_dir) / "test_predictions.parquet")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
        },
        Path(output_dir) / "model.pt",
    )
    with open(Path(output_dir) / "train_args.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                **vars(args),
                "train_class_counts": np.bincount(train_labels, minlength=num_classes).astype(int).tolist(),
                "train_class_prior": compute_class_prior(train_labels, num_classes).tolist(),
                "ce_class_weight_values": ce_class_weight.tolist() if ce_class_weight is not None else None,
                "balance_target_values": balance_target.tolist() if balance_target is not None else None,
                "best_epoch": best_epoch,
                "checkpoint_selection": args.checkpoint_selection,
                "best_selection_key": list(best_selection_key) if best_selection_key is not None else None,
            },
            file,
            indent=2,
        )
    with open(Path(output_dir) / "selection_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint_selection": args.checkpoint_selection,
                "best_epoch": best_epoch,
                "best_selection_key": list(best_selection_key) if best_selection_key is not None else None,
                "history_columns": list(history_df.columns),
            },
            file,
            indent=2,
        )

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
