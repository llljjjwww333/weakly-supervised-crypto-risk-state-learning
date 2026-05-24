from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path

import pandas as pd

from src.evaluation.build_results_manifest import extract_return_semantics, extract_risk_semantics
from src.utils.io import ensure_dir


DEFAULT_SEEDS = [7, 42, 123, 2024, 3407]
LABEL_PATH = Path("data/labels_improved/default/BTCUSDT_labels.parquet")
WINDOW_PATH = Path("data/processed/windows_improved/1h/default/BTCUSDT_win48.parquet")
SUMMARY_PATH = Path("experiments/summary/seed_robustness_5seed.csv")
SUMMARY_AGG_PATH = Path("experiments/summary/seed_robustness_5seed_summary.csv")


MODEL_SPECS = {
    "gru": {
        "display_name": "GRU",
        "source_name": "main_gru",
        "base_dir": Path("experiments/robustness_5seed/gru"),
        "train_args": [
            "--model_type",
            "gru",
            "--hidden_dim",
            "64",
            "--num_layers",
            "1",
            "--load_batch_size",
            "1024",
        ],
    },
    "tcn96": {
        "display_name": "TCN-96x4",
        "source_name": "seed_tcn_96x4",
        "base_dir": Path("experiments/robustness_5seed/tcn96"),
        "train_args": [
            "--model_type",
            "tcn",
            "--hidden_dim",
            "96",
            "--num_layers",
            "4",
            "--load_batch_size",
            "1024",
        ],
    },
    "transformer": {
        "display_name": "Transformer-2L",
        "source_name": "seed_transformer_2l",
        "base_dir": Path("experiments/robustness_5seed/transformer"),
        "train_args": [
            "--model_type",
            "transformer",
            "--hidden_dim",
            "64",
            "--num_layers",
            "2",
            "--transformer_heads",
            "4",
            "--transformer_ff_dim",
            "256",
            "--load_batch_size",
            "128",
        ],
    },
}

LEGACY_RUN_DIRS: dict[str, dict[int, Path]] = {
    "gru": {
        42: Path("experiments/main/gru_btc"),
        7: Path("experiments/robustness/gru_btc_seed7"),
        123: Path("experiments/robustness/gru_btc_seed123"),
    },
    "tcn96": {
        42: Path("experiments/improved/main/tcn_btc_96x4"),
    },
    "transformer": {
        42: Path("experiments/improved/main/transformer_btc"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5-seed BTC robustness for GRU, TCN-96x4, and Transformer-2L.")
    parser.add_argument("--models", nargs="*", default=["gru", "tcn96", "transformer"])
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def ensure_prediction_evaluations(run_dir: Path, source_name: str) -> None:
    pred_path = run_dir / "test_predictions.parquet"
    if not (run_dir / "classification_metrics.csv").exists():
        run(
            [
                sys.executable,
                "-m",
                "src.evaluation.evaluate_classification",
                "--input_path",
                str(pred_path),
                "--output_dir",
                str(run_dir),
            ]
        )
    if not (run_dir / "stability.csv").exists():
        run(
            [
                sys.executable,
                "-m",
                "src.evaluation.evaluate_stability",
                "--input_path",
                str(pred_path),
                "--output_path",
                str(run_dir / "stability.csv"),
            ]
        )

    semantics_dir = run_dir / "risk_semantics"
    significance_dir = run_dir / "risk_significance"
    if not (semantics_dir / "ordering_summary.csv").exists():
        run(
            [
                sys.executable,
                "-m",
                "src.evaluation.evaluate_risk_state_semantics",
                "--label_path",
                str(LABEL_PATH),
                "--prediction_paths",
                str(pred_path),
                "--prediction_names",
                source_name,
                "--output_dir",
                str(semantics_dir),
            ]
        )
    if not (significance_dir / "risk_significance_summary.csv").exists():
        run(
            [
                sys.executable,
                "-m",
                "src.evaluation.evaluate_risk_state_significance",
                "--label_path",
                str(LABEL_PATH),
                "--prediction_paths",
                str(pred_path),
                "--prediction_names",
                source_name,
                "--output_dir",
                str(significance_dir),
            ]
        )


def train_seed(model_key: str, seed: int, epochs: int, skip_existing: bool) -> Path:
    spec = MODEL_SPECS[model_key]
    legacy_dir = LEGACY_RUN_DIRS.get(model_key, {}).get(seed)
    if legacy_dir is not None and legacy_dir.exists():
        ensure_prediction_evaluations(legacy_dir, spec["source_name"])
        return legacy_dir
    run_dir = ensure_dir(spec["base_dir"] / f"seed_{seed}")
    if skip_existing and (run_dir / "test_predictions.parquet").exists():
        ensure_prediction_evaluations(run_dir, spec["source_name"])
        return run_dir

    cmd = [
        sys.executable,
        "-m",
        "src.models.main.train_main",
        "--input_path",
        str(WINDOW_PATH),
        "--output_dir",
        str(run_dir),
        "--train_end",
        "2023-12-31",
        "--valid_end",
        "2024-12-31",
        "--epochs",
        str(epochs),
        "--batch_size",
        "256",
        "--dropout",
        "0.1",
        "--lr",
        "0.001",
        "--weight_decay",
        "0.0001",
        "--continuity_weight",
        "0.15",
        "--balance_weight",
        "0.05",
        "--balance_mode",
        "uniform",
        "--ce_class_weight_mode",
        "none",
        "--volatility_gate_strength",
        "3.0",
        "--checkpoint_selection",
        "valid_macro_f1",
        "--label_path",
        str(LABEL_PATH),
        "--semantic_horizon",
        "24",
        "--semantic_alpha",
        "0.05",
        "--seed",
        str(seed),
    ]
    cmd.extend(spec["train_args"])
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        cpu_env = os.environ.copy()
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""
        print("[retry-cpu]", spec["display_name"], "seed", seed)
        run(cmd, env=cpu_env)
    ensure_prediction_evaluations(run_dir, spec["source_name"])
    return run_dir


def build_row(model_key: str, seed: int, run_dir: Path) -> dict[str, object]:
    spec = MODEL_SPECS[model_key]
    cls = pd.read_csv(run_dir / "classification_metrics.csv").iloc[0]
    stab = pd.read_csv(run_dir / "stability.csv").iloc[0]
    return_sem = extract_return_semantics(run_dir / "risk_semantics" / "ordering_summary.csv", spec["source_name"])
    risk_sem = extract_risk_semantics(
        run_dir / "risk_significance" / "risk_significance_summary.csv", spec["source_name"]
    )
    return {
        "asset": "BTC",
        "model_key": model_key,
        "model": spec["display_name"],
        "seed": seed,
        "macro_f1": float(cls["macro_f1"]),
        "balanced_accuracy": float(cls["balanced_accuracy"]),
        "bull_f1": float(cls["bull_f1"]),
        "bear_f1": float(cls["bear_f1"]),
        "switch_day": float(stab["daily_switch_rate"]),
        "avg_duration": float(stab["avg_state_duration_bars"]),
        "risk_order_checks": risk_sem.get("risk_order_checks"),
        "risk_order_matches": risk_sem.get("risk_order_matches"),
        "risk_order_rate": risk_sem.get("risk_order_rate"),
        "return_order_checks": return_sem.get("return_order_checks"),
        "return_order_matches": return_sem.get("return_order_matches"),
        "return_order_rate": return_sem.get("return_order_rate"),
        "sig_risk_checks": risk_sem.get("sig_risk_checks"),
        "sig_risk_rate": risk_sem.get("sig_risk_rate"),
        "run_dir": str(run_dir),
    }


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["asset", "model_key", "model"], as_index=False)
        .agg(
            seeds=("seed", "count"),
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
    )
    return agg


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for model_key in args.models:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unsupported model key: {model_key}")
        for seed in args.seeds:
            run_dir = train_seed(model_key, seed, args.epochs, args.skip_existing)
            rows.append(build_row(model_key, seed, run_dir))

    detail_df = pd.DataFrame(rows).sort_values(["asset", "model", "seed"])
    summary_df = summarise(detail_df)
    detail_df.to_csv(SUMMARY_PATH, index=False)
    summary_df.to_csv(SUMMARY_AGG_PATH, index=False)
    print(f"[wrote] {SUMMARY_PATH}")
    print(f"[wrote] {SUMMARY_AGG_PATH}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
