from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.evaluation.build_results_manifest import extract_return_semantics, extract_risk_semantics
from src.utils.io import ensure_dir


LABEL_PATH = Path("data/labels_improved/default/BTCUSDT_labels.parquet")
WINDOW_PATH = Path("data/processed/windows_improved/1h/stride48/BTCUSDT_win48_s48.parquet")
SUMMARY_PATH = Path("experiments/summary/saws_stride48_diagnostic.csv")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


RUN_SPECS = [
    {
        "variant": "stride48_standard",
        "semantic_weight": 0.0,
        "output_dir": Path("experiments/saws_stride48/standard"),
    },
    {
        "variant": "stride48_saws_l0p20",
        "semantic_weight": 0.20,
        "output_dir": Path("experiments/saws_stride48/saws_l0p20"),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stride-48 SAWS diagnostic on BTCUSDT.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--load_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic_margin", type=float, default=0.001)
    parser.add_argument("--semantic_min_state_mass", type=float, default=0.1)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd))
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    python_path = merged_env.get("PYTHONPATH", "")
    merged_env["PYTHONPATH"] = str(PROJECT_ROOT) if not python_path else os.pathsep.join([str(PROJECT_ROOT), python_path])
    subprocess.run(cmd, check=True, env=merged_env, cwd=PROJECT_ROOT)


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


def train_one(spec: dict[str, object], args: argparse.Namespace) -> None:
    output_dir: Path = ensure_dir(spec["output_dir"])
    if args.skip_existing and (output_dir / "metrics.csv").exists() and (output_dir / "history.csv").exists():
        ensure_prediction_evaluations(output_dir, str(spec["variant"]))
        return

    cmd = [
        sys.executable,
        "-m",
        "src.models.main.train_main",
        "--input_path",
        str(WINDOW_PATH),
        "--output_dir",
        str(output_dir),
        "--train_end",
        "2023-12-31",
        "--valid_end",
        "2024-12-31",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--load_batch_size",
        str(args.load_batch_size),
        "--model_type",
        "gru",
        "--hidden_dim",
        "64",
        "--num_layers",
        "1",
        "--dropout",
        "0.1",
        "--lr",
        "0.001",
        "--weight_decay",
        "0.0001",
        "--continuity_weight",
        "0.0",
        "--balance_weight",
        "0.05",
        "--balance_mode",
        "uniform",
        "--ce_class_weight_mode",
        "none",
        "--volatility_gate_strength",
        "3.0",
        "--checkpoint_selection",
        "semantic_audit",
        "--label_path",
        str(LABEL_PATH),
        "--semantic_horizon",
        "24",
        "--semantic_alpha",
        "0.05",
        "--semantic_order_weight",
        str(spec["semantic_weight"]),
        "--semantic_margin",
        str(args.semantic_margin),
        "--semantic_min_state_mass",
        str(args.semantic_min_state_mass),
        "--seed",
        str(args.seed),
    ]
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        cpu_env = os.environ.copy()
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""
        cpu_env["FORCE_CPU"] = "1"
        cpu_cmd = cmd.copy()
        idx = cpu_cmd.index("--load_batch_size")
        cpu_cmd[idx + 1] = str(min(args.load_batch_size, 128))
        print("[retry-cpu]", spec["variant"])
        run(cpu_cmd, env=cpu_env)
    ensure_prediction_evaluations(output_dir, str(spec["variant"]))


def build_row(spec: dict[str, object]) -> dict[str, object]:
    output_dir: Path = spec["output_dir"]
    metrics = pd.read_csv(output_dir / "metrics.csv").iloc[0]
    stability = pd.read_csv(output_dir / "stability.csv").iloc[0]
    history = pd.read_csv(output_dir / "history.csv")
    risk_sem = extract_risk_semantics(output_dir / "risk_significance" / "risk_significance_summary.csv", str(spec["variant"]))
    return_sem = extract_return_semantics(output_dir / "risk_semantics" / "ordering_summary.csv", str(spec["variant"]))
    row = {
        "variant": spec["variant"],
        "semantic_order_weight": spec["semantic_weight"],
        "test_macro_f1": float(metrics["macro_f1"]),
        "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
        "test_switch_day": float(stability["daily_switch_rate"]),
        "test_avg_duration": float(stability["avg_state_duration_bars"]),
        "train_semantic_gap_mean_last_epoch": float(history["train_semantic_gap_mean"].dropna().iloc[-1]) if "train_semantic_gap_mean" in history.columns and history["train_semantic_gap_mean"].notna().any() else None,
        "output_dir": str(output_dir),
    }
    row.update(risk_sem)
    row.update(return_sem)
    return row


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for spec in RUN_SPECS:
        train_one(spec, args)
        rows.append(build_row(spec))
    df = pd.DataFrame(rows)
    ensure_dir(SUMMARY_PATH.parent)
    df.to_csv(SUMMARY_PATH, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
