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
WINDOW_PATH = Path("data/processed/windows_improved/1h/default/BTCUSDT_win48.parquet")
BASELINE_RUN_DIR = Path("experiments/main/gru_btc")
SUMMARY_PATH = Path("experiments/summary/saws_sweep.csv")
SUMMARY_AGG_PATH = Path("experiments/summary/saws_sweep_summary.csv")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTCUSDT SAWS sweep on top of the main GRU pipeline.")
    parser.add_argument("--lambdas", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--load_batch_size", type=int, default=1024)
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


def lambda_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def train_variant(
    semantic_weight: float,
    seed: int,
    epochs: int,
    batch_size: int,
    load_batch_size: int,
    semantic_margin: float,
    semantic_min_state_mass: float,
    skip_existing: bool,
) -> tuple[Path, str, str]:
    if semantic_weight == 0.0 and seed == 42 and BASELINE_RUN_DIR.exists():
        ensure_prediction_evaluations(BASELINE_RUN_DIR, "main_gru")
        return BASELINE_RUN_DIR, "gru_standard", "main_gru"

    variant_name = "gru_standard" if semantic_weight == 0.0 else f"gru_saws_l{lambda_tag(semantic_weight)}"
    source_name = variant_name
    run_dir = ensure_dir(Path("experiments/saws_sweep") / variant_name / f"seed_{seed}")
    if skip_existing and (run_dir / "test_predictions.parquet").exists():
        ensure_prediction_evaluations(run_dir, source_name)
        return run_dir, variant_name, source_name

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
        str(batch_size),
        "--load_batch_size",
        str(load_batch_size),
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
        "semantic_audit",
        "--label_path",
        str(LABEL_PATH),
        "--semantic_horizon",
        "24",
        "--semantic_alpha",
        "0.05",
        "--semantic_order_weight",
        str(semantic_weight),
        "--semantic_margin",
        str(semantic_margin),
        "--semantic_min_state_mass",
        str(semantic_min_state_mass),
        "--seed",
        str(seed),
    ]
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        cpu_env = os.environ.copy()
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""
        cpu_env["FORCE_CPU"] = "1"
        cpu_cmd = cmd.copy()
        if "--load_batch_size" in cpu_cmd:
            idx = cpu_cmd.index("--load_batch_size")
            cpu_cmd[idx + 1] = str(min(load_batch_size, 256))
        print("[retry-cpu]", variant_name, "seed", seed)
        run(cpu_cmd, env=cpu_env)

    ensure_prediction_evaluations(run_dir, source_name)
    return run_dir, variant_name, source_name


def build_row(run_dir: Path, variant_name: str, source_name: str, semantic_weight: float, seed: int) -> dict[str, object]:
    cls = pd.read_csv(run_dir / "classification_metrics.csv").iloc[0]
    stab = pd.read_csv(run_dir / "stability.csv").iloc[0]
    risk_sem = extract_risk_semantics(run_dir / "risk_significance" / "risk_significance_summary.csv", source_name)
    return_sem = extract_return_semantics(run_dir / "risk_semantics" / "ordering_summary.csv", source_name)
    row = {
        "variant": variant_name,
        "semantic_order_weight": semantic_weight,
        "seed": seed,
        "macro_f1": float(cls["macro_f1"]),
        "balanced_accuracy": float(cls["balanced_accuracy"]),
        "bull_f1": float(cls.get("bull_f1", float("nan"))),
        "bear_f1": float(cls.get("bear_f1", float("nan"))),
        "switch_day": float(stab["daily_switch_rate"]),
        "avg_duration": float(stab["avg_state_duration_bars"]),
        "run_dir": str(run_dir),
    }
    row.update(risk_sem)
    row.update(return_sem)
    for key in [
        "risk_order_checks",
        "risk_order_matches",
        "risk_order_rate",
        "sig_risk_checks",
        "sig_risk_rate",
        "bootstrap_checks",
        "bootstrap_rate",
        "return_order_checks",
        "return_order_matches",
        "return_order_rate",
    ]:
        row.setdefault(key, None)
    return row


def summarize(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    candidate_numeric_cols = [
        "macro_f1",
        "balanced_accuracy",
        "bull_f1",
        "bear_f1",
        "switch_day",
        "avg_duration",
        "risk_order_rate",
        "sig_risk_rate",
        "bootstrap_rate",
        "return_order_rate",
    ]
    numeric_cols = [column for column in candidate_numeric_cols if column in df.columns]
    agg = (
        df.groupby(["variant", "semantic_order_weight"], dropna=False)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "_".join(part for part in column if part).rstrip("_")
        for column in agg.columns.to_flat_index()
    ]
    return agg


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    variants = [0.0, *args.lambdas]
    for semantic_weight in variants:
        for seed in args.seeds:
            run_dir, variant_name, source_name = train_variant(
                semantic_weight,
                seed,
                args.epochs,
                args.batch_size,
                args.load_batch_size,
                args.semantic_margin,
                args.semantic_min_state_mass,
                args.skip_existing,
            )
            rows.append(build_row(run_dir, variant_name, source_name, semantic_weight, seed))

    summary_df = pd.DataFrame(rows).sort_values(["semantic_order_weight", "seed"]).reset_index(drop=True)
    ensure_dir(SUMMARY_PATH.parent)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    summarize(summary_df.to_dict("records")).to_csv(SUMMARY_AGG_PATH, index=False)
    print(summary_df.to_string(index=False))
    print()
    print(pd.read_csv(SUMMARY_AGG_PATH).to_string(index=False))


if __name__ == "__main__":
    main()
