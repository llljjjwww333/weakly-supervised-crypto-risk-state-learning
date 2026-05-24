from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.evaluation.build_results_manifest import extract_risk_semantics
from src.utils.io import ensure_dir


LABEL_PATH = Path("data/labels_improved/default/BTCUSDT_labels.parquet")
WINDOW_PATH = Path("data/processed/windows_improved/1h/default/BTCUSDT_win48.parquet")
SUMMARY_PATH = Path("experiments/summary/semantic_signal_aware_selector.csv")


RUN_SPECS = [
    {
        "window": "main",
        "selector": "valid_macro_f1",
        "output_dir": Path("experiments/selector_signal_aware/main_macro"),
        "train_end": "2023-12-31",
        "valid_end": "2024-12-31",
    },
    {
        "window": "main",
        "selector": "semantic_audit",
        "output_dir": Path("experiments/selector_signal_aware/main_semantic"),
        "train_end": "2023-12-31",
        "valid_end": "2024-12-31",
    },
    {
        "window": "main",
        "selector": "semantic_signal_aware",
        "output_dir": Path("experiments/selector_signal_aware/main_signal_aware"),
        "train_end": "2023-12-31",
        "valid_end": "2024-12-31",
    },
    {
        "window": "rolling_2024",
        "selector": "valid_macro_f1",
        "output_dir": Path("experiments/selector_signal_aware/rolling2024_macro"),
        "train_end": "2022-12-31",
        "valid_end": "2023-12-31",
        "test_end": "2024-12-31",
    },
    {
        "window": "rolling_2024",
        "selector": "semantic_audit",
        "output_dir": Path("experiments/selector_signal_aware/rolling2024_semantic"),
        "train_end": "2022-12-31",
        "valid_end": "2023-12-31",
        "test_end": "2024-12-31",
    },
    {
        "window": "rolling_2024",
        "selector": "semantic_signal_aware",
        "output_dir": Path("experiments/selector_signal_aware/rolling2024_signal_aware"),
        "train_end": "2022-12-31",
        "valid_end": "2023-12-31",
        "test_end": "2024-12-31",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare macro-F1, semantic, and signal-aware checkpoint selectors.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--load_batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def train_one(spec: dict[str, object], args: argparse.Namespace) -> None:
    output_dir: Path = ensure_dir(spec["output_dir"])
    if args.skip_existing and (output_dir / "metrics.csv").exists() and (output_dir / "selection_summary.json").exists():
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
        str(spec["train_end"]),
        "--valid_end",
        str(spec["valid_end"]),
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
        str(spec["selector"]),
        "--label_path",
        str(LABEL_PATH),
        "--semantic_horizon",
        "24",
        "--semantic_alpha",
        "0.05",
        "--seed",
        str(args.seed),
    ]
    if "test_end" in spec:
        cmd.extend(["--test_end", str(spec["test_end"])])
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        cpu_env = os.environ.copy()
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""
        cpu_env["FORCE_CPU"] = "1"
        cpu_cmd = cmd.copy()
        idx = cpu_cmd.index("--load_batch_size")
        cpu_cmd[idx + 1] = str(min(args.load_batch_size, 256))
        print("[retry-cpu]", spec["window"], spec["selector"])
        run(cpu_cmd, env=cpu_env)


def build_row(spec: dict[str, object]) -> dict[str, object]:
    output_dir: Path = spec["output_dir"]
    metrics = pd.read_csv(output_dir / "metrics.csv").iloc[0]
    selection_summary = json.loads((output_dir / "selection_summary.json").read_text(encoding="utf-8"))
    history = pd.read_csv(output_dir / "history.csv")
    best_epoch = int(selection_summary["best_epoch"])
    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]
    pred_name = "main_gru"
    if spec["selector"] != "valid_macro_f1":
        pred_name = "main_gru"
    risk_sem = extract_risk_semantics(output_dir / "risk_significance" / "risk_significance_summary.csv", pred_name)
    return {
        "window": spec["window"],
        "selector": spec["selector"],
        "best_epoch": best_epoch,
        "test_macro_f1": float(metrics["macro_f1"]),
        "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
        "valid_macro_f1_at_best": float(best_row.get("valid_macro_f1", float("nan"))),
        "valid_sig_risk_rate_at_best": float(best_row.get("valid_significant_risk_layering_rate", 0.0)),
        "valid_risk_order_rate_at_best": float(best_row.get("valid_risk_order_rate", 0.0)),
        "valid_daily_switch_rate_at_best": float(best_row.get("valid_daily_switch_rate", float("nan"))),
        "test_sig_risk_rate": risk_sem.get("sig_risk_rate"),
        "test_risk_order_rate": risk_sem.get("risk_order_rate"),
        "selection_key": json.dumps(selection_summary.get("best_selection_key", [])),
        "output_dir": str(output_dir),
    }


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
