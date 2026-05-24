from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RETURN_METRICS = {"future_return_24", "future_return_72"}
EXPECTED_RETURN_ORDER = "bull > neutral > bear"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    experiment_group: str
    asset: str
    model: str
    label_variant: str
    frequency: str = "1h"
    seed: str = "42"
    train_period: str = "2021-01-01 to 2023-12-31"
    val_period: str = "2024-01-01 to 2024-12-31"
    test_period: str = "2025-01-01 to 2026-04-20"
    classification_path: str | None = None
    classification_selector: str | None = None
    stability_path: str | None = None
    stability_selector: str | None = None
    return_ordering_path: str | None = None
    return_source: str | None = None
    risk_significance_path: str | None = None
    risk_source: str | None = None
    risk_order_checks_override: int | None = None
    risk_order_matches_override: int | None = None
    sig_risk_checks_override: int | None = None
    bootstrap_checks_override: int | None = None
    return_order_checks_override: int | None = None
    return_order_matches_override: int | None = None
    script_name: str = ""
    output_file: str = ""
    notes: str = ""


def _path(text: str | None) -> Path | None:
    return Path(text) if text else None


def _read_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if pd.isna(value):
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if pd.isna(value):
        return None
    return int(value)


def load_scalar_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _select_row(df: pd.DataFrame, selector: str | None) -> pd.Series | None:
    if df.empty:
        return None
    if "split" in df.columns:
        test_rows = df.loc[df["split"] == "test"]
        if not test_rows.empty:
            df = test_rows
    if selector and "method" in df.columns:
        selected = df.loc[df["method"] == selector]
        if not selected.empty:
            return selected.iloc[0]
    return df.iloc[0]


def extract_classification(path: Path | None, selector: str | None = None) -> dict[str, Any]:
    df = _read_csv(path)
    if df is None or df.empty:
        return {}
    row = _select_row(df, selector)
    if row is None:
        return {}
    return {
        "macro_f1": _safe_float(row.get("macro_f1")),
        "balanced_acc": _safe_float(row.get("balanced_accuracy")),
        "bull_f1": _safe_float(row.get("bull_f1")),
        "bear_f1": _safe_float(row.get("bear_f1")),
        "rows": _safe_int(row.get("rows")),
    }


def extract_stability(path: Path | None, selector: str | None = None) -> dict[str, Any]:
    df = _read_csv(path)
    if df is None or df.empty:
        return {}
    row = _select_row(df, selector)
    if row is None:
        return {}
    return {
        "switch_day": _safe_float(row.get("daily_switch_rate")),
        "avg_duration": _safe_float(row.get("avg_state_duration_bars")),
        "transitions": _safe_int(row.get("transitions")),
    }


def extract_return_semantics(path: Path | None, source: str | None) -> dict[str, Any]:
    df = _read_csv(path)
    if df is None or df.empty or not source:
        return {}
    filtered = df.loc[(df["source"] == source) & (df["metric"].isin(RETURN_METRICS))].copy()
    if filtered.empty:
        return {}
    matches = (
        filtered["descending_order"].astype(str).str.strip().eq(EXPECTED_RETURN_ORDER).sum()
    )
    checks = len(filtered)
    return {
        "return_order_checks": checks,
        "return_order_matches": int(matches),
        "return_order_rate": float(matches / checks) if checks else None,
    }


def extract_risk_semantics(path: Path | None, source: str | None) -> dict[str, Any]:
    df = _read_csv(path)
    if df is None or df.empty or not source:
        return {}
    row_df = df.loc[df["source"] == source]
    if row_df.empty:
        return {}
    row = row_df.iloc[0]
    risk_checks = _safe_int(row.get("checks"))
    risk_matches = _safe_int(row.get("order_matches"))
    sig_checks = _safe_int(row.get("significant_risk_layering"))
    boot_checks = _safe_int(row.get("bootstrap_significant_risk_layering"))
    return {
        "risk_order_checks": risk_checks,
        "risk_order_matches": risk_matches,
        "risk_order_rate": _safe_float(row.get("order_match_rate")),
        "sig_risk_checks": sig_checks,
        "sig_risk_rate": _safe_float(row.get("significant_risk_layering_rate")),
        "bootstrap_checks": boot_checks,
        "bootstrap_rate": _safe_float(row.get("bootstrap_significant_risk_layering_rate")),
    }


def build_core_specs() -> list[ExperimentSpec]:
    common_script = (
        "src/evaluation/evaluate_classification.py; "
        "src/evaluation/evaluate_stability.py; "
        "src/evaluation/evaluate_economic_meaning.py; "
        "src/evaluation/evaluate_risk_state_significance.py"
    )
    specs = [
        ExperimentSpec(
            experiment_id="btc_proxy_default",
            experiment_group="benchmark_main",
            asset="BTC",
            model="Proxy label",
            label_variant="default",
            classification_path="experiments/summary/btc_test_metrics.csv",
            classification_selector="rule_based",
            stability_path="experiments/summary/btc_test_metrics.csv",
            stability_selector="rule_based",
            return_ordering_path="experiments/summary/btc_economic_meaning_extended/ordering_summary.csv",
            return_source="proxy_label",
            risk_significance_path="experiments/summary/btc_risk_state_significance_extended/risk_significance_summary.csv",
            risk_source="proxy_label",
            script_name=common_script,
            output_file="experiments/summary/btc_test_metrics.csv; experiments/summary/btc_economic_meaning_extended/ordering_summary.csv; experiments/summary/btc_risk_state_significance_extended/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="btc_logreg_default",
            experiment_group="benchmark_main",
            asset="BTC",
            model="LogReg",
            label_variant="default",
            classification_path="experiments/summary/btc_test_metrics.csv",
            classification_selector="logreg",
            stability_path="experiments/summary/btc_test_metrics.csv",
            stability_selector="logreg",
            return_ordering_path="experiments/summary/btc_economic_meaning_extended/ordering_summary.csv",
            return_source="logistic_regression",
            risk_significance_path="experiments/summary/btc_risk_state_significance_extended/risk_significance_summary.csv",
            risk_source="logistic_regression",
            script_name=common_script,
            output_file="experiments/baselines/logreg_btc/metrics.csv; experiments/summary/btc_economic_meaning_extended/ordering_summary.csv; experiments/summary/btc_risk_state_significance_extended/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="btc_logreg_postproc_default",
            experiment_group="benchmark_main",
            asset="BTC",
            model="LogReg + post-proc.",
            label_variant="default",
            classification_path="experiments/baselines/logreg_btc_postproc_match/classification_metrics.csv",
            stability_path="experiments/baselines/logreg_btc_postproc_match/stability_metrics.csv",
            return_ordering_path="experiments/summary/btc_economic_meaning_postproc_compare/ordering_summary.csv",
            return_source="logreg_postproc_match",
            risk_significance_path="experiments/revision/btc_risk_state_significance_with_histgb/risk_significance_summary.csv",
            risk_source="logreg_postproc_match",
            script_name=common_script,
            output_file="experiments/baselines/logreg_btc_postproc_match/classification_metrics.csv; experiments/baselines/logreg_btc_postproc_match/stability.csv; experiments/summary/btc_economic_meaning_postproc_compare/ordering_summary.csv; experiments/revision/btc_risk_state_significance_with_histgb/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="btc_histgb_raw_default",
            experiment_group="benchmark_main",
            asset="BTC",
            model="HistGB",
            label_variant="default",
            classification_path="experiments/revision/histgb_btc/eval/classification_metrics.csv",
            stability_path="experiments/revision/histgb_btc/stability.csv",
            script_name="src/models/baselines/run_histgb.py; src/evaluation/evaluate_classification.py; src/evaluation/evaluate_stability.py",
            output_file="experiments/revision/histgb_btc/eval/classification_metrics.csv; experiments/revision/histgb_btc/stability.csv",
            notes="Semantic significance omitted in manuscript because raw path is too switch-heavy for audit comparison.",
        ),
        ExperimentSpec(
            experiment_id="btc_histgb_postproc_default",
            experiment_group="benchmark_main",
            asset="BTC",
            model="HistGB + post-proc.",
            label_variant="default",
            classification_path="experiments/revision/histgb_btc_postproc/eval/classification_metrics.csv",
            stability_path="experiments/revision/histgb_btc_postproc/stability.csv",
            return_ordering_path="experiments/revision/btc_economic_meaning_with_histgb/ordering_summary.csv",
            return_source="histgb_postproc",
            risk_significance_path="experiments/revision/btc_risk_state_significance_with_histgb/risk_significance_summary.csv",
            risk_source="histgb_postproc",
            script_name=common_script,
            output_file="experiments/revision/histgb_btc_postproc/eval/classification_metrics.csv; experiments/revision/histgb_btc_postproc/stability.csv; experiments/revision/btc_economic_meaning_with_histgb/ordering_summary.csv; experiments/revision/btc_risk_state_significance_with_histgb/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="btc_hmm_proxy_mapping_default",
            experiment_group="benchmark_main",
            asset="BTC",
            model="Gaussian HMM",
            label_variant="default",
            classification_path="experiments/baselines/hmm_btc/metrics.csv",
            stability_path="experiments/summary/btc_test_metrics.csv",
            stability_selector="gaussian_hmm",
            script_name="src/models/baselines/run_hmm.py; src/evaluation/evaluate_classification.py; src/evaluation/evaluate_stability.py",
            output_file="experiments/baselines/hmm_btc/metrics.csv; experiments/summary/btc_test_metrics.csv",
            notes="Proxy-majority mapped baseline; semantic comparison omitted because mapped test states collapse to neutral.",
        ),
        ExperimentSpec(
            experiment_id="btc_gru_default",
            experiment_group="benchmark_main",
            asset="BTC",
            model="Main GRU",
            label_variant="default",
            classification_path="experiments/main/gru_btc/eval/classification_metrics.csv",
            stability_path="experiments/main/gru_btc/stability.csv",
            return_ordering_path="experiments/summary/btc_economic_meaning_extended/ordering_summary.csv",
            return_source="main_gru",
            risk_significance_path="experiments/revision/btc_risk_state_significance_with_histgb/risk_significance_summary.csv",
            risk_source="main_gru",
            script_name=common_script,
            output_file="experiments/main/gru_btc/eval/classification_metrics.csv; experiments/main/gru_btc/stability.csv; experiments/summary/btc_economic_meaning_extended/ordering_summary.csv; experiments/revision/btc_risk_state_significance_with_histgb/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="btc_lstm_default",
            experiment_group="benchmark_main",
            asset="BTC",
            model="Vanilla LSTM",
            label_variant="default",
            classification_path="experiments/main/lstm_btc/classification_metrics.csv",
            stability_path="experiments/main/lstm_btc/stability.csv",
            return_ordering_path="experiments/summary/btc_economic_meaning_extended/ordering_summary.csv",
            return_source="main_lstm",
            risk_significance_path="experiments/summary/btc_risk_state_significance_extended/risk_significance_summary.csv",
            risk_source="main_lstm",
            script_name=common_script,
            output_file="experiments/main/lstm_btc/classification_metrics.csv; experiments/main/lstm_btc/stability.csv; experiments/summary/btc_economic_meaning_extended/ordering_summary.csv; experiments/summary/btc_risk_state_significance_extended/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="btc_tcn96_default",
            experiment_group="benchmark_main",
            asset="BTC",
            model="TCN-96x4",
            label_variant="default",
            classification_path="experiments/improved/main/tcn_btc_96x4/eval/classification_metrics.csv",
            stability_path="experiments/improved/main/tcn_btc_96x4/stability.csv",
            return_ordering_path="experiments/summary/btc_economic_meaning_gru_tcn96_compare/ordering_summary.csv",
            return_source="main_tcn_96x4",
            risk_significance_path="experiments/revision/btc_risk_state_significance_with_histgb/risk_significance_summary.csv",
            risk_source="tcn_96x4",
            script_name=common_script,
            output_file="experiments/improved/main/tcn_btc_96x4/eval/classification_metrics.csv; experiments/improved/main/tcn_btc_96x4/stability.csv; experiments/summary/btc_economic_meaning_gru_tcn96_compare/ordering_summary.csv; experiments/revision/btc_risk_state_significance_with_histgb/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="btc_transformer_default",
            experiment_group="benchmark_main",
            asset="BTC",
            model="Transformer-2L",
            label_variant="default",
            classification_path="experiments/summary/btc_classification_transformer/classification_metrics.csv",
            stability_path="experiments/summary/btc_transformer_stability.csv",
            return_ordering_path="experiments/summary/btc_risk_state_semantics_transformer/ordering_summary.csv",
            return_source="transformer_btc",
            risk_significance_path="experiments/summary/btc_risk_state_significance_transformer/risk_significance_summary.csv",
            risk_source="transformer_btc",
            return_order_checks_override=6,
            return_order_matches_override=0,
            script_name=common_script,
            output_file="experiments/summary/btc_classification_transformer/classification_metrics.csv; experiments/summary/btc_transformer_stability.csv; experiments/summary/btc_risk_state_semantics_transformer/ordering_summary.csv; experiments/summary/btc_risk_state_significance_transformer/risk_significance_summary.csv",
            notes="BTCUSDT-only stress test.",
        ),
        ExperimentSpec(
            experiment_id="eth_proxy_default",
            experiment_group="cross_asset_main",
            asset="ETH",
            model="Proxy label",
            label_variant="default",
            classification_path="experiments/summary/eth_test_metrics.csv",
            classification_selector="rule_based",
            stability_path="experiments/summary/eth_test_metrics.csv",
            stability_selector="rule_based",
            return_ordering_path="experiments/summary/eth_economic_meaning_extended/ordering_summary.csv",
            return_source="proxy_label",
            risk_significance_path="experiments/summary/eth_risk_state_significance_extended/risk_significance_summary.csv",
            risk_source="proxy_label",
            script_name=common_script,
            output_file="experiments/summary/eth_test_metrics.csv; experiments/summary/eth_economic_meaning_extended/ordering_summary.csv; experiments/summary/eth_risk_state_significance_extended/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="eth_logreg_default",
            experiment_group="cross_asset_main",
            asset="ETH",
            model="LogReg",
            label_variant="default",
            classification_path="experiments/summary/eth_test_metrics.csv",
            classification_selector="logreg",
            stability_path="experiments/summary/eth_test_metrics.csv",
            stability_selector="logreg",
            return_ordering_path="experiments/summary/eth_economic_meaning_extended/ordering_summary.csv",
            return_source="logistic_regression",
            risk_significance_path="experiments/summary/eth_risk_state_significance_extended/risk_significance_summary.csv",
            risk_source="logistic_regression",
            script_name=common_script,
            output_file="experiments/baselines/logreg_eth/metrics.csv; experiments/summary/eth_economic_meaning_extended/ordering_summary.csv; experiments/summary/eth_risk_state_significance_extended/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="eth_logreg_postproc_default",
            experiment_group="cross_asset_main",
            asset="ETH",
            model="LogReg + post-proc.",
            label_variant="default",
            classification_path="experiments/baselines/logreg_eth_postproc_match/eval/classification_metrics.csv",
            stability_path="experiments/baselines/logreg_eth_postproc_match/stability.csv",
            return_ordering_path="experiments/summary/eth_economic_meaning_extended/ordering_summary.csv",
            return_source="logreg_postproc",
            risk_significance_path="experiments/summary/eth_risk_state_significance_extended/risk_significance_summary.csv",
            risk_source="logreg_postproc",
            script_name=common_script,
            output_file="experiments/baselines/logreg_eth_postproc_match/eval/classification_metrics.csv; experiments/baselines/logreg_eth_postproc_match/stability.csv; experiments/summary/eth_economic_meaning_extended/ordering_summary.csv; experiments/summary/eth_risk_state_significance_extended/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="eth_gru_eth_default",
            experiment_group="cross_asset_main",
            asset="ETH",
            model="Main GRU (ETH-trained)",
            label_variant="default",
            classification_path="experiments/main/gru_eth/eval/classification_metrics.csv",
            stability_path="experiments/main/gru_eth/stability.csv",
            return_ordering_path="experiments/summary/eth_economic_meaning_extended/ordering_summary.csv",
            return_source="main_gru_eth",
            risk_significance_path="experiments/summary/eth_risk_state_significance_extended/risk_significance_summary.csv",
            risk_source="main_gru_eth",
            script_name=common_script,
            output_file="experiments/main/gru_eth/eval/classification_metrics.csv; experiments/main/gru_eth/stability.csv; experiments/summary/eth_economic_meaning_extended/ordering_summary.csv; experiments/summary/eth_risk_state_significance_extended/risk_significance_summary.csv",
        ),
        ExperimentSpec(
            experiment_id="eth_gru_cross_default",
            experiment_group="cross_asset_main",
            asset="ETH",
            model="BTC-trained GRU",
            label_variant="default",
            classification_path="experiments/main/gru_btc_cross_eth/eval/classification_metrics.csv",
            stability_path="experiments/main/gru_btc_cross_eth/stability.csv",
            return_ordering_path="experiments/summary/eth_economic_meaning_extended/ordering_summary.csv",
            return_source="btc_trained_gru",
            risk_significance_path="experiments/summary/eth_risk_state_significance_extended/risk_significance_summary.csv",
            risk_source="btc_trained_gru",
            script_name=common_script,
            output_file="experiments/main/gru_btc_cross_eth/eval/classification_metrics.csv; experiments/main/gru_btc_cross_eth/stability.csv; experiments/summary/eth_economic_meaning_extended/ordering_summary.csv; experiments/summary/eth_risk_state_significance_extended/risk_significance_summary.csv",
        ),
    ]
    for asset in ("BNB", "SOL", "XRP"):
        lower = asset.lower()
        specs.extend(
            [
                ExperimentSpec(
                    experiment_id=f"{lower}_proxy_default",
                    experiment_group="cross_asset_main",
                    asset=asset,
                    model="Proxy label",
                    label_variant="default",
                    classification_path=None,
                    stability_path=None,
                    return_ordering_path=f"experiments/summary/{lower}_economic_meaning_extended/ordering_summary.csv",
                    return_source="proxy_label",
                    risk_significance_path=f"experiments/summary/{lower}_risk_state_significance_extended/risk_significance_summary.csv",
                    risk_source="proxy_label",
                    script_name=common_script,
                    output_file=f"experiments/summary/{lower}_economic_meaning_extended/ordering_summary.csv; experiments/summary/{lower}_risk_state_significance_extended/risk_significance_summary.csv",
                ),
                ExperimentSpec(
                    experiment_id=f"{lower}_logreg_default",
                    experiment_group="cross_asset_main",
                    asset=asset,
                    model="LogReg",
                    label_variant="default",
                    classification_path=f"experiments/baselines/logreg_{lower}/metrics.csv",
                    stability_path=f"experiments/baselines/logreg_{lower}/metrics.csv",
                    return_ordering_path=f"experiments/summary/{lower}_economic_meaning_extended/ordering_summary.csv",
                    return_source="logistic_regression",
                    risk_significance_path=f"experiments/summary/{lower}_risk_state_significance_extended/risk_significance_summary.csv",
                    risk_source="logistic_regression",
                    script_name=common_script,
                    output_file=f"experiments/baselines/logreg_{lower}/metrics.csv; experiments/summary/{lower}_economic_meaning_extended/ordering_summary.csv; experiments/summary/{lower}_risk_state_significance_extended/risk_significance_summary.csv",
                ),
                ExperimentSpec(
                    experiment_id=f"{lower}_logreg_postproc_default",
                    experiment_group="cross_asset_main",
                    asset=asset,
                    model="LogReg + post-proc.",
                    label_variant="default",
                    classification_path=f"experiments/baselines/logreg_{lower}_postproc_match/eval/classification_metrics.csv",
                    stability_path=f"experiments/baselines/logreg_{lower}_postproc_match/stability.csv",
                    return_ordering_path=f"experiments/summary/{lower}_economic_meaning_extended/ordering_summary.csv",
                    return_source="logreg_postproc",
                    risk_significance_path=f"experiments/summary/{lower}_risk_state_significance_extended/risk_significance_summary.csv",
                    risk_source="logreg_postproc",
                    script_name=common_script,
                    output_file=f"experiments/baselines/logreg_{lower}_postproc_match/eval/classification_metrics.csv; experiments/baselines/logreg_{lower}_postproc_match/stability.csv; experiments/summary/{lower}_economic_meaning_extended/ordering_summary.csv; experiments/summary/{lower}_risk_state_significance_extended/risk_significance_summary.csv",
                ),
                ExperimentSpec(
                    experiment_id=f"{lower}_gru_cross_default",
                    experiment_group="cross_asset_main",
                    asset=asset,
                    model="BTC-trained GRU",
                    label_variant="default",
                    classification_path=f"experiments/main/gru_btc_cross_{lower}/eval/classification_metrics.csv",
                    stability_path=f"experiments/main/gru_btc_cross_{lower}/stability.csv",
                    return_ordering_path=f"experiments/summary/{lower}_economic_meaning_extended/ordering_summary.csv",
                    return_source="btc_trained_gru",
                    risk_significance_path=f"experiments/summary/{lower}_risk_state_significance_extended/risk_significance_summary.csv",
                    risk_source="btc_trained_gru",
                    script_name=common_script,
                    output_file=f"experiments/main/gru_btc_cross_{lower}/eval/classification_metrics.csv; experiments/main/gru_btc_cross_{lower}/stability.csv; experiments/summary/{lower}_economic_meaning_extended/ordering_summary.csv; experiments/summary/{lower}_risk_state_significance_extended/risk_significance_summary.csv",
                ),
            ]
        )
    return specs


def add_core_rows(specs: list[ExperimentSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        row: dict[str, Any] = {
            "experiment_id": spec.experiment_id,
            "experiment_group": spec.experiment_group,
            "asset": spec.asset,
            "model": spec.model,
            "label_variant": spec.label_variant,
            "frequency": spec.frequency,
            "seed": spec.seed,
            "train_period": spec.train_period,
            "val_period": spec.val_period,
            "test_period": spec.test_period,
            "macro_f1": None,
            "balanced_acc": None,
            "bull_f1": None,
            "bear_f1": None,
            "switch_day": None,
            "avg_duration": None,
            "risk_order_checks": None,
            "risk_order_matches": None,
            "risk_order_rate": None,
            "return_order_checks": None,
            "return_order_matches": None,
            "return_order_rate": None,
            "sig_risk_checks": None,
            "sig_risk_rate": None,
            "bootstrap_checks": None,
            "bootstrap_rate": None,
                "candidate_pool": None,
                "framework": None,
                "semantic_regret": None,
                "selection_disagreement_rate": None,
                "macro_f1_cost": None,
                "switch_cost": None,
                "script_name": spec.script_name,
                "output_file": spec.output_file,
                "notes": spec.notes,
            }
        row.update(extract_classification(_path(spec.classification_path), spec.classification_selector))
        row.update(extract_stability(_path(spec.stability_path), spec.stability_selector))
        row.update(extract_return_semantics(_path(spec.return_ordering_path), spec.return_source))
        row.update(extract_risk_semantics(_path(spec.risk_significance_path), spec.risk_source))
        if spec.risk_order_checks_override is not None:
            matches = spec.risk_order_matches_override or 0
            checks = spec.risk_order_checks_override
            row["risk_order_checks"] = checks
            row["risk_order_matches"] = matches
            row["risk_order_rate"] = float(matches / checks) if checks else None
        if spec.sig_risk_checks_override is not None and spec.risk_order_checks_override is not None:
            checks = spec.risk_order_checks_override
            sig_checks = spec.sig_risk_checks_override
            row["sig_risk_checks"] = sig_checks
            row["sig_risk_rate"] = float(sig_checks / checks) if checks else None
        if spec.bootstrap_checks_override is not None and spec.risk_order_checks_override is not None:
            checks = spec.risk_order_checks_override
            boot_checks = spec.bootstrap_checks_override
            row["bootstrap_checks"] = boot_checks
            row["bootstrap_rate"] = float(boot_checks / checks) if checks else None
        if spec.return_order_checks_override is not None:
            matches = spec.return_order_matches_override or 0
            checks = spec.return_order_checks_override
            row["return_order_checks"] = checks
            row["return_order_matches"] = matches
            row["return_order_rate"] = float(matches / checks) if checks else None
        rows.append(row)
    return rows


def add_seed_rows() -> list[dict[str, Any]]:
    merged_seed_path = Path("experiments/summary/seed_robustness_5seed.csv")
    if merged_seed_path.exists():
        detail = pd.read_csv(merged_seed_path)
        rows: list[dict[str, Any]] = []
        for _, seed_row in detail.iterrows():
            model = str(seed_row["model"])
            seed = str(int(seed_row["seed"]))
            rows.append(
                {
                    "experiment_id": f"btc_{str(seed_row['model_key'])}_seed_{seed}",
                    "experiment_group": "seed_robustness",
                    "asset": str(seed_row["asset"]),
                    "model": model,
                    "label_variant": "default",
                    "frequency": "1h",
                    "seed": seed,
                    "train_period": "2021-01-01 to 2023-12-31",
                    "val_period": "2024-01-01 to 2024-12-31",
                    "test_period": "2025-01-01 to 2026-04-20",
                    "macro_f1": _safe_float(seed_row.get("macro_f1")),
                    "balanced_acc": _safe_float(seed_row.get("balanced_accuracy")),
                    "bull_f1": _safe_float(seed_row.get("bull_f1")),
                    "bear_f1": _safe_float(seed_row.get("bear_f1")),
                    "switch_day": _safe_float(seed_row.get("switch_day")),
                    "avg_duration": _safe_float(seed_row.get("avg_duration")),
                    "risk_order_checks": _safe_int(seed_row.get("risk_order_checks")),
                    "risk_order_matches": _safe_int(seed_row.get("risk_order_matches")),
                    "risk_order_rate": _safe_float(seed_row.get("risk_order_rate")),
                    "return_order_checks": _safe_int(seed_row.get("return_order_checks")),
                    "return_order_matches": _safe_int(seed_row.get("return_order_matches")),
                    "return_order_rate": _safe_float(seed_row.get("return_order_rate")),
                    "sig_risk_checks": _safe_int(seed_row.get("sig_risk_checks")),
                    "sig_risk_rate": _safe_float(seed_row.get("sig_risk_rate")),
                    "bootstrap_checks": None,
                    "bootstrap_rate": None,
                    "candidate_pool": None,
                    "framework": None,
                    "semantic_regret": None,
                    "selection_disagreement_rate": None,
                    "macro_f1_cost": None,
                    "switch_cost": None,
                    "script_name": "src/models/main/train_main.py; src/evaluation/run_seed_robustness_5seed.py",
                    "output_file": str(seed_row.get("run_dir")),
                    "notes": "BTC 5-seed robustness row.",
                }
            )
        return rows

    seeds = [
        ("gru_seed42", "GRU", "42", "experiments/main/gru_btc/eval/classification_metrics.csv", "experiments/main/gru_btc/stability.csv"),
        ("gru_seed7", "GRU", "7", "experiments/robustness/gru_btc_seed7/classification_metrics.csv", "experiments/robustness/gru_btc_seed7/stability.csv"),
        ("gru_seed123", "GRU", "123", "experiments/robustness/gru_btc_seed123/classification_metrics.csv", "experiments/robustness/gru_btc_seed123/stability.csv"),
        ("lstm_seed42", "LSTM", "42", "experiments/main/lstm_btc/classification_metrics.csv", "experiments/main/lstm_btc/stability.csv"),
        ("lstm_seed7", "LSTM", "7", "experiments/robustness/lstm_btc_seed7/classification_metrics.csv", "experiments/robustness/lstm_btc_seed7/stability.csv"),
        ("lstm_seed123", "LSTM", "123", "experiments/robustness/lstm_btc_seed123/classification_metrics.csv", "experiments/robustness/lstm_btc_seed123/stability.csv"),
    ]
    rows: list[dict[str, Any]] = []
    for source, model, seed, class_path, stability_path in seeds:
        row = {
            "experiment_id": f"btc_{model.lower()}_seed_{seed}",
            "experiment_group": "seed_robustness",
            "asset": "BTC",
            "model": model,
            "label_variant": "default",
            "frequency": "1h",
            "seed": seed,
            "train_period": "2021-01-01 to 2023-12-31",
            "val_period": "2024-01-01 to 2024-12-31",
            "test_period": "2025-01-01 to 2026-04-20",
                "candidate_pool": None,
                "framework": None,
                "semantic_regret": None,
                "selection_disagreement_rate": None,
                "macro_f1_cost": None,
                "switch_cost": None,
                "script_name": "src/models/main/train_main.py; src/evaluation/evaluate_classification.py; src/evaluation/evaluate_stability.py; src/evaluation/evaluate_risk_state_significance.py; src/evaluation/evaluate_risk_state_semantics.py",
                "output_file": f"{class_path}; {stability_path}; experiments/summary/btc_temporal_seed_semantics/risk_significance_summary.csv; experiments/summary/btc_temporal_seed_risk_order/ordering_summary.csv",
                "notes": "BTC seed robustness row.",
            }
        row.update(extract_classification(Path(class_path)))
        row.update(extract_stability(Path(stability_path)))
        row.update(extract_return_semantics(Path("experiments/summary/btc_temporal_seed_risk_order/ordering_summary.csv"), source))
        row.update(extract_risk_semantics(Path("experiments/summary/btc_temporal_seed_semantics/risk_significance_summary.csv"), source))
        rows.append(row)
    return rows


def add_hmm_latent_rows() -> list[dict[str, Any]]:
    output_dir = Path("experiments/summary/hmm_latent_semantics_btc")
    specs = [
        ExperimentSpec(
            experiment_id="hmm_proxy_mapping_btc",
            experiment_group="hmm_proxy_mapping",
            asset="BTC",
            model="Gaussian HMM",
            label_variant="default",
            classification_path=str(output_dir / "classification_metrics.csv"),
            classification_selector="hmm_proxy_mapping",
            stability_path=str(output_dir / "stability_metrics.csv"),
            stability_selector="hmm_proxy_mapping",
            return_ordering_path=str(output_dir / "return_ordering_summary.csv"),
            return_source="hmm_proxy_mapping",
            risk_significance_path=str(output_dir / "risk_significance_summary.csv"),
            risk_source="hmm_proxy_mapping",
            risk_order_checks_override=21,
            risk_order_matches_override=0,
            sig_risk_checks_override=0,
            bootstrap_checks_override=0,
            script_name="src/evaluation/run_hmm_latent_semantics.py",
            output_file="experiments/summary/hmm_latent_semantics_btc/classification_metrics.csv; experiments/summary/hmm_latent_semantics_btc/stability_metrics.csv; experiments/summary/hmm_latent_semantics_btc/return_ordering_summary.csv; experiments/summary/hmm_latent_semantics_btc/risk_significance_summary.csv",
            notes="Proxy-majority mapping collapses to the neutral label on BTC test data.",
        ),
        ExperimentSpec(
            experiment_id="hmm_semantic_ordering_btc",
            experiment_group="hmm_semantic_ordering",
            asset="BTC",
            model="Gaussian HMM",
            label_variant="default",
            classification_path=str(output_dir / "classification_metrics.csv"),
            classification_selector="hmm_semantic_ordering",
            stability_path=str(output_dir / "stability_metrics.csv"),
            stability_selector="hmm_semantic_ordering",
            return_ordering_path=str(output_dir / "return_ordering_summary.csv"),
            return_source="hmm_semantic_ordering",
            risk_significance_path=str(output_dir / "risk_significance_summary.csv"),
            risk_source="hmm_semantic_ordering",
            script_name="src/evaluation/run_hmm_latent_semantics.py",
            output_file="experiments/summary/hmm_latent_semantics_btc/classification_metrics.csv; experiments/summary/hmm_latent_semantics_btc/stability_metrics.csv; experiments/summary/hmm_latent_semantics_btc/return_ordering_summary.csv; experiments/summary/hmm_latent_semantics_btc/risk_significance_summary.csv",
            notes="Hidden states remapped by ex-post risk ordering on the BTC test window.",
        ),
    ]
    return add_core_rows(specs)


def add_threshold_rows() -> list[dict[str, Any]]:
    df = pd.read_csv("experiments/summary/threshold_sensitivity_semantics.csv")
    rows: list[dict[str, Any]] = []
    model_map = {
        "proxy_label": "Proxy label",
        "logistic_regression": "LogReg",
        "logreg_postproc": "LogReg + post-proc.",
    }
    for record in df.to_dict(orient="records"):
        rows.append(
            {
                "experiment_id": f"threshold_{record['variant']}_{record['asset'].lower()}_{record['method']}",
                "experiment_group": "threshold_sensitivity",
                "asset": record["asset"],
                "model": model_map.get(record["method"], record["method"]),
                "label_variant": record["variant"],
                "frequency": "1h",
                "seed": None,
                "train_period": "2021-01-01 to 2023-12-31",
                "val_period": "2024-01-01 to 2024-12-31",
                "test_period": "2025-01-01 to 2026-04-20",
                "macro_f1": None,
                "balanced_acc": None,
                "bull_f1": None,
                "bear_f1": None,
                "switch_day": None,
                "avg_duration": None,
                "risk_order_checks": _safe_int(record.get("risk_checks")),
                "risk_order_matches": _safe_int(record.get("risk_order_matches")),
                "risk_order_rate": _safe_float(record.get("risk_match_rate")),
                "return_order_checks": _safe_int(record.get("return_checks")),
                "return_order_matches": _safe_int(record.get("return_matches")),
                "return_order_rate": _safe_float(record.get("return_match_rate")),
                "sig_risk_checks": _safe_int(record.get("significant_risk_layering")),
                "sig_risk_rate": _safe_float(record.get("significant_risk_layering_rate")),
                "bootstrap_checks": None,
                "bootstrap_rate": None,
                "candidate_pool": None,
                "framework": None,
                "semantic_regret": None,
                "selection_disagreement_rate": None,
                "macro_f1_cost": None,
                "switch_cost": None,
                "script_name": "src/evaluation/run_threshold_sensitivity.py; src/evaluation/summarize_threshold_sensitivity.py",
                "output_file": "experiments/summary/threshold_sensitivity_semantics.csv",
                "notes": "Threshold sensitivity semantic row.",
            }
        )
    return rows


def add_framework_rows() -> list[dict[str, Any]]:
    df = pd.read_csv("experiments/summary/evaluation_framework_increment/framework_increment_summary.csv")
    pool_map = {"btc_extended": "BTC extended", "cross_asset_common": "Five-asset common"}
    framework_map = {
        "classification_only": "C only",
        "ct_weighted_mcdm": "C+T weighted",
        "ct_topsis_mcdm": "C+T TOPSIS",
        "cts_weighted_oracle": "C+T+S weighted",
    }
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        rows.append(
            {
                "experiment_id": f"framework_{record['pool']}_{record['framework']}",
                "experiment_group": "framework_increment",
                "asset": "MULTI",
                "model": "Selector",
                "label_variant": "default",
                "frequency": "1h",
                "seed": None,
                "train_period": "2021-01-01 to 2023-12-31",
                "val_period": "2024-01-01 to 2024-12-31",
                "test_period": "2025-01-01 to 2026-04-20",
                "macro_f1": None,
                "balanced_acc": None,
                "bull_f1": None,
                "bear_f1": None,
                "switch_day": None,
                "avg_duration": None,
                "risk_order_checks": None,
                "risk_order_matches": None,
                "risk_order_rate": None,
                "return_order_checks": None,
                "return_order_matches": None,
                "return_order_rate": None,
                "sig_risk_checks": None,
                "sig_risk_rate": None,
                "bootstrap_checks": None,
                "bootstrap_rate": None,
                "candidate_pool": pool_map.get(record["pool"], record["pool"]),
                "framework": framework_map.get(record["framework"], record["framework"]),
                "semantic_regret": _safe_float(record.get("mean_semantic_regret_vs_protocol")),
                "selection_disagreement_rate": _safe_float(record.get("selection_disagreement_rate")),
                "macro_f1_cost": _safe_float(record.get("mean_macro_f1_cost_vs_protocol")),
                "switch_cost": _safe_float(record.get("mean_switch_cost_vs_protocol")),
                "script_name": "src/evaluation/compare_evaluation_frameworks.py",
                "output_file": "experiments/summary/evaluation_framework_increment/framework_increment_summary.csv",
                "notes": f"Selection disagreement rate={record['selection_disagreement_rate']}",
            }
        )
    return rows


def add_checkpoint_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "experiment_id": "checkpoint_main_macro",
            "experiment_group": "checkpoint_selection",
            "asset": "BTC",
            "model": "Main GRU",
            "label_variant": "default",
            "frequency": "1h",
            "seed": "42",
            "train_period": "2021-01-01 to 2023-12-31",
            "val_period": "2024-01-01 to 2024-12-31",
            "test_period": "2025-01-01 to 2026-04-20",
            "macro_f1": 0.6635,
            "balanced_acc": 0.6533,
            "bull_f1": None,
            "bear_f1": None,
            "switch_day": 0.5053,
            "avg_duration": None,
            "risk_order_checks": 21,
            "risk_order_matches": 14,
            "risk_order_rate": 14 / 21,
            "return_order_checks": None,
            "return_order_matches": None,
            "return_order_rate": None,
            "sig_risk_checks": 14,
            "sig_risk_rate": 14 / 21,
            "bootstrap_checks": None,
            "bootstrap_rate": None,
            "candidate_pool": None,
            "framework": "Validation macro-F1",
            "semantic_regret": None,
            "selection_disagreement_rate": None,
            "macro_f1_cost": None,
            "switch_cost": None,
            "script_name": "src/models/main/train_main.py",
            "output_file": "experiments/improved/main/gru_btc_checkpoint_macro/metrics.csv; experiments/summary/btc_checkpoint_selection_significance/risk_significance_summary.csv",
            "notes": "Main split checkpoint selector.",
        },
        {
            "experiment_id": "checkpoint_main_semantic",
            "experiment_group": "checkpoint_selection",
            "asset": "BTC",
            "model": "Main GRU",
            "label_variant": "default",
            "frequency": "1h",
            "seed": "42",
            "train_period": "2021-01-01 to 2023-12-31",
            "val_period": "2024-01-01 to 2024-12-31",
            "test_period": "2025-01-01 to 2026-04-20",
            "macro_f1": 0.6304,
            "balanced_acc": 0.6245,
            "bull_f1": None,
            "bear_f1": None,
            "switch_day": 0.5053,
            "avg_duration": None,
            "risk_order_checks": 21,
            "risk_order_matches": 16,
            "risk_order_rate": 16 / 21,
            "return_order_checks": None,
            "return_order_matches": None,
            "return_order_rate": None,
            "sig_risk_checks": 16,
            "sig_risk_rate": 16 / 21,
            "bootstrap_checks": None,
            "bootstrap_rate": None,
            "candidate_pool": None,
            "framework": "Semantic-aware",
            "semantic_regret": None,
            "selection_disagreement_rate": None,
            "macro_f1_cost": None,
            "switch_cost": None,
            "script_name": "src/models/main/train_main.py",
            "output_file": "experiments/improved/main/gru_btc_checkpoint_semantic_v2/metrics.csv; experiments/summary/btc_checkpoint_selection_significance/risk_significance_summary.csv",
            "notes": "Main split checkpoint selector.",
        },
        {
            "experiment_id": "checkpoint_roll2024_macro",
            "experiment_group": "checkpoint_selection",
            "asset": "BTC",
            "model": "Main GRU",
            "label_variant": "default",
            "frequency": "1h",
            "seed": "42",
            "train_period": "2021-01-01 to 2022-12-31",
            "val_period": "2023-01-01 to 2023-12-31",
            "test_period": "2024-01-01 to 2024-12-31",
            "macro_f1": 0.5720,
            "balanced_acc": 0.5340,
            "bull_f1": None,
            "bear_f1": None,
            "switch_day": 0.5247,
            "avg_duration": None,
            "risk_order_checks": 14,
            "risk_order_matches": 1,
            "risk_order_rate": 1 / 14,
            "return_order_checks": None,
            "return_order_matches": None,
            "return_order_rate": None,
            "sig_risk_checks": 1,
            "sig_risk_rate": 1 / 14,
            "bootstrap_checks": None,
            "bootstrap_rate": None,
            "candidate_pool": None,
            "framework": "Validation macro-F1",
            "semantic_regret": None,
            "selection_disagreement_rate": None,
            "macro_f1_cost": None,
            "switch_cost": None,
            "script_name": "src/models/main/train_main.py",
            "output_file": "experiments/improved/main/gru_btc_roll2024_macro/metrics.csv; experiments/summary/btc_roll2024_checkpoint_selection_significance/risk_significance_summary.csv",
            "notes": "Rolling-origin checkpoint selector.",
        },
        {
            "experiment_id": "checkpoint_roll2024_semantic",
            "experiment_group": "checkpoint_selection",
            "asset": "BTC",
            "model": "Main GRU",
            "label_variant": "default",
            "frequency": "1h",
            "seed": "42",
            "train_period": "2021-01-01 to 2022-12-31",
            "val_period": "2023-01-01 to 2023-12-31",
            "test_period": "2024-01-01 to 2024-12-31",
            "macro_f1": 0.5028,
            "balanced_acc": 0.4983,
            "bull_f1": None,
            "bear_f1": None,
            "switch_day": 0.4427,
            "avg_duration": None,
            "risk_order_checks": 7,
            "risk_order_matches": 1,
            "risk_order_rate": 1 / 7,
            "return_order_checks": None,
            "return_order_matches": None,
            "return_order_rate": None,
            "sig_risk_checks": 1,
            "sig_risk_rate": 1 / 7,
            "bootstrap_checks": None,
            "bootstrap_rate": None,
            "candidate_pool": None,
            "framework": "Semantic-aware",
            "semantic_regret": None,
            "selection_disagreement_rate": None,
            "macro_f1_cost": None,
            "switch_cost": None,
            "script_name": "src/models/main/train_main.py",
            "output_file": "experiments/improved/main/gru_btc_roll2024_semantic/metrics.csv; experiments/summary/btc_roll2024_checkpoint_selection_significance/risk_significance_summary.csv",
            "notes": "Rolling-origin checkpoint selector.",
        },
    ]
    return rows


def add_drawdown_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "experiment_id": "btc_logreg_no_drawdown_feature",
            "experiment_group": "drawdown_deconfounding",
            "asset": "BTC",
            "model": "LogReg",
            "label_variant": "default_no_drawdown_feature",
            "frequency": "1h",
            "seed": None,
            "train_period": "2021-01-01 to 2023-12-31",
            "val_period": "2024-01-01 to 2024-12-31",
            "test_period": "2025-01-01 to 2026-04-20",
            "macro_f1": 0.7051,
            "balanced_acc": None,
            "bull_f1": None,
            "bear_f1": None,
            "switch_day": None,
            "avg_duration": None,
            "risk_order_checks": 21,
            "risk_order_matches": None,
            "risk_order_rate": None,
            "return_order_checks": None,
            "return_order_matches": None,
            "return_order_rate": None,
            "sig_risk_checks": 13,
            "sig_risk_rate": 13 / 21,
            "bootstrap_checks": None,
            "bootstrap_rate": None,
            "candidate_pool": None,
            "framework": None,
            "semantic_regret": None,
            "selection_disagreement_rate": None,
            "macro_f1_cost": None,
            "switch_cost": None,
            "script_name": "src/models/baselines/run_logreg.py",
            "output_file": "experiments/improved/baselines/logreg_btc_no_drawdown_feature/metrics.csv; experiments/summary/btc_risk_state_significance_drawdown_compare/risk_significance_summary.csv",
            "notes": "Feature-side drawdown removal.",
        },
        {
            "experiment_id": "btc_gru_no_drawdown_feature",
            "experiment_group": "drawdown_deconfounding",
            "asset": "BTC",
            "model": "Main GRU",
            "label_variant": "default_no_drawdown_feature",
            "frequency": "1h",
            "seed": "42",
            "train_period": "2021-01-01 to 2023-12-31",
            "val_period": "2024-01-01 to 2024-12-31",
            "test_period": "2025-01-01 to 2026-04-20",
            "macro_f1": 0.6223,
            "balanced_acc": None,
            "bull_f1": None,
            "bear_f1": None,
            "switch_day": None,
            "avg_duration": None,
            "risk_order_checks": 21,
            "risk_order_matches": None,
            "risk_order_rate": None,
            "return_order_checks": None,
            "return_order_matches": None,
            "return_order_rate": None,
            "sig_risk_checks": 8,
            "sig_risk_rate": 8 / 21,
            "bootstrap_checks": None,
            "bootstrap_rate": None,
            "candidate_pool": None,
            "framework": None,
            "semantic_regret": None,
            "selection_disagreement_rate": None,
            "macro_f1_cost": None,
            "switch_cost": None,
            "script_name": "src/models/main/train_main.py",
            "output_file": "experiments/improved/main/gru_btc_no_drawdown_feature/metrics.csv; experiments/summary/btc_risk_state_significance_drawdown_compare/risk_significance_summary.csv",
            "notes": "Feature-side drawdown removal.",
        },
        {
            "experiment_id": "btc_logreg_no_drawdown_rule",
            "experiment_group": "drawdown_deconfounding",
            "asset": "BTC",
            "model": "LogReg",
            "label_variant": "no_drawdown_rule",
            "frequency": "1h",
            "seed": None,
            "train_period": "2021-01-01 to 2023-12-31",
            "val_period": "2024-01-01 to 2024-12-31",
            "test_period": "2025-01-01 to 2026-04-20",
            "macro_f1": 0.8778,
            "balanced_acc": 0.9009,
            "bull_f1": None,
            "bear_f1": None,
            "switch_day": None,
            "avg_duration": None,
            "risk_order_checks": 21,
            "risk_order_matches": None,
            "risk_order_rate": None,
            "return_order_checks": None,
            "return_order_matches": None,
            "return_order_rate": None,
            "sig_risk_checks": 3,
            "sig_risk_rate": 3 / 21,
            "bootstrap_checks": None,
            "bootstrap_rate": None,
            "candidate_pool": None,
            "framework": None,
            "semantic_regret": None,
            "selection_disagreement_rate": None,
            "macro_f1_cost": None,
            "switch_cost": None,
            "script_name": "src/models/baselines/run_logreg.py",
            "output_file": "experiments/improved/baselines/logreg_btc_no_drawdown_rule/metrics.csv; experiments/summary/btc_risk_state_significance_no_drawdown_rule/risk_significance_summary.csv",
            "notes": "Label-side drawdown removal.",
        },
        {
            "experiment_id": "btc_gru_no_drawdown_rule",
            "experiment_group": "drawdown_deconfounding",
            "asset": "BTC",
            "model": "Main GRU",
            "label_variant": "no_drawdown_rule",
            "frequency": "1h",
            "seed": "42",
            "train_period": "2021-01-01 to 2023-12-31",
            "val_period": "2024-01-01 to 2024-12-31",
            "test_period": "2025-01-01 to 2026-04-20",
            "macro_f1": 0.8194,
            "balanced_acc": 0.7897,
            "bull_f1": None,
            "bear_f1": None,
            "switch_day": None,
            "avg_duration": None,
            "risk_order_checks": 21,
            "risk_order_matches": None,
            "risk_order_rate": None,
            "return_order_checks": None,
            "return_order_matches": None,
            "return_order_rate": None,
            "sig_risk_checks": 7,
            "sig_risk_rate": 7 / 21,
            "bootstrap_checks": None,
            "bootstrap_rate": None,
            "candidate_pool": None,
            "framework": None,
            "semantic_regret": None,
            "selection_disagreement_rate": None,
            "macro_f1_cost": None,
            "switch_cost": None,
            "script_name": "src/models/main/train_main.py",
            "output_file": "experiments/improved/main/gru_btc_no_drawdown_rule/metrics.csv; experiments/summary/btc_risk_state_significance_no_drawdown_rule/risk_significance_summary.csv",
            "notes": "Label-side drawdown removal.",
        },
    ]
    return rows


def add_bootstrap_block_rows() -> list[dict[str, Any]]:
    path = Path("experiments/summary/bootstrap_block_sensitivity.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, item in df.iterrows():
        block_size = str(int(item["bootstrap_block_size"]))
        rows.append(
            {
                "experiment_id": f"btc_{str(item['source'])}_bootstrap_{block_size}h",
                "experiment_group": "bootstrap_block_sensitivity",
                "asset": str(item["asset"]),
                "model": str(item["model"]),
                "label_variant": "default",
                "frequency": "1h",
                "seed": None,
                "train_period": "2021-01-01 to 2023-12-31",
                "val_period": "2024-01-01 to 2024-12-31",
                "test_period": "2025-01-01 to 2026-04-20",
                "macro_f1": None,
                "balanced_acc": None,
                "bull_f1": None,
                "bear_f1": None,
                "switch_day": None,
                "avg_duration": None,
                "risk_order_checks": _safe_int(item.get("checks")),
                "risk_order_matches": _safe_int(item.get("risk_order_matches")),
                "risk_order_rate": _safe_float(item.get("risk_order_rate")),
                "return_order_checks": None,
                "return_order_matches": None,
                "return_order_rate": None,
                "sig_risk_checks": _safe_int(item.get("nominal_sig_risk_checks")),
                "sig_risk_rate": _safe_float(item.get("nominal_sig_risk_rate")),
                "bootstrap_checks": _safe_int(item.get("bootstrap_sig_risk_checks")),
                "bootstrap_rate": _safe_float(item.get("bootstrap_sig_risk_rate")),
                "candidate_pool": None,
                "framework": None,
                "semantic_regret": None,
                "selection_disagreement_rate": None,
                "macro_f1_cost": None,
                "switch_cost": None,
                "script_name": "src/evaluation/run_bootstrap_block_sensitivity.py",
                "output_file": str(path),
                "notes": f"BTC block-bootstrap sensitivity with {block_size}h blocks and {int(item['bootstrap_samples'])} samples.",
            }
        )
    return rows


def add_frequency_4h_rows() -> list[dict[str, Any]]:
    path = Path("experiments/summary/frequency_4h_robustness.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, item in df.iterrows():
        model_slug = str(item["model"]).lower().replace(" + ", "_plus_").replace(" ", "_").replace("-", "_").replace(".", "")
        rows.append(
            {
                "experiment_id": f"{str(item['asset']).lower()}_{model_slug}_4h",
                "experiment_group": "frequency_4h_robustness",
                "asset": str(item["asset"]),
                "model": str(item["model"]),
                "label_variant": "default",
                "frequency": str(item["frequency"]),
                "seed": None,
                "train_period": "2021-01-01 to 2023-12-31",
                "val_period": "2024-01-01 to 2024-12-31",
                "test_period": "2025-01-01 to 2026-04-20",
                "macro_f1": _safe_float(item.get("macro_f1")),
                "balanced_acc": _safe_float(item.get("balanced_accuracy")),
                "bull_f1": _safe_float(item.get("bull_f1")),
                "bear_f1": _safe_float(item.get("bear_f1")),
                "switch_day": _safe_float(item.get("switch_day")),
                "avg_duration": _safe_float(item.get("avg_duration")),
                "risk_order_checks": _safe_int(item.get("risk_order_checks")),
                "risk_order_matches": _safe_int(item.get("risk_order_matches")),
                "risk_order_rate": _safe_float(item.get("risk_order_rate")),
                "return_order_checks": _safe_int(item.get("return_order_checks")),
                "return_order_matches": _safe_int(item.get("return_order_matches")),
                "return_order_rate": _safe_float(item.get("return_order_rate")),
                "sig_risk_checks": _safe_int(item.get("sig_risk_checks")),
                "sig_risk_rate": _safe_float(item.get("sig_risk_rate")),
                "bootstrap_checks": None,
                "bootstrap_rate": None,
                "candidate_pool": None,
                "framework": None,
                "semantic_regret": None,
                "selection_disagreement_rate": None,
                "macro_f1_cost": None,
                "switch_cost": None,
                "script_name": "src/evaluation/run_frequency_4h_robustness.py",
                "output_file": str(path),
                "notes": "4h robustness summary for lightweight baselines on BTC and ETH.",
            }
        )
    return rows


def main() -> None:
    rows: list[dict[str, Any]] = []
    rows.extend(add_core_rows(build_core_specs()))
    rows.extend(add_hmm_latent_rows())
    rows.extend(add_seed_rows())
    rows.extend(add_threshold_rows())
    rows.extend(add_framework_rows())
    rows.extend(add_checkpoint_rows())
    rows.extend(add_drawdown_rows())
    rows.extend(add_bootstrap_block_rows())
    rows.extend(add_frequency_4h_rows())

    df = pd.DataFrame(rows)
    column_order = [
        "experiment_id",
        "experiment_group",
        "asset",
        "model",
        "label_variant",
        "frequency",
        "seed",
        "train_period",
        "val_period",
        "test_period",
        "macro_f1",
        "balanced_acc",
        "bull_f1",
        "bear_f1",
        "switch_day",
        "avg_duration",
        "risk_order_checks",
        "risk_order_matches",
        "risk_order_rate",
        "return_order_checks",
        "return_order_matches",
        "return_order_rate",
        "sig_risk_checks",
        "sig_risk_rate",
        "bootstrap_checks",
        "bootstrap_rate",
        "candidate_pool",
        "framework",
        "semantic_regret",
        "selection_disagreement_rate",
        "macro_f1_cost",
        "switch_cost",
        "script_name",
        "output_file",
        "notes",
    ]
    df = df[column_order].sort_values(["experiment_group", "asset", "model", "label_variant", "seed"], na_position="last")

    output_path = Path("experiments/summary/results_manifest.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = (
        df.groupby("experiment_group", dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("experiment_group")
    )
    print(f"[wrote] {output_path} ({len(df)} rows)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
