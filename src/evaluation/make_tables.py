from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.io import ensure_dir


MANIFEST_PATH = Path("experiments/summary/results_manifest.csv")
CSV_DIR = ensure_dir("experiments/summary/tables")
TEX_DIR = ensure_dir("experiments/summary/generated_tables")


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH)


def fmt_decimal(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "---"
    return f"{float(value):.{digits}f}"


def fmt_decimal2(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "---"
    return f"{float(value):.2f}"


def fmt_percent_from_rate(rate: float | int | None, digits: int = 1) -> str:
    if rate is None or pd.isna(rate):
        return "---"
    return f"{100.0 * float(rate):.{digits}f}\\%"


def fmt_percent_fraction(numerator: float | int | None, denominator: float | int | None, digits: int = 1) -> str:
    if numerator is None or denominator in (None, 0) or pd.isna(numerator) or pd.isna(denominator):
        return "---"
    return f"{100.0 * float(numerator) / float(denominator):.{digits}f}\\%"


def fmt_count_rate(numerator: float | int | None, denominator: float | int | None, digits: int = 1) -> str:
    if numerator is None or denominator in (None, 0) or pd.isna(numerator) or pd.isna(denominator):
        return "---"
    return f"{int(numerator)}/{int(denominator)} ({fmt_percent_fraction(numerator, denominator, digits)})"


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)
    return text


def write_outputs(name: str, df: pd.DataFrame, tex: str) -> None:
    df.to_csv(CSV_DIR / f"{name}.csv", index=False)
    (TEX_DIR / f"{name}.tex").write_text(tex + "\n", encoding="utf-8")
    print(f"[wrote] {CSV_DIR / f'{name}.csv'}")
    print(f"[wrote] {TEX_DIR / f'{name}.tex'}")


def wrap_best(text: str, is_best: bool) -> str:
    return f"\\textbf{{{text}}}" if is_best and text != "---" else text


def render_table(lines: Iterable[str]) -> str:
    return "\n".join(lines)


def build_btc_benchmark(df: pd.DataFrame) -> None:
    subset = df.loc[
        (df["experiment_group"] == "benchmark_main")
        & (df["asset"] == "BTC")
        & (df["model"] != "Proxy label")
    ].copy()
    order = [
        "LogReg",
        "LogReg + post-proc.",
        "HistGB",
        "HistGB + post-proc.",
        "Gaussian HMM",
        "Main GRU",
        "Vanilla LSTM",
        "TCN-96x4",
        "Transformer-2L",
    ]
    display_map = {
        "LogReg": "Logistic regression",
        "LogReg + post-proc.": "Logreg + post-proc.",
        "HistGB": "HistGB",
        "HistGB + post-proc.": "HistGB + post-proc.",
        "Gaussian HMM": "Gaussian HMM",
        "Main GRU": "Main GRU",
        "Vanilla LSTM": "Vanilla LSTM",
        "TCN-96x4": "TCN-96x4",
        "Transformer-2L": "Transformer-2L",
    }
    subset["order"] = subset["model"].map({name: idx for idx, name in enumerate(order)})
    subset = subset.sort_values("order")

    learned_for_macro = subset.loc[subset["model"] != "Gaussian HMM"]
    best_macro = learned_for_macro["macro_f1"].max()
    best_bull = learned_for_macro["bull_f1"].max()
    best_bear = learned_for_macro["bear_f1"].max()
    best_switch = learned_for_macro["switch_day"].min()
    best_duration = learned_for_macro["avg_duration"].max()
    best_sig = subset["sig_risk_rate"].max()

    rows = []
    for _, row in subset.iterrows():
        model = row["model"]
        sig_text = "---" if model in {"HistGB", "Gaussian HMM"} else fmt_count_rate(row["sig_risk_checks"], 21)
        rows.append(
            {
                "Method": display_map[model],
                "Macro-F1": fmt_decimal(row["macro_f1"]),
                "Bull F1": fmt_decimal(row["bull_f1"]),
                "Bear F1": fmt_decimal(row["bear_f1"]),
                "Switch/day": fmt_decimal(row["switch_day"]),
                "Avg. duration": fmt_decimal2(row["avg_duration"]),
                "Sig. risk checks": sig_text,
            }
        )

    out_df = pd.DataFrame(rows)

    body_lines = []
    for _, row in subset.iterrows():
        model = row["model"]
        method = display_map[model]
        macro = wrap_best(fmt_decimal(row["macro_f1"]), row["macro_f1"] == best_macro and model != "Gaussian HMM")
        bull = wrap_best(fmt_decimal(row["bull_f1"]), row["bull_f1"] == best_bull and model != "Gaussian HMM")
        bear = wrap_best(fmt_decimal(row["bear_f1"]), row["bear_f1"] == best_bear and model != "Gaussian HMM")
        switch = wrap_best(fmt_decimal(row["switch_day"]), row["switch_day"] == best_switch and model != "Gaussian HMM")
        duration = wrap_best(fmt_decimal2(row["avg_duration"]), row["avg_duration"] == best_duration)
        sig = "---" if model in {"HistGB", "Gaussian HMM"} else fmt_count_rate(row["sig_risk_checks"], 21)
        if model not in {"HistGB", "Gaussian HMM"}:
            sig = wrap_best(sig, row["sig_risk_rate"] == best_sig)
        body_lines.append(
            f"{method} & {macro} & {bull} & {bear} & {switch} & {duration} & {sig} \\\\"
        )

    tex = render_table(
        [
            r"\begin{table*}[t]",
            r"\caption{Condensed BTCUSDT benchmark. The rule-based proxy itself is omitted because it is a non-learned consistency reference. Best learned value per column is shown in bold. Transformer-2L is included only as a BTCUSDT stress-test reference rather than a symmetric multi-asset baseline. Semantic-significance checks are omitted for raw HistGB because its unsmoothed path remains too switch-heavy for the semantic-audit comparison, and for the Gaussian HMM because its mapped test states collapse to the neutral label.}",
            r"\label{tab:btc-compact}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.4pt}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Method & Macro-F1 & Bull F1 & Bear F1 & Switch/day & \shortstack{Avg.\\duration} & \shortstack{Sig.\\risk checks} \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    write_outputs("btc_benchmark", out_df, tex)


def build_semantic_consistency(df: pd.DataFrame) -> None:
    subset = df.loc[df["experiment_group"].isin(["benchmark_main", "cross_asset_main"])].copy()
    order = {
        "BTC": [
            "Proxy label",
            "LogReg",
            "LogReg + post-proc.",
            "HistGB + post-proc.",
            "Main GRU",
            "Vanilla LSTM",
            "TCN-96x4",
            "Transformer-2L",
        ],
        "ETH": [
            "Proxy label",
            "LogReg",
            "LogReg + post-proc.",
            "Main GRU (ETH-trained)",
            "BTC-trained GRU",
        ],
        "BNB": ["Proxy label", "LogReg", "LogReg + post-proc.", "BTC-trained GRU"],
        "SOL": ["Proxy label", "LogReg", "LogReg + post-proc.", "BTC-trained GRU"],
        "XRP": ["Proxy label", "LogReg", "LogReg + post-proc.", "BTC-trained GRU"],
    }
    display_map = {
        "Proxy label": "Proxy label",
        "LogReg": "Logistic regression",
        "LogReg + post-proc.": "Logreg + post-proc.",
        "HistGB + post-proc.": "HistGB + post-proc.",
        "Main GRU": "Main GRU",
        "Vanilla LSTM": "Vanilla LSTM",
        "TCN-96x4": "TCN-96x4",
        "Transformer-2L": "Transformer-2L",
        "Main GRU (ETH-trained)": "Main GRU (ETH-trained)",
        "BTC-trained GRU": "BTC-trained GRU",
    }
    asset_order = ["BTC", "ETH", "BNB", "SOL", "XRP"]
    rows = []
    body_lines: list[str] = []
    for asset in asset_order:
        asset_df = subset.loc[(subset["asset"] == asset) & (subset["model"].isin(order[asset]))].copy()
        asset_df["order"] = asset_df["model"].map({name: idx for idx, name in enumerate(order[asset])})
        asset_df = asset_df.sort_values("order")
        best_risk = asset_df["risk_order_rate"].max()
        best_return = asset_df["return_order_rate"].max()
        best_sig = asset_df["sig_risk_rate"].max()
        for idx, (_, row) in enumerate(asset_df.iterrows()):
            asset_label = asset if idx == 0 else asset
            risk = wrap_best(fmt_percent_from_rate(row["risk_order_rate"]), row["risk_order_rate"] == best_risk)
            ret = wrap_best(fmt_percent_from_rate(row["return_order_rate"]), row["return_order_rate"] == best_return)
            sig = wrap_best(fmt_count_rate(row["sig_risk_checks"], row["risk_order_checks"]), row["sig_risk_rate"] == best_sig)
            method = display_map[row["model"]]
            rows.append(
                {
                    "Asset": asset,
                    "Method": method,
                    "Risk-order consistency": fmt_percent_from_rate(row["risk_order_rate"]),
                    "Return-order consistency": fmt_percent_from_rate(row["return_order_rate"]),
                    "Sig. risk checks": fmt_count_rate(row["sig_risk_checks"], row["risk_order_checks"]),
                }
            )
            body_lines.append(f"{asset_label} & {method} & {risk} & {ret} & {sig} \\\\")
        if asset != asset_order[-1]:
            body_lines.append(r"\midrule")

    tex = render_table(
        [
            r"\begin{table*}[t]",
            r"\caption{Three-dimensional semantic evaluation. Risk-order consistency counts how often the expected order bear $>$ neutral $>$ bull appears across 21 risk checks; return-order consistency counts how often bull $>$ neutral $>$ bear appears across six return checks; significant risk checks count how often the expected risk order is also supported by Kruskal--Wallis and Holm-corrected one-sided Mann--Whitney tests under the nominal independence approximation. Within each asset block, the strongest value on each semantic axis is shown in bold. BTCUSDT block-bootstrap robustness is reported separately in Figure~\ref{fig:btc-bootstrap-robustness}.}",
            r"\label{tab:semantic-consistency}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Asset & Method & \shortstack{Risk-order\\consistency} & \shortstack{Return-order\\consistency} & \shortstack{Sig.\\risk checks} \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    write_outputs("semantic_consistency", pd.DataFrame(rows), tex)


def build_framework_increment(df: pd.DataFrame) -> None:
    subset = df.loc[df["experiment_group"] == "framework_increment"].copy()
    pool_order = ["Five-asset common", "BTC extended"]
    framework_order = ["C only", "C+T weighted", "C+T TOPSIS", "C+T+S weighted"]
    subset["pool_order"] = subset["candidate_pool"].map({name: idx for idx, name in enumerate(pool_order)})
    subset["framework_order"] = subset["framework"].map({name: idx for idx, name in enumerate(framework_order)})
    subset = subset.sort_values(["pool_order", "framework_order"])

    rows = []
    body_lines = []
    current_pool = None
    for _, row in subset.iterrows():
        if current_pool is not None and row["candidate_pool"] != current_pool:
            body_lines.append(r"\midrule")
        current_pool = row["candidate_pool"]
        disagreement = fmt_percent_from_rate(row["selection_disagreement_rate"], digits=0)
        semantic_regret = fmt_decimal(row["semantic_regret"], digits=3)
        macro_cost = fmt_decimal(row["macro_f1_cost"], digits=3)
        switch_cost = fmt_decimal(row["switch_cost"], digits=3)
        body_lines.append(
            f"{row['candidate_pool']} & {row['framework']} & {disagreement} & {semantic_regret} & {macro_cost} & {switch_cost} \\\\"
        )
        rows.append(
            {
                "Candidate pool": row["candidate_pool"],
                "Framework": row["framework"],
                "Disagreement": f"{100 * float(row['selection_disagreement_rate']):.0f}%",
                "Semantic regret": f"{float(row['semantic_regret']):.3f}",
                "Macro-F1 cost": f"{float(row['macro_f1_cost']):.3f}",
                "Switch cost": f"{float(row['switch_cost']):.3f}",
            }
        )

    tex = render_table(
        [
            r"\begin{table*}[t]",
            r"\caption{Increment over conventional multi-criteria evaluation. Disagreement is the fraction of assets in the pool where the framework selects a different method from the semantic protocol selector. Semantic regret is the selected method's lost nominal significant-risk-layering rate relative to the protocol selector; 0.2857 corresponds to 6/21 checks. Positive macro-F1 cost means the conventional framework selected a model with higher proxy-label macro-F1 than the protocol selector. Switch cost is the selected method's daily switching rate minus the protocol selector's rate.}",
            r"\label{tab:framework-increment}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Candidate pool & Framework & Disagreement & Semantic regret & Macro-F1 cost & Switch cost \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    write_outputs("framework_increment", pd.DataFrame(rows), tex)


def build_checkpoint_selection(df: pd.DataFrame) -> None:
    subset = df.loc[df["experiment_group"] == "checkpoint_selection"].copy()
    order = [
        "checkpoint_main_macro",
        "checkpoint_main_semantic",
        "checkpoint_roll2024_macro",
        "checkpoint_roll2024_semantic",
    ]
    epoch_map = {
        "checkpoint_main_macro": 12,
        "checkpoint_main_semantic": 9,
        "checkpoint_roll2024_macro": 12,
        "checkpoint_roll2024_semantic": 8,
    }
    window_map = {
        "checkpoint_main_macro": "Main",
        "checkpoint_main_semantic": "Main",
        "checkpoint_roll2024_macro": "Rolling 2024",
        "checkpoint_roll2024_semantic": "Rolling 2024",
    }
    selector_map = {
        "Validation macro-F1": "Validation macro-F1",
        "Semantic-aware": "Semantic-aware",
    }
    subset["order"] = subset["experiment_id"].map({name: idx for idx, name in enumerate(order)})
    subset = subset.sort_values("order")

    rows = []
    body_lines = []
    for _, row in subset.iterrows():
        if row["experiment_id"] == "checkpoint_roll2024_macro":
            body_lines.append(r"\midrule")
        sig_den = row["risk_order_checks"]
        sig = fmt_count_rate(row["sig_risk_checks"], sig_den)
        body_lines.append(
            f"{window_map[row['experiment_id']]} & {selector_map[row['framework']]} & {epoch_map[row['experiment_id']]} & {fmt_decimal(row['macro_f1'])} & {fmt_decimal(row['balanced_acc'])} & {sig} & {fmt_decimal(row['switch_day'])} \\\\"
        )
        rows.append(
            {
                "Window": window_map[row["experiment_id"]],
                "Selector": selector_map[row["framework"]],
                "Best epoch": epoch_map[row["experiment_id"]],
                "Test macro-F1": fmt_decimal(row["macro_f1"]),
                "Test bal. acc.": fmt_decimal(row["balanced_acc"]),
                "Sig. risk checks": sig,
                "Switch/day": fmt_decimal(row["switch_day"]),
            }
        )

    tex = render_table(
        [
            r"\begin{table*}[t]",
            r"\caption{Checkpoint-selection diagnostic for the BTCUSDT GRU. Main window uses the default split (train 2021--2023, validation 2024, test 2025-01 to 2026-04). Rolling window uses train 2021--2022, validation 2023, and test 2024. Significant risk checks report nominal significant-risk layering counts under the same audit protocol used elsewhere in this repository.}",
            r"\label{tab:checkpoint-selection}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"Window & Selector & Best epoch & Test macro-F1 & Test bal. acc. & Sig. risk checks & Switch/day \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    write_outputs("checkpoint_selection", pd.DataFrame(rows), tex)


def build_drawdown_rule_deconfounding(df: pd.DataFrame) -> None:
    subset = df.loc[
        (df["experiment_group"] == "drawdown_deconfounding")
        & (df["label_variant"].isin(["no_drawdown_rule"]))
    ].copy()
    defaults = df.loc[
        (df["experiment_group"] == "benchmark_main")
        & (df["asset"] == "BTC")
        & (df["model"].isin(["LogReg", "Main GRU"]))
    ][["model", "macro_f1", "sig_risk_checks", "risk_order_checks"]].copy()
    defaults["label_variant"] = "default"
    combined = pd.concat([defaults, subset[["model", "macro_f1", "sig_risk_checks", "risk_order_checks", "label_variant"]]], ignore_index=True)

    order = [
        ("default", "LogReg"),
        ("default", "Main GRU"),
        ("no_drawdown_rule", "LogReg"),
        ("no_drawdown_rule", "Main GRU"),
    ]
    weak_label_map = {
        "default": "Default",
        "no_drawdown_rule": "No drawdown filter",
    }
    model_map = {"LogReg": "Logistic regression", "Main GRU": "Main GRU"}
    combined["order"] = combined.apply(lambda r: order.index((r["label_variant"], r["model"])), axis=1)
    combined = combined.sort_values("order")

    rows = []
    body_lines = []
    for _, row in combined.iterrows():
        body_lines.append(
            f"{weak_label_map[row['label_variant']]} & {model_map[row['model']]} & {fmt_decimal(row['macro_f1'])} & {fmt_count_rate(row['sig_risk_checks'], row['risk_order_checks'])} \\\\"
        )
        rows.append(
            {
                "Weak-label rule": weak_label_map[row["label_variant"]],
                "Model": model_map[row["model"]],
                "Macro-F1": fmt_decimal(row["macro_f1"]),
                "Sig. risk checks": fmt_count_rate(row["sig_risk_checks"], row["risk_order_checks"]),
            }
        )

    tex = render_table(
        [
            r"\begin{table*}[t]",
            r"\caption{BTCUSDT label-side drawdown-rule deconfounding. Removing the drawdown thresholds from the weak-label rule improves proxy-label fit for both learned models, but it substantially weakens statistically significant risk layering.}",
            r"\label{tab:drawdown-rule-deconfounding}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{llcc}",
            r"\toprule",
            r"Weak-label rule & Model & Macro-F1 & Sig. risk checks \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    write_outputs("drawdown_rule_deconfounding", pd.DataFrame(rows), tex)


def main() -> None:
    df = load_manifest()
    build_btc_benchmark(df)
    build_semantic_consistency(df)
    build_framework_increment(df)
    build_checkpoint_selection(df)
    build_drawdown_rule_deconfounding(df)


if __name__ == "__main__":
    main()
