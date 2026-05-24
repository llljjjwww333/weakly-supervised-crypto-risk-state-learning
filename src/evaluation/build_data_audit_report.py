from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.utils.io import ensure_dir


LABEL_DIR = Path("data/labels_improved/default")
WINDOW_DIR = Path("data/processed/windows/1h")
OUTPUT_DIR = ensure_dir("experiments/summary/data_audit")
REPORT_PATH = Path("data_audit_report.md")

TRAIN_END = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")
VAL_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")


def iter_label_paths() -> list[Path]:
    return sorted(
        p for p in LABEL_DIR.glob("*USDT_labels.parquet") if not p.name.startswith("._")
    )


def infer_split(open_time: pd.Series) -> pd.Series:
    return pd.Series(
        pd.Categorical(
            [
                "train" if ts <= TRAIN_END else "val" if ts <= VAL_END else "test"
                for ts in open_time
            ],
            categories=["train", "val", "test"],
            ordered=True,
        ),
        index=open_time.index,
    )


def abnormal_ohlc_mask(df: pd.DataFrame) -> pd.Series:
    max_oc = df[["open", "close"]].max(axis=1)
    min_oc = df[["open", "close"]].min(axis=1)
    return (
        (df["high"] < max_oc)
        | (df["high"] < df["low"])
        | (df["low"] > min_oc)
        | (df["low"] > df["high"])
    )


def build_data_quality_summary(label_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in label_paths:
        asset = path.name.replace("_labels.parquet", "").replace("USDT", "")
        df = pd.read_parquet(path, columns=["open_time", "open", "high", "low", "close"])
        df = df.sort_values("open_time").reset_index(drop=True)
        start = df["open_time"].min()
        end = df["open_time"].max()
        expected_bars = int(((end - start) / pd.Timedelta(hours=1)) + 1)
        actual_bars = int(len(df))
        duplicate_bars = int(df["open_time"].duplicated().sum())
        unique_bars = int(df["open_time"].nunique())
        missing_bars = int(expected_bars - unique_bars)
        abnormal_bars = int(abnormal_ohlc_mask(df).sum())
        rows.append(
            {
                "asset": asset,
                "symbol": f"{asset}USDT",
                "start_time_utc": start,
                "end_time_utc": end,
                "expected_bars": expected_bars,
                "actual_bars": actual_bars,
                "unique_bars": unique_bars,
                "missing_bars": missing_bars,
                "duplicate_bars": duplicate_bars,
                "abnormal_ohlc_bars": abnormal_bars,
            }
        )
    return pd.DataFrame(rows).sort_values("asset")


def build_split_and_label_summary(label_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in label_paths:
        asset = path.name.replace("_labels.parquet", "").replace("USDT", "")
        df = pd.read_parquet(path, columns=["open_time", "proxy_label", "proxy_label_id"])
        df["split"] = infer_split(df["open_time"])
        for split, split_df in df.groupby("split", observed=True):
            counts = split_df["proxy_label"].value_counts()
            rows.append(
                {
                    "asset": asset,
                    "split": split,
                    "rows": int(len(split_df)),
                    "bull_count": int(counts.get("bull", 0)),
                    "neutral_count": int(counts.get("neutral", 0)),
                    "bear_count": int(counts.get("bear", 0)),
                    "bull_share": float((split_df["proxy_label"] == "bull").mean()),
                    "neutral_share": float((split_df["proxy_label"] == "neutral").mean()),
                    "bear_share": float((split_df["proxy_label"] == "bear").mean()),
                }
            )
    split_order = {"train": 0, "val": 1, "test": 2}
    out = pd.DataFrame(rows)
    out["split_order"] = out["split"].map(split_order)
    return out.sort_values(["asset", "split_order"]).drop(columns="split_order")


def build_alignment_samples(label_paths: list[Path], per_asset: int = 2) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in label_paths:
        asset = path.name.replace("_labels.parquet", "").replace("USDT", "")
        window_path = WINDOW_DIR / f"{asset}USDT_win48.parquet"
        table = pq.read_table(
            window_path,
            columns=["open_time", "window_start", "window_end", "proxy_label"],
        ).to_pandas()
        sample_idx = [0, max(0, len(table) // 2)]
        if per_asset > 2:
            extra = pd.Series(range(len(table))).sample(per_asset - 2, random_state=42).tolist()
            sample_idx.extend(extra)
        for idx in sorted(set(sample_idx))[:per_asset]:
            row = table.iloc[idx]
            label_time = pd.Timestamp(row["open_time"])
            rows.append(
                {
                    "asset": asset,
                    "symbol": f"{asset}USDT",
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "label_time": label_time,
                    "future_start": label_time + pd.Timedelta(hours=1),
                    "future_end": label_time + pd.Timedelta(hours=24),
                    "proxy_label": row["proxy_label"],
                    "window_length_hours": int(
                        ((pd.Timestamp(row["window_end"]) - pd.Timestamp(row["window_start"])) / pd.Timedelta(hours=1))
                        + 1
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["asset", "label_time"]).reset_index(drop=True)


def build_report(
    quality_df: pd.DataFrame, split_df: pd.DataFrame, alignment_df: pd.DataFrame
) -> str:
    leakage_lines = [
        "- 历史窗口特征只覆盖 `t` 及以前。`*_win48.parquet` 的 `window_end` 与 `open_time` 相同，说明 48h 窗口以当前标签时点结束，不向未来取值。",
        "- `future_return_*`、`future_vol_24` 等未来指标存在于 `data/labels_improved/default/*USDT_labels.parquet`，用于语义评估与弱标签审计；它们不出现在 `data/processed/windows/1h/*_win48.parquet` 的模型输入窗口中。",
        "- 线性与 HMM 基线在各自训练脚本里都采用 `fit(train)` 后再 `transform(val/test)` 的预处理流程：`run_logreg.py` 使用 `SimpleImputer + StandardScaler + LogisticRegression` 管线，`run_hmm.py` 和 `run_histgb.py` 也只在训练段拟合预处理器。",
        "- 主时序模型训练脚本 `src/models/main/train_main.py` 直接读取预先构造好的历史窗口，不在全样本上额外拟合 scaler，因此没有跨 split 的归一化泄漏路径。",
    ]

    quality_table = quality_df.to_markdown(index=False)
    split_table = split_df.to_markdown(index=False)
    alignment_table = alignment_df.to_markdown(index=False)

    return "\n".join(
        [
            "# Data Audit Report",
            "",
            "## Scope",
            "- Label source: `data/labels_improved/default/*USDT_labels.parquet`",
            "- Window source: `data/processed/windows/1h/*_win48.parquet`",
            "- Splits audited here follow the paper default chronology: train up to 2023-12-31, validation 2024-01-01 to 2024-12-31, test 2025-01-01 onward.",
            "",
            "## Bar Quality Summary",
            quality_table,
            "",
            "## Split And Label Summary",
            split_table,
            "",
            "## Window Alignment Samples",
            alignment_table,
            "",
            "## Leakage Audit",
            *leakage_lines,
        ]
    )


def main() -> None:
    label_paths = iter_label_paths()
    quality_df = build_data_quality_summary(label_paths)
    split_df = build_split_and_label_summary(label_paths)
    alignment_df = build_alignment_samples(label_paths, per_asset=2)

    quality_df.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
    split_df.to_csv(OUTPUT_DIR / "split_and_label_summary.csv", index=False)
    alignment_df.to_csv(OUTPUT_DIR / "alignment_samples.csv", index=False)
    REPORT_PATH.write_text(build_report(quality_df, split_df, alignment_df), encoding="utf-8")

    print(f"[wrote] {OUTPUT_DIR / 'data_quality_summary.csv'}")
    print(f"[wrote] {OUTPUT_DIR / 'split_and_label_summary.csv'}")
    print(f"[wrote] {OUTPUT_DIR / 'alignment_samples.csv'}")
    print(f"[wrote] {REPORT_PATH}")


if __name__ == "__main__":
    main()
