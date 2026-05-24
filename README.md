# Weakly Supervised Crypto Risk-State Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](./requirements.txt)
[![Data](https://img.shields.io/badge/Data-Binance%20Spot%201h-F3BA2F?style=flat-square&logo=binance&logoColor=black)](./data/README.md)
[![Last Commit](https://img.shields.io/github/last-commit/llljjjwww333/weakly-supervised-crypto-risk-state-learning?style=flat-square)](https://github.com/llljjjwww333/weakly-supervised-crypto-risk-state-learning)
[![Repo Size](https://img.shields.io/github/repo-size/llljjjwww333/weakly-supervised-crypto-risk-state-learning?style=flat-square)](https://github.com/llljjjwww333/weakly-supervised-crypto-risk-state-learning)

Code, configuration, scripts, and figures for weakly supervised crypto risk-state learning experiments.

This repository focuses on the reproducible core of the project:

- data download and validation
- feature engineering and weak-label construction
- baseline and temporal model training
- semantic, stability, and framework-comparison evaluation

It intentionally does **not** include raw market data, processed parquet files, trained checkpoints, or large experiment outputs.

![Workflow Overview](figures/main/fig1.svg)

## Repository Layout

```text
configs/        experiment and asset configuration
data/           notes on the expected local data layout
experiments/    note on omitted outputs and how to regenerate them
figures/main/   summary figures and diagnostic plots
scripts/        helper scripts for reruns and figure regeneration
src/            data, feature, model, and evaluation code
requirements.txt
```

## Environment

Recommended:

- Python 3.10+
- `pip install -r requirements.txt`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and validate raw Binance data

```bash
python -m src.data.download_binance_klines --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT --interval 1h --start 2021-01-01 --end 2026-04-20 --out_dir data/raw/spot/1h
python -m src.data.validate_raw_data --input_dir data/raw/spot/1h --interval 1h --report_path data/metadata/raw_validation_report.csv
```

### 3. Build features, weak labels, and windows

```bash
python -m src.features.build_base_table --input_dir data/raw/spot/1h --output_dir data/interim/spot/1h
python -m src.features.make_features --input_dir data/interim/spot/1h --output_dir data/processed/features/1h
python -m src.features.build_labels --input_dir data/processed/features/1h --output_dir data/labels
python -m src.features.build_windows --input_dir data/processed/features/1h --output_dir data/processed/windows/1h --window 48
```

### 4. Run baselines or the main temporal model

```bash
python -m src.models.baselines.run_logreg --input_path data/labels/BTCUSDT_labels.parquet --output_dir experiments/baselines/logreg_btc
python -m src.models.baselines.run_hmm --input_path data/labels/BTCUSDT_labels.parquet --output_dir experiments/baselines/hmm_btc --n_states 3
python -m src.models.main.train_main --input_path data/processed/windows/1h/BTCUSDT_win48.parquet --output_dir experiments/main/gru_btc
```

### 5. Evaluate semantic and stability behavior

```bash
python -m src.evaluation.evaluate_classification --input_path experiments/main/gru_btc/test_predictions.parquet --output_dir experiments/main/gru_btc/eval
python -m src.evaluation.evaluate_stability --input_path experiments/main/gru_btc/test_predictions.parquet --output_path experiments/main/gru_btc/stability.csv
python -m src.evaluation.evaluate_risk_state_semantics
python -m src.evaluation.run_bootstrap_block_sensitivity
```

## What Is Not Included

- raw Binance downloads
- processed parquet tables
- trained model checkpoints
- large experiment output trees

The repository is intended to stay focused on the code and experiment assets needed to reproduce the workflow.
