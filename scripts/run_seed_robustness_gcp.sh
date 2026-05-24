#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-all}"
EPOCHS="${EPOCHS:-12}"
SEEDS="${SEEDS:-7 42 123 2024 3407}"

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

case "$MODEL" in
  gru)
    python -u -m src.evaluation.run_seed_robustness_5seed --models gru --seeds ${SEEDS} --epochs "${EPOCHS}" --skip_existing
    ;;
  tcn96)
    python -u -m src.evaluation.run_seed_robustness_5seed --models tcn96 --seeds ${SEEDS} --epochs "${EPOCHS}" --skip_existing
    ;;
  transformer)
    python -u -m src.evaluation.run_seed_robustness_5seed --models transformer --seeds ${SEEDS} --epochs "${EPOCHS}" --skip_existing
    ;;
  all)
    python -u -m src.evaluation.run_seed_robustness_5seed --models gru tcn96 transformer --seeds ${SEEDS} --epochs "${EPOCHS}" --skip_existing
    ;;
  *)
    echo "Usage: $0 [gru|tcn96|transformer|all]"
    exit 1
    ;;
esac
