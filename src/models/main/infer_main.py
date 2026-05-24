from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from src.models.main.bull_bear_model import ModelConfig, build_model
from src.utils.io import ensure_parent


LABEL_ID_TO_NAME = {0: "bear", 1: "neutral", 2: "bull"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the trained bull/bear temporal model.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_path", required=True, help="Window parquet built by build_windows.py")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()


def stack_batch_features(values: list[list[list[float]]]) -> np.ndarray:
    return np.stack([np.asarray(sequence, dtype=np.float32) for sequence in values], axis=0)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    parquet_file = pq.ParquetFile(args.input_path)
    output_batches: list[pd.DataFrame] = []
    columns = ["open_time", "symbol", "proxy_label", "proxy_label_id", "features"]

    with torch.inference_mode():
        for batch in parquet_file.iter_batches(batch_size=args.batch_size, columns=columns):
            data = batch.to_pydict()
            features = torch.from_numpy(stack_batch_features(data.pop("features"))).to(device)
            logits = model(features)
            proba = F.softmax(logits, dim=-1).cpu().numpy()
            pred = np.argmax(proba, axis=1)

            out = pd.DataFrame(
                {
                    "open_time": pd.to_datetime(data["open_time"], utc=True),
                    "symbol": data["symbol"],
                    "proxy_label": data["proxy_label"],
                    "proxy_label_id": data["proxy_label_id"],
                    "pred_label_id": pred,
                    "pred_label": [LABEL_ID_TO_NAME[int(label)] for label in pred],
                    "proba_bear": proba[:, 0],
                    "proba_neutral": proba[:, 1],
                    "proba_bull": proba[:, 2],
                }
            )
            output_batches.append(out)

    out = pd.concat(output_batches, ignore_index=True)
    output_path = ensure_parent(args.output_path)
    out.to_parquet(output_path, index=False)
    print(f"[saved] {output_path} rows={len(out)}")


if __name__ == "__main__":
    main()
