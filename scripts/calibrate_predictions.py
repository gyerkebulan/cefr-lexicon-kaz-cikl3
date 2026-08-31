#!/usr/bin/env python3
"""Fit one temperature on out-of-fold probabilities and emit calibrated output."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_metrics
from src.research_protocol import (
    load_protocol, sha256_file, verify_dataset_manifest, write_json,
)
from src.utils import CEFR_LABELS


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))

    def loss(log_temperature: float) -> float:
        calibrated = softmax(logits / math.exp(log_temperature))
        return float(-np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1)).mean())

    left, right = math.log(0.05), math.log(10.0)
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = loss(x1), loss(x2)
    for _ in range(80):
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = loss(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = loss(x2)
    return math.exp((left + right) / 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-config", type=Path, required=True)
    parser.add_argument("--proposal-id", required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.research_config)
    frame = pd.read_csv(args.oof_predictions)
    required = {"source_row_id", "fold", "gold_cefr", *[f"prob_{label}" for label in CEFR_LABELS]}
    if not required.issubset(frame.columns):
        raise ValueError(f"OOF predictions missing: {sorted(required - set(frame.columns))}")
    folds = int(protocol["evaluation"]["calibration_folds"])
    if frame["source_row_id"].duplicated().any() or frame["fold"].nunique() != folds:
        raise ValueError(
            f"Calibration requires one out-of-fold prediction per row from exactly {folds} folds"
        )
    dataset_dir = PROJECT_ROOT / protocol["dataset"]["output_dir"]
    verify_dataset_manifest(dataset_dir, protocol_hash=protocol_hash, splits=("train",))
    train = pd.read_csv(dataset_dir / "train.csv")
    if set(frame["source_row_id"]) != set(train["source_row_id"]):
        raise ValueError("OOF predictions must cover the frozen gold training split exactly")
    gold = train.set_index("source_row_id")["cefr"]
    if any(
        gold.loc[source_id] != label
        for source_id, label in zip(frame["source_row_id"], frame["gold_cefr"])
    ):
        raise ValueError("OOF prediction labels do not match the frozen training split")
    columns = [f"prob_{label}" for label in CEFR_LABELS]
    probabilities = frame[columns].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("OOF probabilities must be finite, non-negative, and sum to one")
    label_indices = frame["gold_cefr"].map({label: i for i, label in enumerate(CEFR_LABELS)}).to_numpy()
    temperature = fit_temperature(probabilities, label_indices)
    calibrated = softmax(np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature)
    output = frame.copy()
    output[columns] = calibrated
    output["pred_cefr"] = [CEFR_LABELS[index] for index in calibrated.argmax(axis=1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    before = compute_metrics(
        frame["gold_cefr"].tolist(),
        [CEFR_LABELS[index] for index in probabilities.argmax(axis=1)],
        probabilities=probabilities,
    )
    after = compute_metrics(
        output["gold_cefr"].tolist(), output["pred_cefr"].tolist(), probabilities=calibrated
    )
    manifest = {
        "series_id": protocol["series"]["id"],
        "proposal_id": args.proposal_id,
        "protocol_hash": protocol_hash,
        "fold_count": folds,
        "source_sha256": sha256_file(args.oof_predictions),
        "temperature": temperature,
        "before": before,
        "after": after,
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
