#!/usr/bin/env python3
"""Create strictly filtered weak data from five unanimous teacher runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research_protocol import load_protocol, sha256_file, write_json
from src.utils import CEFR_LABELS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-config", type=Path, required=True)
    parser.add_argument("--gold-train", type=Path, required=True)
    parser.add_argument("--teacher-predictions", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.research_config)
    policy = protocol["derived_supervision"]
    frames = [pd.read_csv(path) for path in args.teacher_predictions]
    forbidden = {"cefr", "gold_cefr"}
    if any(forbidden & set(frame.columns) for frame in frames):
        raise ValueError("Derived supervision input must be external unlabeled material")
    identity = ["source_id", "lemma", "pos"]
    if any(not set(identity).issubset(frame.columns) for frame in frames):
        raise ValueError(f"Every teacher file must include {identity}")
    if any(frame["source_id"].duplicated().any() for frame in frames):
        raise ValueError("External unlabeled source IDs must be unique")
    if any(not frames[0][identity].equals(frame[identity]) for frame in frames[1:]):
        raise ValueError("Teacher files do not describe identical ordered examples")
    if any("seed" not in frame or frame["seed"].nunique() != 1 for frame in frames):
        raise ValueError("Every teacher file must identify exactly one seed")
    expected_seeds = set(protocol["evaluation"]["confirmation_seeds"])
    actual_seeds = {int(frame["seed"].iloc[0]) for frame in frames}
    if actual_seeds != expected_seeds:
        raise ValueError("Teacher files must use exactly the five locked confirmation seeds")
    predictions = pd.concat(
        [frame["pred_cefr"].rename(f"pred_{index}") for index, frame in enumerate(frames)],
        axis=1,
    )
    unanimous = predictions.nunique(axis=1) == 1
    confidence = pd.concat(
        [
            frame.apply(lambda row: row[f"prob_{row['pred_cefr']}"], axis=1).rename(f"p_{index}")
            for index, frame in enumerate(frames)
        ],
        axis=1,
    )
    keep = unanimous & (confidence.min(axis=1) >= float(policy["minimum_probability"]))
    metadata_columns = [
        column for column in frames[0]
        if column not in {"pred_cefr", "seed"} and not column.startswith("prob_")
    ]
    selected = frames[0].loc[keep, metadata_columns].copy()
    selected["cefr"] = predictions.loc[keep, "pred_0"]
    selected["teacher_min_probability"] = confidence.loc[keep].min(axis=1)
    selected["weak_data"] = True
    maximum = int(len(pd.read_csv(args.gold_train)) * policy["maximum_relative_to_gold_train"])
    available = selected["cefr"].value_counts().reindex(CEFR_LABELS, fill_value=0)
    quota = min(maximum // len(CEFR_LABELS), int(available.min()))
    selected = (
        selected.sort_values("teacher_min_probability", ascending=False)
        .groupby("cefr", group_keys=False)
        .head(quota)
        .sort_values(identity)
        .reset_index(drop=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    manifest = {
        "series_id": protocol["series"]["id"],
        "protocol_hash": protocol_hash,
        "teacher_files": {
            str(path): sha256_file(path) for path in args.teacher_predictions
        },
        "gold_train_size": len(pd.read_csv(args.gold_train)),
        "candidate_count": int(keep.sum()),
        "selected_count": len(selected),
        "class_counts": selected["cefr"].value_counts().reindex(CEFR_LABELS, fill_value=0).to_dict(),
        "output_sha256": sha256_file(args.output),
        "weak_data": True,
        "requires_gold_only_ablation": policy["require_gold_only_ablation"],
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
