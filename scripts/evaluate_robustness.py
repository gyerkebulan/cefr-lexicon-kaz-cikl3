#!/usr/bin/env python3
"""Generate deterministic robustness inputs or score robustness predictions."""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_metrics
from src.research_protocol import load_protocol, sha256_file, write_json
from src.utils import CEFR_LABELS


def controlled_noise(text: str, seed: int) -> str:
    candidates = [index for index, character in enumerate(text) if character.isalpha()]
    if not candidates:
        return text
    index = random.Random(seed).choice(candidates)
    return text[:index] + text[index + 1:]


def suffixes() -> list[str]:
    config = yaml.safe_load((PROJECT_ROOT / "config/kaz_suffixes.yaml").read_text(encoding="utf-8"))
    found = []

    def collect(value):
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            if "suffix" in value:
                found.append(value["suffix"])
            else:
                for item in value.values():
                    collect(item)

    collect(config["derivational"])
    collect(config["inflectional"])
    return sorted(set(found), key=lambda suffix: (len(suffix), suffix))


def generate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    variants = []
    endings = suffixes() or ["лар"]
    for row_index, row in frame.iterrows():
        sentence, target = str(row["sentence"]), str(row["target"])
        cases = {
            "clean": (sentence, target, False),
            "unicode_normalization": (
                unicodedata.normalize("NFD", sentence),
                unicodedata.normalize("NFD", target),
                False,
            ),
            "case": (sentence.swapcase(), target.swapcase(), False),
            "punctuation_whitespace": (f"  {sentence}  !", target, False),
            "controlled_character_noise": (controlled_noise(sentence, row_index), target, False),
            "target_repetition": (f"{sentence} {target}", target, False),
            "target_truncation": (("Бейтарап мәтін. " * 160) + target, target, False),
            "context_removal": (target, target, False),
            "context_distraction": (f"{sentence} Бұл қосымша бейтарап сөйлем.", target, False),
            "weak_inflected_surface_forms": (
                f"{sentence} {target}{endings[row_index % len(endings)]}",
                f"{target}{endings[row_index % len(endings)]}",
                True,
            ),
        }
        for scenario, (variant_sentence, variant_target, weak) in cases.items():
            record = row.to_dict()
            record.update({
                "scenario": scenario,
                "sentence": variant_sentence,
                "target": variant_target,
                "weak_data": weak,
                "source_row_index": row_index,
            })
            variants.append(record)
    return pd.DataFrame(variants)


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [f"prob_{label}" for label in CEFR_LABELS]
    return columns if all(column in frame for column in columns) else []


def ensemble_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if len(frames) == 1:
        return frames[0]
    keys = ["source_row_id", "scenario"]
    probability_names = [f"prob_{label}" for label in CEFR_LABELS]
    required = {*keys, "gold_cefr", *probability_names}
    if any(not required.issubset(frame.columns) for frame in frames):
        raise ValueError("Ensemble robustness predictions require aligned IDs and probabilities")
    reference = frames[0][keys + ["gold_cefr"]]
    if any(not reference.equals(frame[keys + ["gold_cefr"]]) for frame in frames[1:]):
        raise ValueError("Ensemble robustness predictions are not aligned")
    output = frames[0].copy()
    probabilities = np.mean(
        [frame[probability_names].to_numpy(dtype=float) for frame in frames], axis=0
    )
    output[probability_names] = probabilities
    output["pred_cefr"] = [CEFR_LABELS[index] for index in probabilities.argmax(axis=1)]
    return output


def metric_record(frame: pd.DataFrame) -> dict:
    columns = probability_columns(frame)
    return compute_metrics(
        frame["gold_cefr"].tolist(),
        frame["pred_cefr"].tolist(),
        probabilities=frame[columns].to_numpy() if columns else None,
    )


def score_predictions(frame: pd.DataFrame, minimum: int) -> dict:
    required = {"gold_cefr", "pred_cefr"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Predictions missing: {sorted(required - set(frame.columns))}")
    invalid = ~frame["gold_cefr"].isin(CEFR_LABELS) | ~frame["pred_cefr"].isin(CEFR_LABELS)
    if invalid.any():
        raise ValueError(f"Predictions contain {int(invalid.sum())} invalid CEFR labels")
    if "scenario" not in frame:
        frame = frame.assign(scenario="clean")
    report = {
        "overall": metric_record(frame),
        "perturbation_macro_f1": metric_record(frame[frame["scenario"] != "clean"])["macro_f1"],
        "scenarios": {},
        "slices": {},
        "support_unit": "unique_source_row_id" if "source_row_id" in frame else "rows",
    }
    for scenario, subset in frame.groupby("scenario"):
        support = subset["source_row_id"].nunique() if "source_row_id" in subset else len(subset)
        report["scenarios"][str(scenario)] = {
            "support": int(support),
            "supported": support >= minimum,
            "metrics": metric_record(subset),
        }
    slice_columns = {
        "cefr": "gold_cefr",
        "pos": "pos",
        "context_availability": "context_available",
    }
    enriched = frame.copy()
    if "freq" in enriched:
        frequency = pd.to_numeric(enriched["freq"], errors="coerce").fillna(0)
        omitted = report.setdefault("omitted_slices", {})
        if frequency.nunique() > 1:
            enriched["frequency_slice"] = pd.cut(
                frequency, [-np.inf, 0, 10, 100, 1000, np.inf],
                labels=["zero", "1-10", "11-100", "101-1000", "1000+"],
            ).astype(str)
            slice_columns["frequency"] = "frequency_slice"
        else:
            omitted["frequency"] = "constant legacy freq"
        if (frequency <= 0).nunique() > 1:
            enriched["oov_slice"] = np.where(frequency <= 0, "oov", "seen")
            slice_columns["oov"] = "oov_slice"
        else:
            omitted["oov"] = "constant legacy freq"
    if "lemma" in enriched:
        enriched["lemma_length_slice"] = pd.cut(
            enriched["lemma"].astype(str).str.len(), [0, 4, 7, 11, np.inf],
            labels=["1-4", "5-7", "8-11", "12+"],
        ).astype(str)
        slice_columns["lemma_length"] = "lemma_length_slice"
    morphology_column = next(
        (
            column for column in ("n_morphemes", "morph_complexity", "suffix_count")
            if column in enriched
        ),
        None,
    )
    if morphology_column is None and "lemma" in enriched:
        endings = suffixes()
        enriched["suffix_count"] = enriched["lemma"].astype(str).map(
            lambda lemma: sum(lemma.endswith(suffix) for suffix in endings)
        )
        morphology_column = "suffix_count"
    if morphology_column:
        enriched["morphology_slice"] = pd.cut(
            pd.to_numeric(enriched[morphology_column], errors="coerce").fillna(0),
            [-np.inf, 1, 2, 3, np.inf],
            labels=["simple", "two", "three", "four_plus"],
        ).astype(str)
        slice_columns["morphology"] = "morphology_slice"
    for slice_name, column in slice_columns.items():
        if column not in enriched:
            continue
        report["slices"][slice_name] = {
            str(value): {
                "support": int(
                    subset["source_row_id"].nunique()
                    if "source_row_id" in subset else len(subset)
                ),
                "supported": (
                    subset["source_row_id"].nunique()
                    if "source_row_id" in subset else len(subset)
                ) >= minimum,
                "metrics": metric_record(subset),
            }
            for value, subset in enriched.groupby(column, dropna=False)
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path)
    mode.add_argument("--predictions", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.research_config)
    if args.input:
        output = generate_inputs(pd.read_csv(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(json.dumps({"rows": len(output), "scenarios": output["scenario"].nunique()}))
        return 0
    minimum = protocol["evaluation"]["guardrail_floors"]["supported_slice_minimum_examples"]
    report = score_predictions(ensemble_predictions(args.predictions), minimum)
    report.update({
        "series_id": protocol["series"]["id"],
        "protocol_hash": protocol_hash,
        "prediction_count": len(args.predictions),
        "prediction_checksums": {
            str(path.resolve()): sha256_file(path) for path in args.predictions
        },
    })
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
