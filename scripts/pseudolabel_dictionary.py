#!/usr/bin/env python3
"""Conservative CEFR pseudolabeling for a supplied dictionary."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VOWELS = set("аәеёиіоөұүуыэюя")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).strip().lower()


def feature_frame(words, input_frequency, surface_frequency, lemma_frequency):
    frame = pd.DataFrame({
        "normalized_lemma": [normalize(word) for word in words],
        "input_frequency": np.asarray(input_frequency, dtype=float),
    })
    frame["log_input_frequency"] = np.log10(frame["input_frequency"].clip(lower=0) + 1)
    frame["log_surface_frequency"] = frame["normalized_lemma"].map(
        lambda word: np.log10(float(surface_frequency.get(word, 0)) + 1)
    )
    frame["log_lemma_frequency"] = frame["normalized_lemma"].map(
        lambda word: np.log10(float(lemma_frequency.get(word, 0)) + 1)
    )
    frame["length"] = frame["normalized_lemma"].str.len().astype(float)
    frame["vowel_count"] = frame["normalized_lemma"].map(
        lambda word: float(sum(character in VOWELS for character in word))
    )
    return frame


def build_model(name: str, config: dict) -> Pipeline:
    numeric = config["models"]["frequency_only"]["numeric_features"]
    transformers = []
    if name in {"morphology_only", "combined"}:
        transformers.append((
            "char",
            TfidfVectorizer(
                analyzer="char", ngram_range=(2, 5), min_df=2,
                max_features=50_000, sublinear_tf=True, lowercase=False,
                dtype=np.float32,
            ),
            "normalized_lemma",
        ))
    if name in {"frequency_only", "combined"}:
        transformers.append(("numeric", StandardScaler(), numeric))
    features = ColumnTransformer(transformers, sparse_threshold=0.3)
    classifier = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=3000,
        solver="lbfgs", random_state=int(config["seed"]),
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def metrics(labels, probabilities, levels, threshold, margin_threshold):
    truth = np.array([levels.index(label) for label in labels], dtype=int)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    selected = (confidence >= threshold) & (margin >= margin_threshold)
    bins = np.linspace(0, 1, 11)
    correct = predicted == truth
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    per_class = f1_score(truth, predicted, labels=range(len(levels)), average=None, zero_division=0)
    high_accuracy = float(correct[selected].mean()) if selected.any() else 0.0
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "per_class_f1": dict(zip(levels, map(float, per_class))),
        "ordinal_mae": float(np.abs(truth - predicted).mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(truth, predicted, weights="quadratic")),
        "expected_calibration_error": float(ece),
        "high_confidence_accuracy": high_accuracy,
        "high_confidence_coverage": float(selected.mean()),
        "high_confidence_count": int(selected.sum()),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=range(len(levels))).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--proposal-id", required=True)
    parser.add_argument("--extracted-target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    levels = config["levels"]
    sources = config["sources"]
    for source in sources.values():
        path = Path(source["path"])
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    target = pd.read_csv(args.extracted_target, keep_default_na=False)
    required = {"id", "source_term", "lemma", "cefr_level", "corpus_frequency"}
    if missing := required - set(target.columns):
        raise ValueError(f"Target is missing columns: {sorted(missing)}")
    if len(target) != int(sources["target_xlsx"]["rows"]):
        raise ValueError("Extracted target row count differs from the locked workbook")
    target["normalized_lemma"] = target["lemma"].map(normalize)
    target["input_frequency"] = pd.to_numeric(
        target["corpus_frequency"], errors="coerce"
    ).fillna(0).clip(lower=0)
    if target["normalized_lemma"].eq("").any():
        raise ValueError("Target contains empty normalized lemmas")

    known = pd.read_csv(sources["known_lexicon"]["path"], keep_default_na=False)
    known["normalized_lemma"] = known["lemma"].map(normalize)
    known = known[known["cefr"].isin(levels) & known["normalized_lemma"].ne("")]
    label_sets = known.groupby("normalized_lemma")["cefr"].agg(
        lambda values: sorted(set(values))
    )
    exact_labels = {
        lemma: labels[0] for lemma, labels in label_sets.items() if len(labels) == 1
    }
    conflicts = {
        lemma: labels for lemma, labels in label_sets.items() if len(labels) > 1
    }
    gold = (
        known[known["normalized_lemma"].isin(exact_labels)]
        .drop_duplicates("normalized_lemma")
        .sort_values("normalized_lemma")
        .reset_index(drop=True)
    )

    surface = json.loads(Path(sources["surface_frequency"]["path"]).read_text(encoding="utf-8"))
    lemma_frequency = json.loads(Path(sources["lemma_frequency"]["path"]).read_text(encoding="utf-8"))
    target_input_frequency = target.groupby("normalized_lemma")["input_frequency"].max().to_dict()
    gold_input_frequency = [
        target_input_frequency.get(word, surface.get(word, 0))
        for word in gold["normalized_lemma"]
    ]
    gold_features = feature_frame(
        gold["normalized_lemma"], gold_input_frequency, surface, lemma_frequency
    )
    labels = gold["cefr"].to_numpy()

    fold = StratifiedKFold(
        n_splits=int(config["validation"]["folds"]), shuffle=True,
        random_state=int(config["seed"]),
    )
    threshold = float(config["pseudo_label_policy"]["minimum_probability"])
    margin_threshold = float(config["pseudo_label_policy"]["minimum_top_two_margin"])
    all_metrics, all_oof = {}, {}
    for name in config["models"]:
        probabilities = np.zeros((len(gold), len(levels)), dtype=float)
        for train_index, validation_index in fold.split(gold_features, labels):
            model = build_model(name, config)
            model.fit(gold_features.iloc[train_index], labels[train_index])
            fold_probabilities = model.predict_proba(gold_features.iloc[validation_index])
            order = [list(model.classes_).index(level) for level in levels]
            probabilities[validation_index] = fold_probabilities[:, order]
        all_oof[name] = probabilities
        all_metrics[name] = metrics(
            labels, probabilities, levels, threshold, margin_threshold
        )

    preference = {name: index for index, name in enumerate(config["models"])}
    selected_name = min(
        all_metrics,
        key=lambda name: (-round(all_metrics[name]["macro_f1"], 6), preference[name]),
    )
    selected_metrics = all_metrics[selected_name]
    gate = (
        selected_metrics["macro_f1"] >= float(config["selection"]["minimum_macro_f1"])
        and selected_metrics["high_confidence_accuracy"]
        >= float(config["selection"]["minimum_high_confidence_accuracy"])
        and selected_metrics["high_confidence_coverage"]
        >= float(config["selection"]["minimum_high_confidence_coverage"])
    )

    unique_target = (
        target.groupby("normalized_lemma", as_index=False)["input_frequency"].max()
        .sort_values("normalized_lemma")
        .reset_index(drop=True)
    )
    unseen = unique_target[
        ~unique_target["normalized_lemma"].isin(label_sets.index)
    ].copy()
    unseen_features = feature_frame(
        unseen["normalized_lemma"], unseen["input_frequency"], surface, lemma_frequency
    )
    final_model = build_model(selected_name, config).fit(gold_features, labels)
    unseen_raw = final_model.predict_proba(unseen_features)
    order = [list(final_model.classes_).index(level) for level in levels]
    unseen_probabilities = unseen_raw[:, order]
    unseen_prediction = unseen_probabilities.argmax(axis=1)
    unseen_confidence = unseen_probabilities.max(axis=1)
    unseen_ordered = np.sort(unseen_probabilities, axis=1)
    unseen_margin = unseen_ordered[:, -1] - unseen_ordered[:, -2]
    unseen_selected = (
        gate & (unseen_confidence >= threshold) & (unseen_margin >= margin_threshold)
    )

    prediction_map = {}
    for row_index, row in unseen.reset_index(drop=True).iterrows():
        word = row["normalized_lemma"]
        probabilities = unseen_probabilities[row_index]
        prediction_map[word] = {
            "assigned_cefr": levels[unseen_prediction[row_index]] if unseen_selected[row_index] else "",
            "label_source": "pseudo_high_confidence" if unseen_selected[row_index] else (
                "abstain_low_confidence" if gate else "abstain_model_gate"
            ),
            "confidence": float(unseen_confidence[row_index]),
            "top_two_margin": float(unseen_margin[row_index]),
            "known_cefr_options": "",
            "manual_review_reason": "",
            **{f"prob_{level}": float(probabilities[index]) for index, level in enumerate(levels)},
        }

    for word, label in exact_labels.items():
        prediction_map[word] = {
            "assigned_cefr": label,
            "label_source": "known_exact",
            "confidence": 1.0,
            "top_two_margin": 1.0,
            "known_cefr_options": label,
            "manual_review_reason": "",
            **{f"prob_{level}": "" for level in levels},
        }
    for word, options in conflicts.items():
        prediction_map[word] = {
            "assigned_cefr": "",
            "label_source": "manual_review_known_conflict",
            "confidence": "",
            "top_two_margin": "",
            "known_cefr_options": "|".join(options),
            "manual_review_reason": "known lemma has multiple CEFR labels without POS",
            **{f"prob_{level}": "" for level in levels},
        }

    additions = target["normalized_lemma"].map(prediction_map).apply(pd.Series)
    output = target.drop(columns=["input_frequency"]).copy()
    output.insert(output.columns.get_loc("cefr_level") + 1, "cefr_level_original", output["cefr_level"])
    for column in additions:
        output[column] = additions[column].values
    output["pseudolabel_model"] = selected_name
    output["pseudolabel_protocol"] = sha256(args.config)
    output["requires_manual_review"] = output["label_source"].eq(
        "manual_review_known_conflict"
    )
    if len(output) != len(target):
        raise RuntimeError("Pseudolabeling changed target row count")
    if output.groupby("normalized_lemma")["assigned_cefr"].nunique().max() > 1:
        raise RuntimeError("Duplicate lemmas received inconsistent labels")
    if (output["cefr_level"] != output["cefr_level_original"]).any():
        raise RuntimeError("Original CEFR values were changed")

    selected_oof = all_oof[selected_name]
    oof = gold[["normalized_lemma", "cefr"]].copy()
    oof["pred_cefr"] = [levels[index] for index in selected_oof.argmax(axis=1)]
    ordered = np.sort(selected_oof, axis=1)
    oof["confidence"] = selected_oof.max(axis=1)
    oof["top_two_margin"] = ordered[:, -1] - ordered[:, -2]
    for index, level in enumerate(levels):
        oof[f"prob_{level}"] = selected_oof[:, index]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "pseudolabeled_dictionary.csv"
    unique_path = args.output_dir / "unique_lemma_predictions.csv"
    oof_path = args.output_dir / "oof_predictions.csv"
    metrics_path = args.output_dir / "metrics.json"
    output.to_csv(output_path, index=False)
    (
        output.drop_duplicates("normalized_lemma")
        .sort_values("normalized_lemma")
        .to_csv(unique_path, index=False)
    )
    oof.to_csv(oof_path, index=False)
    summary = {
        "proposal_id": args.proposal_id,
        "protocol_hash": sha256(args.config),
        "selected_model": selected_name,
        "gate_passed": bool(gate),
        "models": all_metrics,
        "target_rows": len(output),
        "target_unique_lemmas": int(output["normalized_lemma"].nunique()),
        "label_source_counts": output["label_source"].value_counts().to_dict(),
        "assigned_cefr_counts": output.loc[
            output["assigned_cefr"].ne(""), "assigned_cefr"
        ].value_counts().reindex(levels, fill_value=0).to_dict(),
        "original_cefr_unchanged": True,
        "test_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "experiment_id": config["experiment_id"],
        "proposal_id": args.proposal_id,
        "protocol_hash": sha256(args.config),
        "python": "system",
        "sklearn": sklearn.__version__,
        "inputs": {
            str(args.config): sha256(args.config),
            str(args.extracted_target): sha256(args.extracted_target),
            str(Path(__file__).resolve().relative_to(Path.cwd().resolve())): sha256(Path(__file__)),
            **{source["path"]: source["sha256"] for source in sources.values()},
        },
        "outputs": {
            path.name: sha256(path)
            for path in (output_path, unique_path, oof_path, metrics_path)
        },
        "test_accessed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
