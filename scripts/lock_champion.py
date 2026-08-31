#!/usr/bin/env python3
"""Apply the five-seed gate and lock one champion for the sealed final test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research_protocol import (
    assess_guardrails, assess_robustness_guardrails, load_protocol, sha256_file,
    validate_candidate_manifest, write_json,
)
from src.utils import CEFR_LABELS


def load_event(event_id: str) -> dict:
    paths = list((PROJECT_ROOT / ".agent/memory/agents").glob(f"*/*/{event_id}.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one synced immutable event for {event_id}, found {len(paths)}")
    event = json.loads(paths[0].read_text(encoding="utf-8"))
    if event.get("id") != event_id:
        raise ValueError(f"Event filename and ID disagree: {event_id}")
    return event


def verify_review_gate(
    review_ids: list[str],
    rebuttal_id: str,
    decision_id: str,
    *,
    experiment_id: str,
    proposal_id: str,
) -> None:
    reviews = [load_event(event_id) for event_id in review_ids]
    rebuttal, decision = load_event(rebuttal_id), load_event(decision_id)
    events = [*reviews, rebuttal, decision]
    if any(event.get("experiment_id") != experiment_id for event in events):
        raise ValueError("Review gate contains an event from another experiment")
    if any(event.get("kind") != "critique" for event in reviews):
        raise ValueError("Both review events must be critiques")
    if len({event.get("session") for event in reviews}) != 2:
        raise ValueError("Critiques must come from two independent review sessions")
    if any(event.get("agent") != "codex" or event.get("role") != "reviewer" for event in reviews):
        raise ValueError("This series requires two Codex reviewer events")
    finding_ids = {event.get("parent_id") for event in reviews}
    if len(finding_ids) != 1:
        raise ValueError("Critiques must review the same finding")
    finding = load_event(finding_ids.pop())
    if (
        finding.get("kind") != "finding"
        or finding.get("experiment_id") != experiment_id
        or finding.get("parent_id") != proposal_id
    ):
        raise ValueError("Reviewed finding does not belong to the candidate proposal")
    if (
        rebuttal.get("kind") != "rebuttal"
        or rebuttal.get("parent_id") != finding["id"]
        or set(rebuttal.get("responds_to", [])) != set(review_ids)
    ):
        raise ValueError("Author rebuttal must address both critiques")
    if (
        decision.get("kind") != "decision"
        or decision.get("parent_id") != finding["id"]
        or decision.get("decision") != "accept"
    ):
        raise ValueError("Coordinator decision must accept the reviewed finding")


def load_run(path: Path, protocol: dict, protocol_hash: str) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    validate_candidate_manifest(
        run, series_id=protocol["series"]["id"], protocol_hash=protocol_hash
    )
    root = Path(run["project_root"])
    for artifact, expected in run["artifact_checksums"].items():
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected:
            raise ValueError(f"Run artifact checksum failed: {artifact}")
    return run


def predictions(run: dict) -> pd.DataFrame:
    path = next(Path(path) for path in run["artifact_paths"] if path.endswith("_dev.csv"))
    return pd.read_csv(path)


def stratified_bootstrap(
    baseline_frames: list[pd.DataFrame],
    candidate_frames: list[pd.DataFrame],
    samples: int,
    seed: int = 42,
) -> tuple[float, float]:
    gold = baseline_frames[0]["gold_cefr"].to_numpy()
    ids = baseline_frames[0]["source_row_id"].to_numpy()
    if any(
        not np.array_equal(gold, frame["gold_cefr"].to_numpy())
        or not np.array_equal(ids, frame["source_row_id"].to_numpy())
        for frame in baseline_frames + candidate_frames
    ):
        raise ValueError("Confirmation predictions are not aligned")
    groups = [np.flatnonzero(gold == label) for label in CEFR_LABELS]
    rng, deltas = np.random.default_rng(seed), np.empty(samples)
    for sample in range(samples):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        baseline_score = np.mean([
            f1_score(gold[indices], frame["pred_cefr"].to_numpy()[indices],
                     labels=CEFR_LABELS, average="macro", zero_division=0)
            for frame in baseline_frames
        ])
        candidate_score = np.mean([
            f1_score(gold[indices], frame["pred_cefr"].to_numpy()[indices],
                     labels=CEFR_LABELS, average="macro", zero_division=0)
            for frame in candidate_frames
        ])
        deltas[sample] = candidate_score - baseline_score
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def mean_metrics(runs: list[dict]) -> dict:
    scalar_keys = set.intersection(*[
        {key for key, value in run["metrics"].items() if isinstance(value, (int, float))}
        for run in runs
    ])
    result = {
        key: float(np.mean([run["metrics"][key] for run in runs]))
        for key in scalar_keys
    }
    result["per_class"] = {
        label: {
            "f1": float(np.mean([run["metrics"]["per_class"][label]["f1"] for run in runs])),
            "support": runs[0]["metrics"]["per_class"][label]["support"],
        }
        for label in CEFR_LABELS
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-config", type=Path, required=True)
    parser.add_argument("--baseline-manifests", type=Path, nargs=5, required=True)
    parser.add_argument("--candidate-manifests", type=Path, nargs=5, required=True)
    parser.add_argument("--baseline-robustness", type=Path, required=True)
    parser.add_argument("--candidate-robustness", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--review-event", action="append", required=True)
    parser.add_argument("--rebuttal-event", required=True)
    parser.add_argument("--decision-event", required=True)
    parser.add_argument("--user-approved", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.review_event) != 2 or len(set(args.review_event)) != 2:
        raise ValueError("Exactly two independent review event IDs are required")
    protocol, protocol_hash = load_protocol(args.research_config)
    baseline = [load_run(path, protocol, protocol_hash) for path in args.baseline_manifests]
    candidate = [load_run(path, protocol, protocol_hash) for path in args.candidate_manifests]
    required_seeds = set(protocol["evaluation"]["confirmation_seeds"])
    if {run["seed"] for run in baseline} != required_seeds or {run["seed"] for run in candidate} != required_seeds:
        raise ValueError("Both systems must have exactly the five locked seeds")
    if len({run["proposal_id"] for run in candidate}) != 1:
        raise ValueError("Candidate confirmation mixes proposals")
    proposal_id = candidate[0]["proposal_id"]
    baseline_metrics, candidate_metrics = mean_metrics(baseline), mean_metrics(candidate)
    gain = candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    lower, upper = stratified_bootstrap(
        [predictions(run) for run in baseline],
        [predictions(run) for run in candidate],
        protocol["evaluation"]["confirmation"]["paired_bootstrap_samples"],
    )
    failures = assess_guardrails(
        baseline_metrics, candidate_metrics, protocol["evaluation"]["guardrail_floors"]
    )
    class_floor = protocol["evaluation"]["guardrail_floors"]["supported_slice_or_class_f1_regression"]
    for label in CEFR_LABELS:
        regression = baseline_metrics["per_class"][label]["f1"] - candidate_metrics["per_class"][label]["f1"]
        if regression > class_floor:
            failures.append(f"{label} F1 regression {regression:.6f} exceeds {class_floor:.6f}")
    baseline_robustness = json.loads(args.baseline_robustness.read_text(encoding="utf-8"))
    candidate_robustness = json.loads(args.candidate_robustness.read_text(encoding="utf-8"))
    for report in (baseline_robustness, candidate_robustness):
        if (
            report.get("series_id") != protocol["series"]["id"]
            or report.get("protocol_hash") != protocol_hash
        ):
            failures.append("robustness report does not match the locked protocol")
        if report.get("prediction_count") != len(required_seeds):
            failures.append("confirmation robustness must ensemble all five locked seeds")
    failures.extend(assess_robustness_guardrails(
        baseline_robustness,
        candidate_robustness,
        protocol["evaluation"]["guardrail_floors"],
    ))
    for run in candidate:
        for metric in ("invalid_prediction_count", "target_span_failure_count", "nan_count"):
            if run["metrics"].get(metric, 0):
                failures.append(f"seed {run['seed']} has non-zero {metric}")
    required_gain = protocol["evaluation"]["confirmation"]["minimum_mean_macro_f1_gain"]
    if gain < required_gain:
        failures.append(f"mean macro-F1 gain {gain:.6f} is below {required_gain:.6f}")
    if lower <= protocol["evaluation"]["confirmation"]["require_lower_bound_above"]:
        failures.append(f"paired-bootstrap lower bound {lower:.6f} is not above zero")
    calibration = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    if (
        calibration.get("series_id") != protocol["series"]["id"]
        or calibration.get("protocol_hash") != protocol_hash
        or calibration.get("proposal_id") != proposal_id
        or calibration.get("fold_count") != protocol["evaluation"]["calibration_folds"]
    ):
        raise ValueError("Calibration manifest does not match the locked protocol")
    calibrated_output = Path(calibration["output_path"])
    if (
        not calibrated_output.is_file()
        or sha256_file(calibrated_output) != calibration["output_sha256"]
    ):
        raise ValueError("Calibrated prediction checksum failed")
    if failures:
        raise ValueError("Champion gate failed: " + "; ".join(failures))
    verify_review_gate(
        args.review_event,
        args.rebuttal_event,
        args.decision_event,
        experiment_id=protocol["series"]["experiment_id"],
        proposal_id=proposal_id,
    )
    selected = max(candidate, key=lambda run: run["metrics"]["macro_f1"])
    checkpoint = next(path for path in selected["artifact_paths"] if path.endswith("best_model.pt"))
    champion = dict(selected)
    artifact_paths = list(champion["artifact_paths"])
    artifact_checksums = dict(champion["artifact_checksums"])
    for path in (args.calibration_manifest.resolve(), calibrated_output.resolve()):
        artifact_paths.append(str(path))
        artifact_checksums[str(path)] = sha256_file(path)
    champion.update({
        "outcome": "accepted",
        "confirmed_seeds": sorted(required_seeds),
        "metrics": candidate_metrics,
        "mean_macro_f1_gain": gain,
        "paired_bootstrap_95_ci": [lower, upper],
        "selected_checkpoint": checkpoint,
        "all_confirmation_manifests": [str(path) for path in args.candidate_manifests],
        "robustness_reports": [str(args.baseline_robustness), str(args.candidate_robustness)],
        "calibration_manifest": str(args.calibration_manifest),
        "temperature": float(calibration["temperature"]),
        "artifact_paths": artifact_paths,
        "artifact_checksums": artifact_checksums,
        "review_events": args.review_event,
        "rebuttal_event": args.rebuttal_event,
        "decision_event": args.decision_event,
        "user_approved": True,
    })
    write_json(args.output, champion, exclusive=True)
    print(json.dumps({
        "accepted": True, "mean_macro_f1_gain": gain,
        "paired_bootstrap_95_ci": [lower, upper], "selected_seed": selected["seed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
