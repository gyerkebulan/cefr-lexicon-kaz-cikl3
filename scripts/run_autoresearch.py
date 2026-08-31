#!/usr/bin/env python3
"""Run one seed-42 candidate screen under the frozen autoresearch protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research_protocol import (
    assess_guardrails, assess_robustness_guardrails, load_protocol,
    require_gpu_budget, sha256_file, validate_candidate_manifest, write_json,
)
from scripts.evaluate_robustness import score_predictions
from src.utils import CEFR_LABELS


def manifest_path(protocol: dict, proposal: str, seed: int) -> Path:
    return (
        PROJECT_ROOT / "results/autoresearch" / protocol["series"]["id"]
        / proposal / f"seed-{seed}" / "run-manifest.json"
    )


def load_manifest(path: Path, protocol: dict, protocol_hash: str) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_candidate_manifest(
        manifest, series_id=protocol["series"]["id"], protocol_hash=protocol_hash
    )
    return manifest


def run_candidate(args, protocol: dict, protocol_hash: str, seed: int) -> dict:
    command = [
        sys.executable, "scripts/train_dual.py",
        "--research-config", str(args.research_config),
        "--proposal-id", args.proposal_id,
        "--seed", str(seed),
        "--robustness-input", str(args.robustness_input),
        *args.model_args,
    ]
    ceiling = int(protocol["evaluation"]["screening"]["gpu_minutes"] * 60)
    require_gpu_budget(
        PROJECT_ROOT,
        protocol=protocol,
        protocol_hash=protocol_hash,
        requested_seconds=ceiling,
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, timeout=ceiling, check=False
        )
        outcome = "complete" if completed.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        outcome = "interrupted_budget"
    path = manifest_path(protocol, args.proposal_id, seed)
    if outcome != "complete":
        elapsed = time.perf_counter() - started
        failure = {
            "series_id": protocol["series"]["id"],
            "proposal_id": args.proposal_id,
            "seed": seed,
            "protocol_hash": protocol_hash,
            "hypothesis": args.hypothesis,
            "outcome": outcome,
            "command": command,
            "runtime_seconds": elapsed,
            "gpu_seconds": elapsed,
        }
        write_json(path.parent / "failure-manifest.json", failure, exclusive=True)
        raise RuntimeError(f"Candidate {args.proposal_id} ended as {outcome}")
    return load_manifest(path, protocol, protocol_hash)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-config", type=Path, required=True)
    parser.add_argument("--proposal-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-robustness", type=Path, required=True)
    parser.add_argument("--robustness-input", type=Path, required=True)
    parser.add_argument("model_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.research_config)
    baseline = load_manifest(args.baseline_manifest, protocol, protocol_hash)
    seeds = protocol["evaluation"]["screen_seeds"]
    if baseline["seed"] not in seeds:
        raise ValueError("Screening baseline must use the locked screen seed")
    candidates = [run_candidate(args, protocol, protocol_hash, seed) for seed in seeds]
    baseline_robustness = json.loads(args.baseline_robustness.read_text(encoding="utf-8"))
    if (
        baseline_robustness.get("series_id") != protocol["series"]["id"]
        or baseline_robustness.get("protocol_hash") != protocol_hash
    ):
        raise ValueError("Baseline robustness report does not match the locked protocol")
    prediction_path = next(
        Path(path)
        for path in candidates[0]["artifact_paths"]
        if path.endswith("_robustness.csv")
    )
    minimum = protocol["evaluation"]["guardrail_floors"]["supported_slice_minimum_examples"]
    candidate_robustness = score_predictions(pd.read_csv(prediction_path), minimum)
    candidate_robustness.update({
        "series_id": protocol["series"]["id"],
        "protocol_hash": protocol_hash,
        "prediction_count": 1,
        "prediction_checksums": {
            str(prediction_path.resolve()): sha256_file(prediction_path)
        },
    })
    robustness_path = manifest_path(
        protocol, args.proposal_id, seeds[0]
    ).with_name("robustness-report.json")
    write_json(robustness_path, candidate_robustness, exclusive=True)
    baseline_f1 = float(baseline["metrics"]["macro_f1"])
    candidate_f1 = sum(float(run["metrics"]["macro_f1"]) for run in candidates) / len(candidates)
    gain = candidate_f1 - baseline_f1
    floors = protocol["evaluation"]["guardrail_floors"]
    guardrail_failures = [
        failure
        for run in candidates
        for failure in assess_guardrails(baseline["metrics"], run["metrics"], floors)
    ]
    guardrail_failures.extend(assess_robustness_guardrails(
        baseline_robustness,
        candidate_robustness,
        floors,
    ))
    class_tolerance = floors["supported_slice_or_class_f1_regression"]
    for label in CEFR_LABELS:
        regression = (
            baseline["metrics"]["per_class"][label]["f1"]
            - candidates[0]["metrics"]["per_class"][label]["f1"]
        )
        if regression > class_tolerance:
            guardrail_failures.append(
                f"{label} F1 regression {regression:.6f} exceeds {class_tolerance:.6f}"
            )
    for metric in ("invalid_prediction_count", "target_span_failure_count", "nan_count"):
        if candidates[0]["metrics"].get(metric, 0):
            guardrail_failures.append(f"non-zero {metric}")
    threshold = protocol["evaluation"]["screening"]["minimum_macro_f1_gain"]
    outcome = "promote" if gain >= threshold and not guardrail_failures else "discard"
    decision = {
        "series_id": protocol["series"]["id"],
        "proposal_id": args.proposal_id,
        "hypothesis": args.hypothesis,
        "protocol_hash": protocol_hash,
        "seeds": seeds,
        "baseline_macro_f1": baseline_f1,
        "candidate_mean_macro_f1": candidate_f1,
        "gain": gain,
        "required_gain": threshold,
        "guardrail_failures": guardrail_failures,
        "outcome": outcome,
        "run_manifests": [str(manifest_path(protocol, args.proposal_id, seed)) for seed in seeds],
        "robustness_report": str(robustness_path),
    }
    decision_path = (
        PROJECT_ROOT / "results/autoresearch" / protocol["series"]["id"]
        / args.proposal_id / "screen-decision.json"
    )
    write_json(decision_path, decision, exclusive=True)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if outcome == "promote" else 2


if __name__ == "__main__":
    raise SystemExit(main())
