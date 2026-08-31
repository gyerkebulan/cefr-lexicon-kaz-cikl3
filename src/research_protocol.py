"""Frozen-protocol, run-manifest, artifact-integrity, and promotion helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_protocol(path: Path) -> tuple[Dict[str, Any], str]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_lock = config.get("series", {}).get("protocol_lock_hash")
    if expected_lock:
        unlocked = json.loads(json.dumps(config))
        unlocked["series"].pop("protocol_lock_hash")
        if canonical_hash(unlocked) != expected_lock:
            raise ValueError("Locked research configuration was modified")
    root = path.parent.parent
    for relative, expected in config.get("series", {}).get("fixed_file_hashes", {}).items():
        fixed_path = root / relative
        if not fixed_path.is_file() or sha256_file(fixed_path) != expected:
            raise ValueError(f"Fixed protocol file checksum mismatch: {relative}")
    surface = config.get("series", {}).get("research_surface_baseline_hashes", {})
    if surface:
        allowed = set(config["series"]["allowed_research_surface"])
        if set(surface) != allowed:
            raise ValueError("Research-surface files and baseline hashes disagree")
        baseline_dirty = set(config["series"]["protocol_baseline_changed_files"])
        unexpected = set(changed_files(root)) - baseline_dirty - allowed
        if unexpected:
            raise ValueError(
                f"Files outside the locked research surface changed: {sorted(unexpected)}"
            )
    return config, canonical_hash(config)


def verify_dataset_manifest(
    directory: Path, *, protocol_hash: str, splits: Iterable[str]
) -> Dict[str, Any]:
    directory = Path(directory)
    manifest = json.loads((directory / "dataset-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_hash") != protocol_hash:
        raise ValueError("Dataset manifest does not match the locked protocol")
    for split in splits:
        path = directory / f"{split}.csv"
        if sha256_file(path) != manifest["split_hashes"][split]:
            raise ValueError(f"{split} split checksum mismatch")
    return manifest


def gpu_budget_status(
    root: Path, *, protocol: Mapping[str, Any], protocol_hash: str
) -> Dict[str, float]:
    series_id = protocol["series"]["id"]
    result_root = Path(root) / "results" / "autoresearch" / series_id
    used = 0.0
    if result_root.exists():
        for pattern in ("run-manifest.json", "failure-manifest.json", "final-test-manifest.json"):
            for path in result_root.rglob(pattern):
                try:
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if (
                    manifest.get("series_id") == series_id
                    and manifest.get("protocol_hash") == protocol_hash
                ):
                    used += float(manifest.get("gpu_seconds", 0.0))
    total = 3600.0 * float(protocol["budget_gpu_hours"]["total"])
    return {
        "used_gpu_seconds": used,
        "remaining_gpu_seconds": max(0.0, total - used),
        "total_gpu_seconds": total,
    }


def require_gpu_budget(
    root: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    requested_seconds: float = 0.0,
) -> Dict[str, float]:
    status = gpu_budget_status(root, protocol=protocol, protocol_hash=protocol_hash)
    if status["remaining_gpu_seconds"] < requested_seconds:
        raise RuntimeError(
            f"GPU budget exhausted: remaining={status['remaining_gpu_seconds'] / 3600:.3f}h, "
            f"requested={requested_seconds / 3600:.3f}h"
        )
    return status


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def changed_files(root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--short"], cwd=root, text=True
    )
    return sorted(line[3:] for line in output.splitlines() if len(line) > 3)


def research_surface_changes(root: Path, protocol: Mapping[str, Any]) -> Dict[str, str]:
    baseline = protocol["series"].get("research_surface_baseline_hashes", {})
    return {
        relative: sha256_file(Path(root) / relative)
        for relative, expected in baseline.items()
        if sha256_file(Path(root) / relative) != expected
    }


def checksums(paths: Iterable[Path], root: Optional[Path] = None) -> Dict[str, str]:
    result = {}
    for path in paths:
        path = Path(path)
        if path.is_file():
            key = str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
            result[key] = sha256_file(path)
    return result


def write_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return path


def build_run_manifest(
    *,
    root: Path,
    series_id: str,
    proposal_id: str,
    protocol_hash: str,
    seed: int,
    configuration: Mapping[str, Any],
    command: list[str],
    metrics: Mapping[str, Any],
    runtime_seconds: float,
    parameter_count: int,
    peak_vram_bytes: int,
    artifacts: Iterable[Path],
    outcome: str,
) -> Dict[str, Any]:
    root = Path(root)
    return {
        "schema_version": 2,
        "project_root": str(root.resolve()),
        "series_id": series_id,
        "proposal_id": proposal_id,
        "revision": git_revision(root),
        "changed_files": changed_files(root),
        "seed": int(seed),
        "configuration": dict(configuration),
        "protocol_hash": protocol_hash,
        "metrics": dict(metrics),
        "runtime_seconds": float(runtime_seconds),
        "gpu_seconds": float(runtime_seconds) if peak_vram_bytes > 0 else 0.0,
        "parameter_count": int(parameter_count),
        "peak_vram_bytes": int(peak_vram_bytes),
        "command": command,
        "artifact_paths": [str(Path(path).resolve()) for path in artifacts],
        "artifact_checksums": checksums(artifacts, root=root),
        "outcome": outcome,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_candidate_manifest(
    manifest: Mapping[str, Any], *, series_id: str, protocol_hash: str
) -> None:
    required = {
        "project_root", "proposal_id", "revision", "changed_files", "seed", "configuration",
        "protocol_hash", "metrics", "runtime_seconds", "parameter_count",
        "gpu_seconds", "peak_vram_bytes", "artifact_paths", "artifact_checksums", "outcome",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"Run manifest missing required fields: {missing}")
    if manifest.get("series_id") != series_id:
        raise ValueError("Run manifest belongs to another series")
    if manifest.get("protocol_hash") != protocol_hash:
        raise ValueError("Run manifest protocol hash does not match the locked protocol")
    if any(key.startswith("test") for key in manifest.get("metrics", {})):
        raise ValueError("Candidate manifest contains forbidden test metrics")


def assess_guardrails(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    floors: Mapping[str, float],
) -> list[str]:
    failures = []
    checks = {
        "perturbation_macro_f1": floors["perturbation_macro_f1_regression"],
        "ordinal_mae": floors["ordinal_mae_regression"],
        "quadratic_weighted_kappa": floors["quadratic_weighted_kappa_regression"],
        "expected_calibration_error": floors["expected_calibration_error_regression"],
    }
    for metric, tolerance in checks.items():
        if metric not in baseline or metric not in candidate:
            continue
        if metric in {"ordinal_mae", "expected_calibration_error"}:
            regression = float(candidate[metric]) - float(baseline[metric])
        else:
            regression = float(baseline[metric]) - float(candidate[metric])
        if regression > float(tolerance):
            failures.append(f"{metric} regression {regression:.6f} exceeds {tolerance:.6f}")
    return failures


def assess_robustness_guardrails(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    floors: Mapping[str, float],
) -> list[str]:
    failures = []
    tolerance = float(floors["perturbation_macro_f1_regression"])
    regression = (
        float(baseline["perturbation_macro_f1"])
        - float(candidate["perturbation_macro_f1"])
    )
    if regression > tolerance:
        failures.append(
            f"perturbation macro-F1 regression {regression:.6f} exceeds {tolerance:.6f}"
        )
    slice_tolerance = float(floors["supported_slice_or_class_f1_regression"])
    minimum = int(floors["supported_slice_minimum_examples"])
    for section in ("scenarios", "slices"):
        for group, baseline_group in baseline.get(section, {}).items():
            candidate_group = candidate.get(section, {}).get(group)
            if candidate_group is None:
                failures.append(f"missing robustness {section}/{group}")
                continue
            baseline_items = baseline_group if section == "slices" else {"all": baseline_group}
            candidate_items = candidate_group if section == "slices" else {"all": candidate_group}
            for item, baseline_record in baseline_items.items():
                candidate_record = candidate_items.get(item)
                if candidate_record is None:
                    failures.append(f"missing robustness {section}/{group}/{item}")
                    continue
                support = int(baseline_record.get("support", minimum))
                if support < minimum or not baseline_record.get("supported", True):
                    continue
                regression = (
                    float(baseline_record["metrics"]["macro_f1"])
                    - float(candidate_record["metrics"]["macro_f1"])
                )
                if regression > slice_tolerance:
                    failures.append(
                        f"{section}/{group}/{item} F1 regression {regression:.6f} "
                        f"exceeds {slice_tolerance:.6f}"
                    )
    return failures


def authorize_final_test(
    champion_path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    final_state_dir: Path,
) -> Dict[str, Any]:
    champion_path = Path(champion_path)
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    validate_candidate_manifest(
        champion,
        series_id=protocol["series"]["id"],
        protocol_hash=protocol_hash,
    )
    confirmation_seeds = set(protocol["evaluation"]["confirmation_seeds"])
    if set(champion.get("confirmed_seeds", [])) != confirmation_seeds:
        raise ValueError("Champion was not confirmed on every locked seed")
    if champion.get("outcome") != "accepted":
        raise ValueError("Champion manifest is not marked accepted")
    for artifact, expected in champion["artifact_checksums"].items():
        path = Path(artifact)
        if not path.is_absolute():
            path = Path(champion["project_root"]) / path
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"Champion artifact checksum failed: {artifact}")
    final_state_dir = Path(final_state_dir)
    final_state_dir.mkdir(parents=True, exist_ok=True)
    marker = final_state_dir / "final-test-consumed.json"
    if marker.exists():
        raise RuntimeError("The sealed final test has already been consumed for this series")
    write_json(
        marker,
        {
            "series_id": protocol["series"]["id"],
            "protocol_hash": protocol_hash,
            "champion_manifest": str(champion_path),
            "champion_manifest_sha256": sha256_file(champion_path),
            "consumed_at": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )
    return champion
