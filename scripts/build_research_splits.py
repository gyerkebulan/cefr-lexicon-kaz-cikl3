#!/usr/bin/env python3
"""Build and verify the frozen, leakage-resistant autoresearch split series."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research_protocol import canonical_hash, load_protocol, sha256_file, write_json
from src.utils import CEFR_LABELS, normalize_cefr


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", str(value).strip()).casefold()


def sentence_hash(value: str) -> str:
    normalized = " ".join(normalize_text(value).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def is_edit_distance_one(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    i = j = differences = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
        else:
            differences += 1
            j += 1
            if differences > 1:
                return False
    return True


def group_rows(df: pd.DataFrame) -> list[list[int]]:
    union = UnionFind(len(df))
    first_by_lemma, first_by_context = {}, {}
    deletion_index = defaultdict(list)
    for idx, row in df.iterrows():
        lemma = row["normalized_lemma"]
        if lemma in first_by_lemma:
            union.union(idx, first_by_lemma[lemma])
        first_by_lemma.setdefault(lemma, idx)
        context = row["context_hash"]
        if context:
            if context in first_by_context:
                union.union(idx, first_by_context[context])
            first_by_context.setdefault(context, idx)
        signatures = {lemma} | {lemma[:i] + lemma[i + 1:] for i in range(len(lemma))}
        for signature in signatures:
            for other in deletion_index[signature]:
                if is_edit_distance_one(lemma, df.at[other, "normalized_lemma"]):
                    union.union(idx, other)
            deletion_index[signature].append(idx)
    groups = defaultdict(list)
    for idx in range(len(df)):
        groups[union.find(idx)].append(idx)
    return list(groups.values())


def assign_groups(
    df: pd.DataFrame,
    groups: list[list[int]],
    proportions: dict[str, float],
    seed: int,
    stratify_columns: tuple[str, ...] = ("cefr",),
) -> dict[str, list[int]]:
    missing = set(stratify_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing stratification columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    split_names = list(proportions)
    target_size = {name: proportions[name] * len(df) for name in split_names}
    totals = {
        column: Counter(df[column].astype(str)) for column in stratify_columns
    }
    targets = {
        name: {
            column: {
                value: proportions[name] * count
                for value, count in totals[column].items()
            }
            for column in stratify_columns
        }
        for name in split_names
    }
    current_size = Counter()
    current = {
        name: {column: Counter() for column in stratify_columns}
        for name in split_names
    }
    assigned = {name: [] for name in split_names}
    order = list(range(len(groups)))
    rng.shuffle(order)
    order.sort(key=lambda i: len(groups[i]), reverse=True)

    for group_index in order:
        indices = groups[group_index]
        strata = {
            column: Counter(df.loc[indices, column].astype(str))
            for column in stratify_columns
        }
        candidates = []
        for split_index, name in enumerate(split_names):
            next_size = current_size[name] + len(indices)
            current_size_cost = (
                (current_size[name] - target_size[name]) / max(target_size[name], 1)
            ) ** 2
            next_size_cost = (
                (next_size - target_size[name]) / max(target_size[name], 1)
            ) ** 2
            current_balance_cost = sum(
                ((current[name][column][value] - target) / max(target, 1)) ** 2
                for column in stratify_columns
                for value, target in targets[name][column].items()
            )
            next_balance_cost = sum(
                (
                    (current[name][column][value] + strata[column][value] - target)
                    / max(target, 1)
                ) ** 2
                for column in stratify_columns
                for value, target in targets[name][column].items()
            )
            overflow = max(0.0, next_size - target_size[name]) / max(target_size[name], 1)
            delta = next_size_cost - current_size_cost + next_balance_cost - current_balance_cost
            candidates.append((delta + 4 * overflow**2, split_index, name))
        _, _, chosen = min(candidates)
        assigned[chosen].extend(indices)
        current_size[chosen] += len(indices)
        for column in stratify_columns:
            current[chosen][column].update(strata[column])
    return assigned


def verify_splits(frames: dict[str, pd.DataFrame]) -> dict:
    checks = {}
    for column in ["normalized_lemma", "group_id", "context_hash"]:
        for left, right in combinations(frames, 2):
            left_values = set(frames[left][column]) - {""}
            right_values = set(frames[right][column]) - {""}
            overlap = left_values & right_values
            checks[f"{column}_{left}_{right}_overlap"] = len(overlap)
            if overlap:
                raise ValueError(f"{column} crosses {left}/{right}: {len(overlap)} values")
    for left, right in combinations(frames, 2):
        left_lemmas = set(frames[left]["normalized_lemma"])
        right_lemmas = set(frames[right]["normalized_lemma"])
        # ponytail: quadratic validation is fine for this 3.7k-row dataset; index if it grows.
        pair = next(
            (
                (left_lemma, right_lemma)
                for left_lemma in left_lemmas
                for right_lemma in right_lemmas
                if is_edit_distance_one(left_lemma, right_lemma)
            ),
            None,
        )
        checks[f"near_duplicate_{left}_{right}_overlap"] = int(pair is not None)
        if pair:
            raise ValueError(f"Near-duplicate lemma crosses {left}/{right}: {pair}")
    return checks


def resplit_development_pool(protocol, protocol_hash, output_dir):
    """Reassign only verified parent train/dev groups; never open the sealed test."""
    config = protocol["dataset"]
    parent = PROJECT_ROOT / config["development_pool"]
    manifest = json.loads((parent / "dataset-manifest.json").read_text())
    parts, hashes = [], {}
    for split in ("train", "dev"):
        path = parent / f"{split}.csv"
        hashes[split] = sha256_file(path)
        if hashes[split] != config["parent_split_hashes"][split]:
            raise ValueError(f"Parent {split} changed")
        if hashes[split] != manifest["split_hashes"][split]:
            raise ValueError(f"Parent {split} manifest mismatch")
        parts.append(pd.read_csv(path, keep_default_na=False))
    df = pd.concat(parts, ignore_index=True).sort_values("source_row_id").reset_index(drop=True)
    if df["source_row_id"].duplicated().any():
        raise ValueError("Parent source IDs are not unique")
    groups = list(df.groupby("group_id", sort=True).indices.values())
    assigned = assign_groups(
        df, groups, config["proportions"], config["seed"], config["stratify_columns"]
    )
    frames = {name: df.loc[sorted(ids)].reset_index(drop=True) for name, ids in assigned.items()}
    checks = verify_splits(frames)
    if sum(map(len, frames.values())) != len(df):
        raise ValueError("Development pool lost rows")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    result = {
        "series_id": protocol["series"]["id"], "protocol_hash": protocol_hash,
        "parent_split_hashes": hashes,
        "sealed_test_sha256": manifest["split_hashes"]["test"],
        "test_policy": "original test untouched; not opened or copied",
        "evidence_scope": "new paired protocol on previously inspected development data",
        "split_hashes": {name: sha256_file(output_dir / f"{name}.csv") for name in frames},
        "counts": {name: len(frame) for name, frame in frames.items()},
        "stratify_counts": {
            name: {col: frame[col].astype(str).value_counts().to_dict()
                   for col in config["stratify_columns"]}
            for name, frame in frames.items()
        },
        "groups": len(groups), "checks": checks,
    }
    write_json(output_dir / "dataset-manifest.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/autoresearch.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.config)
    config = protocol["dataset"]
    if config.get("development_pool"):
        return resplit_development_pool(
            protocol, protocol_hash, args.output_dir or PROJECT_ROOT / config["output_dir"]
        )
    if config.get("ambiguous_context_policy") != "target_only_for_multi_pos_multi_cefr":
        raise ValueError("Unsupported ambiguous-context policy")
    source = PROJECT_ROOT / config["source"]
    context_path = PROJECT_ROOT / config["context_cache"]
    output_dir = args.output_dir or PROJECT_ROOT / config["output_dir"]
    frozen = config.get("frozen_hashes", {})
    for name, path in (("source_sha256", source), ("context_cache_sha256", context_path)):
        if frozen.get(name) and sha256_file(path) != frozen[name]:
            raise ValueError(f"Frozen {name} does not match {path}")

    raw = pd.read_csv(source)
    raw.insert(0, "source_row_id", np.arange(len(raw), dtype=int))
    invalid = raw["lemma"].isna() | raw["lemma"].astype("string").str.strip().eq("")
    exclusions = pd.DataFrame({
        "source_row_id": raw.loc[invalid, "source_row_id"],
        "reason": "null_or_empty_lemma",
    })
    df = raw.loc[~invalid].copy().reset_index(drop=True)
    df["lemma"] = df["lemma"].astype(str).map(lambda value: unicodedata.normalize("NFC", value.strip()))
    df["normalized_lemma"] = df["lemma"].map(normalize_text)
    df["pos"] = df["pos"].fillna("OTHER").astype(str).str.strip().str.upper()
    df["cefr"] = df["cefr"].map(normalize_cefr)
    invalid_cefr = ~df["cefr"].isin(CEFR_LABELS)
    if invalid_cefr.any():
        extra = pd.DataFrame({
            "source_row_id": df.loc[invalid_cefr, "source_row_id"],
            "reason": "invalid_cefr",
        })
        exclusions = pd.concat([exclusions, extra], ignore_index=True)
        df = df.loc[~invalid_cefr].copy().reset_index(drop=True)

    key_columns = ["normalized_lemma", "pos"]
    label_counts = df.groupby(key_columns)["cefr"].transform("nunique")
    conflict_mask = label_counts > 1
    if conflict_mask.any():
        extra = pd.DataFrame({
            "source_row_id": df.loc[conflict_mask, "source_row_id"],
            "reason": "conflicting_labels_same_lemma_pos",
        })
        exclusions = pd.concat([exclusions, extra], ignore_index=True)
        df = df.loc[~conflict_mask].copy()
    duplicate_mask = df.duplicated(key_columns, keep="first")
    if duplicate_mask.any():
        extra = pd.DataFrame({
            "source_row_id": df.loc[duplicate_mask, "source_row_id"],
            "reason": "duplicate_same_label",
        })
        exclusions = pd.concat([exclusions, extra], ignore_index=True)
        df = df.loc[~duplicate_mask].copy()
    df = df.reset_index(drop=True)

    legacy = pd.concat(
        [pd.read_csv(PROJECT_ROOT / config["legacy_frequency_splits"] / f"{name}.csv")
         for name in ("train", "dev", "test")],
        ignore_index=True,
    )
    legacy = legacy.loc[legacy["lemma"].notna()].copy()
    legacy["normalized_lemma"] = legacy["lemma"].map(normalize_text)
    legacy["pos"] = legacy["pos"].fillna("OTHER").astype(str).str.strip().str.upper()
    frequencies = legacy.groupby(["normalized_lemma", "pos"])["freq"].max()
    df["freq"] = [
        float(frequencies.get((lemma, pos), 0))
        for lemma, pos in zip(df["normalized_lemma"], df["pos"])
    ]

    context_map = json.loads(context_path.read_text(encoding="utf-8"))
    normalized_context = {
        normalize_text(key): (
            next((item for item in value if isinstance(item, str) and item.strip()), "")
            if isinstance(value, list) else str(value)
        )
        for key, value in context_map.items()
    }
    normalized_context = {
        key: value for key, value in normalized_context.items() if value.strip()
    }
    df["target"] = df["lemma"]
    df["sentence"] = df["normalized_lemma"].map(normalized_context).fillna(df["lemma"])
    df["context_available"] = df["normalized_lemma"].isin(normalized_context)
    ambiguity = df.groupby("normalized_lemma").agg(
        pos_count=("pos", "nunique"), cefr_count=("cefr", "nunique")
    )
    ambiguous_lemmas = set(
        ambiguity.index[(ambiguity.pos_count > 1) & (ambiguity.cefr_count > 1)]
    )
    ambiguous_context = df["normalized_lemma"].isin(ambiguous_lemmas)
    df.loc[ambiguous_context, "sentence"] = df.loc[ambiguous_context, "target"]
    df.loc[ambiguous_context, "context_available"] = False
    df["context_hash"] = [
        sentence_hash(sentence) if available else ""
        for sentence, available in zip(df["sentence"], df["context_available"])
    ]

    groups = group_rows(df)
    stratify_columns = config.get("stratify_columns", ["cefr"])
    assigned = assign_groups(
        df, groups, config["proportions"], int(config["seed"]), stratify_columns
    )
    root_by_index = {}
    for group_id, indices in enumerate(sorted(groups, key=lambda values: min(values))):
        for idx in indices:
            root_by_index[idx] = group_id
    df["group_id"] = [root_by_index[idx] for idx in range(len(df))]
    frames = {
        name: df.loc[sorted(indices)].reset_index(drop=True)
        for name, indices in assigned.items()
    }
    checks = verify_splits(frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "source_row_id", "lemma", "normalized_lemma", "pos", "cefr", "freq",
        "sentence", "target", "context_available", "context_hash", "group_id",
    ]
    for name, frame in frames.items():
        frame[output_columns].to_csv(output_dir / f"{name}.csv", index=False)
    exclusions.to_csv(output_dir / "exclusions.csv", index=False)
    hashes = {
        name: sha256_file(output_dir / f"{name}.csv")
        for name in ("train", "dev", "test")
    }
    dataset_hash = canonical_hash(hashes)
    preprocessing_hash = canonical_hash({
        "normalization": config["normalization"],
        "near_duplicate_edit_distance": config["near_duplicate_edit_distance"],
        "grouping": config["grouping"],
        "ambiguous_context_policy": config["ambiguous_context_policy"],
    })
    locked_outputs = {
        "preprocessing_sha256": preprocessing_hash,
        "dataset_sha256": dataset_hash,
        "splits": hashes,
    }
    for name, actual in locked_outputs.items():
        if frozen.get(name) and frozen[name] != actual:
            raise ValueError(f"Frozen {name} does not reproduce: {actual}")
    manifest = {
        "series_id": protocol["series"]["id"],
        "protocol_hash": protocol_hash,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256_file(source),
        "context_cache_sha256": sha256_file(context_path),
        "preprocessing_hash": preprocessing_hash,
        "split_hashes": hashes,
        "dataset_hash": dataset_hash,
        "counts": {name: len(frame) for name, frame in frames.items()},
        "class_counts": {
            name: frame["cefr"].value_counts().reindex(CEFR_LABELS, fill_value=0).to_dict()
            for name, frame in frames.items()
        },
        "stratify_columns": stratify_columns,
        "stratify_counts": {
            name: {
                column: frame[column].astype(str).value_counts().sort_index().to_dict()
                for column in stratify_columns
            }
            for name, frame in frames.items()
        },
        "exclusions": exclusions["reason"].value_counts().to_dict(),
        "ambiguous_contexts_removed": {
            "lemmas": len(ambiguous_lemmas),
            "rows": int(ambiguous_context.sum()),
        },
        "groups": len(groups),
        "checks": checks,
    }
    write_json(output_dir / "dataset-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
