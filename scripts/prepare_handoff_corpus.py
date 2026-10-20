#!/usr/bin/env python3
"""Create deterministic, word-disjoint and exact-sentence-clean corpus splits."""

import argparse
import csv
import hashlib
import json
import random
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ("A1", "A2", "B1", "B2", "C1")
SEED = 42
MAX_PER_WORD = 10


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def prepare(archive: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if name.endswith("full_results.jsonl")]
        if len(members) != 1:
            raise ValueError(f"expected one full_results.jsonl member, got {len(members)}")
        labels_by_word = defaultdict(set)
        raw_rows = []
        totals = Counter()
        for line in zf.open(members[0]):
            entry = json.loads(line)
            word = norm(str(entry["word"]))
            label = str(entry["cefr"]).strip().upper()
            if not word or label not in LABELS:
                raise ValueError("invalid word or CEFR label")
            labels_by_word[word].add(label)
            totals["entries"] += 1
            for sentence in entry["sentences"]:
                totals["input_rows"] += 1
                text = norm(str(sentence["text"]))
                if not text:
                    totals["empty_sentence"] += 1
                    continue
                if word not in text:
                    totals["target_not_found"] += 1
                    continue
                sentence_hash = digest(text.encode("utf-8"))
                row_id = digest(f"{word}\0{label}\0{sentence_hash}".encode())
                raw_rows.append((word, label, sentence_hash, row_id, text))

    conflicting_words = {word for word, labels in labels_by_word.items() if len(labels) != 1}
    words_by_label = defaultdict(list)
    for word, labels in labels_by_word.items():
        if word not in conflicting_words:
            words_by_label[next(iter(labels))].append(word)

    rng = random.Random(SEED)
    word_split = {}
    for label in LABELS:
        words = sorted(words_by_label[label])
        rng.shuffle(words)
        n_test = round(0.15 * len(words))
        n_dev = round(0.15 * len(words))
        for word in words[:n_test]:
            word_split[word] = "test"
        for word in words[n_test:n_test + n_dev]:
            word_split[word] = "dev"
        for word in words[n_test + n_dev:]:
            word_split[word] = "train"

    priority = {"test": 0, "dev": 1, "train": 2}
    sentence_owner = {}
    for word, _, sentence_hash, _, _ in raw_rows:
        if word in conflicting_words:
            continue
        split = word_split[word]
        old = sentence_owner.get(sentence_hash)
        if old is None or priority[split] < priority[old]:
            sentence_owner[sentence_hash] = split

    unique_rows = {}
    drops = Counter()
    for word, label, sentence_hash, row_id, text in raw_rows:
        if word in conflicting_words:
            drops["conflicting_word_label"] += 1
            continue
        split = word_split[word]
        if sentence_owner[sentence_hash] != split:
            drops["cross_split_sentence"] += 1
            continue
        key = (split, word, label, sentence_hash)
        if key in unique_rows:
            drops["duplicate_target_sentence"] += 1
            continue
        unique_rows[key] = (row_id, text)

    grouped = defaultdict(list)
    for (split, word, label, sentence_hash), (row_id, text) in unique_rows.items():
        grouped[(split, word)].append((row_id, text, sentence_hash, label))

    selected = {split: [] for split in ("train", "dev", "test")}
    for (split, word), rows in grouped.items():
        for row_id, text, sentence_hash, label in sorted(rows)[:MAX_PER_WORD]:
            selected[split].append({
                "id": row_id,
                "sentence": text,
                "target": word,
                "lemma": word,
                "cefr": label,
                "sentence_hash": sentence_hash,
            })

    if any(not selected[name] for name in selected):
        raise ValueError("at least one split is empty")
    for name, rows in selected.items():
        if {row["cefr"] for row in rows} != set(LABELS):
            raise ValueError(f"split {name} does not contain all CEFR labels")

    output.mkdir(parents=True)
    counts = {}
    split_hashes = {}
    for name, rows in selected.items():
        rows.sort(key=lambda row: row["id"])
        split_hashes[name] = digest("\n".join(row["id"] for row in rows).encode())
        counts[name] = {
            "rows": len(rows),
            "words": len({row["lemma"] for row in rows}),
            "labels": dict(sorted(Counter(row["cefr"] for row in rows).items())),
        }
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("id", "sentence", "target", "lemma", "cefr", "sentence_hash"))
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "archive_sha256": archive_hash,
        "member": members[0],
        "seed": SEED,
        "split_policy": "surface-word stratified 70/15/15; exact normalized sentence hashes assigned test then dev then train; one conflicting-label word group excluded; unmatched lemma rows excluded; duplicate target-sentence pairs removed; cap 10 rows per word by SHA256 ordering",
        "max_rows_per_word": MAX_PER_WORD,
        "totals": dict(totals),
        "dropped": dict(drops),
        "conflicting_word_groups": len(conflicting_words),
        "word_groups_by_split": {name: sum(split == name for split in word_split.values()) for name in ("train", "dev", "test")},
        "counts": counts,
        "split_sha256": split_hashes,
        "limitations": ["source/document identifiers absent", "near-duplicate documents not verifiable", "target occurrence approximated by normalized lemma substring"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.archive, args.output)
    print(json.dumps({"counts": manifest["counts"], "dropped": manifest["dropped"], "split_sha256": manifest["split_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
