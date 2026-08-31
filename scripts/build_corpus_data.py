#!/usr/bin/env python3
"""Build versioned lemma frequencies and multi-sentence context resources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "dataset/raw/monolingual"
DEFAULT_LEXICON = PROJECT_ROOT / "dataset/processed/combined_lexicon.csv"
TOKEN = re.compile(r"[a-zа-яәғқңөұүһі]+", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_lemmas(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if "lemma" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no lemma column")
        return {
            str(row["lemma"]).strip().lower()
            for row in reader
            if str(row.get("lemma", "")).strip()
        }


def build_context_sentences(
    sentence_files: list[Path], target_lemmas: set[str], max_contexts: int
) -> tuple[dict[str, list[str]], dict]:
    """Keep source-balanced contexts; zero means all unique matched sentences."""
    if max_contexts < 0:
        raise ValueError("max_contexts must be nonnegative (0 means unlimited)")
    # ponytail: hold contexts in RAM; use a disk-backed index if corpora outgrow memory.
    candidates = defaultdict(lambda: defaultdict(dict if max_contexts == 0 else list))
    for path in sentence_files:
        source = str(path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                parts = line.rstrip("\n").split("\t", 1)
                sentence = " ".join((parts[1] if len(parts) == 2 else parts[0]).split())
                if not sentence:
                    continue
                for lemma in set(TOKEN.findall(sentence.lower())) & target_lemmas:
                    heap = candidates[lemma][source]
                    if (sentence in heap if max_contexts == 0 else
                            any(existing == sentence for _, existing in heap)):
                        continue
                    rank = int.from_bytes(
                        hashlib.sha256(
                            f"{lemma}\0{source}\0{sentence}".encode("utf-8")
                        ).digest(),
                        "big",
                    )
                    item = (-rank, sentence)
                    if max_contexts == 0:
                        heap[sentence] = item
                    elif len(heap) < max_contexts:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)

    context_map, source_counts = {}, {}
    for lemma in sorted(candidates):
        queues = {
            source: deque(sentence for _, sentence in sorted(
                heap.values() if max_contexts == 0 else heap,
                key=lambda item: -item[0],
            ))
            for source, heap in candidates[lemma].items()
        }
        selected, selected_sources, seen = [], set(), set()
        while (max_contexts == 0 or len(selected) < max_contexts) and any(queues.values()):
            for source in sorted(queues):
                if not queues[source]:
                    continue
                sentence = queues[source].popleft()
                if sentence in seen:
                    continue
                selected.append(sentence)
                selected_sources.add(source)
                seen.add(sentence)
                if max_contexts and len(selected) == max_contexts:
                    break
        if selected:
            context_map[lemma] = selected
            source_counts[lemma] = len(selected_sources)

    counts = Counter(map(len, context_map.values()))
    diversity = Counter(source_counts.values())
    stats = {
        "target_lemmas": len(target_lemmas),
        "covered_lemmas": len(context_map),
        "coverage": len(context_map) / len(target_lemmas) if target_lemmas else 0.0,
        "stored_sentences": sum(count * lemmas for count, lemmas in counts.items()),
        "max_contexts_per_lemma": max_contexts or None,
        "contexts_per_lemma": {str(key): value for key, value in sorted(counts.items())},
        "sources_per_lemma": {str(key): value for key, value in sorted(diversity.items())},
    }
    return context_map, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzer", type=Path)
    parser.add_argument("--contexts-only", action="store_true",
                        help="Build contexts without recomputing frequency resources")
    parser.add_argument("--analyzer-commit", default="unknown")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--legacy-contexts", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-contexts", type=int, default=10,
                        help="Maximum sentences per lemma; 0 keeps all unique matches")
    args = parser.parse_args()
    if args.max_contexts < 0:
        parser.error("--max-contexts must be nonnegative")
    if not args.contexts_only and args.analyzer is None:
        parser.error("--analyzer is required unless --contexts-only is set")

    word_files = [] if args.contexts_only else sorted(args.corpus_dir.rglob("*-words.txt"))
    sentence_files = sorted(args.corpus_dir.rglob("*-sentences.txt"))
    if not sentence_files or (not args.contexts_only and not word_files):
        raise FileNotFoundError(f"No Leipzig word/sentence files under {args.corpus_dir}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    target_lemmas = load_lemmas(args.lexicon)
    payloads, frequency_stats, analyzer_record = {}, None, None
    if not args.contexts_only:
        # Reuse the tested HFST aggregation from the corpus reconstruction experiment.
        from scripts.reconstruct_corpus_features import build_frequencies, load_analyzer

        analyzer = load_analyzer(args.analyzer)
        surface, lemma, frequency_stats = build_frequencies(analyzer, word_files)
        if sum(surface.values()) != frequency_stats["accepted_tokens"]:
            raise RuntimeError("Surface counts failed the token-total check")
        if sum(lemma.values()) != frequency_stats["accepted_tokens"]:
            raise RuntimeError("Lemma counts failed the token-total check")
        payloads.update(surface_freqs=surface, lemma_freqs=lemma)
        frequency_stats.update(surface_entries=len(surface), lemma_entries=len(lemma))
        analyzer_record = {**file_record(args.analyzer), "apertium_kaz_commit": args.analyzer_commit}
    contexts, context_stats = build_context_sentences(
        sentence_files, target_lemmas, args.max_contexts
    )
    recovered = 0
    if args.legacy_contexts:
        legacy = json.loads(args.legacy_contexts.read_text(encoding="utf-8"))
        for raw_lemma, raw_values in legacy.items():
            normalized_lemma = str(raw_lemma).strip().lower()
            if normalized_lemma not in target_lemmas or normalized_lemma in contexts:
                continue
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            sentence = next(
                (
                    " ".join(value.split())
                    for value in values
                    if isinstance(value, str)
                    and normalized_lemma in set(TOKEN.findall(value.lower()))
                ),
                "",
            )
            if sentence:
                contexts[normalized_lemma] = [sentence]
                recovered += 1
        counts = Counter(map(len, contexts.values()))
        context_stats.update({
            "covered_lemmas": len(contexts),
            "coverage": len(contexts) / len(target_lemmas),
            "stored_sentences": sum(count * lemmas for count, lemmas in counts.items()),
            "contexts_per_lemma": {str(key): value for key, value in sorted(counts.items())},
            "legacy_recovered_lemmas": recovered,
        })
        context_stats["sources_per_lemma"]["1"] = (
            context_stats["sources_per_lemma"].get("1", 0) + recovered
        )
    if any(not values or (args.max_contexts and len(values) > args.max_contexts)
           for values in contexts.values()):
        raise RuntimeError("Context output failed its cardinality check")

    args.output_dir.mkdir(parents=True)
    payloads["lemma_sentences"] = contexts
    outputs = {name: args.output_dir / f"{name}.json" for name in payloads}
    for name, payload in payloads.items():
        write_json(outputs[name], payload)
    manifest = {
        "schema_version": 2,
        "frequency_semantics": None if args.contexts_only else "HFST-lemmatized Leipzig word-table token counts",
        "context_semantics": ("all unique exact-token-matched sentences; deterministic source-balanced order"
                              if args.max_contexts == 0 else "deterministic source-balanced sentence samples"),
        "analyzer": analyzer_record,
        "lexicon": file_record(args.lexicon),
        "inputs": {
            "word_files": [file_record(path) for path in word_files],
            "sentence_files": [file_record(path) for path in sentence_files],
            "legacy_contexts": file_record(args.legacy_contexts) if args.legacy_contexts else None,
        },
        "frequency_stats": frequency_stats,
        "context_stats": context_stats,
        "outputs": {name: file_record(path) for name, path in outputs.items()},
        "builder": file_record(Path(__file__)),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
