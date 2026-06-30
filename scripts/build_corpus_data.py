#!/usr/bin/env python3
"""
Build corpus frequency map and context sentence cache from Leipzig corpora.

Reads all *-words.txt and *-sentences.txt files from the monolingual corpus
directory and produces:
  1. artifacts/frequencies/lemma_freqs.json  — {word: total_count}
  2. artifacts/context/lemma_sentences.json  — {lemma: example_sentence}
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "dataset" / "raw" / "monolingual"
FREQ_OUTPUT = PROJECT_ROOT / "artifacts" / "frequencies" / "lemma_freqs.json"
CONTEXT_OUTPUT = PROJECT_ROOT / "artifacts" / "context" / "lemma_sentences.json"
LEXICON_PATH = PROJECT_ROOT / "dataset" / "processed" / "combined_lexicon.json"

KAZAKH_TOKEN_RE = re.compile(r"[a-zа-яәғқңөұүһі]+", re.IGNORECASE)


def build_frequencies():
    """Merge all Leipzig *-words.txt files into a single frequency dict."""
    freq = Counter()
    word_files = sorted(CORPUS_DIR.rglob("*-words.txt"))
    print(f"Found {len(word_files)} word frequency files")

    for wf in word_files:
        print(f"  Processing {wf.name}...")
        with wf.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    word = parts[1].lower().strip()
                    try:
                        count = int(parts[2])
                    except ValueError:
                        continue
                    # Skip punctuation-only entries
                    if KAZAKH_TOKEN_RE.search(word):
                        freq[word] += count

    FREQ_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with FREQ_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(dict(freq), f, ensure_ascii=False, indent=2)
    print(f"Saved {len(freq)} word frequencies to {FREQ_OUTPUT}")
    return freq


def build_context_sentences(target_lemmas=None):
    """Find one example sentence per lemma from Leipzig sentence files."""
    if target_lemmas is None:
        # Load from lexicon or splits
        target_lemmas = set()
        for split in ["train", "dev", "test"]:
            csv_path = PROJECT_ROOT / "dataset" / "splits" / f"{split}.csv"
            if csv_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                target_lemmas.update(df["lemma"].astype(str).str.lower().tolist())
        if LEXICON_PATH.exists():
            with LEXICON_PATH.open(encoding="utf-8") as f:
                lexicon = json.load(f)
            for entry in lexicon:
                target_lemmas.add(str(entry.get("word", "")).lower().strip())

    if not target_lemmas:
        print("No target lemmas found, skipping context sentence extraction")
        return {}

    print(f"Finding context sentences for {len(target_lemmas)} lemmas...")
    remaining = set(target_lemmas)
    context_map = {}

    sent_files = sorted(CORPUS_DIR.rglob("*-sentences.txt"))
    print(f"Scanning {len(sent_files)} sentence files...")

    for sf in sent_files:
        print(f"  Scanning {sf.name}...")
        with sf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                sentence = parts[1] if len(parts) == 2 else parts[0]
                tokens = set(KAZAKH_TOKEN_RE.findall(sentence.lower()))
                matches = tokens & remaining
                for lemma in matches:
                    context_map[lemma] = sentence
                remaining -= matches
                if not remaining:
                    break
        if not remaining:
            break

    CONTEXT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CONTEXT_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(context_map, f, ensure_ascii=False, indent=2)

    coverage = len(context_map)
    total = len(target_lemmas)
    print(f"Context coverage: {coverage}/{total} ({100.0 * coverage / max(total, 1):.1f}%)")
    print(f"Saved to {CONTEXT_OUTPUT}")
    return context_map


if __name__ == "__main__":
    print("=== Building corpus frequency map ===")
    freq = build_frequencies()

    print("\n=== Building context sentence cache ===")
    # Load lexicon lemmas from combined_final.json if available
    target_lemmas = set()
    alt_lexicon = Path("/Users/zhantore/Documents/work/cefr-classification-kk/combined_final.json")
    if alt_lexicon.exists():
        with alt_lexicon.open(encoding="utf-8") as f:
            lexicon = json.load(f)
        for entry in lexicon:
            target_lemmas.add(str(entry.get("word", "")).lower().strip())
        print(f"Loaded {len(target_lemmas)} lemmas from combined_final.json")

    build_context_sentences(target_lemmas if target_lemmas else None)
    print("\nDone!")
