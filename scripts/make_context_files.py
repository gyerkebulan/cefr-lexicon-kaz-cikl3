#!/usr/bin/env python3
"""Build multi-context files for train rotation and dev/test evaluation from the context pool (lemma -> list of sentences).

    python make_context_files.py --dataset dataset/research/ar-cycle3-clean3563-20261020 \
        --pool artifacts/corpus-resources-v2-20260823/lemma_sentences.json --max-contexts 10

Rule: a pool sentence is kept for a lemma only if the same sentence (NFC, casefold, whitespace-normalised) occurs in no pool list
or chosen sentence of a lemma from another split. Only rows with context_available = True receive extra contexts.
Outputs next to the dataset: train_contexts.json, dev_contexts.json, test_contexts.json, context-files-audit.json.
Format (read by scripts/train_dual.py): {"<lemma, lower case>": ["sentence", ...]}.
"""
import argparse, hashlib, json, unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


def norm(v):
    return unicodedata.normalize("NFC", str(v).strip()).casefold()


def key(sentence):
    return hashlib.sha256(" ".join(norm(sentence).split()).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--max-contexts", type=int, default=10)
    args = ap.parse_args()

    frames = {n: pd.read_csv(args.dataset / f"{n}.csv", keep_default_na=False) for n in ("train", "dev", "test")}
    pool_raw = json.loads(args.pool.read_text(encoding="utf-8"))
    pool = {norm(k): [s for s in (v if isinstance(v, list) else [v]) if isinstance(s, str) and s.strip()] for k, v in pool_raw.items()}

    owner = defaultdict(set)  # sentence key -> splits that could use it
    for n, f in frames.items():
        for lemma, sentence, ok in zip(f.normalized_lemma, f.sentence, f.context_available):
            if str(ok).lower() == "true":
                owner[key(sentence)].add(n)
                for s in pool.get(lemma, []):
                    owner[key(s)].add(n)

    audit = {"max_contexts": args.max_contexts, "pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest()}
    for n, f in frames.items():
        out, removed, extra = {}, 0, 0
        for lemma, sentence, ok in zip(f.normalized_lemma, f.sentence, f.context_available):
            if str(ok).lower() != "true":
                continue
            seen, items = set(), []
            for s in [sentence] + pool.get(lemma, []):
                k = key(s)
                if k in seen:
                    continue
                if owner[k] != {n}:
                    removed += 1
                    continue
                seen.add(k); items.append(s.strip())
                if len(items) == args.max_contexts:
                    break
            if items:
                out[lemma] = items
                extra += len(items) - 1
        (args.dataset / f"{n}_contexts.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        audit[n] = {"lemmas": len(out), "extra_contexts": extra, "removed_shared_sentences": removed,
                    "mean_contexts": round(sum(map(len, out.values())) / max(len(out), 1), 3)}
    keys = {n: {key(s) for v in json.loads((args.dataset / f"{n}_contexts.json").read_text(encoding="utf-8")).values() for s in v}
            for n in frames}
    audit["cross_split_sentence_overlap"] = {f"{a}_{b}": len(keys[a] & keys[b]) for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    (args.dataset / "context-files-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    if any(audit["cross_split_sentence_overlap"].values()):
        raise SystemExit("Sentence overlap between splits")


if __name__ == "__main__":
    main()
