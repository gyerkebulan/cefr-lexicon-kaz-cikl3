#!/usr/bin/env python3
"""Build the cycle 3 grouped train/dev/test sets from words_clean3563.xlsx together with context sentences.

Run from the root of the cefr-lexicon-kaz repository on the machine that holds the context cache and legacy frequency splits:

    python build_cycle3_dataset.py \
        --source words_clean3563.xlsx \
        --context-cache artifacts/context/lemma_sentences.json \
        --legacy-splits dataset/splits \
        --out-dir dataset/research/ar-cycle3-clean3563-20261020

Groups (a group always goes to one split) join rows that share: the normalized word, a link in the `pair` column,
an edit distance <= 1 between words of length >= 6, or the same context sentence (hash). Allocation is by quota over
CEFR level, ambiguity, part of speech and group size. All overlap checks (word, group, near duplicate, pair partner,
context hash) must be zero, otherwise the script stops.
Output columns follow scripts/build_research_splits.py:
source_row_id, lemma, normalized_lemma, pos, cefr, freq, sentence, target, context_available, context_hash, group_id.
"""
import argparse, hashlib, json, unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

POS_MAP = {"зат есім": "NOUN", "етістік": "VERB", "сын есім": "ADJ", "үстеу": "ADV", "есімдік": "PRON", "сан есім": "NUM",
           "шылау": "ADP", "модаль сөз": "MODAL", "одағай": "INTJ", "еліктеу": "OTHER", "көмекші сөз": "AUX"}
LEVELS = ["A1", "A2", "B1", "B2", "C1"]
SEED, MINLEN, PROP = 42, 6, {"train": 0.70, "dev": 0.15, "test": 0.15}
STRAT = ("cefr", "amb", "pos", "gsize")


def norm(v):
    return unicodedata.normalize("NFC", str(v).strip()).casefold()


def sentence_hash(v):
    return hashlib.sha256(" ".join(norm(v).split()).encode("utf-8")).hexdigest()


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ed1(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    i = j = d = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1; j += 1
        else:
            d += 1; j += 1
            if d > 1:
                return False
    return True


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def assign_groups(df, groups, proportions, seed, strat):
    rng = np.random.default_rng(seed)
    names = list(proportions)
    d = df.assign(all_="all")
    cols = list(strat) + ["all_"]
    totals = {c: Counter(d[c].astype(str)) for c in cols}
    quota = {n: {c: {v: proportions[n] * k for v, k in totals[c].items()} for c in cols} for n in names}
    got = {n: {c: Counter() for c in cols} for n in names}
    order = list(range(len(groups)))
    rng.shuffle(order)
    order.sort(key=lambda g: len(groups[g]), reverse=True)
    assigned = {n: [] for n in names}
    for g in order:
        idx = groups[g]
        comp = {c: Counter(d.loc[idx, c].astype(str)) for c in cols}
        best, best_score = None, None
        for n in names:
            score = sum(k * (quota[n][c][v] - got[n][c][v]) / (quota[n][c][v] + 1.0) for c in cols for v, k in comp[c].items())
            if best_score is None or score > best_score:
                best, best_score = n, score
        assigned[best].extend(idx)
        for c in cols:
            got[best][c].update(comp[c])
    return assigned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--context-cache", type=Path, required=True)
    ap.add_argument("--legacy-splits", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_excel(args.source)
    unknown = sorted(set(df.pos) - set(POS_MAP))
    if unknown:
        raise ValueError(f"Unmapped parts of speech: {unknown}")
    df.insert(0, "source_row_id", range(len(df)))
    df["lemma"] = df.word.map(lambda v: unicodedata.normalize("NFC", str(v).strip()))
    df["normalized_lemma"] = df.lemma.map(norm)
    df["pos"] = df.pos.map(POS_MAP)
    df["cefr"] = df.cefr_level.astype(str).str.strip().str.upper()
    assert set(df.cefr) <= set(LEVELS)
    cnt = df.groupby("normalized_lemma").size()
    df["amb"] = ((df.normalized_lemma.map(cnt) > 1) | df.note.notna()).astype(int)

    freq_pos, freq_lemma = {}, {}
    if args.legacy_splits:
        old = pd.concat([pd.read_csv(args.legacy_splits / f"{n}.csv") for n in ("train", "dev", "test")], ignore_index=True)
        old = old[old.lemma.notna()].copy()
        old["nl"] = old.lemma.map(norm)
        old["pos"] = old.pos.fillna("OTHER").astype(str).str.strip().str.upper()
        freq_pos = old.groupby(["nl", "pos"]).freq.max().to_dict()
        freq_lemma = old.groupby("nl").freq.max().to_dict()
    df["freq"] = [float(freq_pos.get((nl, p), freq_lemma.get(nl, 0.0))) for nl, p in zip(df.normalized_lemma, df.pos)]

    cache = json.loads(args.context_cache.read_text(encoding="utf-8"))
    ctx = {}
    for key, value in cache.items():
        text = next((x for x in value if isinstance(x, str) and x.strip()), "") if isinstance(value, list) else str(value)
        if text.strip():
            ctx[norm(key)] = text
    df["target"] = df.lemma
    df["sentence"] = df.normalized_lemma.map(ctx).fillna(df.lemma)
    df["context_available"] = df.normalized_lemma.isin(ctx)
    a = df.groupby("normalized_lemma").agg(pc=("pos", "nunique"), cc=("cefr", "nunique"))
    ambiguous = set(a.index[(a.pc > 1) & (a.cc > 1)])
    m = df.normalized_lemma.isin(ambiguous)
    df.loc[m, "sentence"] = df.loc[m, "target"]
    df.loc[m, "context_available"] = False
    df["context_hash"] = [sentence_hash(s) if ok else "" for s, ok in zip(df.sentence, df.context_available)]

    # ---- groups ----
    uf = UF(len(df)); first = {}
    for i, w in enumerate(df.normalized_lemma):
        if w in first:
            uf.union(i, first[w])
        first.setdefault(w, i)
    for i, p in enumerate(df.pair):
        if isinstance(p, str):
            for tok in (norm(t) for t in p.split("/")):
                if tok in first:
                    uf.union(i, first[tok])
    byhash = {}
    for i, h in enumerate(df.context_hash):
        if h:
            if h in byhash:
                uf.union(i, byhash[h])
            byhash.setdefault(h, i)
    index = defaultdict(list)
    for i, w in enumerate(df.normalized_lemma):
        if len(w) < MINLEN:
            continue
        for sig in {w} | {w[:k] + w[k + 1:] for k in range(len(w))}:
            for j in index[sig]:
                if df.normalized_lemma[j] != w and ed1(w, df.normalized_lemma[j]):
                    uf.union(i, j)
            index[sig].append(i)
    roots = defaultdict(list)
    for i in range(len(df)):
        roots[uf.find(i)].append(i)
    groups = sorted(roots.values(), key=min)
    gid = {i: k for k, g in enumerate(groups) for i in g}
    df["group_id"] = [gid[i] for i in range(len(df))]
    df["gsize"] = df.group_id.map(df.groupby("group_id").size()).clip(upper=4)
    print(f"groups {len(groups)}, largest {sorted((len(g) for g in groups), reverse=True)[:6]}")

    asg = assign_groups(df, groups, PROP, SEED, STRAT)
    out = {n: df.loc[sorted(ix)].reset_index(drop=True) for n, ix in asg.items()}

    # ---- checks ----
    checks = {}
    for l, r in combinations(out, 2):
        for col in ("normalized_lemma", "group_id"):
            checks[f"{col}_{l}_{r}_overlap"] = len(set(out[l][col]) & set(out[r][col]))
        checks[f"context_hash_{l}_{r}_overlap"] = len((set(out[l].context_hash) - {""}) & (set(out[r].context_hash) - {""}))
        rw = {w for w in out[r].normalized_lemma if len(w) >= MINLEN}
        sig = defaultdict(list)
        for w in rw:
            for s_ in {w} | {w[:k] + w[k + 1:] for k in range(len(w))}:
                sig[s_].append(w)
        hit = 0
        for w in (w for w in out[l].normalized_lemma if len(w) >= MINLEN):
            if any(x != w and ed1(w, x) for s_ in ({w} | {w[:k] + w[k + 1:] for k in range(len(w))}) for x in sig.get(s_, [])):
                hit += 1
        checks[f"near_duplicate_len{MINLEN}plus_{l}_{r}_overlap"] = hit
    where = {w: n for n, f in out.items() for w in f.normalized_lemma}
    checks["pair_partners_in_different_splits"] = sum(
        1 for p, w in zip(df.pair, df.normalized_lemma)
        if isinstance(p, str) and len({where[w]} | {where[norm(t)] for t in p.split("/") if norm(t) in where}) > 1)
    print("checks:", checks)

    cols = ["source_row_id", "lemma", "normalized_lemma", "pos", "cefr", "freq", "sentence", "target", "context_available",
            "context_hash", "group_id"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, f in out.items():
        f[cols].to_csv(args.out_dir / f"{n}.csv", index=False)
    manifest = {
        "series_id": "ar-cycle3-clean3563-20261020", "source_sha256": sha256_file(args.source),
        "context_cache_sha256": sha256_file(args.context_cache),
        "protocol": {"seed": SEED, "proportions": PROP, "stratify": list(STRAT), "min_len_edit_distance": MINLEN,
                     "group_links": ["same word", "pair column", "same context sentence hash", "edit distance <= 1 (len >= 6)"]},
        "counts": {n: len(f) for n, f in out.items()}, "groups_total": len(groups),
        "class_counts": {n: f.cefr.value_counts().reindex(LEVELS, fill_value=0).to_dict() for n, f in out.items()},
        "context_available_share": {n: round(float(f.context_available.mean()), 4) for n, f in out.items()},
        "ambiguous_lemmas": len(ambiguous), "checks": checks,
        "output_sha256": {n: sha256_file(args.out_dir / f"{n}.csv") for n in out}, "sealed_test_copied": False,
    }
    (args.out_dir / "dataset-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        raise SystemExit(f"Overlap checks failed: {bad}")
    print(json.dumps({k: manifest[k] for k in ("counts", "class_counts", "context_available_share")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
