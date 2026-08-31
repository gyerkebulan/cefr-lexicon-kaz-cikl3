#!/usr/bin/env python3
"""Reconstruct corpus-derived CEFR features without mutating source splits."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import hfst
import numpy as np
import pandas as pd

KAZAKH_WORD = re.compile(r"^[а-яәғқңөұүһі]+$", re.IGNORECASE)
TOKEN = re.compile(r"[а-яәғқңөұүһі]+", re.IGNORECASE)
TAGS = re.compile(r"<([^>]+)>")
CASE_TAGS = {"nom", "gen", "dat", "acc", "abl", "loc", "ins"}
POSS_TAGS = {"px1sg", "px2sg", "px3sp", "px1pl", "px2pl", "px3pl"}
TENSE_TAGS = {"aor", "past", "pres", "fut", "ifi", "ifi_evid", "hab", "hab_evid"}
DERIV_TAGS = {
    "attr", "subst", "advl", "ger", "ger_past", "ger_perf", "ger_ppot",
    "ger_abs", "ger_fut", "gpr_past", "gpr_perf", "gpr_ppot", "gpr_fut",
    "gpr_impf", "gpr_pot",
}
POS_TAGS = {
    "n", "v", "adj", "adv", "num", "prn", "det", "post", "cnjcoo",
    "cnjsub", "ij", "cop", "vaux", "abbr",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_analyzer(path: Path):
    stream = hfst.HfstInputStream(str(path))
    analyzer = stream.read()
    stream.close()
    analyzer.invert()
    analyzer.convert(hfst.ImplementationType.HFST_OLW_TYPE)
    return analyzer


def parse_analysis(raw: str):
    raw = raw.replace("@_EPSILON_SYMBOL_@", "")
    parts = raw.split("+")
    lemma = re.sub(r"<[^>]+>", "", parts[0]).strip().lower()
    tags = [tag for part in parts for tag in TAGS.findall(part)]
    return lemma, tags


def analyses(analyzer, word: str):
    return [parse_analysis(raw) for raw, _ in analyzer.lookup(word.lower())]


def build_frequencies(analyzer, word_files):
    surface_counts, lemma_counts = Counter(), Counter()
    rows = accepted_tokens = recognized_rows = recognized_tokens = 0
    for path in word_files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                word = parts[1].strip().lower()
                if not KAZAKH_WORD.fullmatch(word):
                    continue
                try:
                    count = int(parts[2])
                except ValueError:
                    continue
                if count <= 0:
                    continue
                parsed = analyses(analyzer, word)
                lemma = min(parsed, key=lambda item: len(item[1]))[0] if parsed else word
                surface_counts[word] += count
                lemma_counts[lemma or word] += count
                rows += 1
                accepted_tokens += count
                recognized_rows += bool(parsed)
                recognized_tokens += count if parsed else 0
    return surface_counts, lemma_counts, {
        "accepted_type_rows": rows,
        "accepted_tokens": accepted_tokens,
        "recognized_type_rows": recognized_rows,
        "recognized_type_ratio": recognized_rows / rows if rows else 0.0,
        "recognized_tokens": recognized_tokens,
        "recognized_token_ratio": recognized_tokens / accepted_tokens if accepted_tokens else 0.0,
    }


def morph_features(analyzer, word: str):
    parsed = analyses(analyzer, word)
    result = {
        "morph_recognized": 0, "n_analyses": 0, "n_morphemes_min": 0,
        "n_morphemes_max": 0, "n_morphemes_avg": 0.0,
        "derivational_depth": 0, "has_copula": 0, "has_case": 0,
        "has_possession": 0, "has_plural": 0, "has_tense": 0,
        "n_unique_pos": 0, "morph_ambiguity": 0.0,
        "morph_complexity_score": 0.0, "apertium_pos_n": 0,
        "apertium_pos_v": 0, "apertium_pos_adj": 0,
        "apertium_pos_adv": 0, "apertium_pos_other": 0,
    }
    if not parsed:
        return result
    tag_lists = [tags for _, tags in parsed]
    morphemes = [len(tags) for tags in tag_lists]
    pos = {tag for tags in tag_lists for tag in tags if tag in POS_TAGS}
    derivational_depth = max(sum(tag in DERIV_TAGS for tag in tags) for tags in tag_lists)
    result.update({
        "morph_recognized": 1,
        "n_analyses": len(parsed),
        "n_morphemes_min": min(morphemes),
        "n_morphemes_max": max(morphemes),
        "n_morphemes_avg": float(np.mean(morphemes)),
        "derivational_depth": derivational_depth,
        "has_copula": int(any("cop" in tags for tags in tag_lists)),
        "has_case": int(any(CASE_TAGS.intersection(tags) for tags in tag_lists)),
        "has_possession": int(any(POSS_TAGS.intersection(tags) for tags in tag_lists)),
        "has_plural": int(any("pl" in tags for tags in tag_lists)),
        "has_tense": int(any(TENSE_TAGS.intersection(tags) for tags in tag_lists)),
        "n_unique_pos": len(pos),
        "morph_ambiguity": math.log1p(len(parsed)),
        "apertium_pos_n": int("n" in pos),
        "apertium_pos_v": int("v" in pos),
        "apertium_pos_adj": int("adj" in pos),
        "apertium_pos_adv": int("adv" in pos),
        "apertium_pos_other": int(bool(pos - {"n", "v", "adj", "adv"})),
    })
    result["morph_complexity_score"] = (
        result["n_morphemes_avg"] * 0.3
        + derivational_depth * 0.3
        + len(pos) * 0.2
        + result["morph_ambiguity"] * 0.2
    )
    return result


def build_document_features(sentence_files, targets):
    document_frequency, term_frequency = Counter(), Counter()
    source_presence = defaultdict(set)
    documents = 0
    for source_index, path in enumerate(sentence_files):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                tokens = [token.lower() for token in TOKEN.findall(parts[1])]
                matched = [token for token in tokens if token in targets]
                term_frequency.update(matched)
                for token in set(matched):
                    document_frequency[token] += 1
                    source_presence[token].add(source_index)
                documents += 1
    features = {}
    for lemma in targets:
        df = document_frequency[lemma]
        tf = term_frequency[lemma]
        idf = math.log(documents / df) if df else math.log(documents)
        features[lemma] = {
            "idf": idf,
            "df_ratio": df / documents,
            "log_df": math.log(df + 1),
            "tf_idf": (1 + math.log(tf)) * idf if tf else 0.0,
            "doc_spread": len(source_presence[lemma]),
        }
    return features, {
        "documents": documents,
        "covered_targets": sum(document_frequency[lemma] > 0 for lemma in targets),
        "target_count": len(targets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--word-file", type=Path, action="append", required=True)
    parser.add_argument("--sentence-file", type=Path, action="append", default=[])
    args = parser.parse_args()

    split_frames = {}
    targets = set()
    for name in ("train", "dev", "test"):
        path = args.splits_dir / f"{name}.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        missing = {"lemma", "pos", "cefr"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        split_frames[name] = frame
        targets.update(frame["lemma"].astype(str).str.strip().str.lower())

    analyzer = load_analyzer(args.analyzer)
    surface, lemma, frequency_stats = build_frequencies(analyzer, args.word_file)
    document, document_stats = build_document_features(args.sentence_file, targets)
    morph = {target: morph_features(analyzer, target) for target in targets}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "surface_freqs.json").write_text(
        json.dumps(surface, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "lemma_freqs.json").write_text(
        json.dumps(lemma, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "idf_features.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    coverage = {}
    for name, source in split_frames.items():
        frame = source.copy()
        normalized = frame["lemma"].astype(str).str.strip().str.lower()
        frame["freq"] = normalized.map(surface).fillna(0).astype(float)
        frame["form_freq"] = frame["freq"]
        frame["lemma_freq_raw"] = normalized.map(lemma).fillna(0).astype(float)
        frame["lemma_in_freq_lexicon"] = (frame["lemma_freq_raw"] > 0).astype(float)
        frame["log_freq_lemma"] = np.log10(frame["lemma_freq_raw"] + 1)
        for column in ("idf", "df_ratio", "log_df", "tf_idf", "doc_spread"):
            frame[column] = normalized.map(lambda value: document[value][column])
        morph_frame = pd.DataFrame(normalized.map(morph).tolist())
        for column in morph_frame:
            frame[column] = morph_frame[column].values
        pd.testing.assert_frame_equal(
            frame[["lemma", "pos", "cefr"]], source[["lemma", "pos", "cefr"]]
        )
        output = args.output_dir / f"{name}.csv"
        frame.to_csv(output, index=False)
        coverage[name] = {
            "rows": len(frame),
            "surface_frequency": int((frame["freq"] > 0).sum()),
            "lemma_frequency": int((frame["lemma_freq_raw"] > 0).sum()),
            "morphology": int(frame["morph_recognized"].sum()),
            "document_frequency": int((frame["df_ratio"] > 0).sum()),
            "sha256": sha256(output),
        }

    inputs = [args.analyzer, *args.word_file, *args.sentence_file]
    inputs.extend(args.splits_dir / f"{name}.csv" for name in ("train", "dev", "test"))
    manifest = {
        "analyzer_commit": "95c6dd0d8536ee69a7058634b03a3e82100b6b6e",
        "inputs": {str(path): sha256(path) for path in inputs},
        "frequency": frequency_stats,
        "document_frequency": document_stats,
        "coverage": coverage,
        "outputs": {
            name: sha256(args.output_dir / name)
            for name in ("surface_freqs.json", "lemma_freqs.json", "idf_features.json")
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    assert parse_analysis("бала<n><pl><nom>+е<cop>") == (
        "бала", ["n", "pl", "nom", "cop"]
    )
    main()
