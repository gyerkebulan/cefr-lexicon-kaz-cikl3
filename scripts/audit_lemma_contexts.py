#!/usr/bin/env python3
"""Write an aggregate-only audit of CEFR lemma, POS and context structure."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd

LEVELS = {label: index for index, label in enumerate(("A1", "A2", "B1", "B2", "C1"))}


def normalize(value: object) -> str:
    return unicodedata.normalize("NFC", str(value).strip()).casefold()


def split_stats(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "labels": {
            label: int(count)
            for label, count in frame["cefr"].value_counts().sort_index().items()
        },
        "context_available": int(frame["context_available"].sum()),
        "target_only": int((~frame["context_available"]).sum()),
        "pos_other": int((frame["pos"] == "OTHER").sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    source = pd.read_csv(args.source)
    source = source.loc[source["lemma"].notna()].copy()
    source["normalized_lemma"] = source["lemma"].map(normalize)
    source["pos"] = source["pos"].fillna("OTHER").astype(str).str.upper()
    source["cefr"] = source["cefr"].astype(str).str.upper()
    source["level"] = source["cefr"].map(LEVELS)
    lemma_groups = source.groupby("normalized_lemma")
    key_groups = source.groupby(["normalized_lemma", "pos"])
    lemma_summary = lemma_groups.agg(
        rows=("cefr", "size"), pos_count=("pos", "nunique"),
        cefr_count=("cefr", "nunique"), min_level=("level", "min"),
        max_level=("level", "max"),
    )

    train, dev = pd.read_csv(args.train), pd.read_csv(args.dev)
    for frame in (train, dev):
        frame["context_available"] = frame["context_available"].astype(bool)
        frame["pos"] = frame["pos"].fillna("OTHER").astype(str).str.upper()
    available = pd.concat([train, dev], ignore_index=True)
    available_summary = available.groupby("normalized_lemma").agg(
        pos_count=("pos", "nunique"), cefr_count=("cefr", "nunique")
    )

    raw_contexts = json.loads(args.contexts.read_text(encoding="utf-8"))
    lengths = [
        len(value) if isinstance(value, list) else int(bool(str(value).strip()))
        for value in raw_contexts.values()
    ]
    wordform_fields = sorted(
        {"word", "wordform", "surface", "surface_form", "form"} & set(source.columns)
    )
    conflicts = key_groups["cefr"].nunique()
    repeated_contexts = (
        available.loc[available["context_available"], "context_hash"].value_counts()
    )
    report = {
        "source": {
            "rows": int(len(source)),
            "unique_lemmas": int(source["normalized_lemma"].nunique()),
            "multi_pos_lemmas": int((lemma_summary["pos_count"] > 1).sum()),
            "multi_cefr_lemmas": int((lemma_summary["cefr_count"] > 1).sum()),
            "multi_pos_multi_cefr_lemmas": int(
                ((lemma_summary["pos_count"] > 1) & (lemma_summary["cefr_count"] > 1)).sum()
            ),
            "same_lemma_pos_conflicts": int((conflicts > 1).sum()),
            "cefr_span_ge_2_lemmas": int(
                ((lemma_summary["max_level"] - lemma_summary["min_level"]) >= 2).sum()
            ),
        },
        "locked_train_dev": {
            "train": split_stats(train),
            "dev": split_stats(dev),
            "multi_pos_multi_cefr_lemmas": int(
                ((available_summary["pos_count"] > 1) & (available_summary["cefr_count"] > 1)).sum()
            ),
            "reused_nonempty_context_hashes": int((repeated_contexts > 1).sum()),
            "rows_using_reused_context_hashes": int(repeated_contexts[repeated_contexts > 1].sum()),
        },
        "multi_context_resource": {
            "lemmas": int(len(raw_contexts)),
            "sentences": int(sum(lengths)),
            "mean_sentences_per_lemma": float(sum(lengths) / len(lengths)),
            "lemmas_with_multiple_sentences": int(sum(length > 1 for length in lengths)),
            "lemmas_with_ten_sentences": int(sum(length == 10 for length in lengths)),
        },
        "wordform_experiment": {
            "available_surface_fields": wordform_fields,
            "status": (
                "possible" if wordform_fields
                else "not_identifiable: source has lemma, POS and CEFR only; independently labelled wordforms are absent"
            ),
        },
    }

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_stats = report["source"]
    context_stats = report["multi_context_resource"]
    lines = [
        "# Аудит лемм, POS и контекстов",
        "",
        "Аудит содержит только агрегаты; sealed test не открывался.",
        "",
        "## Результат",
        "",
        f"- В источнике {source_stats['rows']} строк и {source_stats['unique_lemmas']} уникальных лемм.",
        f"- {source_stats['multi_pos_multi_cefr_lemmas']} лемм имеют одновременно несколько POS и CEFR; это кандидаты на смысловую неоднозначность, а не автоматические ошибки.",
        f"- Конфликтов CEFR внутри строгого ключа lemma+POS: {source_stats['same_lemma_pos_conflicts']}.",
        f"- Лемм с размахом минимум два CEFR-шага: {source_stats['cefr_span_ge_2_lemmas']}.",
        f"- Новый ресурс содержит {context_stats['sentences']} предложений для {context_stats['lemmas']} лемм; у {context_stats['lemmas_with_multiple_sentences']} лемм больше одного контекста.",
        "- Эксперимент по отдельным словоформам пока неидентифицируем: в источнике нет независимо размеченного поля словоформы.",
        "",
        "## Вывод для следующего опыта",
        "",
        "Можно честно проверить multi-context aggregation для несложных lemma+POS групп. Нельзя делать вывод о разных CEFR у словоформ одной леммы без нового датасета парадигм.",
    ]
    (args.output_dir / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
