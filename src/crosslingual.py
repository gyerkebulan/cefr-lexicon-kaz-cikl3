"""
Cross-lingual CEFR utilities.

Translation (kk→ru/tr/en) with caching, CEFR lexicon loading for
Russian (TORFL), Turkish (TFLLex), and English (EFLLex),
and silver label projection.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("cefr")

CEFR_NUMERIC = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


# =============================================================================
# Translation Functions
# =============================================================================

def _load_marian_model(model_name):
    """Load MarianMT model and tokenizer with GPU support."""
    from transformers import MarianMTModel, MarianTokenizer
    import torch

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Loaded {model_name} on {device}")
    return model, tokenizer, device


def translate_batch(
    lemmas: List[str],
    model_name: str,
    cache_path: Optional[Path] = None,
    batch_size: int = 32,
) -> Dict[str, str]:
    """Translate a batch of lemmas using a MarianMT model with caching."""
    import torch

    # Load cache if exists
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        # Only translate missing lemmas
        missing = [l for l in lemmas if l not in cached]
        if not missing:
            logger.info(f"All {len(lemmas)} translations loaded from cache")
            return cached
        logger.info(f"Cache has {len(cached)} entries, translating {len(missing)} new")
    else:
        cached = {}
        missing = list(lemmas)

    if not missing:
        return cached

    model, tokenizer, device = _load_marian_model(model_name)

    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for word, translation in zip(batch, decoded):
            cached[word] = translation.lower().strip()

        if (i + batch_size) % 200 == 0 or (i + batch_size) >= len(missing):
            logger.info(f"  Translated {min(i + batch_size, len(missing))}/{len(missing)}")

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cached, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached translations to {cache_path}")

    return cached


def translate_kk_to_en(lemmas: List[str], cache_path: Optional[Path] = None) -> Dict[str, str]:
    """Translate Kazakh → English via Turkic-English multilingual model."""
    return translate_batch(lemmas, "Helsinki-NLP/opus-mt-trk-en", cache_path)


def translate_kk_to_ru(lemmas: List[str], cache_path: Optional[Path] = None) -> Dict[str, str]:
    """Translate Kazakh → Russian using issai/tilmash (NLLB-based)."""
    return _translate_tilmash(lemmas, "ru", cache_path)


def _translate_tilmash(
    lemmas: List[str], tgt_lang: str = "ru",
    cache_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Translate using issai/tilmash (NLLB-based kk→ru/en/tr)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        missing = [l for l in lemmas if l not in cached]
        if not missing:
            logger.info(f"All {len(lemmas)} translations loaded from cache")
            return cached
        logger.info(f"Cache has {len(cached)} entries, translating {len(missing)} new")
    else:
        cached = {}
        missing = list(lemmas)

    if not missing:
        return cached

    model_name = "issai/tilmash"
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Loaded {model_name} on {device}")

    # tilmash uses NLLB language codes
    lang_map = {"ru": "rus_Cyrl", "en": "eng_Latn", "tr": "tur_Latn"}
    tgt_code = lang_map.get(tgt_lang, tgt_lang)
    tokenizer.src_lang = "kaz_Cyrl"
    batch_size = 32

    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=50,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for word, translation in zip(batch, decoded):
            cached[word] = translation.lower().strip()

        if (i + batch_size) % 200 == 0 or (i + batch_size) >= len(missing):
            logger.info(f"  Translated {min(i + batch_size, len(missing))}/{len(missing)}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cached, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached translations to {cache_path}")

    return cached


def translate_kk_to_en_via_ru(
    lemmas: List[str], cache_path: Optional[Path] = None
) -> Dict[str, str]:
    """Translate Kazakh → English via Russian pivot (kk→ru→en).

    Uses tilmash for kk→ru (high-quality NLLB model) then
    opus-mt-ru-en for ru→en. This gives better English translations
    than direct kk→en via opus-mt-trk-en.
    """
    logger.info("Translating kk→en via Russian pivot (tilmash kk→ru + opus-mt ru→en)")

    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        missing = [l for l in lemmas if l not in cached]
        if not missing:
            logger.info(f"All {len(lemmas)} pivot translations loaded from cache")
            return cached
    else:
        cached = {}
        missing = list(lemmas)

    if not missing:
        return cached

    # Step 1: kk → ru (tilmash, reuse existing kk_ru cache)
    ru_cache = cache_path.parent / "kk_ru_translations.json" if cache_path else None
    kk_to_ru = translate_kk_to_ru(missing, ru_cache)

    # Step 2: ru → en (opus-mt-ru-en)
    ru_words = list(set(kk_to_ru.values()))
    ru_to_en_cache = cache_path.parent / "ru_en_pivot.json" if cache_path else None
    ru_to_en = translate_batch(ru_words, "Helsinki-NLP/opus-mt-ru-en", ru_to_en_cache)

    # Chain: kk → ru → en
    for word in missing:
        ru_word = kk_to_ru.get(word, "")
        cached[word] = ru_to_en.get(ru_word, "")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cached, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached pivot translations to {cache_path}")

    return cached


def translate_kk_to_tr(lemmas: List[str], cache_path: Optional[Path] = None) -> Dict[str, str]:
    """Translate Kazakh → Turkish via pivot (kk→en→tr)."""
    logger.info("Translating kk→tr via English pivot (trk-en + en-tr)")
    return _translate_pivot(lemmas, "en", "tr", cache_path)


def _translate_pivot(
    lemmas: List[str], pivot_lang: str, tgt_lang: str,
    cache_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Two-hop translation: kk → pivot → target.

    Uses opus-mt-trk-en for kk→en (Turkic multilingual model),
    then opus-mt-en-{tgt} for the second leg.
    """
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        missing = [l for l in lemmas if l not in cached]
        if not missing:
            return cached
    else:
        cached = {}
        missing = list(lemmas)

    # Step 1: kk → pivot (use Turkic multilingual model for kk→en)
    pivot_cache = cache_path.parent / f"kk_{pivot_lang}_pivot.json" if cache_path else None
    if pivot_lang == "en":
        kk_to_pivot = translate_batch(missing, "Helsinki-NLP/opus-mt-trk-en", pivot_cache)
    else:
        kk_to_pivot = translate_batch(missing, f"Helsinki-NLP/opus-mt-kk-{pivot_lang}", pivot_cache)

    # Step 2: pivot → target
    pivot_words = list(set(kk_to_pivot.values()))
    pivot_to_tgt = translate_batch(
        pivot_words, f"Helsinki-NLP/opus-mt-{pivot_lang}-{tgt_lang}", None
    )

    for word in missing:
        pivot_word = kk_to_pivot.get(word, "")
        cached[word] = pivot_to_tgt.get(pivot_word, "")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cached, f, ensure_ascii=False, indent=2)

    return cached


# =============================================================================
# CEFR Lexicon Loaders
# =============================================================================

def load_russian_cefr(path: Path) -> Dict[str, str]:
    """Load Russian CEFR lexicon.

    Supports:
    - JSON: {"word": "A1", ...}
    - Kelly project XLS (columns: Lemma, CEFR, POS, Frq abs, Frq ipm)
    - Simple CSV (columns: word, level)
    """
    path = Path(path)
    if not path.exists():
        logger.error(f"Russian CEFR file not found: {path}")
        return {}

    if path.suffix == ".json":
        with open(path) as f:
            result = json.load(f)
        logger.info(f"Loaded {len(result)} Russian CEFR entries from {path}")
        return result

    if path.suffix in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Try common column patterns
    word_col = None
    level_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("word", "lemma", "слово"):
            word_col = c
        if cl in ("level", "cefr", "уровень"):
            level_col = c

    if word_col is None:
        word_col = df.columns[0]
    if level_col is None:
        level_col = df.columns[1]

    result = {}
    for _, row in df.iterrows():
        word = str(row[word_col]).lower().strip()
        level = str(row[level_col]).upper().strip()
        if level in CEFR_NUMERIC:
            result[word] = level
    logger.info(f"Loaded {len(result)} Russian CEFR entries from {path}")
    return result


def load_tfllex(path: Path) -> Dict[str, str]:
    """Load Turkish CEFR lexicon (TFLLex, same format as EFLLex TSV)."""
    path = Path(path)
    if not path.exists():
        logger.error(f"TFLLex file not found: {path}")
        return {}

    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)

    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)

    # EFLLex/TFLLex TSV: word column + per-level frequency columns
    word_col = df.columns[0]
    level_cols = [c for c in df.columns if c in CEFR_NUMERIC]

    if level_cols:
        result = {}
        for _, row in df.iterrows():
            word = str(row[word_col]).lower().strip()
            best_level = max(
                level_cols,
                key=lambda c: float(row[c]) if pd.notna(row[c]) else 0,
            )
            if pd.notna(row[best_level]) and float(row[best_level]) > 0:
                result[word] = best_level
        logger.info(f"Loaded {len(result)} Turkish CEFR entries from {path}")
        return result

    # Fallback: try word,level columns
    for c in df.columns:
        cl = c.lower()
        if cl in ("cefr", "level"):
            result = {}
            for _, row in df.iterrows():
                result[str(row[word_col]).lower()] = str(row[c]).upper()
            return result

    logger.warning(f"Could not parse TFLLex format: {list(df.columns)}")
    return {}


# =============================================================================
# Silver Label Generation
# =============================================================================

def generate_silver_labels(
    translations: Dict[str, str],
    cefr_lexicon: Dict[str, str],
    source_lang: str = "unknown",
) -> pd.DataFrame:
    """Project CEFR levels from source language to Kazakh via translations.

    Returns DataFrame with columns: lemma_kk, translation, cefr_projected, source_lang
    """
    rows = []
    for kk_word, translation in translations.items():
        tgt_level = cefr_lexicon.get(translation, "")
        if tgt_level and tgt_level in CEFR_NUMERIC:
            rows.append({
                "lemma_kk": kk_word,
                "translation": translation,
                "cefr_projected": tgt_level,
                "source_lang": source_lang,
            })

    df = pd.DataFrame(rows)
    logger.info(
        f"Generated {len(df)} silver labels from {source_lang} "
        f"({len(df)}/{len(translations)} = {len(df)/max(len(translations),1)*100:.1f}% coverage)"
    )
    return df


def merge_silver_sources(
    silver_dfs: Dict[str, pd.DataFrame],
    min_agreement: int = 2,
) -> pd.DataFrame:
    """Merge silver labels from multiple sources.

    Keeps labels where at least min_agreement sources agree, or takes
    the majority vote.
    """
    if not silver_dfs:
        return pd.DataFrame()

    # Collect all projected levels per lemma
    all_projections = {}
    for source, df in silver_dfs.items():
        for _, row in df.iterrows():
            lemma = row["lemma_kk"]
            level = row["cefr_projected"]
            if lemma not in all_projections:
                all_projections[lemma] = []
            all_projections[lemma].append((source, level))

    rows = []
    for lemma, projections in all_projections.items():
        levels = [p[1] for p in projections]
        sources = [p[0] for p in projections]
        n_sources = len(projections)

        # Majority vote
        from collections import Counter
        level_counts = Counter(levels)
        majority_level, majority_count = level_counts.most_common(1)[0]

        rows.append({
            "lemma_kk": lemma,
            "cefr_projected": majority_level,
            "n_sources": n_sources,
            "agreement": majority_count,
            "sources": ",".join(sources),
            "confident": majority_count >= min_agreement,
        })

    df = pd.DataFrame(rows)
    n_confident = df["confident"].sum() if len(df) > 0 else 0
    logger.info(
        f"Merged silver labels: {len(df)} total, "
        f"{n_confident} with agreement >= {min_agreement}"
    )
    return df
