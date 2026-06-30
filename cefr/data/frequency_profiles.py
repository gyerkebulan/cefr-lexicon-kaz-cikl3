"""
Frequency profile computation for silver lexicon generation.

Computes per-level frequency statistics for lemmas from processed corpora.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .scrape_learner_corpus import CEFR_LEVELS
from .preprocess import load_processed_corpus

logger = logging.getLogger(__name__)


def build_frequency_profiles(
    input_dir: Path = Path("data/processed"),
    output_path: Path = Path("data/processed/frequency_profiles.json"),
) -> Dict[str, Dict]:
    """
    Compute per-level frequency profiles for all lemmas.
    
    For each lemma+POS combination, computes:
    - Raw count at each level
    - Frequency per million tokens
    - Number of distinct texts (dispersion)
    """
    # Per-level statistics
    level_lemma_counts: Dict[str, Counter] = defaultdict(Counter)
    level_text_presence: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    level_token_totals: Dict[str, int] = {}
    
    for level in CEFR_LEVELS:
        try:
            corpus = load_processed_corpus(level, input_dir)
        except FileNotFoundError:
            logger.warning(f"Skipping {level}: corpus not found")
            level_token_totals[level] = 0
            continue
        
        total_tokens = 0
        
        for text in corpus:
            text_id = text.get("text_id", "unknown")
            
            for sent in text.get("sentences", []):
                for analysis in sent.get("analysis", []):
                    lemma = analysis.get("lemma", "").lower()
                    pos = analysis.get("pos", "UNK")
                    
                    if not lemma:
                        continue
                    
                    key = f"{lemma}_{pos}"
                    level_lemma_counts[level][key] += 1
                    level_text_presence[level][key].add(text_id)
                    total_tokens += 1
        
        level_token_totals[level] = total_tokens
        logger.info(f"{level}: {total_tokens} tokens, {len(level_lemma_counts[level])} unique lemmas")
    
    # Build frequency profiles
    all_lemma_keys = set()
    for level in CEFR_LEVELS:
        all_lemma_keys.update(level_lemma_counts[level].keys())
    
    frequency_profiles = {}
    
    for lemma_key in all_lemma_keys:
        parts = lemma_key.rsplit("_", 1)
        lemma = parts[0] if len(parts) > 1 else lemma_key
        pos = parts[1] if len(parts) > 1 else "UNK"
        
        frequencies = {}
        text_counts = {}
        total_occurrences = 0
        
        for level in CEFR_LEVELS:
            raw_count = level_lemma_counts[level].get(lemma_key, 0)
            total_tokens = level_token_totals.get(level, 0)
            
            if total_tokens > 0 and raw_count > 0:
                freq_per_million = (raw_count / total_tokens) * 1_000_000
            else:
                freq_per_million = 0.0
            
            num_texts = len(level_text_presence[level].get(lemma_key, set()))
            
            frequencies[level] = freq_per_million
            text_counts[level] = num_texts
            total_occurrences += raw_count
        
        frequency_profiles[lemma_key] = {
            "lemma": lemma,
            "pos": pos,
            "frequencies": frequencies,
            "text_counts": text_counts,
            "total_occurrences": total_occurrences,
        }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(frequency_profiles, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(frequency_profiles)} frequency profiles to {output_path}")
    
    return frequency_profiles


def load_frequency_profiles(
    path: Path = Path("data/processed/frequency_profiles.json"),
) -> Dict[str, Dict]:
    """Load frequency profiles from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frequency profiles not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def frequency_stats(profiles: Dict[str, Dict]) -> Dict:
    """Compute summary statistics for frequency profiles."""
    total = len(profiles)
    by_pos = Counter(p["pos"] for p in profiles.values())
    
    # Count lemmas appearing at each level
    level_coverage = {level: 0 for level in CEFR_LEVELS}
    for profile in profiles.values():
        for level in CEFR_LEVELS:
            if profile["frequencies"].get(level, 0) > 0:
                level_coverage[level] += 1
    
    return {
        "total_lemmas": total,
        "by_pos": dict(by_pos),
        "level_coverage": level_coverage,
    }


__all__ = [
    "build_frequency_profiles",
    "load_frequency_profiles",
    "frequency_stats",
]
