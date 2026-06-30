"""
Entry condition algorithm for silver lexicon generation.

Implements CEFRLex-style cumulative filtering to assign
CEFR entry levels to lemmas based on frequency profiles.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .scrape_learner_corpus import CEFR_LEVELS
from .frequency_profiles import load_frequency_profiles

logger = logging.getLogger(__name__)

# Default thresholds (empirically tunable)
DEFAULT_THRESHOLDS = {
    "min_freq_per_million": 1.0,
    "min_text_count": 2,
    "min_total_occurrences": 3,
}


def assign_entry_level(
    profile: Dict,
    thresholds: Dict = DEFAULT_THRESHOLDS,
) -> Optional[str]:
    """
    Determine CEFR entry level for a lemma using cumulative filtering.
    
    Returns the LOWEST level where the lemma meets both:
    - Minimum frequency per million tokens
    - Minimum number of distinct texts (dispersion)
    """
    min_freq = thresholds.get("min_freq_per_million", 1.0)
    min_texts = thresholds.get("min_text_count", 2)
    
    frequencies = profile.get("frequencies", {})
    text_counts = profile.get("text_counts", {})
    
    # Check levels in order (A1 → C1)
    for level in CEFR_LEVELS:
        freq = frequencies.get(level, 0)
        texts = text_counts.get(level, 0)
        
        if freq >= min_freq and texts >= min_texts:
            return level
    
    # Fallback: highest level where lemma appears
    for level in reversed(CEFR_LEVELS):
        if frequencies.get(level, 0) > 0:
            return level
    
    return None


def compute_entry_strength(profile: Dict, entry_level: str) -> float:
    """
    Compute entry strength score for confidence weighting.
    
    Higher when lemma has high count and high dispersion at entry level.
    """
    total = profile.get("total_occurrences", 0)
    text_count = profile.get("text_counts", {}).get(entry_level, 0)
    
    if total == 0 or text_count == 0:
        return 0.0
    
    # log(count) * dispersion, normalized
    return min(1.0, (math.log(total + 1) * text_count) / 20.0)


def build_silver_lexicon(
    frequency_profiles: Dict[str, Dict],
    thresholds: Dict = DEFAULT_THRESHOLDS,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    """
    Build silver lexicon with entry-based CEFR labels.
    
    Args:
        frequency_profiles: Output of build_frequency_profiles()
        thresholds: Entry condition thresholds
        overrides: Manual level overrides for specific lemmas
        
    Returns:
        Silver lexicon mapping lemma_key to entry info
    """
    overrides = overrides or {}
    silver_lexicon = {}
    
    for lemma_key, profile in frequency_profiles.items():
        # Check for manual override
        if lemma_key in overrides:
            entry_level = overrides[lemma_key]
        else:
            entry_level = assign_entry_level(profile, thresholds)
        
        if entry_level is None:
            continue
        
        entry_strength = compute_entry_strength(profile, entry_level)
        
        silver_lexicon[lemma_key] = {
            "lemma": profile["lemma"],
            "pos": profile["pos"],
            "entry_level": entry_level,
            "entry_strength": entry_strength,
            "frequencies": profile["frequencies"],
            "text_counts": profile["text_counts"],
            "total_occurrences": profile["total_occurrences"],
        }
    
    return silver_lexicon


def validate_silver_lexicon(lexicon: Dict[str, Dict]) -> Dict:
    """
    Run sanity checks on silver lexicon.
    
    Returns issues and statistics.
    """
    from collections import Counter
    
    level_counts = Counter(item["entry_level"] for item in lexicon.values())
    total = len(lexicon)
    
    # Distribution check (soft, not targets)
    distribution = {level: level_counts.get(level, 0) / max(1, total) * 100 for level in CEFR_LEVELS}
    
    # Common words that should be A1/A2
    common_words = ["мен", "сен", "ол", "біз", "және", "бірақ", "үшін", "бар", "жоқ"]
    common_check = {}
    
    for word in common_words:
        matches = [k for k, v in lexicon.items() if v["lemma"] == word]
        if matches:
            level = lexicon[matches[0]]["entry_level"]
            common_check[word] = {
                "level": level,
                "warning": level not in ["A1", "A2"],
            }
    
    # Rare items assigned to A1
    a1_items = [v for v in lexicon.values() if v["entry_level"] == "A1"]
    rare_a1 = [v for v in a1_items if v["total_occurrences"] < 5]
    
    return {
        "total_lemmas": total,
        "level_distribution": distribution,
        "level_counts": dict(level_counts),
        "common_word_check": common_check,
        "rare_a1_count": len(rare_a1),
        "rare_a1_samples": [v["lemma"] for v in rare_a1[:10]],
    }


def save_silver_lexicon(
    lexicon: Dict[str, Dict],
    path: Path = Path("data/processed/silver_lexicon.json"),
) -> None:
    """Save silver lexicon to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved silver lexicon ({len(lexicon)} entries) to {path}")


def load_silver_lexicon(
    path: Path = Path("data/processed/silver_lexicon.json"),
) -> Dict[str, Dict]:
    """Load silver lexicon from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Silver lexicon not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Common function word overrides (A1 defaults)
FUNCTION_WORD_OVERRIDES = {
    "мен_PRON": "A1",
    "сен_PRON": "A1",
    "ол_PRON": "A1",
    "біз_PRON": "A1",
    "сіз_PRON": "A1",
    "олар_PRON": "A1",
    "және_CONJ": "A1",
    "бірақ_CONJ": "A1",
    "немесе_CONJ": "A1",
    "үшін_ADP": "A1",
    "бар_ADJ": "A1",
    "жоқ_ADJ": "A1",
    "бұл_DET": "A1",
    "сол_DET": "A1",
}


__all__ = [
    "assign_entry_level",
    "compute_entry_strength",
    "build_silver_lexicon",
    "validate_silver_lexicon",
    "save_silver_lexicon",
    "load_silver_lexicon",
    "DEFAULT_THRESHOLDS",
    "FUNCTION_WORD_OVERRIDES",
]
