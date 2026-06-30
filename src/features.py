"""
Feature extraction for CEFR classification.
Orthographic, morphological, and frequency features.
"""

import json
import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .utils import load_suffix_inventory

logger = logging.getLogger("cefr")


# =============================================================================
# Vowel Sets (for harmony checking)
# =============================================================================

FRONT_VOWELS = {'е', 'ө', 'ү', 'і', 'ә', 'і', 'и', 'э'}  # Front/soft
BACK_VOWELS = {'а', 'о', 'ұ', 'ы', 'у'}    # Back/hard
ALL_VOWELS = FRONT_VOWELS | BACK_VOWELS
RARE_CHARS = set('ңұәүөқғһі')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEMMA_FREQ_FILES = [
    PROJECT_ROOT / "artifacts" / "frequencies" / "lemma_freqs.json",
    PROJECT_ROOT / "artifacts" / "frequencies" / "frequencies_full.json",
    PROJECT_ROOT / "artifacts" / "frequencies" / "expanded_freqs_full.json",
    PROJECT_ROOT / "artifacts" / "frequencies" / "expanded_freqs_100k.json",
    PROJECT_ROOT / "artifacts" / "frequencies" / "expanded_freqs.json",
    PROJECT_ROOT / "dataset" / "external" / "lemma_freqs.json",
    PROJECT_ROOT / "dataset" / "external" / "frequencies_full.json",
    PROJECT_ROOT / "dataset" / "external" / "expanded_freqs_full.json",
    PROJECT_ROOT / "dataset" / "external" / "expanded_freqs_100k.json",
    PROJECT_ROOT / "dataset" / "external" / "expanded_freqs.json",
]

_lemma_freq_cache: Optional[Dict[str, float]] = None
_paradigm_stats_cache: Optional[Dict[str, Dict]] = None
_suffix_productivity_cache: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = {}

DEFAULT_PARADIGM_STATS_FILES = [
    PROJECT_ROOT / "artifacts" / "frequencies" / "paradigm_stats.json",
    PROJECT_ROOT / "dataset" / "external" / "paradigm_stats.json",
]


# =============================================================================
# Suffix Inventory
# =============================================================================

class SuffixInventory:
    """
    Manages suffix lists for a language.
    Supports greedy right-to-left matching with stacking.
    """
    
    def __init__(self, lang: str = "kaz"):
        self.lang = lang
        self.inventory = load_suffix_inventory(lang)
        
        # Build flat suffix lists (longest first for greedy matching)
        self.derivational: List[Tuple[str, str]] = []  # (suffix, type)
        self.inflectional: List[Tuple[str, str]] = []  # (suffix, category)
        self.suffix_metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        self._build_suffix_lists()
    
    def _build_suffix_lists(self) -> None:
        """Build sorted suffix lists from inventory."""
        # Derivational
        for item in self.inventory.get("derivational", []):
            if isinstance(item, dict):
                suffix = item["suffix"]
                stype = item.get("type", "deriv")
                self.derivational.append((suffix, stype))
                self.suffix_metadata[(suffix, f"deriv:{stype}")] = dict(item)
            else:
                suffix = str(item)
                self.derivational.append((suffix, "deriv"))
                self.suffix_metadata[(suffix, "deriv:deriv")] = {"suffix": suffix, "type": "deriv"}
        
        # Inflectional (nested structure)
        infl = self.inventory.get("inflectional", {})
        for category, items in infl.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        suffix = item["suffix"]
                        self.inflectional.append((suffix, category))
                        metadata = dict(item)
                        metadata["category"] = category
                        self.suffix_metadata[(suffix, f"infl:{category}")] = metadata
                    else:
                        suffix = str(item)
                        self.inflectional.append((suffix, category))
                        self.suffix_metadata[(suffix, f"infl:{category}")] = {
                            "suffix": suffix,
                            "category": category,
                        }
        
        # Sort by length (longest first) for greedy matching
        self.derivational.sort(key=lambda x: len(x[0]), reverse=True)
        self.inflectional.sort(key=lambda x: len(x[0]), reverse=True)
        
        # All suffixes (for quick matching)
        self.all_suffixes = set(s for s, _ in self.derivational + self.inflectional)
        
        logger.info(
            f"Loaded suffix inventory: {len(self.derivational)} derivational, "
            f"{len(self.inflectional)} inflectional"
        )
    
    def match_suffixes(self, word: str) -> Dict[str, Any]:
        """
        Greedy right-to-left suffix matching.
        
        Returns dict with:
            - suffix_count: total matched suffixes
            - deriv_count: derivational suffix count
            - infl_count: inflectional suffix count
            - stem: remaining word after suffix removal
            - matched: list of (suffix, type/category) tuples
        """
        word = word.lower().strip()
        matched = []
        remaining = word
        
        # Keep matching until no more suffixes found
        changed = True
        while changed and len(remaining) > 2:
            changed = False
            
            # Try inflectional first (they tend to be outermost)
            for suffix, category in self.inflectional:
                if remaining.endswith(suffix) and len(remaining) > len(suffix):
                    tag = f"infl:{category}"
                    matched.append((suffix, tag, self.suffix_metadata.get((suffix, tag), {})))
                    remaining = remaining[:-len(suffix)]
                    changed = True
                    break
            
            if changed:
                continue
            
            # Then try derivational
            for suffix, stype in self.derivational:
                if remaining.endswith(suffix) and len(remaining) > len(suffix):
                    tag = f"deriv:{stype}"
                    matched.append((suffix, tag, self.suffix_metadata.get((suffix, tag), {})))
                    remaining = remaining[:-len(suffix)]
                    changed = True
                    break
        
        deriv_count = sum(1 for _, t, _ in matched if t.startswith("deriv:"))
        infl_count = sum(1 for _, t, _ in matched if t.startswith("infl:"))
        
        return {
            "suffix_count": len(matched),
            "deriv_count": deriv_count,
            "infl_count": infl_count,
            "stem": remaining,
            "stem_len": len(remaining),
            "matched": [(suffix, tag) for suffix, tag, _ in matched],
            "matched_detailed": matched,
        }


def load_lemma_frequency_map(
    lemma_freq_file: Optional[Path] = None,
) -> Dict[str, float]:
    """Load lemma frequencies from the best available JSON file."""
    global _lemma_freq_cache

    if lemma_freq_file is None and _lemma_freq_cache is not None:
        return _lemma_freq_cache

    if lemma_freq_file is not None:
        candidates = [Path(lemma_freq_file)]
    else:
        env_override = os.environ.get("CEFR_LEMMA_FREQ_FILE")
        if env_override:
            candidates = [Path(env_override), *DEFAULT_LEMMA_FREQ_FILES]
        else:
            candidates = DEFAULT_LEMMA_FREQ_FILES
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        freq_map = {str(k).strip().lower(): float(v) for k, v in raw.items()}
        if lemma_freq_file is None:
            _lemma_freq_cache = freq_map
        logger.info(f"Loaded lemma frequency map with {len(freq_map)} entries from {path.name}")
        return freq_map

    logger.info("No external lemma frequency map found; falling back to row-level frequencies only")
    return {}


def load_paradigm_stats(
    paradigm_file: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Load pre-computed paradigm statistics from JSON."""
    global _paradigm_stats_cache

    if paradigm_file is None and _paradigm_stats_cache is not None:
        return _paradigm_stats_cache

    candidates = [Path(paradigm_file)] if paradigm_file else DEFAULT_PARADIGM_STATS_FILES
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        # Normalize keys to lowercase
        stats = {str(k).strip().lower(): v for k, v in stats.items()}
        if paradigm_file is None:
            _paradigm_stats_cache = stats
        logger.info(f"Loaded paradigm stats for {len(stats)} lemmas from {path.name}")
        return stats

    logger.info("No paradigm stats file found; paradigm features will be zeros")
    return {}


def build_suffix_productivity_stats(
    suffix_inventory: SuffixInventory,
    lemma_freq_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate suffix productivity over a lemma-frequency lexicon."""
    lemma_freq_map = lemma_freq_map or {}
    cache_key = (suffix_inventory.lang, len(lemma_freq_map))
    if cache_key in _suffix_productivity_cache:
        return _suffix_productivity_cache[cache_key]

    type_counts: Counter[str] = Counter()
    freq_sums: Dict[str, float] = defaultdict(float)

    for lemma, freq in lemma_freq_map.items():
        match_result = suffix_inventory.match_suffixes(lemma)
        seen_suffixes: Set[str] = set()
        for suffix, _, _ in match_result.get("matched_detailed", []):
            if suffix in seen_suffixes:
                continue
            type_counts[suffix] += 1
            freq_sums[suffix] += float(freq)
            seen_suffixes.add(suffix)

    stats = {
        "type_count": dict(type_counts),
        "freq_sum": {suffix: float(val) for suffix, val in freq_sums.items()},
    }
    _suffix_productivity_cache[cache_key] = stats
    return stats


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


REFRESHABLE_FREQ_PREFIXES = (
    "freq_",
    "log_freq",
    "lemma_freq",
    "paradigm_",
    "subword_",
    "idf",
    "df_",
    "tf_idf",
    "doc_spread",
)
REFRESHABLE_FREQ_COLUMNS = {
    "rel_freq",
    "in_corpus",
    "has_rare_subword",
    "subword_count",
}


def is_frequency_feature_column(column: str) -> bool:
    if column in REFRESHABLE_FREQ_COLUMNS:
        return True
    return any(column.startswith(prefix) for prefix in REFRESHABLE_FREQ_PREFIXES)


def _safe_max(values: List[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _safe_min(values: List[float]) -> float:
    return float(np.min(values)) if values else 0.0


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def vowel_harmony_ok(word: str) -> int:
    """
    Check vowel harmony.
    Returns 1 if all vowels are front OR all back, else 0.
    """
    word = word.lower()
    vowels_in_word = [c for c in word if c in ALL_VOWELS]
    
    if not vowels_in_word:
        return 1  # No vowels = trivially harmonic
    
    front = sum(1 for v in vowels_in_word if v in FRONT_VOWELS)
    back = sum(1 for v in vowels_in_word if v in BACK_VOWELS)
    
    return 1 if (front == 0 or back == 0) else 0


def count_syllables(word: str) -> int:
    """Estimate syllable count based on vowels."""
    word = word.lower()
    return max(1, sum(1 for c in word if c in ALL_VOWELS))


def extract_orthographic_features(word: str) -> Dict[str, float]:
    """Extract basic orthographic features."""
    word = word.lower().strip()
    
    vowel_count = sum(1 for c in word if c in ALL_VOWELS)
    consonant_count = sum(1 for c in word if c.isalpha() and c not in ALL_VOWELS)
    
    return {
        "char_len": len(word),
        "vowel_count": vowel_count,
        "consonant_count": consonant_count,
        "syllable_count": max(1, vowel_count),
        "vowel_ratio": vowel_count / max(1, len(word)),
        "rare_char_count": sum(1 for c in word if c in RARE_CHARS),
        "is_long": 1 if len(word) >= 8 else 0,
    }


def extract_morph_features(
    word: str,
    suffix_inventory: Optional[SuffixInventory] = None,
    suffix_productivity: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Extract morphological complexity features.
    """
    word = word.lower().strip()
    suffix_productivity = suffix_productivity or {"type_count": {}, "freq_sum": {}}
    
    # Suffix matching
    if suffix_inventory:
        match_result = suffix_inventory.match_suffixes(word)
        suffix_count = match_result["suffix_count"]
        deriv_count = match_result["deriv_count"]
        infl_count = match_result["infl_count"]
        stem_len = match_result["stem_len"]
        matched_detailed = match_result.get("matched_detailed", [])
    else:
        # Fallback: estimate based on length
        suffix_count = max(0, len(word) // 3 - 1)
        deriv_count = suffix_count // 2
        infl_count = suffix_count - deriv_count
        stem_len = len(word) - suffix_count * 2
        matched_detailed = []
    
    # Harmony
    harmony_ok = vowel_harmony_ok(word)
    
    # Composite complexity score
    # Formula: 0.3*suffix_count + 0.3*deriv + 0.2*len_norm + 0.2*infl
    len_norm = min(1.0, len(word) / 15.0)
    morph_complexity = (
        0.3 * min(1.0, suffix_count / 4.0) +
        0.3 * min(1.0, deriv_count / 2.0) +
        0.2 * len_norm +
        0.2 * min(1.0, infl_count / 3.0)
    )

    suffix_lengths = [len(suffix) for suffix, _, _ in matched_detailed]
    suffix_productivity_logs = [
        math.log1p(suffix_productivity["type_count"].get(suffix, 0))
        for suffix, _, _ in matched_detailed
    ]
    suffix_family_freq_logs = [
        math.log1p(suffix_productivity["freq_sum"].get(suffix, 0.0))
        for suffix, _, _ in matched_detailed
    ]

    rich_features = {
        "has_suffix_abstract": 0,
        "has_suffix_agent": 0,
        "has_suffix_privative": 0,
        "has_suffix_adjective": 0,
        "has_suffix_verbalize": 0,
        "has_suffix_diminutive": 0,
        "case_genitive_like": 0,
        "case_dative_like": 0,
        "case_accusative_like": 0,
        "case_locative_like": 0,
        "case_ablative_like": 0,
        "case_instrumental_like": 0,
        "poss_1sg_like": 0,
        "poss_2sg_like": 0,
        "poss_3sg_like": 0,
        "poss_1pl_like": 0,
        "poss_2pl_formal_like": 0,
        "verb_past_like": 0,
        "verb_participle_like": 0,
        "verb_present_like": 0,
        "verb_future_like": 0,
        "verb_infinitive_like": 0,
        "morph_ngram_deriv_infl": 0,
        "morph_ngram_plural_case": 0,
        "morph_ngram_possessive_case": 0,
        "morph_ngram_deriv_case": 0,
    }

    morph_tags: List[str] = []
    for suffix, tag, metadata in matched_detailed:
        if tag.startswith("deriv:"):
            deriv_type = tag.split(":", 1)[1]
            rich_features[f"has_suffix_{deriv_type}"] = 1
            morph_tags.append(f"deriv_{deriv_type}")
        elif tag == "infl:case":
            case = str(metadata.get("case", "")).upper()
            case_map = {
                "GEN": "case_genitive_like",
                "DAT": "case_dative_like",
                "ACC": "case_accusative_like",
                "LOC": "case_locative_like",
                "ABL": "case_ablative_like",
                "INS": "case_instrumental_like",
            }
            if case in case_map:
                rich_features[case_map[case]] = 1
            morph_tags.append(f"infl_case_{case.lower()}" if case else "infl_case")
        elif tag == "infl:possessive":
            person = str(metadata.get("person", ""))
            person_map = {
                "1sg": "poss_1sg_like",
                "2sg": "poss_2sg_like",
                "3sg": "poss_3sg_like",
                "1pl": "poss_1pl_like",
                "2pl_formal": "poss_2pl_formal_like",
            }
            if person in person_map:
                rich_features[person_map[person]] = 1
            morph_tags.append(f"infl_possessive_{person}" if person else "infl_possessive")
        elif tag == "infl:verbal":
            tense_or_form = str(metadata.get("tense") or metadata.get("form") or "").upper()
            tense_map = {
                "PAST": "verb_past_like",
                "PAST_PART": "verb_participle_like",
                "PRES": "verb_present_like",
                "FUT": "verb_future_like",
                "INF": "verb_infinitive_like",
            }
            if tense_or_form in tense_map:
                rich_features[tense_map[tense_or_form]] = 1
            morph_tags.append(f"infl_verbal_{tense_or_form.lower()}" if tense_or_form else "infl_verbal")
        else:
            morph_tags.append(tag.replace(":", "_"))

    for left, right in zip(morph_tags, morph_tags[1:]):
        if left.startswith("deriv_") and right.startswith("infl_"):
            rich_features["morph_ngram_deriv_infl"] = 1
        if left == "infl_plural" and right.startswith("infl_case_"):
            rich_features["morph_ngram_plural_case"] = 1
        if left.startswith("infl_possessive_") and right.startswith("infl_case_"):
            rich_features["morph_ngram_possessive_case"] = 1
        if left.startswith("deriv_") and right.startswith("infl_case_"):
            rich_features["morph_ngram_deriv_case"] = 1

    return {
        "suffix_count": suffix_count,
        "deriv_suffix_count": deriv_count,
        "infl_suffix_count": infl_count,
        "stem_len": stem_len,
        "has_deriv": 1 if deriv_count > 0 else 0,
        "has_infl": 1 if infl_count > 0 else 0,
        "vowel_harmony_ok": harmony_ok,
        "morph_complexity": morph_complexity,
        "suffix_stack_depth": suffix_count,
        "suffix_mean_len": _safe_mean(suffix_lengths),
        "suffix_max_len": _safe_max(suffix_lengths),
        "suffix_productivity_mean": _safe_mean(suffix_productivity_logs),
        "suffix_productivity_max": _safe_max(suffix_productivity_logs),
        "suffix_productivity_min": _safe_min(suffix_productivity_logs),
        "suffix_family_freq_mean": _safe_mean(suffix_family_freq_logs),
        "suffix_family_freq_max": _safe_max(suffix_family_freq_logs),
        **rich_features,
    }


def extract_freq_features(
    freq: float,
    total_corpus_tokens: int = 90_000_000
) -> Dict[str, float]:
    """
    Extract frequency-based features.
    """
    freq = max(0, float(freq))
    
    log_freq = math.log10(freq + 1)
    rel_freq = freq / total_corpus_tokens
    in_corpus = 1 if freq > 0 else 0
    
    # Frequency band (0=rare, 1=low, 2=mid, 3=high)
    if freq == 0:
        freq_band = 0
    elif freq < 100:
        freq_band = 1
    elif freq < 10000:
        freq_band = 2
    else:
        freq_band = 3
    
    return {
        "log_freq": log_freq,
        "rel_freq": rel_freq,
        "in_corpus": in_corpus,
        "freq_band": freq_band,
    }


# =============================================================================
# Subword Frequency Features (BPE/WordPiece)
# =============================================================================

# Lazy-loaded tokenizer to avoid importing at module level
_subword_tokenizer = None
_subword_freq_cache = {}  # Cache subword frequencies


def get_subword_tokenizer():
    """Get mBERT tokenizer (lazy loaded)."""
    global _subword_tokenizer
    if os.environ.get("CEFR_USE_HF_TOKENIZER", "0") != "1":
        return None
    if _subword_tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _subword_tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-multilingual-cased", 
                use_fast=True
            )
            logger.info("Loaded mBERT subword tokenizer")
        except Exception as e:
            logger.warning(f"Could not load mBERT tokenizer: {e}")
            _subword_tokenizer = False  # Mark as unavailable
    return _subword_tokenizer if _subword_tokenizer else None


def tokenize_to_subwords(word: str) -> List[str]:
    """Tokenize word into subword pieces using mBERT tokenizer."""
    tokenizer = get_subword_tokenizer()
    if tokenizer is None:
        # Fallback: simple character n-gram splitting
        return [word[i:i+3] for i in range(0, len(word), 3)] if len(word) > 3 else [word]
    
    try:
        # Get tokens without special tokens
        tokens = tokenizer.tokenize(word)
        return tokens if tokens else [word]
    except Exception:
        return [word]


def extract_subword_freq_features(
    word: str,
    subword_freq_dict: Optional[Dict[str, float]] = None,
    total_corpus_tokens: int = 90_000_000
) -> Dict[str, float]:
    """
    Extract subword-based frequency features.
    
    Features:
        - log_freq_subword_mean: Mean log-frequency of subwords
        - log_freq_subword_min: Min log-frequency (captures rare parts)
        - log_freq_subword_max: Max log-frequency
        - subword_count: Number of subwords
        - has_rare_subword: Binary flag if any subword is rare
    """
    subwords = tokenize_to_subwords(word.lower())
    
    if not subwords:
        return {
            "log_freq_subword_mean": 0.0,
            "log_freq_subword_min": 0.0,
            "log_freq_subword_max": 0.0,
            "subword_count": 1,
            "has_rare_subword": 0,
        }
    
    # Get frequencies for each subword
    subword_freqs = []
    for sw in subwords:
        # Remove WordPiece prefix markers for lookup
        clean_sw = sw.replace("##", "")
        if subword_freq_dict and clean_sw in subword_freq_dict:
            freq = subword_freq_dict[clean_sw]
        else:
            # Use length-based proxy if no freq dict
            # Longer subwords tend to be rarer
            freq = max(1, 10000 / (len(clean_sw) ** 2))
        subword_freqs.append(freq)
    
    # Compute log frequencies
    log_freqs = [math.log10(f + 1) for f in subword_freqs]
    
    return {
        "log_freq_subword_mean": float(np.mean(log_freqs)),
        "log_freq_subword_min": float(np.min(log_freqs)),
        "log_freq_subword_max": float(np.max(log_freqs)),
        "subword_count": len(subwords),
        "has_rare_subword": 1 if min(subword_freqs) < 100 else 0,
    }


def extract_form_freq_features(
    form: str,
    form_freq: float = 0,
    total_corpus_tokens: int = 90_000_000
) -> Dict[str, float]:
    """
    Extract surface form frequency features.
    
    Features:
        - log_freq_form: Log frequency of the surface form
        - freq_band_form: Frequency band (0-3) for surface form
    """
    freq = max(0, float(form_freq))
    log_freq = math.log10(freq + 1)
    
    # Frequency band
    if freq == 0:
        freq_band = 0
    elif freq < 100:
        freq_band = 1
    elif freq < 10000:
        freq_band = 2
    else:
        freq_band = 3
    
    return {
        "log_freq_form": log_freq,
        "freq_band_form": freq_band,
    }


def extract_lemma_freq_features(
    lemma: str,
    lemma_freq: float = 0,
    lemmatizer = None
) -> Dict[str, float]:
    """
    Extract lemma-based frequency features.
    
    Features:
        - log_freq_lemma: Log frequency of the lemmatized form
        - lemma_differs: Binary flag if lemma differs from form
    """
    freq = max(0, float(lemma_freq))
    log_freq = math.log10(freq + 1)
    
    return {
        "lemma_freq_raw": freq,
        "lemma_in_freq_lexicon": 1 if freq > 0 else 0,
        "log_freq_lemma": log_freq,
    }


def extract_paradigm_features(
    lemma: str,
    paradigm_stats: Optional[Dict[str, Dict]] = None,
) -> Dict[str, float]:
    """
    Extract paradigm-based features from pre-computed paradigm statistics.

    These features capture how richly a lemma is attested across inflected
    forms in the corpus — a strong signal for CEFR level in agglutinative
    languages where everyday words have many attested forms.

    Features:
        - paradigm_size: number of distinct surface forms attested
        - log_freq_aggregated: log10(sum of all form frequencies + 1)
        - paradigm_entropy: Shannon entropy of freq distribution across forms
        - paradigm_freq_concentration: fraction of total freq from top-3 forms
        - paradigm_freq_max: log10(freq of most common form + 1)
    """
    if not paradigm_stats or lemma not in paradigm_stats:
        return {
            "paradigm_size": 0,
            "log_freq_aggregated": 0.0,
            "paradigm_entropy": 0.0,
            "paradigm_freq_concentration": 1.0,
            "paradigm_freq_max": 0.0,
        }

    stats = paradigm_stats[lemma]
    return {
        "paradigm_size": float(stats.get("paradigm_size", 0)),
        "log_freq_aggregated": float(stats.get("log_freq_aggregated", 0)),
        "paradigm_entropy": float(stats.get("paradigm_entropy", 0)),
        "paradigm_freq_concentration": float(stats.get("paradigm_freq_concentration", 1.0)),
        "paradigm_freq_max": math.log10(float(stats.get("paradigm_freq_max", 0)) + 1),
    }


def extract_combined_freq_features(
    log_freq_form: float,
    log_freq_lemma: float,
    log_freq_subword_mean: float
) -> Dict[str, float]:
    """
    Compute combined frequency features.

    Features:
        - log_freq_combined_max: Max of all frequency signals
        - log_freq_combined_weighted: Weighted combination
    """
    # Max combination (most generous estimate)
    log_freq_max = max(log_freq_form, log_freq_lemma, log_freq_subword_mean)

    # Weighted combination: once true lemma counts are available,
    # lemma frequency is the most reliable signal in an agglutinative setting.
    log_freq_weighted = (
        0.25 * log_freq_form +
        0.55 * log_freq_lemma +
        0.2 * log_freq_subword_mean
    )

    return {
        "log_freq_combined_max": log_freq_max,
        "log_freq_combined_weighted": log_freq_weighted,
    }


def extract_all_features(
    lemma: str,
    pos: str = "NOUN",
    freq: float = 0,
    suffix_inventory: Optional[SuffixInventory] = None,
    form: str = None,
    form_freq: float = 0,
    lemma_freq: float = 0,
    subword_freq_dict: Optional[Dict[str, float]] = None,
    suffix_productivity: Optional[Dict[str, Dict[str, float]]] = None,
    paradigm_stats: Optional[Dict[str, Dict]] = None,
) -> Dict[str, float]:
    """
    Extract all features for a single word.

    Args:
        lemma: Lemma/dictionary form of the word
        pos: Part of speech
        freq: Base frequency (legacy, kept for compatibility)
        suffix_inventory: SuffixInventory for morphological analysis
        form: Surface form (if different from lemma)
        form_freq: Frequency of the surface form
        lemma_freq: Frequency of the lemma
        subword_freq_dict: Dict mapping subwords to frequencies
        paradigm_stats: Pre-computed paradigm stats dict (lemma -> stats)
    """
    features = {}
    
    # Use form if provided, otherwise use lemma
    word = form if form else lemma
    
    # Orthographic
    features.update(extract_orthographic_features(lemma))
    
    # Morphological
    features.update(extract_morph_features(
        lemma,
        suffix_inventory=suffix_inventory,
        suffix_productivity=suffix_productivity,
    ))
    
    # Base frequency (legacy)
    features.update(extract_freq_features(freq))
    
    # Surface form frequency
    features.update(extract_form_freq_features(word, form_freq))
    
    # Subword frequency features
    features.update(extract_subword_freq_features(word, subword_freq_dict))
    
    # Lemma frequency (from lemmatizer - may be noisy)
    features.update(extract_lemma_freq_features(lemma, lemma_freq))

    # Paradigm features (aggregated frequency, form diversity, entropy)
    features.update(extract_paradigm_features(lemma, paradigm_stats))

    # Combined frequency
    features.update(extract_combined_freq_features(
        features.get('log_freq_form', 0),
        features.get('log_freq_lemma', 0),
        features.get('log_freq_subword_mean', 0)
    ))

    freq_signal = features.get("log_freq_lemma", features.get("log_freq", 0.0))
    features["freq_zero"] = 1.0 if freq_signal == 0 else 0.0
    features["freq_low"] = 1.0 if 0 < freq_signal < 1.5 else 0.0
    features["freq_mid"] = 1.0 if 1.5 <= freq_signal < 3.0 else 0.0
    features["freq_high"] = 1.0 if freq_signal >= 3.0 else 0.0
    features["freq_x_complexity"] = freq_signal * features.get("morph_complexity", 0.0)
    features["freq_x_charlen"] = freq_signal * features.get("char_len", 0.0)
    features["freq_x_syllable"] = freq_signal * features.get("syllable_count", 0.0)
    features["freq_gap_form_lemma"] = features.get("log_freq_lemma", 0.0) - features.get("log_freq_form", 0.0)
    
    # POS (one-hot)
    pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV', 'OTHER']
    pos_upper = pos.upper() if pos else 'OTHER'
    for p in pos_list:
        features[f"pos_{p}"] = 1 if pos_upper == p else 0
    features["is_content_word"] = 1 if pos_upper in {"NOUN", "VERB", "ADJ", "ADV"} else 0
    features["freq_x_noun"] = freq_signal * features["pos_NOUN"]
    features["freq_x_verb"] = freq_signal * features["pos_VERB"]
    features["freq_x_adj"] = freq_signal * features["pos_ADJ"]
    features["morphemes_per_char"] = features.get("suffix_count", 0.0) / max(1.0, features.get("char_len", 1.0))
    features["deriv_ratio"] = features.get("deriv_suffix_count", 0.0) / max(1.0, features.get("suffix_count", 1.0))
    
    return features


# =============================================================================
# DataFrame Augmentation
# =============================================================================

def augment_with_features(
    df: pd.DataFrame,
    lexicon_df: Optional[pd.DataFrame] = None,
    suffix_inventory: Optional[SuffixInventory] = None,
    lang: str = "kaz",
    lemma_freq_file: Optional[Path] = None,
    refresh_frequency_features: bool = True,
) -> pd.DataFrame:
    """
    Add all features to a dataframe.
    
    Args:
        df: DataFrame with 'lemma' column
        lexicon_df: Optional lexicon with 'freq' column for lookup
        suffix_inventory: SuffixInventory instance (created if None)
        lang: Language code for suffix loading
        lemma_freq_file: Optional explicit lemma-frequency JSON path
        refresh_frequency_features: If True, overwrite existing frequency columns
        
    Returns:
        DataFrame with added feature columns
    """
    df = df.copy()

    # Load suffix inventory if needed
    if suffix_inventory is None:
        suffix_inventory = SuffixInventory(lang)
    
    # Build freq lookup if lexicon provided, then backfill with the best external lemma map.
    freq_lookup: Dict[str, float] = {}
    if lexicon_df is not None and 'freq' in lexicon_df.columns:
        for _, row in lexicon_df.iterrows():
            freq_lookup[str(row['lemma']).lower()] = float(row['freq'])
        logger.info(f"Built frequency lookup with {len(freq_lookup)} entries")
    freq_lookup.update(load_lemma_frequency_map(lemma_freq_file))
    if freq_lookup and "lemma" in df.columns:
        lemma_series = df["lemma"].astype(str).str.lower()
        covered = int(lemma_series.isin(freq_lookup).sum())
        logger.info(
            f"Lemma frequency coverage: {covered}/{len(df)} "
            f"({100.0 * covered / max(len(df), 1):.1f}%)"
        )
    suffix_productivity = build_suffix_productivity_stats(suffix_inventory, freq_lookup)

    # Load paradigm stats (aggregated lemma frequencies + form diversity)
    p_stats = load_paradigm_stats()

    # Extract features for each row
    feature_rows = []
    for _, row in df.iterrows():
        lemma = str(row.get('lemma', '')).lower()
        pos = str(row.get('pos', 'NOUN'))
        freq = float(row.get('freq', 0) or 0)
        lemma_freq = freq_lookup.get(lemma, float(row.get('lemma_freq_raw', row.get('freq', 0)) or 0))
        form = str(row.get('target') or row.get('form') or lemma).lower()
        form_freq = float(row.get('form_freq', row.get('freq', 0)) or 0)

        features = extract_all_features(
            lemma=lemma,
            pos=pos,
            freq=freq,
            suffix_inventory=suffix_inventory,
            form=form,
            form_freq=form_freq,
            lemma_freq=lemma_freq,
            suffix_productivity=suffix_productivity,
            paradigm_stats=p_stats,
        )
        feature_rows.append(features)
    
    # Merge features (only columns not already present)
    feature_df = pd.DataFrame(feature_rows)
    new_cols = [c for c in feature_df.columns if c not in df.columns]
    overwrite_cols = [
        c for c in feature_df.columns
        if c in df.columns and refresh_frequency_features and is_frequency_feature_column(c)
    ]

    result = df.reset_index(drop=True)
    if new_cols:
        result = pd.concat([result, feature_df[new_cols]], axis=1)
    if overwrite_cols:
        for col in overwrite_cols:
            result[col] = feature_df[col].values

    if new_cols or overwrite_cols:
        logger.info(
            f"Feature merge: added={len(new_cols)} columns, "
            f"refreshed_frequency={len(overwrite_cols)} columns"
        )
    else:
        logger.info("No new or refreshable feature columns to merge")

    # Post-hoc interaction features can reuse any existing HFST columns in the split.
    freq_col = "log_freq_lemma" if "log_freq_lemma" in result.columns else "log_freq"
    if freq_col in result.columns:
        result["freq_zero"] = (result[freq_col] == 0).astype(float)
        result["freq_low"] = ((result[freq_col] > 0) & (result[freq_col] < 1.5)).astype(float)
        result["freq_mid"] = ((result[freq_col] >= 1.5) & (result[freq_col] < 3.0)).astype(float)
        result["freq_high"] = (result[freq_col] >= 3.0).astype(float)
        if "morph_complexity_score" in result.columns:
            result["freq_x_hfst_complexity"] = result[freq_col] * result["morph_complexity_score"]
        if "morph_recognized" in result.columns:
            result["recognized_x_freq"] = result["morph_recognized"] * result[freq_col]
            result["not_recognized_zero_freq"] = (
                (result["morph_recognized"] == 0) & (result[freq_col] == 0)
            ).astype(float)
    if "n_morphemes_avg" in result.columns and "char_len" in result.columns:
        result["hfst_morphemes_per_char"] = result["n_morphemes_avg"] / result["char_len"].clip(lower=1)
    if "derivational_depth" in result.columns and "n_morphemes_avg" in result.columns:
        result["hfst_deriv_ratio"] = result["derivational_depth"] / result["n_morphemes_avg"].clip(lower=1)
    
    return result


def get_feature_columns() -> List[str]:
    """Get list of all feature column names."""
    return [
        # Orthographic
        "char_len", "vowel_count", "consonant_count",
        "syllable_count", "vowel_ratio",
        "rare_char_count", "is_long",
        # Morphological (heuristic)
        "suffix_count", "deriv_suffix_count", "infl_suffix_count",
        "stem_len", "has_deriv", "has_infl",
        "vowel_harmony_ok", "morph_complexity",
        "suffix_stack_depth", "suffix_mean_len", "suffix_max_len",
        "suffix_productivity_mean", "suffix_productivity_max", "suffix_productivity_min",
        "suffix_family_freq_mean", "suffix_family_freq_max",
        "has_suffix_abstract", "has_suffix_agent", "has_suffix_privative",
        "has_suffix_adjective", "has_suffix_verbalize", "has_suffix_diminutive",
        "case_genitive_like", "case_dative_like", "case_accusative_like",
        "case_locative_like", "case_ablative_like", "case_instrumental_like",
        "poss_1sg_like", "poss_2sg_like", "poss_3sg_like",
        "poss_1pl_like", "poss_2pl_formal_like",
        "verb_past_like", "verb_participle_like", "verb_present_like",
        "verb_future_like", "verb_infinitive_like",
        "morph_ngram_deriv_infl", "morph_ngram_plural_case",
        "morph_ngram_possessive_case", "morph_ngram_deriv_case",
        "morphemes_per_char", "deriv_ratio",
        # Morphological (Apertium analyzer)
        "morph_recognized", "n_analyses",
        "n_morphemes_min", "n_morphemes_max", "n_morphemes_avg",
        "derivational_depth", "has_copula",
        "has_case", "has_possession", "has_plural", "has_tense",
        "n_unique_pos", "morph_ambiguity", "morph_complexity_score",
        "hfst_morphemes_per_char", "hfst_deriv_ratio",
        # Apertium POS
        "apertium_pos_n", "apertium_pos_v", "apertium_pos_adj",
        "apertium_pos_adv", "apertium_pos_other",
        # Frequency (base/legacy)
        "log_freq", "rel_freq", "in_corpus", "freq_band",
        # Surface form frequency
        "log_freq_form", "freq_band_form",
        # Subword frequency (BPE/WordPiece based)
        "log_freq_subword_mean", "log_freq_subword_min", "log_freq_subword_max",
        "subword_count", "has_rare_subword",
        # Lemma frequency
        "lemma_freq_raw", "lemma_in_freq_lexicon", "log_freq_lemma",
        # Paradigm features (aggregated lemma frequency + form diversity)
        "paradigm_size", "log_freq_aggregated",
        "paradigm_entropy", "paradigm_freq_concentration", "paradigm_freq_max",
        # Combined frequency
        "log_freq_combined_max", "log_freq_combined_weighted", "freq_gap_form_lemma",
        "freq_zero", "freq_low", "freq_mid", "freq_high",
        "freq_x_complexity", "freq_x_charlen", "freq_x_syllable",
        "freq_x_noun", "freq_x_verb", "freq_x_adj",
        "freq_x_hfst_complexity", "recognized_x_freq", "not_recognized_zero_freq",
        # IDF / document frequency features
        "idf", "df_ratio", "log_df", "tf_idf", "doc_spread",
        # Content word flag
        "is_content_word",
        # POS
        "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_OTHER",
    ]


# =============================================================================
# Feature Groups for Ablation Experiments
# =============================================================================

FEATURE_GROUPS = {
    "freq_only": [
        "log_freq", "rel_freq", "in_corpus", "freq_band",
        "lemma_freq_raw", "lemma_in_freq_lexicon", "log_freq_lemma",
        "paradigm_size", "log_freq_aggregated",
        "paradigm_entropy", "paradigm_freq_concentration", "paradigm_freq_max",
        "freq_zero", "freq_low", "freq_mid", "freq_high",
    ],
    "ortho_only": [
        "char_len", "vowel_count", "consonant_count",
        "syllable_count", "vowel_ratio",
    ],
    "morph_only": [
        "suffix_count", "deriv_suffix_count", "infl_suffix_count",
        "stem_len", "has_deriv", "has_infl",
        "vowel_harmony_ok", "morph_complexity",
        "suffix_productivity_mean", "suffix_productivity_max",
        "has_suffix_abstract", "has_suffix_agent", "has_suffix_privative",
        "has_suffix_adjective", "has_suffix_verbalize",
        "case_genitive_like", "case_dative_like", "case_accusative_like",
        "case_locative_like", "case_ablative_like", "case_instrumental_like",
        "morph_ngram_deriv_infl", "morph_ngram_plural_case",
        "morph_ngram_possessive_case", "morph_ngram_deriv_case",
        "morphemes_per_char", "deriv_ratio",
    ],
    "subword_only": [
        "log_freq_subword_mean", "log_freq_subword_min", "log_freq_subword_max",
        "subword_count", "has_rare_subword",
    ],
    "pos_only": [
        "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_OTHER",
    ],
    "lemma_full": [
        # Orthographic
        "char_len", "vowel_count", "consonant_count",
        "syllable_count", "vowel_ratio",
        # Morphological
        "suffix_count", "deriv_suffix_count", "infl_suffix_count",
        "stem_len", "has_deriv", "has_infl",
        "vowel_harmony_ok", "morph_complexity",
        # Frequency (base/legacy)
        "log_freq", "rel_freq", "in_corpus", "freq_band",
        # Surface form frequency
        "log_freq_form", "freq_band_form",
        # Subword frequency
        "log_freq_subword_mean", "log_freq_subword_min", "log_freq_subword_max",
        "subword_count", "has_rare_subword",
        # Lemma frequency
        "lemma_freq_raw", "lemma_in_freq_lexicon", "log_freq_lemma",
        # Paradigm features
        "paradigm_size", "log_freq_aggregated",
        "paradigm_entropy", "paradigm_freq_concentration", "paradigm_freq_max",
        # Combined frequency
        "log_freq_combined_max", "log_freq_combined_weighted", "freq_gap_form_lemma",
        "freq_x_complexity", "freq_x_charlen", "freq_x_syllable",
        "freq_x_noun", "freq_x_verb", "freq_x_adj",
        # POS
        "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_OTHER",
    ],
}


# Feature groups mapped to actual column names in existing train/test CSV splits
FEATURE_GROUPS_SPLITS = {
    "freq_only": [
        "log_freq", "rel_freq", "in_corpus",
        "log_freq_lemma", "lemma_freq_raw", "lemma_in_freq_lexicon",
        "paradigm_size", "log_freq_aggregated",
        "paradigm_entropy", "paradigm_freq_concentration", "paradigm_freq_max",
    ],
    "ortho_only": [
        "char_len", "vowel_count", "consonant_count",
        "syllable_count", "is_long", "rare_char_count",
    ],
    "morph_only": [
        "morph_recognized", "n_analyses", "n_morphemes_avg", "derivational_depth",
        "morph_complexity_score", "suffix_count", "deriv_suffix_count",
        "suffix_productivity_mean", "has_suffix_abstract", "has_suffix_agent",
        "case_genitive_like", "case_dative_like", "morph_ngram_deriv_infl",
    ],
    "pos_only": [
        "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_OTHER",
    ],
    "ortho_morph": [
        "char_len", "vowel_count", "consonant_count",
        "syllable_count", "is_long", "rare_char_count",
        "morph_recognized", "n_analyses", "n_morphemes_avg", "derivational_depth",
        "morph_complexity_score", "suffix_count", "deriv_suffix_count",
        "suffix_productivity_mean", "has_suffix_abstract", "has_suffix_agent",
    ],
    "lemma_full": [
        # Orthographic
        "char_len", "vowel_count", "consonant_count",
        "syllable_count", "is_long", "rare_char_count",
        # Morphological
        "morph_recognized", "n_analyses", "n_morphemes_avg", "derivational_depth",
        "morph_complexity_score", "suffix_count", "deriv_suffix_count",
        "suffix_productivity_mean", "has_suffix_abstract", "has_suffix_agent",
        # Frequency
        "log_freq", "rel_freq", "in_corpus", "log_freq_lemma",
        "paradigm_size", "log_freq_aggregated",
        "paradigm_entropy", "paradigm_freq_concentration", "paradigm_freq_max",
        "freq_zero", "freq_low", "freq_mid", "freq_high",
        # POS
        "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_OTHER",
        # Other
        "is_content_word", "freq_x_complexity", "morphemes_per_char",
    ],
}


def get_feature_group(group_name: str) -> List[str]:
    """Get feature column list for a named group.
    
    Args:
        group_name: One of the keys in FEATURE_GROUPS.
        
    Returns:
        List of feature column names.
        
    Raises:
        ValueError: If group_name is not recognized.
    """
    if group_name not in FEATURE_GROUPS:
        raise ValueError(
            f"Unknown feature group '{group_name}'. "
            f"Available: {list(FEATURE_GROUPS.keys())}"
        )
    return FEATURE_GROUPS[group_name]


# =============================================================================
# Wordform-Level Feature Extraction (for Task B: lemma vs wordform)
# =============================================================================

def extract_wordform_features(
    surface: str,
    lemma: str,
    lemma_cefr: str,
    pos: str = "NOUN",
    surface_freq: float = 0,
    lemma_freq: float = 0,
    suffix_inventory: Optional[SuffixInventory] = None,
) -> Dict[str, float]:
    """
    Extract features for a wordform, producing three feature sets:
    - Lemma-only (Model L): features derived from the lemma string
    - Morph-augmented (Model LMorph): lemma features + surface morph features
    - Surface-only: features derived from the surface form string
    
    Prefix conventions:
        lem_*   : extracted from the lemma
        surf_*  : extracted from the surface form
        delta_* : difference between surface and lemma
    """
    features = {}
    
    # === LEMMA CEFR (numeric encoding) ===
    cefr_to_num = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
    features["lem_cefr_num"] = cefr_to_num.get(lemma_cefr, 0)
    
    # === LEMMA-DERIVED FEATURES (Model L) ===
    lem_ortho = extract_orthographic_features(lemma)
    for k, v in lem_ortho.items():
        features[f"lem_{k}"] = v
    
    features["lem_log_freq"] = math.log10(lemma_freq + 1) if lemma_freq > 0 else 0
    
    # POS one-hot
    pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV', 'OTHER']
    pos_upper = pos.upper() if pos else 'OTHER'
    for p in pos_list:
        features[f"pos_{p}"] = 1 if pos_upper == p else 0
    
    # === SURFACE-DERIVED FEATURES ===
    surf_ortho = extract_orthographic_features(surface)
    for k, v in surf_ortho.items():
        features[f"surf_{k}"] = v
    
    surf_morph = extract_morph_features(surface, suffix_inventory)
    for k, v in surf_morph.items():
        features[f"surf_{k}"] = v
    
    features["surf_log_freq"] = math.log10(surface_freq + 1) if surface_freq > 0 else 0
    
    # === DELTA FEATURES (surface minus lemma) ===
    features["delta_char_len"] = features["surf_char_len"] - features["lem_char_len"]
    features["delta_syllable_count"] = (
        features["surf_syllable_count"] - features["lem_syllable_count"]
    )
    features["lemma_differs"] = 0 if surface.lower() == lemma.lower() else 1
    
    return features


# Column name lists for the three models in Task B
WORDFORM_MODEL_FEATURES = {
    "model_l": [
        # Lemma-only features (NO lem_cefr_num — that's the target!)
        "lem_char_len", "lem_vowel_count",
        "lem_consonant_count", "lem_syllable_count", "lem_vowel_ratio",
        "lem_log_freq",
        "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_OTHER",
    ],
    "model_lmorph": [
        # Lemma features + surface morphology
        "lem_char_len", "lem_vowel_count",
        "lem_consonant_count", "lem_syllable_count", "lem_vowel_ratio",
        "lem_log_freq",
        "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_OTHER",
        # Surface morph additions
        "surf_suffix_count", "surf_deriv_suffix_count", "surf_infl_suffix_count",
        "surf_morph_complexity", "surf_log_freq",
        "delta_char_len", "delta_syllable_count", "lemma_differs",
    ],
    "model_surface": [
        # Surface-only features
        "surf_char_len", "surf_vowel_count", "surf_consonant_count",
        "surf_syllable_count", "surf_vowel_ratio",
        "surf_suffix_count", "surf_deriv_suffix_count", "surf_infl_suffix_count",
        "surf_stem_len", "surf_has_deriv", "surf_has_infl",
        "surf_vowel_harmony_ok", "surf_morph_complexity",
        "surf_log_freq",
        "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_OTHER",
    ],
}


# =============================================================================
# Debug / Analysis Utilities
# =============================================================================

def print_feature_stats(df: pd.DataFrame, sample_n: int = 10) -> None:
    """Print feature statistics for debugging."""
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    
    for col in feature_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            print(f"  {col:25s}: min={float(vals.min()):6.2f}, max={float(vals.max()):6.2f}, "
                  f"mean={float(vals.mean()):6.2f}, std={float(vals.std()):6.2f}")
    
    print("\n" + "-"*60)
    print(f"Sample words (n={sample_n}):")
    print("-"*60)
    
    sample = df.sample(min(sample_n, len(df)), random_state=42)
    for _, row in sample.iterrows():
        lemma = row.get('lemma', 'N/A')
        cefr = row.get('cefr', 'N/A')
        morph = row.get('morph_complexity', 0)
        suff = row.get('suffix_count', 0)
        freq = row.get('log_freq', 0)
        print(f"  {lemma:20s} | CEFR={cefr} | morph={morph:.2f} | "
              f"suffix={int(suff)} | log_freq={freq:.2f}")
    print()
