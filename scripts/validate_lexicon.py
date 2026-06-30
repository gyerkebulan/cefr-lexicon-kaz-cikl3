"""
Validate and clean the CEFR lexicon.

This script:
1. Validates schema (non-empty lemma, valid level, valid POS)
2. Normalizes POS tags to a closed set
3. Resolves conflicts (same lemma_POS with different levels → keep lowest)
4. Removes noisy entries
5. Outputs statistics
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Valid CEFR levels (ordered lowest to highest)
VALID_LEVELS = ["A1", "A2", "B1", "B2", "C1"]
LEVEL_ORDER = {level: i for i, level in enumerate(VALID_LEVELS)}

# POS tag normalization map
POS_NORMALIZE = {
    # Standard tags
    "NOUN": "NOUN",
    "VERB": "VERB",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "NUM": "NUM",
    "PRON": "PRON",
    # Map variations to standard
    "AUX": "VERB",      # Auxiliary verbs → VERB
    "ADP": "OTHER",     # Adpositions
    "CONJ": "OTHER",    # Conjunctions
    "PART": "OTHER",    # Particles
    "INTJ": "OTHER",    # Interjections
    "MODAL": "VERB",    # Modal verbs → VERB
    "DET": "OTHER",     # Determiners
}

# Kazakh letter patterns for validation
KAZAKH_PATTERN = re.compile(r'^[а-яәғқңөұүһіА-ЯӘҒҚҢӨҰҮҺІ\-\s]+$')


def is_valid_lemma(lemma: str) -> bool:
    """Check if lemma is valid Kazakh word."""
    if not lemma or len(lemma.strip()) < 1:
        return False
    # Must contain Kazakh/Cyrillic letters
    if not KAZAKH_PATTERN.match(lemma):
        return False
    # Reject if it's mostly numbers or punctuation
    if len(lemma) < 2:
        return False
    return True


def normalize_pos(pos: str) -> str:
    """Normalize POS tag to closed set."""
    pos_upper = pos.upper().strip()
    return POS_NORMALIZE.get(pos_upper, "OTHER")


def validate_and_clean(lexicon: Dict) -> Tuple[Dict, Dict]:
    """
    Validate and clean the lexicon.
    
    Returns:
        cleaned_lexicon: Dict with valid, cleaned entries
        stats: Dict with validation statistics
    """
    stats = {
        "total_input": len(lexicon),
        "invalid_lemma": 0,
        "invalid_level": 0,
        "invalid_pos": 0,
        "normalized_pos": 0,
        "conflicts_resolved": 0,
        "total_output": 0,
    }
    
    # First pass: collect all entries, normalize, and detect conflicts
    lemma_pos_entries = defaultdict(list)  # lemma_pos -> [(level, original_key), ...]
    invalid_entries = []
    
    for key, entry in lexicon.items():
        lemma = entry.get("lemma", "").strip().lower()
        pos = entry.get("pos", "")
        level = entry.get("entry_level", "")
        
        # Validate lemma
        if not is_valid_lemma(lemma):
            stats["invalid_lemma"] += 1
            invalid_entries.append((key, "invalid_lemma", lemma))
            continue
        
        # Validate level
        if level not in VALID_LEVELS:
            stats["invalid_level"] += 1
            invalid_entries.append((key, "invalid_level", level))
            continue
        
        # Normalize POS
        original_pos = pos
        normalized_pos = normalize_pos(pos)
        if normalized_pos != original_pos:
            stats["normalized_pos"] += 1
        
        # Collect for conflict detection
        new_key = f"{lemma}_{normalized_pos}"
        lemma_pos_entries[new_key].append((level, key, entry))
    
    # Log invalid entries
    if invalid_entries:
        logger.warning(f"Found {len(invalid_entries)} invalid entries:")
        for key, reason, value in invalid_entries[:10]:
            logger.warning(f"  {key}: {reason} = '{value}'")
        if len(invalid_entries) > 10:
            logger.warning(f"  ... and {len(invalid_entries) - 10} more")
    
    # Second pass: resolve conflicts (keep lowest level)
    cleaned_lexicon = {}
    conflicts = []
    
    for new_key, entries in lemma_pos_entries.items():
        if len(entries) > 1:
            # Conflict: multiple levels for same lemma_pos
            levels = [e[0] for e in entries]
            if len(set(levels)) > 1:
                stats["conflicts_resolved"] += 1
                conflicts.append((new_key, levels))
            
            # Keep entry with lowest level
            entries.sort(key=lambda x: LEVEL_ORDER[x[0]])
        
        # Take the first (lowest level) entry
        level, original_key, original_entry = entries[0]
        lemma = new_key.rsplit("_", 1)[0]
        pos = new_key.rsplit("_", 1)[1]
        
        cleaned_lexicon[new_key] = {
            "lemma": lemma,
            "pos": pos,
            "entry_level": level,
        }
    
    # Log conflicts
    if conflicts:
        logger.info(f"Resolved {len(conflicts)} conflicts (kept lowest level):")
        for key, levels in conflicts[:5]:
            logger.info(f"  {key}: {levels} → {min(levels, key=lambda x: LEVEL_ORDER[x])}")
        if len(conflicts) > 5:
            logger.info(f"  ... and {len(conflicts) - 5} more")
    
    stats["total_output"] = len(cleaned_lexicon)
    return cleaned_lexicon, stats


def compute_statistics(lexicon: Dict) -> Dict:
    """Compute lexicon statistics."""
    level_counts = Counter()
    pos_counts = Counter()
    level_pos_counts = defaultdict(Counter)
    
    for key, entry in lexicon.items():
        level = entry["entry_level"]
        pos = entry["pos"]
        
        level_counts[level] += 1
        pos_counts[pos] += 1
        level_pos_counts[level][pos] += 1
    
    return {
        "total": len(lexicon),
        "by_level": dict(level_counts),
        "by_pos": dict(pos_counts),
        "by_level_pos": {level: dict(counts) for level, counts in level_pos_counts.items()},
    }


def main():
    input_path = Path("dataset/processed/silver_lexicon.json")
    output_path = Path("dataset/processed/lexicon_validated.json")
    stats_path = Path("dataset/processed/lexicon_stats.json")
    
    # Load lexicon
    logger.info(f"Loading lexicon from {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        lexicon = json.load(f)
    
    # Validate and clean
    logger.info("Validating and cleaning...")
    cleaned_lexicon, validation_stats = validate_and_clean(lexicon)
    
    # Compute statistics
    logger.info("Computing statistics...")
    stats = compute_statistics(cleaned_lexicon)
    stats["validation"] = validation_stats
    
    # Save cleaned lexicon
    logger.info(f"Saving cleaned lexicon to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_lexicon, f, ensure_ascii=False, indent=2)
    
    # Save statistics
    logger.info(f"Saving statistics to {stats_path}")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("LEXICON VALIDATION SUMMARY")
    print("="*50)
    print(f"Input entries:  {validation_stats['total_input']}")
    print(f"Output entries: {validation_stats['total_output']}")
    print(f"Invalid lemmas: {validation_stats['invalid_lemma']}")
    print(f"Invalid levels: {validation_stats['invalid_level']}")
    print(f"POS normalized: {validation_stats['normalized_pos']}")
    print(f"Conflicts resolved: {validation_stats['conflicts_resolved']}")
    print()
    print("Distribution by CEFR level:")
    for level in VALID_LEVELS:
        count = stats["by_level"].get(level, 0)
        pct = count / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {level}: {count:>5} ({pct:>5.1f}%)")
    print()
    print("Distribution by POS:")
    for pos, count in sorted(stats["by_pos"].items(), key=lambda x: -x[1]):
        pct = count / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {pos}: {count:>5} ({pct:>5.1f}%)")
    print("="*50)


if __name__ == "__main__":
    main()
