"""
Build supervised dataset from lexicon for CEFR classification.

Features:
1. Orthographic: char_len, vowel_count, consonant_count, rare_char_count, syllable_count, is_long
2. POS: one-hot encoding + is_content_word
3. Morphology-shaped: suffix_like_len, has_verb_noun_suffix, suffix_chunk_count
4. Frequency (optional): log_freq, freq_per_million, dispersion
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple

# Kazakh vowels
VOWELS = set('аәеиоөұүыіэуАӘЕИОӨҰҮЫІЭУ')

# Rare/special Kazakh characters
RARE_CHARS = set('ңұәүөқғһіҢҰӘҮӨҚҒҺІ')

# Common derivational suffixes (simplified)
VERB_NOUN_SUFFIXES = [
    'лық', 'лік', 'дық', 'дік', 'тық', 'тік',  # -лық/-лік (noun forming)
    'шы', 'ші',  # -шы/-ші (agent)
    'шылық', 'шілік',  # -шылық (abstract noun)
    'ма', 'ме', 'ба', 'бе', 'па', 'пе',  # negation/noun
    'лау', 'леу', 'дау', 'деу', 'тау', 'теу',  # verbal nouns
    'ғыш', 'гіш', 'қыш', 'кіш',  # agent/instrument
]

# Content word POS
# Content word POS
CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}

# Corpus Stats (approximate)
TOTAL_CORPUS_TOKENS = 90_000_000



def count_vowels(word: str) -> int:
    """Count vowels in a word."""
    return sum(1 for c in word if c in VOWELS)


def count_rare_chars(word: str) -> int:
    """Count rare/special Kazakh characters."""
    return sum(1 for c in word if c in RARE_CHARS)


def count_syllables(word: str) -> int:
    """Heuristic syllable count based on vowel groups."""
    # Count vowel groups (consecutive vowels = 1 syllable)
    in_vowel = False
    count = 0
    for c in word.lower():
        if c in VOWELS:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    return max(1, count)  # At least 1 syllable


def has_verb_noun_suffix(word: str) -> bool:
    """Check if word ends with common derivational suffix."""
    word_lower = word.lower()
    return any(word_lower.endswith(suffix) for suffix in VERB_NOUN_SUFFIXES)


def count_suffix_chunks(word: str) -> int:
    """Count how many derivational suffixes are detected."""
    word_lower = word.lower()
    count = 0
    for suffix in VERB_NOUN_SUFFIXES:
        if suffix in word_lower:
            count += 1
    return min(count, 3)  # Cap at 3


def estimate_suffix_len(word: str) -> int:
    """Estimate suffix length (word length minus guessed stem)."""
    # Simple heuristic: stem is first 3-4 chars
    stem_guess = min(len(word), 4)
    return max(0, len(word) - stem_guess)


def extract_features(entry: Dict, freq_map: Dict[str, int] = None) -> Dict:
    """Extract all features for a lexicon entry."""
    lemma = entry['lemma']
    pos = entry['pos']
    
    # Frequency features
    count = 0
    if freq_map:
        count = freq_map.get(lemma, 0)
    
    log_freq = np.log10(count + 1)
    rel_freq = count / TOTAL_CORPUS_TOKENS
    in_corpus = 1 if count > 0 else 0
    
    char_len = len(lemma)
    vowel_count = count_vowels(lemma)
    consonant_count = char_len - vowel_count
    
    features = {
        # Orthographic
        'char_len': char_len,
        'vowel_count': vowel_count,
        'consonant_count': consonant_count,
        'rare_char_count': count_rare_chars(lemma),
        'syllable_count': count_syllables(lemma),
        'is_long': 1 if char_len >= 8 else 0,
        
        # POS one-hot
        'pos_NOUN': 1 if pos == 'NOUN' else 0,
        'pos_VERB': 1 if pos == 'VERB' else 0,
        'pos_ADJ': 1 if pos == 'ADJ' else 0,
        'pos_ADV': 1 if pos == 'ADV' else 0,
        'pos_OTHER': 1 if pos not in CONTENT_POS else 0,
        'is_content_word': 1 if pos in CONTENT_POS else 0,
        
        # Morphology-shaped
        'suffix_like_len': estimate_suffix_len(lemma),
        'has_verb_noun_suffix': 1 if has_verb_noun_suffix(lemma) else 0,
        'suffix_chunk_count': count_suffix_chunks(lemma),
        # Morphology-shaped
        'suffix_like_len': estimate_suffix_len(lemma),
        'has_verb_noun_suffix': 1 if has_verb_noun_suffix(lemma) else 0,
        'suffix_chunk_count': count_suffix_chunks(lemma),
        
        # Frequency
        'log_freq': log_freq,
        'rel_freq': rel_freq,
        'in_corpus': in_corpus,
    }
    
    return features


def build_dataset(lexicon_path: Path, output_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """Build full dataset with features and train/dev/test splits."""
    
    # Load lexicon
    with lexicon_path.open('r', encoding='utf-8') as f:
        lexicon = json.load(f)
        
    # Load frequencies if available
    freq_map = {}
    freq_path = Path('dataset/external/frequencies_full.json')
    if freq_path.exists():
        with freq_path.open('r', encoding='utf-8') as f:
            freq_map = json.load(f)
        print(f"Loaded {len(freq_map)} frequency entries from {freq_path.name}.")
    else:
        # Fallback to old one or warned
        old_path = Path('dataset/external/qazcorpus_freqs.json')
        if old_path.exists():
             print(f"WARNING: Full frequency file not found. Falling back to {old_path.name}")
             with old_path.open('r', encoding='utf-8') as f:
                freq_map = json.load(f)
        else:
             print("WARNING: External frequency file not found. Frequency features will be 0.")
    
    # Extract features for each entry
    rows = []
    for key, entry in lexicon.items():
        features = extract_features(entry, freq_map)
        features['lemma'] = entry['lemma']
        features['pos'] = entry['pos']
        features['cefr_level'] = entry['entry_level']
        features['key'] = key
        rows.append(features)
    
    df = pd.DataFrame(rows)
    
    # Define feature columns
    feature_cols = [
        'char_len', 'vowel_count', 'consonant_count', 'rare_char_count',
        'syllable_count', 'is_long',
        'pos_NOUN', 'pos_VERB', 'pos_ADJ', 'pos_ADV', 'pos_OTHER',
        'is_content_word',
        'suffix_like_len', 'has_verb_noun_suffix', 'suffix_chunk_count',
        'log_freq', 'rel_freq', 'in_corpus',
    ]
    
    # Stratified split: 70% train, 15% dev, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['cefr_level'], random_state=42
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['cefr_level'], random_state=42
    )
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / 'train.csv', index=False)
    dev_df.to_csv(output_dir / 'dev.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    # Compute statistics
    stats = {
        'total': len(df),
        'train': len(train_df),
        'dev': len(dev_df),
        'test': len(test_df),
        'feature_cols': feature_cols,
        'by_level': df['cefr_level'].value_counts().to_dict(),
        'train_by_level': train_df['cefr_level'].value_counts().to_dict(),
        'dev_by_level': dev_df['cefr_level'].value_counts().to_dict(),
        'test_by_level': test_df['cefr_level'].value_counts().to_dict(),
    }
    
    with (output_dir / 'dataset_stats.json').open('w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return df, stats


def main():
    lexicon_path = Path('dataset/processed/lexicon_validated.json')
    output_dir = Path('dataset/splits')
    
    print("Building dataset from lexicon...")
    df, stats = build_dataset(lexicon_path, output_dir)
    
    print("\n" + "="*50)
    print("DATASET CREATED")
    print("="*50)
    print(f"Total entries: {stats['total']}")
    print(f"Train: {stats['train']} ({stats['train']/stats['total']*100:.1f}%)")
    print(f"Dev:   {stats['dev']} ({stats['dev']/stats['total']*100:.1f}%)")
    print(f"Test:  {stats['test']} ({stats['test']/stats['total']*100:.1f}%)")
    print()
    print("Distribution by CEFR level:")
    for level in ['A1', 'A2', 'B1', 'B2', 'C1']:
        train_n = stats['train_by_level'].get(level, 0)
        dev_n = stats['dev_by_level'].get(level, 0)
        test_n = stats['test_by_level'].get(level, 0)
        print(f"  {level}: train={train_n}, dev={dev_n}, test={test_n}")
    print()
    print(f"Feature columns ({len(stats['feature_cols'])}):")
    for col in stats['feature_cols']:
        print(f"  - {col}")
    print("="*50)
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
