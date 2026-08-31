"""
Data preparation utilities for CEFR classification.
Loading, validation, and train/dev/test splitting.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import CEFR_LABELS, normalize_cefr

logger = logging.getLogger("cefr")


# =============================================================================
# Data Loading
# =============================================================================

def load_lexicon(path: Path, validate: bool = True) -> pd.DataFrame:
    """
    Load type-level lexicon from CSV.
    
    Required columns: lemma, pos, cefr
    Optional columns: freq (default: 0)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {path}")
    
    df = pd.read_csv(path, encoding='utf-8')
    logger.info(f"Loaded lexicon: {len(df)} rows from {path.name}")
    
    if validate:
        df = validate_lexicon(df)
    
    return df


def load_context(path: Path, validate: bool = True) -> pd.DataFrame:
    """
    Load context-level dataset from CSV.
    
    Required columns: sentence, target, lemma, cefr
    Optional columns: pos (default: "UNK")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")
    
    df = pd.read_csv(path, encoding='utf-8')
    logger.info(f"Loaded context data: {len(df)} rows from {path.name}")
    
    if validate:
        df = validate_context(df)
    
    return df


# =============================================================================
# Validation
# =============================================================================

def validate_lexicon(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean lexicon dataframe."""
    required = ['lemma', 'pos', 'cefr']
    
    # Check required columns
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Lexicon missing required columns: {missing}")
    
    # Handle optional freq column
    if 'freq' not in df.columns:
        logger.warning("Lexicon missing 'freq' column, defaulting to 0")
        df['freq'] = 0
    else:
        df['freq'] = pd.to_numeric(df['freq'], errors='coerce').fillna(0)
    
    # Normalize CEFR labels
    df['cefr'] = df['cefr'].apply(normalize_cefr)
    
    # Filter invalid CEFR
    valid_mask = df['cefr'].isin(CEFR_LABELS)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Dropping {invalid_count} rows with invalid CEFR labels")
        df = df[valid_mask].copy()
    
    # Clean text columns
    df['lemma'] = df['lemma'].astype(str).str.strip().str.lower()
    df['pos'] = df['pos'].astype(str).str.strip().str.upper()
    
    # Normalize POS to standard set
    known_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'NUM', 'CONJ', 'PART', 'INTJ'}
    df['pos'] = df['pos'].apply(lambda p: p if p in known_pos else 'OTHER')
    
    # Drop duplicates (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=['lemma', 'pos'], keep='first')
    if len(df) < before:
        logger.info(f"Removed {before - len(df)} duplicate lemma-pos pairs")
    
    return df.reset_index(drop=True)


def validate_context(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean context dataframe."""
    required = ['sentence', 'target', 'lemma', 'cefr']
    
    # Check required columns
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Context data missing required columns: {missing}")
    
    # Handle optional pos column
    if 'pos' not in df.columns:
        logger.warning("Context data missing 'pos' column, defaulting to 'UNK'")
        df['pos'] = 'UNK'
    
    # Normalize CEFR labels
    df['cefr'] = df['cefr'].apply(normalize_cefr)
    
    # Filter invalid CEFR
    valid_mask = df['cefr'].isin(CEFR_LABELS)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Dropping {invalid_count} rows with invalid CEFR labels")
        df = df[valid_mask].copy()
    
    # Clean text columns
    df['sentence'] = df['sentence'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()
    df['lemma'] = df['lemma'].astype(str).str.strip().str.lower()
    df['pos'] = df['pos'].astype(str).str.strip().str.upper()
    
    # Drop rows with empty sentences
    empty_mask = df['sentence'].str.len() == 0
    if empty_mask.sum() > 0:
        logger.warning(f"Dropping {empty_mask.sum()} rows with empty sentences")
        df = df[~empty_mask].copy()
    
    # Add unique ID if not present
    if 'id' not in df.columns:
        df['id'] = range(len(df))
    
    return df.reset_index(drop=True)


# =============================================================================
# Splitting
# =============================================================================

def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    dev_size: float = 0.15,
    stratify_col: str = 'cefr',
    lemma_disjoint: bool = False,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/dev/test splits.
    
    Args:
        df: Input dataframe
        test_size: Fraction for test set
        dev_size: Fraction for dev set  
        stratify_col: Column to stratify by
        lemma_disjoint: If True, ensure no lemma appears in multiple splits
        seed: Random seed
        
    Returns:
        train, dev, test DataFrames
    """
    if lemma_disjoint:
        return _create_lemma_disjoint_splits(df, test_size, dev_size, seed)
    
    # Standard stratified split
    stratify = df[stratify_col] if stratify_col in df.columns else None
    
    # First split: train+dev vs test
    train_dev, test = train_test_split(
        df, test_size=test_size, stratify=stratify, random_state=seed
    )
    
    # Second split: train vs dev
    dev_ratio = dev_size / (1 - test_size)
    stratify_td = train_dev[stratify_col] if stratify_col in train_dev.columns else None
    
    train, dev = train_test_split(
        train_dev, test_size=dev_ratio, stratify=stratify_td, random_state=seed
    )
    
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)


def _create_lemma_disjoint_splits(
    df: pd.DataFrame,
    test_size: float,
    dev_size: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create splits where no lemma appears in multiple sets."""
    import numpy as np
    np.random.seed(seed)
    
    # Get unique lemmas with their CEFR distribution
    lemmas = df.groupby('lemma').agg({
        'cefr': lambda x: x.mode().iloc[0] if len(x) > 0 else 'B1'
    }).reset_index()
    
    # Split lemmas
    lemma_stratify = lemmas['cefr']
    lemmas_train_dev, lemmas_test = train_test_split(
        lemmas, test_size=test_size, stratify=lemma_stratify, random_state=seed
    )
    
    dev_ratio = dev_size / (1 - test_size)
    lemma_stratify_td = lemmas_train_dev['cefr']
    lemmas_train, lemmas_dev = train_test_split(
        lemmas_train_dev, test_size=dev_ratio, stratify=lemma_stratify_td, random_state=seed
    )
    
    # Map back to instances
    train = df[df['lemma'].isin(lemmas_train['lemma'])].copy()
    dev = df[df['lemma'].isin(lemmas_dev['lemma'])].copy()
    test = df[df['lemma'].isin(lemmas_test['lemma'])].copy()
    
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)


# =============================================================================
# Label Quality Utilities
# =============================================================================

def print_label_distribution(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print CEFR label distribution."""
    print(f"\n{'='*50}")
    print(f"{name} Label Distribution (n={len(df)})")
    print('='*50)
    
    counts = df['cefr'].value_counts().reindex(CEFR_LABELS, fill_value=0)
    for label, count in counts.items():
        pct = 100 * count / len(df)
        bar = '█' * int(pct / 2)
        print(f"  {label}: {count:5d} ({pct:5.1f}%) {bar}")
    print()


def detect_suspicious_examples(
    df: pd.DataFrame,
    lexicon_df: Optional[pd.DataFrame] = None,
    freq_threshold: int = 10000
) -> pd.DataFrame:
    """
    Flag suspicious label combinations.
    
    Returns rows where:
    - A1/A2 word has very high frequency (might be common word)
    - C1 word has very low frequency (might be rare variant)
    """
    suspicious = []
    
    if 'freq' in df.columns:
        # A1/A2 with high freq
        mask_low_level_high_freq = (
            df['cefr'].isin(['A1', 'A2']) & 
            (df['freq'] > freq_threshold)
        )
        suspicious.append(df[mask_low_level_high_freq].copy())
        
        # C1 with very low freq (might be labeling error)
        mask_high_level_zero_freq = (
            (df['cefr'] == 'C1') &
            (df['freq'] == 0)
        )
        if mask_high_level_zero_freq.any():
            sample = df[mask_high_level_zero_freq].head(50)
            suspicious.append(sample)
    
    if suspicious:
        result = pd.concat(suspicious, ignore_index=True).drop_duplicates()
        logger.info(f"Found {len(result)} suspicious examples")
        return result
    
    return pd.DataFrame()


def dump_uncertain_predictions(
    predictions_df: pd.DataFrame,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Get top N least confident predictions.
    
    Expects columns: gold_cefr, pred_cefr, and optionally probability columns.
    """
    # If probabilities available, use entropy
    prob_cols = [c for c in predictions_df.columns if c.startswith('prob_')]
    
    if prob_cols:
        import numpy as np
        probs = predictions_df[prob_cols].values
        probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        predictions_df = predictions_df.copy()
        predictions_df['uncertainty'] = entropy
        return predictions_df.nlargest(top_n, 'uncertainty')
    
    # Fallback: just return misclassified examples
    mistakes = predictions_df[predictions_df['gold_cefr'] != predictions_df['pred_cefr']]
    return mistakes.head(top_n)


# =============================================================================
# Convenience Functions
# =============================================================================

def _normalize_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy split column names without accessing another split."""
    if 'cefr_level' in df.columns and 'cefr' not in df.columns:
        df['cefr'] = df['cefr_level']
    if 'word' in df.columns and 'lemma' not in df.columns:
        df['lemma'] = df['word']
    return df


def load_train_dev(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load only train and dev; candidate processes must use this boundary."""
    processed_dir = Path(processed_dir)
    train = _normalize_split_columns(pd.read_csv(processed_dir / 'train.csv'))
    dev = _normalize_split_columns(pd.read_csv(processed_dir / 'dev.csv'))
    logger.info(f"Loaded candidate splits: train={len(train)}, dev={len(dev)}")
    return train, dev


def load_test(processed_dir: Path) -> pd.DataFrame:
    """Load the sealed test split. Only the final-test command may call this."""
    processed_dir = Path(processed_dir)
    test = _normalize_split_columns(pd.read_csv(processed_dir / 'test.csv'))
    logger.info(f"Loaded sealed test split: test={len(test)}")
    return test


def load_splits(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Legacy loader. Research candidate commands must use :func:`load_train_dev`."""
    logger.warning("load_splits opens test data; do not use it in autoresearch candidate mode")
    train, dev = load_train_dev(processed_dir)
    test = load_test(processed_dir)
    return train, dev, test


def save_splits(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path
) -> None:
    """Save train/dev/test splits to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(output_dir / 'train.csv', index=False)
    dev.to_csv(output_dir / 'dev.csv', index=False)
    test.to_csv(output_dir / 'test.csv', index=False)
    
    logger.info(f"Saved splits to {output_dir}")
    print_label_distribution(train, "Train")
    print_label_distribution(dev, "Dev")
    print_label_distribution(test, "Test")
