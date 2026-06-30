#!/usr/bin/env python3
"""
Preprocess raw data into train/dev/test splits.

Usage:
    python scripts/preprocess.py --data-type lexicon
    python scripts/preprocess.py --data-type context
    python scripts/preprocess.py --data-type context --lemma-disjoint
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, load_paths, setup_logging, set_seed
from src.data_prep import (
    load_lexicon, load_context, create_splits, save_splits,
    print_label_distribution, detect_suspicious_examples
)


def main():
    parser = argparse.ArgumentParser(description="Preprocess CEFR data")
    parser.add_argument(
        "--data-type", 
        choices=["lexicon", "context"],
        default="lexicon",
        help="Type of data to process"
    )
    parser.add_argument(
        "--lemma-disjoint",
        action="store_true",
        help="Create lemma-disjoint splits (no lemma in multiple sets)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config (default: config/training.yaml)"
    )
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    paths = load_paths()
    config = load_config("training")
    set_seed(config.get("seed", 42))
    
    logger.info(f"Processing {args.data_type} data")
    
    # Load data
    if args.data_type == "lexicon":
        raw_path = Path(paths["data"]["lexicon"])
        if not raw_path.exists():
            logger.error(f"Lexicon file not found: {raw_path}")
            logger.info("Please place your kaz_lexicon.csv in data/raw/")
            return 1
        df = load_lexicon(raw_path)
    else:
        raw_path = Path(paths["data"]["context"])
        if not raw_path.exists():
            logger.error(f"Context file not found: {raw_path}")
            logger.info("Please place your kaz_context.csv in data/raw/")
            return 1
        df = load_context(raw_path)
    
    # Print initial distribution
    print_label_distribution(df, "Full Dataset")
    
    # Check for suspicious examples
    suspicious = detect_suspicious_examples(df)
    if len(suspicious) > 0:
        logger.warning(f"Found {len(suspicious)} suspicious examples (see above)")
        print("\nSample suspicious examples:")
        print(suspicious.head(10).to_string())
    
    # Create splits
    lemma_disjoint = args.lemma_disjoint or config.get("lemma_disjoint", False)
    logger.info(f"Creating splits (lemma_disjoint={lemma_disjoint})")
    
    train, dev, test = create_splits(
        df,
        test_size=config.get("test_size", 0.15),
        dev_size=config.get("dev_size", 0.15),
        stratify_col=config.get("stratify_col", "cefr"),
        lemma_disjoint=lemma_disjoint,
        seed=config.get("seed", 42)
    )
    
    # Save
    output_dir = Path(paths["data"]["processed_dir"])
    save_splits(train, dev, test, output_dir)
    
    # Summary
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"  Data type: {args.data_type}")
    print(f"  Total samples: {len(df)}")
    print(f"  Train: {len(train)} ({100*len(train)/len(df):.1f}%)")
    print(f"  Dev: {len(dev)} ({100*len(dev)/len(df):.1f}%)")
    print(f"  Test: {len(test)} ({100*len(test)/len(df):.1f}%)")
    print(f"  Output: {output_dir}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
