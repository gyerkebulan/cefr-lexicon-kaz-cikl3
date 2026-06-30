#!/usr/bin/env python3
"""
Train baseline models for CEFR classification.

Usage:
    python scripts/train_baselines.py --variant majority
    python scripts/train_baselines.py --variant freq_only
    python scripts/train_baselines.py --variant full_feature
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, load_paths, setup_logging, set_seed
from src.data_prep import load_splits, print_label_distribution
from src.features import SuffixInventory, augment_with_features, print_feature_stats
from src.models.baselines import get_baseline_model, prepare_features
from src.metrics import (
    compute_metrics, compute_confusion_matrix,
    save_metrics, save_predictions, update_run_index,
    print_metrics_summary, save_confusion_matrix_figure
)


def main():
    parser = argparse.ArgumentParser(description="Train baseline CEFR models")
    parser.add_argument(
        "--variant",
        choices=["majority", "freq_only", "full_feature", "full_feature_lr", "full_feature_gb"],
        default="full_feature",
        help="Baseline variant to train"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to processed data directory"
    )
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    paths = load_paths()
    config = load_config("training")
    set_seed(config.get("seed", 42))
    
    model_name = args.variant
    logger.info(f"Training baseline: {model_name}")
    
    # Load data
    data_dir = args.data_dir or Path(paths["data"]["processed_dir"])
    train, dev, test = load_splits(data_dir)
    
    print_label_distribution(train, "Train")
    
    # Always backfill missing features so new columns land on older split files.
    logger.info("Backfilling feature columns...")
    suffix_inv = SuffixInventory(config.get("lang", "kaz"))
    train = augment_with_features(train, suffix_inventory=suffix_inv)
    dev = augment_with_features(dev, suffix_inventory=suffix_inv)
    test = augment_with_features(test, suffix_inventory=suffix_inv)
    
    # Prepare data - use ALL available features, not just the expected ones
    meta_cols = {'lemma', 'pos', 'cefr', 'cefr_level', 'key'}
    available_features = [c for c in train.columns if c not in meta_cols]

    logger.info(f"Using all {len(available_features)} available features")
    logger.info(f"Features: {available_features}")

    # Extract feature matrices using available features
    X_train = train[available_features].values.astype('float32')
    y_train = train['cefr'].values if 'cefr' in train.columns else train['cefr_level'].values

    X_dev = dev[available_features].values.astype('float32')
    y_dev = dev['cefr'].values if 'cefr' in dev.columns else dev['cefr_level'].values

    X_test = test[available_features].values.astype('float32')
    y_test = test['cefr'].values if 'cefr' in test.columns else test['cefr_level'].values

    # Handle NaN
    import numpy as np
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_dev = np.nan_to_num(X_dev, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    logger.info(f"Feature matrix: {X_train.shape}")

    # Train model
    model = get_baseline_model(model_name)
    logger.info(f"Training {model.name}...")

    if model_name == "freq_only":
        model.fit(X_train, y_train, feature_cols=available_features)
    else:
        model.fit(X_train, y_train)
    
    # Evaluate
    dev_preds = model.predict(X_dev)
    test_preds = model.predict(X_test)
    
    dev_metrics = compute_metrics(y_dev.tolist(), dev_preds.tolist())
    test_metrics = compute_metrics(y_test.tolist(), test_preds.tolist())
    
    # Print summary
    print_metrics_summary(
        model_name=model_name,
        train_size=len(train),
        dev_size=len(dev),
        test_size=len(test),
        dev_metrics=dev_metrics,
        test_metrics=test_metrics,
        output_dir=Path(paths["results"]["dir"])
    )
    
    # Save outputs
    results_dir = Path(paths["results"]["dir"])
    
    # Metrics JSON
    save_metrics(
        metrics={**dev_metrics, "split": "dev"},
        model_name=f"{model_name}_dev",
        output_dir=results_dir / "metrics"
    )
    save_metrics(
        metrics={**test_metrics, "split": "test"},
        model_name=f"{model_name}_test",
        output_dir=results_dir / "metrics"
    )
    
    # Predictions CSV
    test['pred_cefr'] = test_preds
    test['gold_cefr'] = test['cefr']
    
    # Add probability columns if available
    if model.predict_proba(X_test) is not None:
        probs = model.predict_proba(X_test)
        from src.utils import CEFR_LABELS
        for i, label in enumerate(CEFR_LABELS):
            if i < probs.shape[1]:
                test[f'prob_{label}'] = probs[:, i]
    
    save_predictions(test, model_name, results_dir / "predictions", split="test")
    
    # Confusion matrix
    cm = compute_confusion_matrix(y_test.tolist(), test_preds.tolist())
    save_confusion_matrix_figure(cm, model_name, results_dir / "figures")
    
    # Update run index
    update_run_index(
        model_name=model_name,
        config="training.yaml",
        dev_metrics=dev_metrics,
        test_metrics=test_metrics,
        index_path=Path(paths["results"]["run_index"])
    )
    
    # Print feature importance for tree-based models
    if hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()
        if importance:
            print("\nTop 10 Feature Importance:")
            for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
                print(f"  {name:25s}: {imp:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
