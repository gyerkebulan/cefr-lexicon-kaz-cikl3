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
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import CEFR_LABELS, load_config, load_paths, setup_logging, set_seed
from src.data_prep import load_train_dev, print_label_distribution
from src.features import SuffixInventory, augment_with_features, print_feature_stats
from src.models.baselines import get_baseline_model, prepare_features
from src.metrics import (
    compute_metrics, compute_confusion_matrix,
    save_metrics, save_predictions, update_run_index,
    print_metrics_summary, save_confusion_matrix_figure
)
from src.research_protocol import (
    build_run_manifest, load_protocol, verify_dataset_manifest, write_json,
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--research-config", type=Path, default=None)
    parser.add_argument("--proposal-id", type=str, default=None)
    args = parser.parse_args()
    started_at = time.perf_counter()

    # Setup
    logger = setup_logging()
    paths = load_paths()
    config = load_config("training")
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)
    protocol = protocol_hash = None
    if args.research_config:
        protocol, protocol_hash = load_protocol(args.research_config)
        if not args.proposal_id:
            raise ValueError("--proposal-id is required with --research-config")

    model_name = f"{args.variant}_s{seed}"
    logger.info(f"Training baseline: {model_name}")

    # Load data
    data_dir = (
        args.data_dir
        or (PROJECT_ROOT / protocol["dataset"]["output_dir"] if protocol else None)
        or Path(paths["data"]["processed_dir"])
    )
    train, dev = load_train_dev(data_dir)
    if protocol:
        verify_dataset_manifest(data_dir, protocol_hash=protocol_hash, splits=("train", "dev"))
    
    print_label_distribution(train, "Train")
    
    # Always backfill missing features so new columns land on older split files.
    logger.info("Backfilling feature columns...")
    suffix_inv = SuffixInventory(config.get("lang", "kaz"))
    train = augment_with_features(train, suffix_inventory=suffix_inv)
    dev = augment_with_features(dev, suffix_inventory=suffix_inv)
    
    # Prepare data - use ALL available features, not just the expected ones
    available_features = [
        column for column in train.columns
        if column not in {"source_row_id", "group_id"}
        and train[column].dtype.kind in "biufc"
    ]

    logger.info(f"Using all {len(available_features)} available features")
    logger.info(f"Features: {available_features}")

    # Extract feature matrices using available features
    X_train = train[available_features].values.astype('float32')
    y_train = train['cefr'].values if 'cefr' in train.columns else train['cefr_level'].values

    X_dev = dev[available_features].values.astype('float32')
    y_dev = dev['cefr'].values if 'cefr' in dev.columns else dev['cefr_level'].values

    # Handle NaN
    import numpy as np
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_dev = np.nan_to_num(X_dev, nan=0.0)

    logger.info(f"Feature matrix: {X_train.shape}")

    # Train model
    model = get_baseline_model(args.variant)
    logger.info(f"Training {model.name}...")

    training_started = time.perf_counter()
    if args.variant == "freq_only":
        model.fit(X_train, y_train, feature_cols=available_features)
    else:
        model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - training_started
    
    # Evaluate
    inference_started = time.perf_counter()
    dev_preds = model.predict(X_dev)
    dev_probs = model.predict_proba(X_dev)
    if dev_probs is None:
        priors = train["cefr"].value_counts(normalize=True).reindex(CEFR_LABELS, fill_value=0)
        dev_probs = np.tile(priors.to_numpy(), (len(dev), 1))
    elif hasattr(model.model, "classes_"):
        order = [list(model.model.classes_).index(label) for label in CEFR_LABELS]
        dev_probs = dev_probs[:, order]
    dev_metrics = compute_metrics(
        y_dev.tolist(), dev_preds.tolist(), probabilities=dev_probs
    )
    dev_metrics["inference_latency_ms_per_example"] = (
        1000 * (time.perf_counter() - inference_started) / len(dev)
    )
    dev_metrics.update({
        "training_seconds": training_seconds,
        "invalid_prediction_count": 0,
        "target_span_failure_count": 0,
        "nan_count": 0,
    })
    
    # Print summary
    print_metrics_summary(
        model_name=model_name,
        train_size=len(train),
        dev_size=len(dev),
        dev_metrics=dev_metrics,
        output_dir=Path(paths["results"]["dir"])
    )
    
    # Save outputs
    results_dir = Path(paths["results"]["dir"])
    
    # Metrics JSON
    metrics_path = save_metrics(
        metrics={**dev_metrics, "split": "dev"},
        model_name=f"{model_name}_dev",
        output_dir=results_dir / "metrics"
    )
    dev['pred_cefr'] = dev_preds
    dev['gold_cefr'] = dev['cefr']
    
    # Add probability columns if available
    for index, label in enumerate(CEFR_LABELS):
        dev[f'prob_{label}'] = dev_probs[:, index]
    
    predictions_path = save_predictions(
        dev, model_name, results_dir / "predictions", split="dev"
    )
    
    # Confusion matrix
    cm = compute_confusion_matrix(y_dev.tolist(), dev_preds.tolist())
    figure_path = save_confusion_matrix_figure(cm, model_name, results_dir / "figures")
    
    # Update run index
    update_run_index(
        model_name=model_name,
        config="training.yaml",
        dev_metrics=dev_metrics,
        index_path=Path(paths["results"]["run_index"])
    )
    if protocol:
        artifacts = [metrics_path, predictions_path] + ([figure_path] if figure_path else [])
        manifest = build_run_manifest(
            root=PROJECT_ROOT,
            series_id=protocol["series"]["id"],
            proposal_id=args.proposal_id,
            protocol_hash=protocol_hash,
            seed=seed,
            configuration={"variant": args.variant, "data_dir": str(data_dir)},
            command=[sys.executable, *sys.argv],
            metrics=dev_metrics,
            runtime_seconds=time.perf_counter() - started_at,
            parameter_count=0,
            peak_vram_bytes=0,
            artifacts=artifacts,
            outcome="diagnostic_complete",
        )
        run_dir = (
            results_dir / "autoresearch" / protocol["series"]["id"]
            / args.proposal_id / f"seed-{seed}"
        )
        write_json(run_dir / "run-manifest.json", manifest, exclusive=True)
    
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
