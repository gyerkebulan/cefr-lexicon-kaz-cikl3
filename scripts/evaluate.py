#!/usr/bin/env python3
"""
Evaluate trained models and generate analysis outputs.

Usage:
    python scripts/evaluate.py --model full_feature
    python scripts/evaluate.py --model dual_gated --generate-report
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.utils import load_paths, setup_logging, CEFR_LABELS
from src.metrics import (
    compute_metrics, compute_confusion_matrix,
    save_confusion_matrix_figure
)
from src.data_prep import dump_uncertain_predictions


def load_predictions(model_name: str, results_dir: Path, split: str = "test") -> pd.DataFrame:
    """Load predictions CSV for a model."""
    pred_path = results_dir / "predictions" / f"{model_name}_{split}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    return pd.read_csv(pred_path)


def load_metrics(model_name: str, results_dir: Path, split: str = "test") -> dict:
    """Load metrics JSON for a model."""
    metrics_path = results_dir / "metrics" / f"{model_name}_{split}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compare_models(model_names: list, results_dir: Path) -> pd.DataFrame:
    """Compare metrics across multiple models."""
    rows = []
    for name in model_names:
        try:
            metrics = load_metrics(name, results_dir)
            rows.append({
                "model": name,
                "accuracy": metrics.get("accuracy", 0),
                "macro_f1": metrics.get("macro_f1", 0),
                "ordinal_mae": metrics.get("ordinal_mae", 0),
            })
        except FileNotFoundError:
            continue
    
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def analyze_predictions(df: pd.DataFrame) -> dict:
    """Analyze predictions for insights."""
    analysis = {}
    
    # Error by CEFR level
    if 'gold_cefr' in df.columns and 'pred_cefr' in df.columns:
        df['correct'] = df['gold_cefr'] == df['pred_cefr']
        
        level_accuracy = {}
        for level in CEFR_LABELS:
            subset = df[df['gold_cefr'] == level]
            if len(subset) > 0:
                level_accuracy[level] = subset['correct'].mean()
        
        analysis['per_level_accuracy'] = level_accuracy
        
        # Confusion patterns
        errors = df[~df['correct']]
        if len(errors) > 0:
            confusion_pairs = errors.groupby(['gold_cefr', 'pred_cefr']).size()
            analysis['top_confusions'] = confusion_pairs.sort_values(ascending=False).head(10).to_dict()
    
    # Morph complexity correlation
    if 'morph_complexity' in df.columns and 'correct' in df.columns:
        correct_morph = df[df['correct']]['morph_complexity'].mean()
        incorrect_morph = df[~df['correct']]['morph_complexity'].mean()
        analysis['morph_complexity'] = {
            'correct_mean': round(correct_morph, 4),
            'incorrect_mean': round(incorrect_morph, 4)
        }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Evaluate CEFR models")
    parser.add_argument("--model", type=str, help="Model name to evaluate")
    parser.add_argument("--compare", nargs="+", help="List of models to compare")
    parser.add_argument("--generate-report", action="store_true", help="Generate detailed report")
    parser.add_argument("--uncertain-top-n", type=int, default=50, help="Top N uncertain predictions")
    args = parser.parse_args()
    
    logger = setup_logging()
    paths = load_paths()
    results_dir = Path(paths["results"]["dir"])
    
    # Compare models
    if args.compare:
        logger.info("Comparing models...")
        comparison = compare_models(args.compare, results_dir)
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(comparison.to_string(index=False))
        print()
        
        comparison.to_csv(results_dir / "model_comparison.csv", index=False)
        logger.info(f"Saved comparison to {results_dir}/model_comparison.csv")
        return 0
    
    # Single model evaluation
    if not args.model:
        logger.error("Please specify --model or --compare")
        return 1
    
    try:
        predictions = load_predictions(args.model, results_dir)
        metrics = load_metrics(args.model, results_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Print metrics
    print("\n" + "="*60)
    print(f"EVALUATION: {args.model}")
    print("="*60)
    print(f"  Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"  Macro-F1:    {metrics.get('macro_f1', 0):.4f}")
    print(f"  Ordinal MAE: {metrics.get('ordinal_mae', 0):.4f}")
    
    # Per-class metrics
    if 'per_class' in metrics:
        print("\n  Per-class F1:")
        for level in CEFR_LABELS:
            if level in metrics['per_class']:
                f1 = metrics['per_class'][level].get('f1', 0)
                support = metrics['per_class'][level].get('support', 0)
                print(f"    {level}: {f1:.4f} (n={support})")
    
    # Confusion matrix
    if 'gold_cefr' in predictions.columns and 'pred_cefr' in predictions.columns:
        cm = compute_confusion_matrix(
            predictions['gold_cefr'].tolist(),
            predictions['pred_cefr'].tolist()
        )
        print("\n  Confusion Matrix:")
        print(cm.to_string())
        
        # Save figure
        save_confusion_matrix_figure(cm, args.model, results_dir / "figures")
    
    # Analysis
    if args.generate_report:
        analysis = analyze_predictions(predictions)
        
        print("\n" + "-"*60)
        print("ANALYSIS")
        print("-"*60)
        
        if 'per_level_accuracy' in analysis:
            print("\n  Accuracy by CEFR Level:")
            for level, acc in analysis['per_level_accuracy'].items():
                print(f"    {level}: {acc:.4f}")
        
        if 'top_confusions' in analysis:
            print("\n  Top Confusion Pairs:")
            for (gold, pred), count in list(analysis['top_confusions'].items())[:5]:
                print(f"    {gold} → {pred}: {count}")
        
        if 'morph_complexity' in analysis:
            mc = analysis['morph_complexity']
            print(f"\n  Morph Complexity (correct): {mc['correct_mean']:.4f}")
            print(f"  Morph Complexity (wrong):   {mc['incorrect_mean']:.4f}")
        
        # Uncertain predictions
        uncertain = dump_uncertain_predictions(predictions, top_n=args.uncertain_top_n)
        if len(uncertain) > 0:
            uncertain_path = results_dir / "predictions" / f"{args.model}_uncertain.csv"
            uncertain.to_csv(uncertain_path, index=False)
            logger.info(f"Saved {len(uncertain)} uncertain predictions to {uncertain_path}")
    
    print("="*60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
