"""
Evaluation metrics for CEFR classification.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, cohen_kappa_score
)

from .utils import CEFR_LABELS, CEFR_TO_ORDINAL

logger = logging.getLogger("cefr")


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None,
    probabilities: Optional[np.ndarray] = None,
    abstention_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute all classification metrics.
    
    Returns:
        Dict with accuracy, macro_f1, per_class metrics, ordinal metrics
    """
    labels = labels or CEFR_LABELS
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    # Per-class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
    
    # Ordinal metrics (treating CEFR as 1-5 scale)
    ordinal_true = [CEFR_TO_ORDINAL.get(y, 3) for y in y_true]
    ordinal_pred = [CEFR_TO_ORDINAL.get(y, 3) for y in y_pred]
    
    ordinal_errors = np.abs(np.array(ordinal_true) - np.array(ordinal_pred))
    ordinal_mae = np.mean(ordinal_errors)
    far_error_rate = np.mean(ordinal_errors > 1)
    if len(set(ordinal_true)) == len(set(ordinal_pred)) == 1:
        qwk = float(np.array_equal(ordinal_true, ordinal_pred))
    else:
        qwk = cohen_kappa_score(
            ordinal_true, ordinal_pred, weights="quadratic",
            labels=list(range(1, len(labels) + 1)),
        )
    
    # Pearson correlation
    if len(set(ordinal_true)) > 1 and len(set(ordinal_pred)) > 1:
        ordinal_corr = np.corrcoef(ordinal_true, ordinal_pred)[0, 1]
    else:
        ordinal_corr = 0.0
    
    result = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "ordinal_mae": float(ordinal_mae),
        "quadratic_weighted_kappa": float(qwk) if np.isfinite(qwk) else 0.0,
        "far_error_rate": float(far_error_rate),
        "ordinal_correlation": float(ordinal_corr) if not np.isnan(ordinal_corr) else 0.0,
    }
    if probabilities is None:
        return result

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(y_true), len(labels)):
        raise ValueError(
            f"Expected probabilities shape {(len(y_true), len(labels))}, got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities contain NaN, infinity, or negative values")
    row_sums = probabilities.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("Probability rows must have positive mass")
    probabilities = probabilities / row_sums[:, None]
    label_to_index = {label: index for index, label in enumerate(labels)}
    indices = np.array([label_to_index[label] for label in y_true])
    one_hot = np.eye(len(labels))[indices]
    clipped = np.clip(probabilities, 1e-12, 1.0)
    confidence = probabilities.max(axis=1)
    correct = np.asarray(y_true) == np.asarray(y_pred)
    bin_edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    thresholds = abstention_thresholds or [0.5, 0.6, 0.7, 0.8, 0.9]
    result.update({
        "negative_log_likelihood": float(-np.log(clipped[np.arange(len(indices)), indices]).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "expected_calibration_error": float(ece),
        "mean_entropy": float(-(clipped * np.log(clipped)).sum(axis=1).mean()),
        "mean_confidence": float(confidence.mean()),
        "abstention": {
            f"{threshold:.2f}": {
                "coverage": float((confidence >= threshold).mean()),
                "selective_accuracy": (
                    float(correct[confidence >= threshold].mean())
                    if (confidence >= threshold).any() else None
                ),
            }
            for threshold in thresholds
        },
    })
    return result


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None
) -> pd.DataFrame:
    """Compute confusion matrix as DataFrame."""
    labels = labels or CEFR_LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def save_metrics(
    metrics: Dict[str, Any],
    model_name: str,
    output_dir: Path,
    config_path: Optional[str] = None,
    extra_info: Optional[Dict] = None
) -> Path:
    """
    Save metrics to JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "model": model_name,
        "config_path": config_path,
        "timestamp": datetime.now().isoformat(),
        **metrics
    }
    
    if extra_info:
        result.update(extra_info)
    
    output_path = output_dir / f"{model_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved metrics to {output_path}")
    return output_path


def save_predictions(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    split: str = "test"
) -> Path:
    """
    Save predictions to CSV.
    
    Expected columns: id, sentence/lemma, gold_cefr, pred_cefr, features...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{model_name}_{split}.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved predictions to {output_path}")
    return output_path


def update_run_index(
    model_name: str,
    config: str,
    dev_metrics: Dict[str, Any],
    index_path: Path,
    test_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append to run index CSV.
    """
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    row = {
        "model_name": model_name,
        "config": config,
        "dev_accuracy": dev_metrics.get("accuracy", 0),
        "dev_macro_f1": dev_metrics.get("macro_f1", 0),
        "timestamp": datetime.now().isoformat()
    }
    if test_metrics is not None:
        row.update({
            "test_accuracy": test_metrics.get("accuracy", 0),
            "test_macro_f1": test_metrics.get("macro_f1", 0),
        })
    
    # Append to existing or create new
    if index_path.exists():
        existing = pd.read_csv(index_path)
        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    df.to_csv(index_path, index=False)
    logger.info(f"Updated run index: {index_path}")


def print_metrics_summary(
    model_name: str,
    train_size: int,
    dev_size: int,
    dev_metrics: Dict[str, Any],
    test_size: Optional[int] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None
) -> None:
    """Print human-readable metrics summary."""
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Model: {model_name}")
    sizes = f"  Train: {train_size} | Dev: {dev_size}"
    if test_size is not None:
        sizes += f" | Test: {test_size}"
    print(sizes)
    print()
    print(f"  Dev  Accuracy: {dev_metrics['accuracy']:.4f}")
    print(f"  Dev  Macro-F1: {dev_metrics['macro_f1']:.4f}")
    print()
    if test_metrics is not None:
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Test Ordinal MAE: {test_metrics.get('ordinal_mae', 0):.4f}")
    
    if output_dir:
        print(f"\n  Saved to: {output_dir}")
    
    print("="*60 + "\n")


def save_confusion_matrix_figure(
    cm: pd.DataFrame,
    model_name: str,
    output_dir: Path
) -> Path:
    """Save confusion matrix as image."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix: {model_name}')
        
        output_path = output_dir / f"{model_name}_confusion.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved confusion matrix to {output_path}")
        return output_path
    
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping figure")
        return None
