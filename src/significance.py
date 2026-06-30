"""
Statistical significance tests for pairwise model comparison.

Implements:
  - Paired bootstrap test on Macro-F1 (Koehn, 2004)
  - McNemar's test on per-instance correctness

Usage:
    python -m src.significance \
        --pred_a results/predictions/charcnn_test.csv \
        --pred_b results/predictions/morph_only_test.csv \
        --n_resamples 10000 --seed 42
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from .utils import CEFR_LABELS

logger = logging.getLogger("cefr")

COMPARISONS = [
    ("CharCNN", "Morph-only"),
    ("Morph-only", "Context-only"),
    ("LR", "Gated fusion"),
    ("Context-only", "Gated fusion"),
    ("Concat", "Gated fusion"),
    ("Gated", "Ensemble"),
]


def paired_bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
    labels: list = None,
) -> Dict[str, float]:
    """Paired bootstrap test on Macro-F1 (Koehn, 2004).

    Returns observed delta, two-sided p-value, and 95% CI.
    """
    labels = labels or CEFR_LABELS
    rng = np.random.RandomState(seed)
    n = len(y_true)

    f1_a = f1_score(y_true, y_pred_a, labels=labels, average="macro", zero_division=0)
    f1_b = f1_score(y_true, y_pred_b, labels=labels, average="macro", zero_division=0)
    observed_delta = f1_a - f1_b

    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        pa = y_pred_a[idx]
        pb = y_pred_b[idx]
        fa = f1_score(yt, pa, labels=labels, average="macro", zero_division=0)
        fb = f1_score(yt, pb, labels=labels, average="macro", zero_division=0)
        deltas[i] = fa - fb

    # Two-sided p-value: fraction of bootstrap deltas at least as extreme
    p_value = np.mean(np.abs(deltas - deltas.mean()) >= np.abs(observed_delta - deltas.mean()))

    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)

    return {
        "delta_f1": float(observed_delta),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, float]:
    """McNemar's test on per-instance correctness.

    Uses exact binomial test when discordant count < 25,
    otherwise chi-squared with continuity correction.
    """
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)

    # n_ab: A correct, B wrong; n_ba: A wrong, B correct
    n_ab = int(np.sum((correct_a == 1) & (correct_b == 0)))
    n_ba = int(np.sum((correct_a == 0) & (correct_b == 1)))

    n_discordant = n_ab + n_ba

    if n_discordant < 25:
        # Exact binomial test
        p_value = float(stats.binom_test(n_ab, n_discordant, 0.5))
        chi2 = float((n_ab - n_ba) ** 2 / max(n_discordant, 1))
    else:
        # Chi-squared with continuity correction
        chi2 = float((abs(n_ab - n_ba) - 1) ** 2 / (n_ab + n_ba))
        p_value = float(1.0 - stats.chi2.cdf(chi2, df=1))

    return {
        "chi2": chi2,
        "p_value": p_value,
        "n_ab": n_ab,
        "n_ba": n_ba,
    }


def run_pairwise_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Dict:
    """Run both significance tests for a single model pair."""
    pbs = paired_bootstrap_macro_f1(y_true, y_pred_a, y_pred_b, n_resamples, seed)
    mcn = mcnemar_test(y_true, y_pred_a, y_pred_b)

    return {
        "comparison": f"{name_a} vs. {name_b}",
        "bootstrap": pbs,
        "mcnemar": mcn,
    }


def format_results_table(results: list) -> str:
    """Format pairwise results as a text table matching paper Table 7."""
    header = (
        f"{'Comparison (A vs. B)':<35} "
        f"{'ΔF1':>7} {'p':>7} {'95% CI':>20}  "
        f"{'χ²':>7} {'p':>7} {'n_AB/n_BA':>10}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        pbs = r["bootstrap"]
        mcn = r["mcnemar"]
        sig_pbs = "*" if pbs["p_value"] < 0.05 else ("**" if pbs["p_value"] < 0.01 else "")
        sig_mcn = "**" if mcn["p_value"] < 0.01 else ("*" if mcn["p_value"] < 0.05 else "")
        ci = f"[{pbs['ci_lower']:+.3f}, {pbs['ci_upper']:+.3f}]"
        lines.append(
            f"{r['comparison']:<35} "
            f"{pbs['delta_f1']:+.3f}{sig_pbs:2s} "
            f"{pbs['p_value']:.3f}  {ci:>20}  "
            f"{mcn['chi2']:7.2f} "
            f"{mcn['p_value']:.3f}{sig_mcn:2s} "
            f"{mcn['n_ab']:>4}/{mcn['n_ba']:<4}"
        )

    lines.append(sep)
    return "\n".join(lines)


def load_predictions(path: Path) -> pd.DataFrame:
    """Load a predictions CSV with columns: gold_cefr, pred_cefr."""
    df = pd.read_csv(path)
    for col in ("gold_cefr", "pred_cefr"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Pairwise significance tests")
    parser.add_argument("--pred_a", type=Path, required=True, help="Predictions CSV for model A")
    parser.add_argument("--pred_b", type=Path, required=True, help="Predictions CSV for model B")
    parser.add_argument("--name_a", type=str, default="Model A")
    parser.add_argument("--name_b", type=str, default="Model B")
    parser.add_argument("--n_resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df_a = load_predictions(args.pred_a)
    df_b = load_predictions(args.pred_b)

    assert len(df_a) == len(df_b), "Prediction files must have the same number of instances"

    y_true = df_a["gold_cefr"].values
    y_pred_a = df_a["pred_cefr"].values
    y_pred_b = df_b["pred_cefr"].values

    result = run_pairwise_test(
        y_true, y_pred_a, y_pred_b,
        args.name_a, args.name_b,
        args.n_resamples, args.seed,
    )
    print(format_results_table([result]))


if __name__ == "__main__":
    main()
