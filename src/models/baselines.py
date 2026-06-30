"""
Baseline models for CEFR classification.
Majority, frequency-only, full feature-based.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..utils import CEFR_LABELS
from ..features import get_feature_columns

logger = logging.getLogger("cefr")


class BaselineModel:
    """Base class for baseline models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None


class MajorityBaseline(BaselineModel):
    """Always predicts the most frequent class."""
    
    def __init__(self):
        super().__init__("majority")
        self.model = DummyClassifier(strategy='most_frequent')
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        logger.info(f"Majority class: {self.model.class_prior_}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class FrequencyOnlyModel(BaselineModel):
    """Classifier using only frequency features."""

    FREQ_FEATURES = ["log_freq", "rel_freq", "in_corpus", "freq_band", "log_freq_lemma"]

    def __init__(self):
        super().__init__("freq_only")
        self.model = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_indices = None

    def _get_freq_features(self, X: np.ndarray, all_feature_cols: List[str]) -> np.ndarray:
        """Extract only frequency-related features."""
        indices = [i for i, c in enumerate(all_feature_cols) if c in self.FREQ_FEATURES]
        self.feature_indices = indices
        if len(indices) > 0:
            logger.info(f"Using {len(indices)} frequency features: {[all_feature_cols[i] for i in indices]}")
            return X[:, indices]
        else:
            # Fallback: use first 3 columns (assuming they contain frequency info)
            logger.warning("No frequency features found in feature list, using first 3 columns")
            self.feature_indices = list(range(min(3, X.shape[1])))
            return X[:, self.feature_indices]
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str] = None) -> None:
        feature_cols = feature_cols or get_feature_columns()
        X_freq = self._get_freq_features(X, feature_cols)
        X_scaled = self.scaler.fit_transform(X_freq)
        self.model.fit(X_scaled, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feature_indices:
            X_freq = X[:, self.feature_indices]
        else:
            X_freq = X[:, :4]
        X_scaled = self.scaler.transform(X_freq)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.feature_indices:
            X_freq = X[:, self.feature_indices]
        else:
            X_freq = X[:, :4]
        X_scaled = self.scaler.transform(X_freq)
        return self.model.predict_proba(X_scaled)


class FullFeatureModel(BaselineModel):
    """Classifier using all morphology + frequency features."""
    
    def __init__(self, model_type: str = "rf"):
        super().__init__("full_feature")
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gb":
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42
            )
        else:  # logistic
            self.model = LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                random_state=42
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Log feature importance for RF
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_idx = np.argsort(importances)[-5:][::-1]
            feature_cols = get_feature_columns()
            logger.info("Top 5 features: " + ", ".join(
                f"{feature_cols[i] if i < len(feature_cols) else f'f{i}'}={importances[i]:.3f}" 
                for i in top_idx
            ))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self, feature_cols: List[str] = None) -> Dict[str, float]:
        """Get feature importance (for tree-based models)."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        feature_cols = feature_cols or get_feature_columns()
        importances = self.model.feature_importances_
        
        return {
            feature_cols[i] if i < len(feature_cols) else f"feature_{i}": float(imp)
            for i, imp in enumerate(importances)
        }


def get_baseline_model(variant: str) -> BaselineModel:
    """
    Factory function to create baseline model by variant name.
    
    Args:
        variant: One of "majority", "freq_only", "full_feature"
    """
    if variant == "majority":
        return MajorityBaseline()
    elif variant == "freq_only":
        return FrequencyOnlyModel()
    elif variant == "full_feature":
        return FullFeatureModel(model_type="rf")
    elif variant == "full_feature_lr":
        model = FullFeatureModel(model_type="logistic")
        model.name = "full_feature_lr"
        return model
    elif variant == "full_feature_gb":
        model = FullFeatureModel(model_type="gb")
        model.name = "full_feature_gb"
        return model
    else:
        raise ValueError(f"Unknown baseline variant: {variant}")


def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and label vector y from DataFrame.
    """
    feature_cols = feature_cols or get_feature_columns()
    
    # Get available feature columns
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(f"No feature columns found in DataFrame. Have: {list(df.columns)}")
    
    X = df[available].values.astype(np.float32)
    y = df['cefr'].values
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    
    return X, y
