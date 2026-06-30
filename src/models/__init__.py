"""
CEFR Models package.
"""

from .baselines import (
    BaselineModel,
    MajorityBaseline,
    FrequencyOnlyModel,
    FullFeatureModel,
    get_baseline_model,
    prepare_features
)

from .context_encoder import ContextEncoder, ContextTokenizer, ContextOnlyCEFR
from .morph_encoder import CharCNNEncoder, MorphFeatureEncoder, CombinedMorphEncoder, CharVocab
from .fusion import ConcatFusion, GatedFusion, get_fusion_module
from .dual_encoder import DualEncoderCEFR, DualEncoderDataset, collate_dual_encoder

__all__ = [
    # Baselines
    "BaselineModel",
    "MajorityBaseline", 
    "FrequencyOnlyModel",
    "FullFeatureModel",
    "get_baseline_model",
    "prepare_features",
    # Context
    "ContextEncoder",
    "ContextTokenizer",
    "ContextOnlyCEFR",
    # Morph
    "CharCNNEncoder",
    "MorphFeatureEncoder",
    "CombinedMorphEncoder",
    "CharVocab",
    # Fusion
    "ConcatFusion",
    "GatedFusion",
    "get_fusion_module",
    # Dual
    "DualEncoderCEFR",
    "DualEncoderDataset",
    "collate_dual_encoder",
]
