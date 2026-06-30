"""
Kazakh CEFR Classification Package.

Word-level CEFR classification for Kazakh using
lexicon-based features and morphological analysis.
"""

from . import data
from . import morphology

__version__ = "0.2.0"

__all__ = ["data", "morphology"]
