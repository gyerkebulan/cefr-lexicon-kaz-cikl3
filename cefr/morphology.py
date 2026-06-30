"""
Morphological analysis interface for Kazakh.

Provides an abstract interface for morphological analysis with
pluggable backends (Apertium, KazNLP, Stanza, or placeholder).
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# Kazakh-specific character pattern
KAZAKH_PATTERN = re.compile(r"[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ]+", re.IGNORECASE)


@dataclass
class MorphAnalysis:
    """Result of morphological analysis for a single word."""
    
    surface: str  # Original word form
    lemma: str  # Dictionary form
    pos: str  # Part of speech
    num_morphemes: int = 1
    
    # Binary features
    has_case: bool = False
    has_plural: bool = False
    has_possessive: bool = False
    
    # Optional detailed features (if analyzer supports)
    case_type: Optional[str] = None  # NOM, ACC, DAT, GEN, LOC, ABL, INS
    tense: Optional[str] = None  # For verbs
    num_derivations: int = 0
    
    # Raw analyzer output for debugging
    raw_tags: List[str] = field(default_factory=list)
    
    @property
    def complexity_index(self) -> float:
        """Derived complexity score (0-1 scale, higher = more complex)."""
        return min(1.0, (
            self.num_morphemes * 0.25 +
            self.has_case * 0.15 +
            self.has_possessive * 0.2 +
            self.num_derivations * 0.2 +
            (len(self.surface) / 20) * 0.2  # Normalized length
        ))


class MorphAnalyzer(ABC):
    """Abstract interface for morphological analyzers."""
    
    @abstractmethod
    def analyze(self, word: str) -> MorphAnalysis:
        """Analyze a single word."""
        ...
    
    @abstractmethod
    def analyze_batch(self, words: List[str]) -> List[MorphAnalysis]:
        """Analyze multiple words."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the analyzer backend."""
        ...


class PlaceholderAnalyzer(MorphAnalyzer):
    """
    Placeholder analyzer using heuristics.
    
    Used when no proper morphological analyzer is available.
    Estimates morpheme count based on word length and suffix patterns.
    """
    
    # Common Kazakh suffixes for heuristic analysis
    CASE_SUFFIXES = {
        "да", "де", "та", "те",  # Locative
        "ға", "ге", "қа", "ке",  # Dative
        "дан", "ден", "тан", "тен",  # Ablative
        "мен", "бен", "пен",  # Instrumental
        "ны", "ні", "ды", "ді", "ты", "ті",  # Accusative
        "ның", "нің", "дың", "дің", "тың", "тің",  # Genitive
    }
    
    POSSESSIVE_SUFFIXES = {
        "м", "ым", "ім",  # 1sg
        "ң", "ың", "ің",  # 2sg
        "сы", "сі",  # 3sg
        "мыз", "міз",  # 1pl
        "сыңдар", "сіңдер",  # 2pl
    }
    
    PLURAL_SUFFIXES = {"лар", "лер", "дар", "дер", "тар", "тер"}
    
    POS_HEURISTICS = {
        "у": "VERB",  # Infinitive ending
        "ды": "VERB", "ді": "VERB", "ты": "VERB", "ті": "VERB",  # Past
        "ған": "VERB", "ген": "VERB",  # Past participle
    }
    
    @property
    def name(self) -> str:
        return "placeholder"
    
    def analyze(self, word: str) -> MorphAnalysis:
        word_lower = word.lower().strip()
        
        # Estimate morphemes based on length
        num_morphemes = max(1, len(word_lower) // 3)
        
        # Check for case markers
        has_case = any(word_lower.endswith(s) for s in self.CASE_SUFFIXES)
        case_type = None
        if has_case:
            for suffix in self.CASE_SUFFIXES:
                if word_lower.endswith(suffix):
                    if suffix in {"да", "де", "та", "те"}:
                        case_type = "LOC"
                    elif suffix in {"ға", "ге", "қа", "ке"}:
                        case_type = "DAT"
                    elif suffix in {"дан", "ден", "тан", "тен"}:
                        case_type = "ABL"
                    elif suffix in {"мен", "бен", "пен"}:
                        case_type = "INS"
                    elif suffix in {"ны", "ні", "ды", "ді", "ты", "ті"}:
                        case_type = "ACC"
                    elif suffix in {"ның", "нің", "дың", "дің", "тың", "тің"}:
                        case_type = "GEN"
                    break
        
        # Check for possessive
        has_possessive = any(word_lower.endswith(s) for s in self.POSSESSIVE_SUFFIXES)
        
        # Check for plural
        has_plural = any(word_lower.endswith(s) for s in self.PLURAL_SUFFIXES)
        
        # Guess POS
        pos = "NOUN"  # Default
        for ending, tag in self.POS_HEURISTICS.items():
            if word_lower.endswith(ending):
                pos = tag
                break
        
        # Estimate lemma (rough: strip common suffixes)
        lemma = word_lower
        for suffix_set in [self.CASE_SUFFIXES, self.POSSESSIVE_SUFFIXES, self.PLURAL_SUFFIXES]:
            for suffix in sorted(suffix_set, key=len, reverse=True):
                if lemma.endswith(suffix) and len(lemma) > len(suffix) + 2:
                    lemma = lemma[:-len(suffix)]
                    break
        
        return MorphAnalysis(
            surface=word,
            lemma=lemma,
            pos=pos,
            num_morphemes=num_morphemes,
            has_case=has_case,
            case_type=case_type,
            has_plural=has_plural,
            has_possessive=has_possessive,
            num_derivations=max(0, num_morphemes - 2),
        )
    
    def analyze_batch(self, words: List[str]) -> List[MorphAnalysis]:
        return [self.analyze(w) for w in words]


# Global analyzer instance (configurable)
_analyzer: Optional[MorphAnalyzer] = None


def get_analyzer() -> MorphAnalyzer:
    """Get the configured morphological analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = PlaceholderAnalyzer()
        logger.warning(
            "Using placeholder morphological analyzer. "
            "Configure a proper backend (Apertium/Stanza) for production."
        )
    return _analyzer


def set_analyzer(analyzer: MorphAnalyzer) -> None:
    """Set the morphological analyzer backend."""
    global _analyzer
    _analyzer = analyzer
    logger.info(f"Morphological analyzer set to: {analyzer.name}")


def analyze_word(word: str) -> MorphAnalysis:
    """Analyze a single word using the configured analyzer."""
    return get_analyzer().analyze(word)


def analyze_words(words: List[str]) -> List[MorphAnalysis]:
    """Analyze multiple words using the configured analyzer."""
    return get_analyzer().analyze_batch(words)


def extract_features(analysis: MorphAnalysis) -> Dict[str, float]:
    """Extract feature vector from morphological analysis."""
    return {
        "num_morphemes": float(analysis.num_morphemes),
        "has_case": float(analysis.has_case),
        "has_plural": float(analysis.has_plural),
        "has_possessive": float(analysis.has_possessive),
        "num_derivations": float(analysis.num_derivations),
        "word_length": float(len(analysis.surface)),
        "complexity_index": analysis.complexity_index,
    }


def is_kazakh_word(word: str) -> bool:
    """Check if a word contains Kazakh/Cyrillic characters."""
    return bool(KAZAKH_PATTERN.fullmatch(word))


__all__ = [
    "MorphAnalysis",
    "MorphAnalyzer",
    "PlaceholderAnalyzer",
    "get_analyzer",
    "set_analyzer",
    "analyze_word",
    "analyze_words",
    "extract_features",
    "is_kazakh_word",
]
