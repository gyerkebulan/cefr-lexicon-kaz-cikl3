"""
Data utilities for Kazakh CEFR classification.

Provides access to learner corpus, preprocessing, frequency profiling,
and silver lexicon generation.
"""

from .scrape_learner_corpus import (
    CEFR_LEVELS,
    download_all_levels,
    load_all_levels,
    load_learner_texts,
    corpus_stats,
)

from .preprocess import (
    clean_text,
    segment_sentences,
    tokenize,
    process_all_levels,
    load_processed_corpus,
    quality_check,
)

from .frequency_profiles import (
    build_frequency_profiles,
    load_frequency_profiles,
    frequency_stats,
)

from .entry_condition import (
    assign_entry_level,
    build_silver_lexicon,
    validate_silver_lexicon,
    save_silver_lexicon,
    load_silver_lexicon,
    DEFAULT_THRESHOLDS,
    FUNCTION_WORD_OVERRIDES,
)


__all__ = [
    # Constants
    "CEFR_LEVELS",
    "DEFAULT_THRESHOLDS",
    "FUNCTION_WORD_OVERRIDES",
    # Corpus
    "download_all_levels",
    "load_all_levels",
    "load_learner_texts",
    "corpus_stats",
    # Preprocessing
    "clean_text",
    "segment_sentences",
    "tokenize",
    "process_all_levels",
    "load_processed_corpus",
    "quality_check",
    # Frequency
    "build_frequency_profiles",
    "load_frequency_profiles",
    "frequency_stats",
    # Silver lexicon
    "assign_entry_level",
    "build_silver_lexicon",
    "validate_silver_lexicon",
    "save_silver_lexicon",
    "load_silver_lexicon",
]
