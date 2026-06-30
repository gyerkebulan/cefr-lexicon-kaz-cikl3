"""
Text preprocessing for Kazakh learner corpus.

Provides sentence segmentation, tokenization, and text cleaning
for Kazakh texts.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .scrape_learner_corpus import CEFR_LEVELS, load_learner_texts

logger = logging.getLogger(__name__)

# Kazakh sentence terminators
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')

# Word tokenization (Kazakh Cyrillic + common punctuation)
WORD_PATTERN = re.compile(r"[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ]+", re.IGNORECASE)

# Noise patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
EXCESSIVE_WHITESPACE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """Remove noise and normalize whitespace."""
    text = URL_PATTERN.sub('', text)
    text = EMAIL_PATTERN.sub('', text)
    text = EXCESSIVE_WHITESPACE.sub(' ', text)
    return text.strip()


def segment_sentences(text: str) -> List[str]:
    """Split Kazakh text into sentences."""
    sentences = SENTENCE_SPLIT_PATTERN.split(text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> List[str]:
    """Extract word tokens from text."""
    return WORD_PATTERN.findall(text)


def analyze_text(text: str) -> Dict:
    """Analyze a single text, returning structured data."""
    from ..morphology import analyze_words, extract_features
    
    clean = clean_text(text)
    sentences = segment_sentences(clean)
    
    processed_sentences = []
    for sent_idx, sent in enumerate(sentences):
        tokens = tokenize(sent)
        analyses = analyze_words(tokens)
        
        processed_sentences.append({
            "sent_id": sent_idx,
            "text": sent,
            "tokens": tokens,
            "analysis": [
                {
                    "surface": a.surface,
                    "lemma": a.lemma,
                    "pos": a.pos,
                    "features": extract_features(a),
                }
                for a in analyses
            ],
        })
    
    return {
        "sentences": processed_sentences,
        "num_sentences": len(processed_sentences),
        "num_tokens": sum(len(s["tokens"]) for s in processed_sentences),
    }


def process_level_corpus(
    level: str,
    input_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed"),
) -> Dict:
    """Process all texts for a CEFR level."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    texts = load_learner_texts(level, input_dir)
    
    processed = []
    total_tokens = 0
    
    for idx, text_data in enumerate(texts):
        content = text_data.get("content", "")
        if not content.strip():
            continue
        
        analyzed = analyze_text(content)
        
        processed.append({
            "text_id": f"{level}_{idx}",
            "level": level,
            "original_id": text_data.get("id", str(idx)),
            **analyzed,
        })
        
        total_tokens += analyzed["num_tokens"]
    
    # Save processed corpus
    output_path = output_dir / f"level_{level}_corpus.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    
    stats = {
        "level": level,
        "num_texts": len(processed),
        "total_tokens": total_tokens,
        "avg_tokens_per_text": total_tokens // max(1, len(processed)),
    }
    
    logger.info(f"Processed {level}: {stats['num_texts']} texts, {stats['total_tokens']} tokens")
    return stats


def process_all_levels(
    input_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed"),
) -> Dict[str, Dict]:
    """Process all CEFR levels."""
    stats = {}
    for level in CEFR_LEVELS:
        try:
            stats[level] = process_level_corpus(level, input_dir, output_dir)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {level}: {e}")
            stats[level] = {"level": level, "num_texts": 0, "total_tokens": 0}
    return stats


def load_processed_corpus(
    level: str,
    input_dir: Path = Path("data/processed"),
) -> List[Dict]:
    """Load processed corpus for a level."""
    path = Path(input_dir) / f"level_{level}_corpus.json"
    if not path.exists():
        raise FileNotFoundError(f"Processed corpus not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def quality_check(corpus: List[Dict], min_tokens: int = 20) -> Dict:
    """Run quality checks on processed corpus."""
    issues = {
        "short_texts": [],
        "empty_texts": [],
        "parse_failures": [],
    }
    
    for text in corpus:
        text_id = text.get("text_id", "unknown")
        num_tokens = text.get("num_tokens", 0)
        
        if num_tokens == 0:
            issues["empty_texts"].append(text_id)
        elif num_tokens < min_tokens:
            issues["short_texts"].append((text_id, num_tokens))
        
        # Check for analysis failures
        for sent in text.get("sentences", []):
            failed = sum(1 for a in sent.get("analysis", []) if a.get("pos") == "UNK")
            if failed > 0:
                issues["parse_failures"].append((text_id, sent.get("sent_id"), failed))
    
    return issues


__all__ = [
    "clean_text",
    "segment_sentences",
    "tokenize",
    "analyze_text",
    "process_level_corpus",
    "process_all_levels",
    "load_processed_corpus",
    "quality_check",
]
