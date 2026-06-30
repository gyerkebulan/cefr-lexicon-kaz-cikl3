"""
Extract CEFR word lists from Abai Institute PDF textbooks.

Follows user instructions:
- Extract from Alphabetical and Thematical sections only
- Format: "word - POS"
- Save as JSON: {"word": "...", "cefr": "A1", "pos": "..."}
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import PyPDF2

logger = logging.getLogger(__name__)

# Page ranges: (start_alpha, start_thematic, end_thematic)
PAGE_RANGES = {
    "A1": (7, 18, 29),
    "A2": (6, 25, 44),
    "B1": (7, 37, 68),
    "B2": (7, 50, 91),
    "C1": (7, 60, 114),
}

# POS tags in Kazakh
POS_TAGS = {
    "зат есім": "NOUN",
    "етістік": "VERB",
    "сын есім": "ADJ",
    "сан есім": "NUM",
    "үстеу": "ADV",
    "есімдік": "PRON",
    "шылау": "ADP",
    "одағай": "INTJ",
    "модаль сөз": "MODAL",
    "модаль": "MODAL",
    "көмекші сөз": "AUX",
}


def extract_pages(pdf_path: Path, start_page: int, end_page: int) -> str:
    """Extract text from a range of pages (1-indexed)."""
    text_content = []
    
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        # Convert to 0-indexed
        for i in range(start_page - 1, min(end_page, len(reader.pages))):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text_content.append(page_text)
    
    return "\n".join(text_content)


def parse_word_pos(text: str) -> List[Tuple[str, str]]:
    """
    Parse "word - POS" or "word POS" patterns from text.
    
    Returns list of (word, pos) tuples.
    """
    entries = []
    
    # Clean text
    text = re.sub(r'\s+', ' ', text)
    
    # Split by common delimiters
    for pos_kz, pos_en in POS_TAGS.items():
        # Pattern: word followed by POS tag
        pattern = rf'([а-яәғқңөұүһіА-ЯӘҒҚҢӨҰҮҺІ][а-яәғқңөұүһі\-/()]+)\s*{re.escape(pos_kz)}'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for match in matches:
            word = match.strip().lower()
            # Clean up
            word = re.sub(r'\s+', ' ', word)
            if word and len(word) > 1:
                entries.append((word, pos_en))
    
    return entries


def extract_words_from_level(level: str, pdf_dir: Path) -> List[Dict]:
    """Extract all words for a specific CEFR level."""
    pdf_path = pdf_dir / f"textbook_{level}.pdf"
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return []
    
    start_alpha, start_thematic, end_thematic = PAGE_RANGES[level]
    
    # Extract alphabetical section (from start_alpha to start_thematic - 1)
    alpha_text = extract_pages(pdf_path, start_alpha, start_thematic - 1)
    alpha_entries = parse_word_pos(alpha_text)
    
    # Extract thematical section (from start_thematic to end_thematic)
    thematic_text = extract_pages(pdf_path, start_thematic, end_thematic)
    thematic_entries = parse_word_pos(thematic_text)
    
    # Combine and deduplicate
    seen = set()
    words = []
    
    for word, pos in alpha_entries + thematic_entries:
        key = (word, pos)
        if key not in seen:
            seen.add(key)
            words.append({
                "word": word,
                "pos": pos,
                "cefr": level,
                "source": "alphabetical" if (word, pos) in alpha_entries else "thematical"
            })
    
    logger.info(f"{level}: Extracted {len(words)} unique words")
    return words


def extract_all_levels(
    pdf_dir: Path = Path("data/raw/pdfs"),
    output_path: Path = Path("data/processed/silver_lexicon_raw.json"),
) -> Dict[str, int]:
    """Extract words from all levels and save as JSON."""
    
    all_words = []
    stats = {}
    
    for level in PAGE_RANGES.keys():
        words = extract_words_from_level(level, pdf_dir)
        all_words.extend(words)
        stats[level] = len(words)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(all_words)} total words to {output_path}")
    return stats


def build_lexicon_from_extracted(
    input_path: Path = Path("data/processed/silver_lexicon_raw.json"),
    output_path: Path = Path("data/processed/silver_lexicon.json"),
) -> Dict:
    """
    Build final silver lexicon from extracted words.
    
    Handles words appearing in multiple levels by taking the lowest level.
    """
    with input_path.open("r", encoding="utf-8") as f:
        all_words = json.load(f)
    
    # Group by word+pos, keep lowest level
    level_order = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5}
    lexicon = {}
    
    for entry in all_words:
        key = f"{entry['word']}_{entry['pos']}"
        
        if key not in lexicon or level_order[entry['cefr']] < level_order[lexicon[key]['entry_level']]:
            lexicon[key] = {
                "lemma": entry['word'],
                "pos": entry['pos'],
                "entry_level": entry['cefr'],
                "entry_strength": 1.0,  # High confidence for official textbook
                "frequencies": {entry['cefr']: 1000.0},  # Placeholder
                "text_counts": {entry['cefr']: 10},  # Placeholder
                "total_occurrences": 100,  # Placeholder
            }
    
    # Save
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Built lexicon with {len(lexicon)} entries")
    return {"total": len(lexicon)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = extract_all_levels()
    print(f"Extraction stats: {stats}")
    lex_stats = build_lexicon_from_extracted()
    print(f"Lexicon stats: {lex_stats}")
