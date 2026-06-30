"""
Download and extract text from Abai Institute PDF textbooks.

Processes official "Lexical-Grammatical Minimum" textbooks to create
a CEFR-labeled corpus.
"""

from __future__ import annotations

import logging
import re
import ssl
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import PyPDF2

logger = logging.getLogger(__name__)

# Official textbook URLs
PDF_SOURCES = {
    "A1": "https://abai.institute/storage/2025/04/03/7086/01JQXJ5R5M55N74478Z3E1V6C4.pdf",
    "A2": "https://abai.institute/storage/2025/04/03/7088/01JQXJ719067YD4SSPF2RNQNY4.pdf",
    "B1": "https://abai.institute/storage/2025/04/03/7090/01JQXJ878DM2YJ9FV98227Y3B8.pdf",
    "B2": "https://abai.institute/storage/2025/04/03/7092/01JQXJ9FZY2BR5WF49TJY932GS.pdf",
    "C1": "https://abai.institute/storage/2025/04/03/7094/01JQXJB1YNW5QTB7RYM451AX01.pdf",
}

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PDF_DIR = DEFAULT_RAW_DIR / "pdfs"

# Patterns to clean (page numbers, headers, exercise prompts)
CLEANING_PATTERNS = [
    (re.compile(r'^\s*\d+\s*$', re.MULTILINE), ''),  # Page numbers
    (re.compile(r'Тапсырма.*?:', re.IGNORECASE), ''),  # "Task:"
    (re.compile(r'Жаттығу.*?:', re.IGNORECASE), ''),  # "Exercise:"
    (re.compile(r'^\s*[А-ЯӘҒҚҢӨҰҮҺІ]+\s*$', re.MULTILINE), ''),  # All caps headers (simple)
]


def download_pdf(level: str, url: str, output_dir: Path) -> Path:
    """Download PDF for a specific level."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"textbook_{level}.pdf"
    
    if filename.exists():
        logger.info(f"PDF for {level} already exists: {filename}")
        return filename
    
    logger.info(f"Downloading {level} PDF from {url}...")
    
    # Bypass SSL verification if needed (common for some inst edu sites)
    context = ssl._create_unverified_context()
    
    with urllib.request.urlopen(url, context=context) as response, open(filename, 'wb') as out_file:
        out_file.write(response.read())
    
    logger.info(f"Saved to {filename}")
    return filename


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from PDF."""
    text_content = []
    
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            logger.info(f"Extracting text from {pdf_path.name} ({num_pages} pages)...")
            
            for i, page in enumerate(reader.pages):
                # Skip typical front matter (first 5-10 pages) if extracted text is sparse
                # For now, extract everything and rely on cleaning
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                    
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""
    
    return "\n".join(text_content)


def clean_extracted_text(text: str) -> str:
    """Clean common artifacts from extracted PDF text."""
    cleaned = text
    for pattern, replacement in CLEANING_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def split_into_documents(text: str, level: str, chunk_size: int = 500) -> List[Dict]:
    """
    Split extracted text into logical 'documents' to simulate checking dispersion.
    
    Since we have one giant textbook, we simulate separate documents by splitting 
    into chunks of roughly 'chunk_size' words.
    """
    words = text.split()
    total_words = len(words)
    documents = []
    
    for i in range(0, total_words, chunk_size):
        chunk = words[i:i + chunk_size]
        if len(chunk) < 50:  # Skip tiny chunks
            continue
            
        doc_content = " ".join(chunk)
        documents.append({
            "id": f"{level}_doc_{i // chunk_size + 1}",
            "level": level,
            "content": doc_content,
            "source": f"textbook_{level}.pdf_chunk_{i // chunk_size + 1}"
        })
    
    return documents


def process_pdfs(
    levels: List[str] = list(PDF_SOURCES.keys()),
    raw_dir: Path = DEFAULT_RAW_DIR,
    pdf_dir: Path = DEFAULT_PDF_DIR,
) -> Dict[str, int]:
    """Download PDFs, extract text, and save as processed JSONs."""
    import json
    
    stats = {}
    
    for level in levels:
        if level not in PDF_SOURCES:
            logger.warning(f"No PDF source for {level}")
            continue
            
        try:
            # 1. Download
            pdf_path = download_pdf(level, PDF_SOURCES[level], pdf_dir)
            
            # 2. Extract
            raw_text = extract_text_from_pdf(pdf_path)
            if not raw_text:
                logger.warning(f"No text extracted for {level}")
                continue
                
            # 3. Clean
            cleaned_text = clean_extracted_text(raw_text)
            
            # 4. Split
            documents = split_into_documents(cleaned_text, level)
            
            # 5. Save
            output_path = raw_dir / f"{level}_texts.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            
            stats[level] = len(documents)
            logger.info(f"Saved {len(documents)} simulated documents for {level}")
            
        except Exception as e:
            logger.error(f"Error processing {level}: {e}")
            stats[level] = 0
            
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_pdfs()
