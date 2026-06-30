"""
Normalize PDF-extracted text files.

Fixes:
1. Joins lines that were broken mid-sentence (PDF extraction artifacts)
2. Preserves empty lines between separate texts
3. Outputs clean texts separated by double newlines
"""

import re
from pathlib import Path
from typing import List

INPUT_DIR = Path("dataset/raw/pdfs/textbooks")
OUTPUT_DIR = Path("dataset/processed/texts")

LEVELS = ["A1", "A2", "B1", "B2", "C1"]


def normalize_text(raw_text: str) -> List[str]:
    """
    Normalize raw PDF-extracted text.
    
    Returns list of individual texts (paragraphs).
    """
    # Split by empty lines (text separators)
    chunks = re.split(r'\n\s*\n', raw_text)
    
    normalized_texts = []
    
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # Join lines within the chunk (remove mid-sentence breaks)
        # Replace single newlines with spaces
        lines = chunk.split('\n')
        joined_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if previous line ended mid-word (hyphenation)
            if joined_lines and joined_lines[-1].endswith('-'):
                # Remove hyphen and join without space
                joined_lines[-1] = joined_lines[-1][:-1] + line
            else:
                joined_lines.append(line)
        
        # Join all lines with spaces
        normalized = ' '.join(joined_lines)
        
        # Clean up extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        if normalized:
            normalized_texts.append(normalized)
    
    return normalized_texts


def process_level(level: str) -> int:
    """Process a single level file."""
    input_path = INPUT_DIR / f"{level}.txt"
    output_path = OUTPUT_DIR / f"{level}_texts.json"
    
    if not input_path.exists():
        print(f"  {level}: File not found, skipping")
        return 0
    
    # Read raw text
    with input_path.open("r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Normalize
    texts = normalize_text(raw_text)
    
    # Save as JSON (list of text objects)
    import json
    output_data = [
        {
            "id": f"{level}_{i:04d}",
            "text": text,
            "level": level,
        }
        for i, text in enumerate(texts)
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Also save as plain text (for quick viewing)
    txt_output = OUTPUT_DIR / f"{level}_texts.txt"
    with txt_output.open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n\n")
    
    return len(texts)


def main():
    print("Normalizing text corpus...")
    print()
    
    total = 0
    for level in LEVELS:
        count = process_level(level)
        print(f"  {level}: {count} texts")
        total += count
    
    print()
    print(f"Total: {total} texts")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
