"""
Scrape word frequencies from QazCorpus.kz search interface.
URL pattern: https://qazcorpus.kz/allqorpus/index4.php?soz={word}
"""

import time
import json
import random
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import re

# Config
INPUT_FILE = "dataset/splits/train.csv"
OUTPUT_FILE = "dataset/external/qazcorpus_freqs.json"
DELAY_RANGE = (0.5, 1.5)

# New endpoint discovered
BASE_URL = "https://qazcorpus.kz/allqorpus/sozsany11.php"

def get_word_count(word):
    """Query qazcorpus JSON API for word count."""
    try:
        response = requests.get(BASE_URL, params={"soz": word}, timeout=10)
        
        if response.status_code != 200:
            print(f"Error {response.status_code} for {word}")
            return None
            
        # Parse JSON response: {"count": 123}
        data = response.json()
        return data.get("count", 0)
        
    except Exception as e:
        print(f"Exception for {word}: {e}")
        return None

def main():
    # 1. Load lexicon to get unique lemmas
    print("Loading lexicon...")
    # Load all splits to ensure coverage
    lemmas = set()
    for split in ['train', 'dev', 'test']:
        path = Path(f"dataset/splits/{split}.csv")
        if path.exists():
            df = pd.read_csv(path)
            lemmas.update(df['lemma'].astype(str).tolist())
    
    print(f"Found {len(lemmas)} unique lemmas to scrape.")
    
    # 2. Check for existing progress
    results = {}
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing records.")
        
    # Filter out already done
    todo = [w for w in lemmas if w not in results]
    print(f"Remaining to scrape: {len(todo)}")
    
    # 3. Scrape loop
    # We'll do a small batch first to verify it works
    session = requests.Session()
    
    for i, word in enumerate(tqdm(todo)):
        count = get_word_count(word)
        
        if count is not None:
            results[word] = count
        else:
            # On error, maybe retry or skip. We record -1 for error to distinguish from 0?
            # actually None usually means connection error. 0 means not found.
            pass

        # Save every 50 requests
        if i % 50 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        time.sleep(random.uniform(*DELAY_RANGE))
        
    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()
