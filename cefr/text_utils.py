import re

WORD_PATTERN = re.compile(r"[^\W\d_]+(?:[''‐-][^\W\d_]+)*", re.UNICODE)
CYRILLIC_WORD = re.compile(r"^[а-яёіїңғүұәөһіъь]+(?:[''‐-][а-яёіїңғүұәөһіъь]+)*$", re.IGNORECASE)

def tokenize_words(text):
    """Return non-empty tokens captured by the shared word pattern."""
    return WORD_PATTERN.findall(text)

def is_cyrillic_token(token):
    t = token.strip().lower()
    return bool(t and CYRILLIC_WORD.fullmatch(t.replace("'", "'")))

__all__ = ["tokenize_words", "is_cyrillic_token"]
