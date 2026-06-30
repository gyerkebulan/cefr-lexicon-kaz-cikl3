"""
Lemmatization module for Kazakh text.
Supports multiple backends: Apertium, KazNLP, and simple rule-based.
"""

import logging
import subprocess
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("cefr")


# =============================================================================
# Abstract Base Lemmatizer
# =============================================================================

class BaseLemmatizer(ABC):
    """Abstract base class for lemmatizers."""
    
    name: str = "base"
    
    @abstractmethod
    def lemmatize(self, word: str) -> str:
        """Return the lemma of a word."""
        pass
    
    @abstractmethod
    def analyze(self, word: str) -> Dict:
        """Return full morphological analysis."""
        pass
    
    def lemmatize_batch(self, words: List[str]) -> List[str]:
        """Lemmatize a batch of words."""
        return [self.lemmatize(w) for w in words]
    
    def is_available(self) -> bool:
        """Check if this lemmatizer is available."""
        return True


# =============================================================================
# Apertium Lemmatizer (Rule-based FST)
# =============================================================================

class ApertiumLemmatizer(BaseLemmatizer):
    """
    Apertium-based lemmatizer for Kazakh.
    Uses the apertium-kaz finite-state transducer.
    
    Install:
        pip install apertium
        # Then in Python:
        import apertium
        apertium.installer.install_module('kaz')
    """
    
    name = "apertium"
    
    def __init__(self):
        self._analyzer = None
        self._available = None
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        try:
            import apertium
            self._analyzer = apertium.Analyzer('kaz')
            self._available = True
            logger.info("Apertium Kazakh analyzer loaded")
        except Exception as e:
            logger.warning(f"Apertium not available: {e}")
            self._available = False
        
        return self._available
    
    def analyze(self, word: str) -> Dict:
        """
        Full morphological analysis.
        Returns dict with lemma, pos, tags, etc.
        """
        if not self.is_available():
            return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': ''}
        
        try:
            analyses = self._analyzer.analyze(word)
            if not analyses:
                return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': ''}
            
            # Parse first analysis (most likely)
            # Format: ^word/lemma<tag1><tag2>...$
            raw = str(analyses[0])
            
            # Extract lemma (before first <)
            match = re.search(r'/([^<]+)', raw)
            lemma = match.group(1) if match else word
            
            # Extract tags
            tags = re.findall(r'<([^>]+)>', raw)
            
            # Determine POS from tags
            pos_map = {
                'n': 'NOUN', 'v': 'VERB', 'adj': 'ADJ', 'adv': 'ADV',
                'prn': 'PRON', 'num': 'NUM', 'post': 'ADP', 'cnjcoo': 'CONJ',
                'ij': 'INTJ', 'det': 'DET'
            }
            pos = 'OTHER'
            for tag in tags:
                if tag in pos_map:
                    pos = pos_map[tag]
                    break
            
            return {
                'lemma': lemma,
                'pos': pos,
                'tags': tags,
                'raw': raw
            }
        except Exception as e:
            logger.debug(f"Apertium analysis failed for '{word}': {e}")
            return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': ''}
    
    def lemmatize(self, word: str) -> str:
        """Get just the lemma."""
        return self.analyze(word)['lemma']


# =============================================================================
# HFST-based Apertium Lemmatizer (compiled from apertium-kaz)
# =============================================================================

class HFSTLemmatizer(BaseLemmatizer):
    """
    HFST-based lemmatizer using compiled apertium-kaz transducer.
    
    Compiles apertium-kaz.kaz.lexc + .twol into a finite-state transducer
    for accurate Kazakh morphological analysis.
    
    Requires:
        pip install hfst
        git clone https://github.com/apertium/apertium-kaz
    """
    
    name = "hfst"
    
    # POS tag mapping (Apertium tagset)
    POS_MAP = {
        'n': 'NOUN', 'np': 'NOUN', 'prop': 'NOUN',
        'v': 'VERB', 'vaux': 'VERB',
        'adj': 'ADJ',
        'adv': 'ADV',
        'prn': 'PRON',
        'num': 'NUM',
        'post': 'ADP',
        'cnjcoo': 'CONJ', 'cnjsub': 'CONJ', 'cnjadv': 'CONJ',
        'ij': 'INTJ',
        'det': 'DET',
        'abbr': 'OTHER',
    }
    
    # Tags that indicate inflectional morphology
    INFL_TAGS = {
        'nom', 'gen', 'dat', 'acc', 'abl', 'loc', 'ins',  # case
        'pl',  # number
        'p1', 'p2', 'p3',  # person
        'sg',  # singular
        'px1sg', 'px2sg', 'px3sp', 'px1pl', 'px2pl', 'px3pl',  # possessive
        'imp', 'aor', 'past', 'ifi', 'pres', 'fut',  # TAM
        'cond', 'opt',  # mood
    }
    
    # Tags that indicate derivational morphology
    DERIV_TAGS = {
        'ger', 'ger_past', 'ger_perf', 'ger_ppot', 'ger_abs', 'ger_fut',
        'gpr_past', 'gpr_impf', 'gpr_pot', 'gpr_ppot', 'gpr_fut',
        'prc_perf', 'prc_impf', 'prc_plan', 'prc_fplan',
        'caus', 'pass', 'coop',
        'subst', 'attr',
        'sim', 'advl',
    }
    
    def __init__(self, apertium_kaz_dir: str = None):
        self._analyzer = None
        self._available = None
        self._apertium_kaz_dir = apertium_kaz_dir
    
    def _find_apertium_kaz_dir(self) -> Optional[Path]:
        """Locate the apertium-kaz directory."""
        project_dir = Path(__file__).parent.parent
        candidates = [
            Path(self._apertium_kaz_dir) if self._apertium_kaz_dir else None,
            project_dir / 'apertium-kaz',
            Path.cwd() / 'apertium-kaz',
        ]
        for c in candidates:
            if c and c.exists() and (c / 'apertium-kaz.kaz.lexc').exists():
                return c
        return None
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        try:
            import hfst
        except ImportError:
            logger.warning("hfst Python package not installed (pip install hfst)")
            self._available = False
            return False
        
        kaz_dir = self._find_apertium_kaz_dir()
        if not kaz_dir:
            logger.warning("apertium-kaz directory not found. Clone from: "
                           "https://github.com/apertium/apertium-kaz")
            self._available = False
            return False
        
        # Check for pre-compiled analyzer
        cached = kaz_dir / 'kaz.analyzer.hfst'
        if cached.exists():
            try:
                istr = hfst.HfstInputStream(str(cached))
                self._analyzer = istr.read()
                self._analyzer.invert()
                self._analyzer.minimize()
                logger.info(f"Loaded cached HFST analyzer from {cached}")
                self._available = True
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached analyzer: {e}")
        
        # Compile from source
        try:
            logger.info("Compiling apertium-kaz transducer (one-time)...")
            lexc_file = str(kaz_dir / 'apertium-kaz.kaz.lexc')
            twol_file = str(kaz_dir / 'apertium-kaz.kaz.twol')
            twol_out = str(kaz_dir / 'kaz.twol.hfst')
            
            fst = hfst.compile_lexc_file(lexc_file)
            logger.info(f"  Lexc compiled: {fst.number_of_states()} states")
            
            hfst.compile_twolc_file(twol_file, twol_out)
            twol_stream = hfst.HfstInputStream(twol_out)
            rules = []
            while not twol_stream.is_eof():
                rules.append(twol_stream.read())
            logger.info(f"  Twol compiled: {len(rules)} rules")
            
            fst.compose_intersect(tuple(rules))
            fst.minimize()
            logger.info(f"  Composed FST: {fst.number_of_states()} states")
            
            # Save compiled analyzer
            ostr = hfst.HfstOutputStream(filename=str(cached))
            ostr.write(fst)
            ostr.flush()
            ostr.close()
            logger.info(f"  Saved compiled analyzer to {cached}")
            
            # Create analyzer (inverted for lookup)
            self._analyzer = hfst.HfstTransducer(fst)
            self._analyzer.invert()
            self._analyzer.minimize()
            
            self._available = True
        except Exception as e:
            logger.warning(f"HFST compilation failed: {e}")
            self._available = False
        
        return self._available
    
    def _parse_analysis(self, raw: str) -> Dict:
        """Parse an apertium-style analysis string into structured form."""
        # Clean epsilon symbols
        raw_clean = re.sub(r'@_EPSILON_SYMBOL_@', '', raw)
        
        # Split on '+' to handle copula/auxiliary attachments
        # e.g. "бала<n><pl><nom>+е<cop><aor><p3><sg>"
        parts = raw_clean.split('+')
        
        # Parse main analysis (first part)
        main = parts[0]
        tags = re.findall(r'<([^>]+)>', main)
        lemma_match = re.match(r'^([^<]+)', main)
        lemma = lemma_match.group(1) if lemma_match else ''
        
        # Determine POS
        pos = 'OTHER'
        for tag in tags:
            if tag in self.POS_MAP:
                pos = self.POS_MAP[tag]
                break
        
        # Count inflectional vs derivational tags
        infl_count = sum(1 for t in tags if t in self.INFL_TAGS)
        deriv_count = sum(1 for t in tags if t in self.DERIV_TAGS)
        
        # Add tags from auxiliary parts
        all_tags = list(tags)
        for part in parts[1:]:
            aux_tags = re.findall(r'<([^>]+)>', part)
            all_tags.extend(aux_tags)
            infl_count += sum(1 for t in aux_tags if t in self.INFL_TAGS)
        
        return {
            'lemma': lemma,
            'pos': pos,
            'tags': all_tags,
            'infl_count': infl_count,
            'deriv_count': deriv_count,
            'raw': raw_clean,
        }
    
    def analyze(self, word: str) -> Dict:
        """Full morphological analysis."""
        if not self.is_available():
            return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': '',
                    'infl_count': 0, 'deriv_count': 0}
        
        try:
            results = self._analyzer.lookup(word.lower())
            if not results:
                return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': '',
                        'infl_count': 0, 'deriv_count': 0}
            
            # Parse all analyses and pick the best
            parsed = []
            for raw_str, weight in results:
                p = self._parse_analysis(raw_str)
                if p['lemma']:
                    parsed.append(p)
            
            if not parsed:
                return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': '',
                        'infl_count': 0, 'deriv_count': 0}
            
            # Prefer: shortest analysis (fewest tags = most basic reading)
            # with non-copula interpretation
            best = min(parsed, key=lambda x: len(x['tags']))
            return best
            
        except Exception as e:
            logger.debug(f"HFST analysis failed for '{word}': {e}")
            return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': '',
                    'infl_count': 0, 'deriv_count': 0}
    
    def lemmatize(self, word: str) -> str:
        """Get just the lemma."""
        return self.analyze(word)['lemma']


# =============================================================================
# KazNLP Lemmatizer (Neural/Statistical)
# =============================================================================

class KazNLPLemmatizer(BaseLemmatizer):
    """
    KazNLP-based lemmatizer for Kazakh.
    Uses the KazNLP morphological analyzer (AnalyzerDD).
    
    Install:
        git clone https://github.com/makazhan/kaznlp.git /mnt/data/kaznlp
        # Add to PYTHONPATH or copy to working directory
    
    The analyzer returns morphological analyses with tagged structure:
        алма_R_ZE сы_S3 н_C4  -> lemma: алма, pos: ZE (noun)
    """
    
    name = "kaznlp"
    
    # POS tag mapping (KLC tagset)
    POS_MAP = {
        'ZE': 'NOUN',      # Зат есім (Noun)
        'ZEQ': 'NOUN',     # Proper noun
        'SE': 'ADJ',       # Сын есім (Adjective)
        'ET': 'VERB',      # Етістік (Verb)
        'ETK': 'VERB',     # Transitive verb
        'US': 'ADV',       # Үстеу (Adverb)
        'SN': 'NUM',       # Сан есім (Numeral)
        'ES': 'PRON',      # Есімдік (Pronoun)
    }
    
    def __init__(self, kaznlp_path: str = None):
        self._analyzer = None
        self._available = None
        self.kaznlp_path = kaznlp_path
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        import sys
        import os
        
        # Find the kaznlp clone directory
        project_dir = Path(__file__).parent.parent  # cefr-classification-kk
        possible_kaznlp_roots = [
            project_dir / 'kaznlp',                    # Local clone in project
            Path('/mnt/data/kaznlp'),                  # VM location
            Path(os.getcwd()) / 'kaznlp',              # Current directory
        ]
        
        # Add kaznlp to path if found
        kaznlp_root = None
        for root in possible_kaznlp_roots:
            if root.exists() and (root / 'kaznlp' / 'morphology' / 'analyzers.py').exists():
                kaznlp_root = root
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                    logger.info(f"Added {root} to sys.path")
                break
        
        if kaznlp_root is None:
            logger.warning("KazNLP repo not found. Clone from: https://github.com/makazhan/kaznlp")
            self._available = False
            return self._available
        
        try:
            # Now try importing
            from kaznlp.morphology.analyzers import AnalyzerDD
            
            self._analyzer = AnalyzerDD()
            
            # Find model directory
            model_path = kaznlp_root / 'kaznlp' / 'morphology' / 'mdl'
            if model_path.exists():
                self._analyzer.load_model(str(model_path))
                logger.info(f"KazNLP model loaded from {model_path}")
                self._available = True
            else:
                logger.warning(f"KazNLP model not found at {model_path}")
                self._available = False
                
        except ImportError as e:
            logger.warning(f"KazNLP import failed: {e}")
            self._available = False
        except Exception as e:
            logger.warning(f"KazNLP not available: {e}")
            self._available = False
        
        return self._available
    
    def analyze(self, word: str) -> Dict:
        """Full morphological analysis."""
        if not self.is_available():
            return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': ''}
        
        try:
            # Analyze word
            is_covered, analyses = self._analyzer.analyze(word.lower())
            
            if not analyses:
                return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': ''}
            
            # If not covered (OOV), fall back to rule-based
            if not is_covered:
                # OOV word - the analysis will just be "word_R_X"
                # Use rule-based stripping as fallback
                fallback = RuleBasedLemmatizer()
                return fallback.analyze(word)
            
            # Parse analyses and rank them
            # For CEFR lexical items, prefer NOUN/ADJ analyses over VERB
            # (most graded vocabulary items are nouns/adjectives)
            parsed = []
            for analysis in analyses:
                parts = analysis.split()
                if not parts:
                    continue
                    
                first_part = parts[0]
                
                # Extract lemma and POS (format: lemma_R_POS)
                if '_R_' in first_part:
                    lemma = first_part.split('_R_')[0]
                    pos_tag = first_part.split('_R_')[1] if len(first_part.split('_R_')) > 1 else 'X'
                else:
                    lemma = first_part.split('_')[0]
                    pos_tag = 'X'
                
                pos = self.POS_MAP.get(pos_tag, 'OTHER')
                
                # Calculate preference score (higher = preferred)
                # Prefer: NOUN > ADJ > others > VERB (verbs often wrong for lexical items)
                score = 0
                if pos == 'NOUN':
                    score = 3
                elif pos == 'ADJ':
                    score = 2
                elif pos == 'VERB':
                    score = 0  # Lower priority for verbs
                else:
                    score = 1
                
                # Prefer longer lemmas (more specific)
                score += len(lemma) * 0.1
                
                parsed.append({
                    'lemma': lemma,
                    'pos': pos,
                    'pos_tag': pos_tag,
                    'tags': parts[1:],
                    'raw': analysis,
                    'is_covered': is_covered,
                    'score': score
                })
            
            # Sort by score (descending) and return best
            if parsed:
                best = max(parsed, key=lambda x: x['score'])
                del best['score']
                del best['pos_tag']
                return best
                
        except Exception as e:
            logger.debug(f"KazNLP analysis failed for '{word}': {e}")
        
        return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': ''}
    
    def lemmatize(self, word: str) -> str:
        """Get just the lemma."""
        return self.analyze(word)['lemma']


# =============================================================================
# Rule-based Suffix Stripper (Fallback)
# =============================================================================

class RuleBasedLemmatizer(BaseLemmatizer):
    """
    Simple rule-based suffix stripper for Kazakh.
    Uses known suffix patterns to estimate lemmas.
    This is a fallback when Apertium/KazNLP are not available.
    """
    
    name = "rule_based"
    
    # Common Kazakh suffixes (ordered by length, longest first)
    SUFFIXES = [
        # Plural + case combinations
        'тарымызда', 'теріміздe', 'дарымызда', 'деріміздe',
        'ларымыздан', 'леріміздeн',
        # Possessive + case
        'ымызда', 'іміздe', 'ыңызда', 'іңізде',
        'ымыздан', 'іміздeн',
        # Case suffixes
        'дан', 'ден', 'тан', 'тен', 'нан', 'нен',  # ablative
        'да', 'де', 'та', 'те',  # locative
        'ға', 'ге', 'қа', 'ке', 'на', 'не',  # dative
        'ды', 'ді', 'ты', 'ті', 'ны', 'ні',  # accusative
        'дың', 'дің', 'тың', 'тің', 'ның', 'нің',  # genitive
        # Plural
        'тар', 'тер', 'дар', 'дер', 'лар', 'лер',
        # Possessive
        'ым', 'ім', 'ың', 'ің', 'ы', 'і', 'сы', 'сі',
        'мыз', 'міз', 'ңыз', 'ңіз',
        # Verbal
        'ған', 'ген', 'қан', 'кен',
        'атын', 'етін', 'йтын', 'йтін',
        'у', 'ю',
    ]
    
    def __init__(self, min_stem_len: int = 2):
        self.min_stem_len = min_stem_len
        # Sort suffixes by length (longest first)
        self._suffixes = sorted(self.SUFFIXES, key=len, reverse=True)
    
    def is_available(self) -> bool:
        return True  # Always available
    
    def analyze(self, word: str) -> Dict:
        """Analyze by stripping suffixes."""
        word_lower = word.lower()
        stripped = []
        current = word_lower
        
        # Iteratively strip suffixes
        changed = True
        while changed and len(current) > self.min_stem_len:
            changed = False
            for suffix in self._suffixes:
                if current.endswith(suffix) and len(current) - len(suffix) >= self.min_stem_len:
                    stripped.append(suffix)
                    current = current[:-len(suffix)]
                    changed = True
                    break
        
        return {
            'lemma': current,
            'pos': 'UNK',  # Can't determine POS with rules alone
            'tags': stripped,
            'raw': f"{word} -> {current} (stripped: {'+'.join(stripped) if stripped else 'none'})"
        }
    
    def lemmatize(self, word: str) -> str:
        """Get the estimated lemma."""
        return self.analyze(word)['lemma']


# =============================================================================
# Stanza/Trankit Lemmatizer (Universal Dependencies)
# =============================================================================

class StanzaLemmatizer(BaseLemmatizer):
    """
    Stanza-based lemmatizer using Universal Dependencies models.
    
    Install:
        pip install stanza
        # Then in Python:
        import stanza
        stanza.download('kk')  # Kazakh
    """
    
    name = "stanza"
    
    def __init__(self):
        self._nlp = None
        self._available = None
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        try:
            import stanza
            # Try to load Kazakh model
            self._nlp = stanza.Pipeline('kk', processors='tokenize,lemma', use_gpu=False, verbose=False)
            self._available = True
            logger.info("Stanza Kazakh pipeline loaded")
        except Exception as e:
            logger.warning(f"Stanza not available for Kazakh: {e}")
            self._available = False
        
        return self._available
    
    def analyze(self, word: str) -> Dict:
        """Full analysis using Stanza."""
        if not self.is_available():
            return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': ''}
        
        try:
            doc = self._nlp(word)
            if doc.sentences and doc.sentences[0].words:
                w = doc.sentences[0].words[0]
                return {
                    'lemma': w.lemma or word,
                    'pos': w.upos or 'UNK',
                    'tags': [w.feats] if w.feats else [],
                    'raw': str(w)
                }
        except Exception as e:
            logger.debug(f"Stanza analysis failed for '{word}': {e}")
        
        return {'lemma': word, 'pos': 'UNK', 'tags': [], 'raw': ''}
    
    def lemmatize(self, word: str) -> str:
        """Get just the lemma."""
        return self.analyze(word)['lemma']


# =============================================================================
# Lemmatizer Factory
# =============================================================================

LEMMATIZERS = {
    'apertium': ApertiumLemmatizer,
    'hfst': HFSTLemmatizer,
    'kaznlp': KazNLPLemmatizer,
    'stanza': StanzaLemmatizer,
    'rule_based': RuleBasedLemmatizer,
}


def get_lemmatizer(name: str = 'auto') -> BaseLemmatizer:
    """
    Get a lemmatizer by name.
    
    Args:
        name: One of 'apertium', 'kaznlp', 'stanza', 'rule_based', or 'auto'.
              'auto' tries them in order of preference.
    
    Returns:
        Configured lemmatizer instance.
    """
    if name == 'auto':
        # Try in order of preference
        preference_order = ['hfst', 'apertium', 'stanza', 'kaznlp', 'rule_based']
        for lemma_name in preference_order:
            lemmatizer = LEMMATIZERS[lemma_name]()
            if lemmatizer.is_available():
                logger.info(f"Auto-selected lemmatizer: {lemma_name}")
                return lemmatizer
        # Fallback to rule-based (always available)
        return RuleBasedLemmatizer()
    
    if name not in LEMMATIZERS:
        raise ValueError(f"Unknown lemmatizer: {name}. Available: {list(LEMMATIZERS.keys())}")
    
    return LEMMATIZERS[name]()


def get_available_lemmatizers() -> List[str]:
    """Get list of available lemmatizers on this system."""
    available = []
    for name, cls in LEMMATIZERS.items():
        try:
            lemmatizer = cls()
            if lemmatizer.is_available():
                available.append(name)
        except:
            pass
    return available


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def lemmatize_dataframe(
    df,
    lemmatizer: BaseLemmatizer,
    word_col: str = 'lemma',
    output_col: str = 'lemma_clean'
) -> None:
    """
    Add lemmatized column to a DataFrame.
    
    Args:
        df: DataFrame to modify (in place)
        lemmatizer: Lemmatizer instance
        word_col: Column containing words to lemmatize
        output_col: Name of output column
    """
    from tqdm import tqdm
    
    words = df[word_col].tolist()
    lemmas = []
    
    for word in tqdm(words, desc=f"Lemmatizing with {lemmatizer.name}"):
        lemmas.append(lemmatizer.lemmatize(str(word)))
    
    df[output_col] = lemmas
    logger.info(f"Added '{output_col}' column using {lemmatizer.name}")


def compare_lemmatizers(words: List[str]) -> Dict[str, List[str]]:
    """
    Compare output of all available lemmatizers on a word list.
    
    Args:
        words: List of words to lemmatize
    
    Returns:
        Dict mapping lemmatizer name to list of lemmas
    """
    results = {}
    
    for name, cls in LEMMATIZERS.items():
        try:
            lemmatizer = cls()
            if lemmatizer.is_available():
                results[name] = lemmatizer.lemmatize_batch(words)
        except Exception as e:
            logger.warning(f"Failed to compare {name}: {e}")
    
    return results
