"""
Utility functions for CEFR classification.
Config loading, logging setup, and reproducibility.
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config(config_name: str = "training") -> Dict[str, Any]:
    """Load a config file from the config/ directory."""
    config_path = PROJECT_ROOT / "config" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_yaml(config_path)


def load_paths() -> Dict[str, Any]:
    """Load the paths configuration."""
    return load_config("paths")


def load_suffix_inventory(lang: str = "kaz") -> Dict[str, Any]:
    """Load suffix inventory for a language."""
    suffix_path = PROJECT_ROOT / "config" / f"{lang}_suffixes.yaml"
    if not suffix_path.exists():
        logging.warning(f"Suffix inventory not found for lang={lang}, using empty")
        return {"derivational": [], "inflectional": {}, "vowels": {}}
    return load_yaml(suffix_path)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    name: str = "cefr"
) -> logging.Logger:
    """Setup logging with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if os.environ.get("CEFR_SKIP_TORCH_SEED", "0") == "1":
        return
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed


def get_device() -> str:
    """Get the best available device (cuda/mps/cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# CEFR label utilities
CEFR_LABELS = ["A1", "A2", "B1", "B2", "C1"]
CEFR_TO_INT = {label: i for i, label in enumerate(CEFR_LABELS)}
INT_TO_CEFR = {i: label for i, label in enumerate(CEFR_LABELS)}
CEFR_TO_ORDINAL = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5}


def normalize_cefr(label: str) -> str:
    """Normalize CEFR label to standard format."""
    label = str(label).strip().upper()
    if label in CEFR_LABELS:
        return label
    # Try common variations
    mapping = {
        "A-1": "A1", "A-2": "A2", "B-1": "B1", "B-2": "B2", "C-1": "C1",
        "1": "A1", "2": "A2", "3": "B1", "4": "B2", "5": "C1",
    }
    return mapping.get(label, label)
