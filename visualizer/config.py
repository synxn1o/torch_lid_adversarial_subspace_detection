"""
Configuration settings for the visualization utility
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from core.config import DATA_DIR

# Base directories
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "visualizer" / "outputs"
ADV_DIR = BASE_DIR / "experiments" / "adversarial_data"
ANALYSIS_DIR = BASE_DIR / "experiments" / "analysis_data"

# MNIST-specific configuration
MNIST_CONFIG = {
    "dataset": "mnist",
    "num_classes": 10,
    "image_shape": (28, 28),
    "channels": 1,
    "attack_params": {
        "eps": 0.30,
        "eps_iter": 0.010
    }
}

# Toy-specific configuration
TOY_CONFIG = {
    "dataset": "toy",
    "num_classes": 2,
    "image_shape": None,
    "channels": None,
    "attack_params": {
        "eps": 0.2,
        "eps_iter": 0.02
    }
}

# CIFAR-specific configuration
CIFAR_CONFIG = {
    "dataset": "cifar",
    "num_classes": 10,
    "image_shape": (32, 32),
    "channels": 3,
    "attack_params": {
        "eps": 0.03,
        "eps_iter": 0.005
    }
}

# Available attacks
ATTACKS = ["fgsm", "bim-a", "bim-b", "jsma", "cw-l2", "cw-lid"]

# Available characteristics
CHARACTERISTICS = ["lid", "kd", "km", "tda"]

# Visualization settings
VISUALIZATION_CONFIG = {
    "figure_dpi": 300,
    "font_size": 12,
    "color_palette": "colorblind",
    "image_grid_rows": 4,
    "image_grid_cols": 4,
    "sample_limit": 1000,  # Maximum samples to load for analysis
    "batch_size": 100,     # Processing batch size
}

# Color palette presets by plot type
PALETTES = {
    "categorical": "colorblind",
    "sequential": "crest",
    "diverging": "BrBG",
}

# Plot style presets
PLOT_STYLES = {
    "presentation": {
        "figsize": (16, 12),
        "font_size": 16,
        "line_width": 3,
        "marker_size": 12
    },
    "paper": {
        "figsize": (8, 6),
        "font_size": 10,
        "line_width": 1.5,
        "marker_size": 6
    },
    "web": {
        "figsize": (12, 8),
        "font_size": 14,
        "line_width": 2,
        "marker_size": 8
    }
}

# File patterns
FILE_PATTERNS = {
    "adversarial": "adv_{attack}.npy",
    "characteristics": "{char}_{attack}.npy",
    "model": "model_{dataset}.pth",
    "training_log": "training_{dataset}.log"
}

# Output formats
OUTPUT_FORMATS = ["png", "pdf", "svg", "html"]

# Cache configuration
CACHE_CONFIG = {
    "enabled": True,
    "dir": OUTPUT_DIR / "cache",
    "max_size": "10GB"
}

# Performance settings
PERFORMANCE = {
    "max_workers": 4,
    "memory_limit": "8GB",
    "use_gpu": True,
    "chunk_size": 500
}

def get_data_file_path(pattern: str, **kwargs) -> str:
    """Get full path for data file based on pattern"""
    filename = pattern.format(**kwargs)
    dataset = kwargs.get('dataset', 'mnist')
    
    if "model" in pattern:
        return str(DATA_DIR / filename)
    elif "adv_" in pattern:
        return str(ADV_DIR / dataset / filename)
    else:
        return str(ANALYSIS_DIR / dataset / filename)

def get_output_path(category: str, filename: str, create_dir: bool = True, base_dir: Optional[Path] = None) -> str:
    """Get output path for visualization file"""
    root = base_dir if base_dir is not None else OUTPUT_DIR
    output_subdir = root / category
    if create_dir:
        output_subdir.mkdir(parents=True, exist_ok=True)
    return str(output_subdir / filename)

def validate_dataset(dataset: str) -> bool:
    """Validate dataset name"""
    return dataset.lower() in ["mnist", "cifar", "svhn", "fashion_mnist", "toy"]

def validate_attack(attack: str) -> bool:
    """Validate attack name"""
    return attack.lower() in ATTACKS or attack.lower() == "all"

def validate_characteristic(char: str) -> bool:
    """Validate characteristic name"""
    return char.lower() in CHARACTERISTICS or char.lower() == "all"

def get_dataset_config(dataset: str) -> Dict:
    """Get configuration for specific dataset"""
    if dataset == "mnist":
        return MNIST_CONFIG
    elif dataset == "toy":
        return TOY_CONFIG
    elif dataset == "cifar":
        return CIFAR_CONFIG
    else:
        return {}

# Default configuration for MNIST-only operation
DEFAULT_CONFIG = {
    "dataset": "mnist",
    "attacks": ATTACKS,  # All available attacks
    "characteristics": CHARACTERISTICS,  # All characteristics
    "output_dir": str(OUTPUT_DIR),
    "format": "png",
    "dpi": VISUALIZATION_CONFIG["figure_dpi"],
    "interactive": False,
    "cache": True,
    "sample_limit": VISUALIZATION_CONFIG["sample_limit"]
}