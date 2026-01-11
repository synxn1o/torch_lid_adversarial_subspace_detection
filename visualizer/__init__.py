"""
Adversarial ML Visualization Utility
Provides comprehensive visual analysis for adversarial machine learning research
"""

__version__ = "1.0.0"
__author__ = "Adversarial ML Team"

# Import key components for easy access
from .config import (
    ATTACKS, CHARACTERISTICS, MNIST_CONFIG, 
    get_data_file_path, get_output_path, validate_dataset
)
from .data_loaders import (
    load_original_data, load_adversarial_data, 
    load_characteristics_data, load_model_predictions,
    load_training_metrics, load_all_characteristics,
    check_required_files
)
from .visualizers import (
    AdversarialVisualizer, 
    ModelVisualizer, 
    DetectionVisualizer,
    BaseVisualizer
)
from .utils import parse_arguments, setup_environment, print_banner

# Main entry point
from .main import main

__all__ = [
    # Version info
    "__version__", "__author__",
    
    # Config
    "ATTACKS", "CHARACTERISTICS", "MNIST_CONFIG",
    "get_data_file_path", "get_output_path", "validate_dataset",
    
    # Data loaders
    "load_original_data", "load_adversarial_data",
    "load_characteristics_data", "load_model_predictions",
    "load_training_metrics", "load_all_characteristics",
    "check_required_files",
    
    # Visualizers
    "BaseVisualizer", "AdversarialVisualizer",
    "ModelVisualizer", "DetectionVisualizer",
    
    # Utils
    "parse_arguments", "setup_environment", "print_banner",
    
    # Main
    "main"
]