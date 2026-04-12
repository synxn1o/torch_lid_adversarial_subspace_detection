"""Core library for adversarial example generation and detection.

Modules:
    attacks — Adversarial attack implementations (FGSM, BIM, JSMA, CarliniL2)
    data_loaders — Unified data loading for MNIST, CIFAR, SVHN, Toy
    detectors — Adversarial detectors (LID, KD, KM, TDA)
    models — CNN architectures and ModelWrapper
    utils — LID estimation, noise generation, distance computation
"""

from .attacks import FGSM, BIM, JSMA, CarliniL2
from .config import DATA_DIR, get_model_path, get_results_dir
from .data_loaders import get_dataloader, loader_to_numpy
from .detectors import LIDDetector, KDDetector, KMDetector, TDADetector
from .models import ModelWrapper, get_model
from .utils import mle_batch, kmean_batch, get_noisy_samples, lid_adv_term

__all__ = [
    'FGSM', 'BIM', 'JSMA', 'CarliniL2',
    'DATA_DIR', 'get_model_path', 'get_results_dir',
    'get_dataloader', 'loader_to_numpy',
    'LIDDetector', 'KDDetector', 'KMDetector', 'TDADetector',
    'ModelWrapper', 'get_model',
    'mle_batch', 'kmean_batch', 'get_noisy_samples', 'lid_adv_term',
]
