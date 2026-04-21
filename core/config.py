"""Centralized path configuration for the adversarial detection pipeline."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TOY_DATA_DIR = DATA_DIR / "toy"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
RESULTS_DIR = BASE_DIR / "results"

FILE_PATTERNS = {
    "model":          "model_{dataset}.pth",
    "adversarial":    "adv_{attack}.npy",
    "noisy":          "Noisy_{dataset}_{attack}.npy",
    "characteristic": "{char}_{dataset}_{attack}.npy",
    "tda":            "tda_{dataset}_{attack}.json",
}


def get_model_path(dataset_name, model_dir=None):
    base = Path(model_dir) if model_dir else DATA_DIR
    return base / FILE_PATTERNS["model"].format(dataset=dataset_name)


def get_dataloader_root():
    return DATA_DIR


def get_toy_dataset_path():
    return TOY_DATA_DIR / "circle_dataset.pkl"


def get_results_dir(output_dir=None, mkdir=True):
    p = Path(output_dir) if output_dir else RESULTS_DIR
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p


def get_experiment_dir(experiment_name=None):
    if experiment_name:
        return EXPERIMENTS_DIR / experiment_name
    return EXPERIMENTS_DIR
