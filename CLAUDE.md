# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PyTorch implementation of "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality" (Ma et al., ICLR 2018). Detects adversarial examples by characterizing the Local Intrinsic Dimensionality (LID) of adversarial subspaces.

Paper: https://arxiv.org/abs/1801.02613

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Type checking (targets conda env "ML" via pyrightconfig.json)
pyright

# Run experiments (use conda env ML — has ripser/persim)
conda run -n ML python experiments/run_dataset.py -d mnist -a fgsm
conda run -n ML python experiments/run_tda.py -d mnist -n 500
conda run -n ML python experiments/run_toy.py
conda run -n ML python experiments/train_model.py --dataset mnist

# Tests
python -m pytest experiments/apr11/test_integration.py -v
python -m pytest experiments/apr11/test_integration.py::test_tda_has_avg_death -v  # single test

# Visualization
python -m visualizer.main --mode all --dataset mnist
python -m visualizer.main --mode tda --dataset mnist --attack fgsm

# Re-plot apr11 experiment from existing data (adversarial + characteristics + TDA)
conda run -n ML python experiments/apr11/replot_all.py         # all datasets
conda run -n ML python experiments/apr11/replot_all.py -d mnist # single dataset
```

## Architecture

4-stage pipeline: **Train → Attack → Detect → Visualize**

### `core/` — Library modules

- **attacks.py** — `BaseAttack` → `FGSM`, `BIM` (modes: first/last), `JSMA`, `CarliniL2` (supports CW-LID variant). All implement `generate(x, y)` returning tensors.
- **detectors.py** — `BaseDetector` → `LIDDetector`, `KDDetector`, `KMDetector`, `TDADetector`, `PersistenceImageDetector`. All implement `fit(train_loader)` (optional) and `detect(x)` returning scores.
- **models.py** — CNN architectures (`MNISTModel`, `CIFARModel`, `SVHNModel`, `ToyModel`) + `ModelWrapper` that attaches forward hooks to Conv2d/Linear/BatchNorm/MaxPool2d layers for feature extraction. Conv layers use Global Average Pooling. `get_model()` factory selects architecture by dataset name.
- **data_loaders.py** — `get_dataloader(dataset)` for MNIST/CIFAR/SVHN/Toy, `loader_to_numpy()`.
- **utils.py** — `mle_batch()` (MLE for LID), `kmean_batch()`, `get_noisy_samples()`, `lid_adv_term()`. Contains `STDEVS` dict (per-dataset, per-attack noise std).
- **tda_utils.py** — `extract_layer_activations()`, `compute_correlation_distance()`, `run_tda()`, `bottleneck_distance()`, `persistence_diagrams_to_images()`. Pure numpy, no torch dependency.
- **config.py** — `DATA_DIR`, `RESULTS_DIR`, `EXPERIMENTS_DIR`, `FILE_PATTERNS` dict, `get_model_path()`, `get_results_dir()`.

### `experiments/` — Runner scripts

- **run_dataset.py** — Unified pipeline for any dataset (MNIST/CIFAR/SVHN/FashionMNIST/Toy). Contains `ATTACK_EPS` and `CLIP_RANGES` dicts.
- **run_mnist.py** — MNIST-specific (single attack).
- **run_toy.py** — 2D circle dataset (all attacks).
- **run_tda.py** — TDA topological analysis.
- **train_model.py** — Training with Adam + cosine annealing scheduler.
- Dated subdirectories (apr11/, apr21/) contain standalone experiment scripts.

### `visualizer/` — Plotting toolkit (seaborn-based)

- All plotting uses seaborn functions (`sns.histplot`, `sns.barplot`, `sns.lineplot`, `sns.scatterplot`, `sns.heatmap`) — not raw `ax.bar()`, `ax.hist()`, etc.
- **config.py** — `PALETTES` dict: `{"categorical": "colorblind", "sequential": "crest", "diverging": "BrBG"}`. Access via `self._palette(kind)` in visualizer classes.
- **visualizers.py** — `BaseVisualizer`, `AdversarialVisualizer`, `ModelVisualizer`, `DetectionVisualizer`. Every public method accepts `title: Optional[str] = None`.
- **tda_visualizers.py** — `TDAVisualizer` (active, exported). All 6 methods accept `title: Optional[str] = None`.
- Shadowed `TDAVisualizer` in `visualizers.py` is dead code (not exported) — do not refactor.
- Style presets: `presentation` (16x12), `paper` (8x6), `web` (12x8). Default DPI: 300.

## Conventions

- Data formats: models as `.pth`, adversarial/characteristic data as `.npy`, TDA results as `.json`.
- Pre-generated adversarial `.npy` files live in `experiments/adversarial_data/<dataset>/adv_<attack>.npy`.
- Experiment scripts use `sys.path.append(...)` at top to import core. Some also monkey-patch `visualizer.config` paths (`ANALYSIS_DIR`, `ADV_DIR`, `OUTPUT_DIR`) before importing visualizer modules.
- Dated experiment output goes in `experiments/mmdd/`.
- Color palettes: `colorblind` (categorical), `crest` (sequential), `BrBG` (diverging).

## Anti-patterns

- **Do not** regenerate adversarial examples or retrain models — reuse existing `.npy` files and saved `.pth` models.
- **Do not** modify `core/` from experiment scripts — core is a shared library.
- **Do not** accumulate GPU tensors in lists during extraction — pre-allocate numpy arrays, write from forward hooks.
- **Do not** add bare `except Exception` handlers (34+ already exist in visualizer).
- **Do not** use raw `ax.bar()` / `ax.hist()` / `ax.plot()` in visualizer — use seaborn equivalents.
- **Do not** hardcode colors in visualizer — use `self._palette()` and `sns.color_palette()`.
- **Always** use `torch.no_grad()`, `gc.collect()`, `torch.cuda.empty_cache()` for memory management in extraction loops.
- **Always** check `ripser` import before using `TDADetector`.
- **Always** call `setup_environment()` before visualization.
- **Always** `del out, batch` after each `model(batch)` call in loops; `del model` + `gc.collect()` + `empty_cache()` after extraction completes.
