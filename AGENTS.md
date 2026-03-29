# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-29
**Commit:** 60fd80e
**Branch:** main

## OVERVIEW

PyTorch implementation of ICLR 2018 paper "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality". Detects adversarial examples via LID, KD, KM, and TDA detectors.

**Stack**: PyTorch, NumPy, SciPy, scikit-learn, matplotlib/seaborn, ripser (optional TDA)

## STRUCTURE

```
.
├── core/                   # Library: attacks, detectors, models, data loaders
├── visualizer/             # Visualization toolkit (adversarial, model, detection, TDA)
├── experiments/            # End-to-end pipelines: run_mnist, run_toy, run_tda
├── .old/                   # LEGACY TensorFlow code - DO NOT EDIT
├── data/                   # Model weights (.pth), generated data (.npy)
├── results/                # Experiment outputs (mnist/, toy/, tda/)
├── plans/                  # Refactoring plans (reference only)
└── toy_example/            # 2D circle dataset data
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new attack | `core/attacks.py` | Inherit `BaseAttack`, implement `generate(x, y)` |
| Add new detector | `core/detectors.py` | Inherit `BaseDetector`, implement `fit()` + `detect()` |
| Add new model | `core/models.py` | Add class, register in `get_model()` factory |
| Run MNIST pipeline | `experiments/run_mnist.py` | `-a` for attack, `-b` for batch size |
| Run TDA analysis | `experiments/run_tda.py` | Requires `ripser` package |
| Visualize results | `visualizer/main.py` | `python -m visualizer.main --mode all` |
| Legacy code | `.old/` | TensorFlow original - reference only |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `FGSM`, `BIM`, `JSMA`, `CarliniL2` | Class | `core/attacks.py` | Adversarial attacks |
| `LIDDetector`, `KDDetector`, `KMDetector`, `TDADetector` | Class | `core/detectors.py` | Detection methods |
| `ModelWrapper`, `get_model` | Class/Func | `core/models.py` | Model abstraction |
| `get_dataloader`, `loader_to_numpy` | Func | `core/data_loaders.py` | Data I/O |
| `mle_batch`, `kmean_batch`, `get_noisy_samples` | Func | `core/utils.py` | Math utilities |
| `AdversarialVisualizer`, `ModelVisualizer`, `DetectionVisualizer` | Class | `visualizer/visualizers.py` | Visualization |

## CONVENTIONS

- **sys.path hacks**: Entry points use `sys.path.append` (no `setup.py` exists)
- **No requirements.txt**: Dependencies listed in README only
- **No test suite**: `.old/` has legacy tests, active codebase has none
- **Data format**: `.npy` files with last column = label (1=adversarial, 0=clean)
- **Model format**: `.pth` PyTorch state dicts

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** use `.old/` code (legacy TensorFlow, not maintained)
- **DO NOT** add bare `except Exception` handlers (34+ instances in visualizer)
- **NEVER** assume `y_target` is pre-set in attacks (validate first)
- **ALWAYS** check `ripser` availability before using `TDADetector`

## COMMANDS

```bash
# MNIST experiment
python experiments/run_mnist.py -a fgsm

# Toy experiment
python experiments/run_toy.py

# TDA analysis (requires ripser)
python experiments/run_tda.py -d mnist -n 500

# Visualization
python -m visualizer.main --mode all --dataset mnist
```

## NOTES

- Paper: https://arxiv.org/abs/1801.02613
- `core/` has proper `__init__.py` with `__all__` exports
- `visualizer/visualizers.py` is 1067 lines — largest file
- `plans/` contains refactoring documentation (reference only)
- No CI/CD, no Makefile, no Dockerfile configured
