# Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality

PyTorch implementation of the ICLR 2018 paper by Ma et al. [^1]. Detects adversarial examples by characterizing the Local Intrinsic Dimensionality (LID) of adversarial subspaces.

Original paper: https://arxiv.org/abs/1801.02613

## Directory Structure

```
├── core/                   # Core library modules
│   ├── attacks.py          # Adversarial attack implementations
│   ├── data_loaders.py     # Unified data loading for all datasets
│   ├── detectors.py        # LID, KD, KM, TDA detectors
│   ├── models.py           # CNN architectures + ModelWrapper
│   └── utils.py            # LID calculation, noisy sample generation
├── experiments/            # Experiment runner scripts
│   ├── run_mnist.py        # MNIST detection pipeline
│   ├── run_toy.py          # Toy dataset detection pipeline
│   └── run_tda.py          # TDA topological analysis pipeline
├── visualizer/             # Visualization toolkit (see visualizer/README.md)
├── data/                   # Trained models (.pth) and generated data (.npy)
├── results/                # Experiment results (mnist/, toy/, tda/)
├── plans/                  # Refactoring plans
└── .old/                   # Legacy TensorFlow implementation
```

## Quick Start

### 1. Generate Toy Dataset
```bash
python toy_example/generate_dataset.py
```

### 2. Train Model
```bash
python experiments/run_mnist.py --attack fgsm
```

The experiment scripts handle the full pipeline: load data, load model, generate adversarial examples, extract characteristics, and save results.

## Usage

### MNIST Experiment
```bash
# Run with FGSM attack (default)
python experiments/run_mnist.py

# Run with specific attack
python experiments/run_mnist.py -a bim-a
python experiments/run_mnist.py -a cw-l2
python experiments/run_mnist.py -a cw-lid

# Attacks: fgsm, bim-a, bim-b, jsma, cw-l2, cw-lid
```

### Toy Dataset Experiment
```bash
python experiments/run_toy.py
```
Runs FGSM, BIM (first/last), and JSMA attacks on the 2D circle dataset.

### TDA Topological Analysis
```bash
python experiments/run_tda.py -d mnist -n 500
python experiments/run_tda.py -d toy
```

### Visualization
```bash
# All visualizations for MNIST
python -m visualizer.main --mode all --dataset mnist

# Specific visualization modes
python -m visualizer.main --mode adversarial --dataset mnist --attack fgsm
python -m visualizer.main --mode model --dataset mnist
python -m visualizer.main --mode detection --dataset mnist
python -m visualizer.main --mode tda --dataset mnist
```

See [visualizer/README.md](visualizer/README.md) for full visualization documentation.

## Core Library

### Attacks (`core/attacks.py`)
| Class | Description |
|-------|-------------|
| `FGSM` | Fast Gradient Sign Method |
| `BIM` | Basic Iterative Method (modes: `first`, `last`) |
| `JSMA` | Jacobian-based Saliency Map Attack |
| `CarliniL2` | Carlini & Wagner L2 (supports `use_lid=True` for CW-LID) |

### Detectors (`core/detectors.py`)
| Class | Description |
|-------|-------------|
| `LIDDetector` | Local Intrinsic Dimensionality using MLE estimation |
| `KDDetector` | Kernel Density estimation per class |
| `KMDetector` | K-Means distance to class centroids |
| `TDADetector` | Topological Data Analysis using persistent homology |

### Data & Models
| Module | Description |
|--------|-------------|
| `core/models.py` | CNN architectures (MNIST, CIFAR, SVHN, Toy) + `ModelWrapper` with feature extraction hooks |
| `core/data_loaders.py` | `get_dataloader()` for MNIST/CIFAR/SVHN/Toy, `loader_to_numpy()` |
| `core/utils.py` | `mle_batch()`, `kmean_batch()`, `get_noisy_samples()`, `lid_adv_term()` |

See [core/README.md](core/README.md) and [experiments/README.md](experiments/README.md) for full API documentation.

## Development

- **Framework**: PyTorch (`torch`, `torchvision`)
- **Data format**: Models as `.pth`, adversarial/characteristic data as `.npy`, TDA results as `.json`
- **Dependencies**: `torch`, `torchvision`, `numpy`, `scipy`, `sklearn`, `tqdm`, `matplotlib`
- **TDA dependency**: `ripser` (optional, for TDA detector)

---

[^1] Ma, X., Li, B., Wang, Y., Erfani, S. M., Wijewickrema, S., Schoenebeck, G., Song, D., Houle, M. E., & Bailey, J. (2018). Characterizing adversarial subspaces using local intrinsic dimensionality. ICLR 2018. https://arxiv.org/abs/1801.02613
