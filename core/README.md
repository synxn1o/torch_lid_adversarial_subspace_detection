# Core Module

Central library for adversarial example generation and detection.

## Modules

### `attacks.py` — Adversarial Attacks

All attacks inherit from `BaseAttack(model_wrapper)` and implement a `generate(x, y)` method.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `FGSM` | `eps`, `clip_min`, `clip_max` | Single-step gradient sign attack |
| `BIM` | `eps`, `eps_iter`, `nb_iter`, `mode` | Iterative gradient attack. Mode `first` saves the first successful adversarial; `last` saves the final iteration. |
| `JSMA` | `theta`, `gamma`, `clip_min`, `clip_max` | Saliency map attack. Very slow (processes sample-by-sample). `gamma` controls perturbation budget as fraction of features. |
| `CarliniL2` | `confidence`, `max_iterations`, `use_lid` | Optimization-based attack. Set `use_lid=True` for CW-LID variant. |

```python
from core.models import get_model
from core.attacks import FGSM, BIM

model = get_model('mnist', 'data/model_mnist.pth', device='cuda')
attack = FGSM(model, eps=0.3)
x_adv = attack.generate(x, y)
```

### `data_loaders.py` — Data Loading

| Function | Description |
|----------|-------------|
| `get_dataloader(dataset_name, batch_size, train, download, augmentation)` | Returns a PyTorch DataLoader. Supported: `mnist`, `cifar`, `svhn`, `toy`. |
| `loader_to_numpy(loader)` | Converts a DataLoader to `(X, y)` numpy arrays. |

```python
from core.data_loaders import get_dataloader, loader_to_numpy

train_loader = get_dataloader('mnist', batch_size=128, train=True)
X_train, y_train = loader_to_numpy(train_loader)
```

### `detectors.py` — Adversarial Detectors

All detectors inherit from `BaseDetector(model_wrapper)` with `fit(train_loader)` and `detect(x)` methods.

| Class | `fit()` | `detect(x)` | Output Shape |
|-------|---------|-------------|-------------|
| `LIDDetector(k=20)` | Not required | `x` or `(x, x_clean_ref)` | `(n_samples, n_layers)` |
| `KDDetector(bandwidth=None)` | Required | Penultimate layer density | `(n_samples, 1)` |
| `KMDetector(k=20)` | Required | Mean k-NN distance | `(n_samples, 1)` |
| `TDADetector(maxdim=1)` | Not required | Returns dict with `features`, `diagrams`, `correlation_matrix` | dict |

Default bandwidths: `mnist: 3.7926`, `cifar: 0.26`, `svhn: 1.00`, `toy: 0.2`.

```python
from core.detectors import LIDDetector, KDDetector

lid = LIDDetector(model, k=20)
lid_scores = lid.detect(x_samples)

kd = KDDetector(model)
kd.fit(train_loader)
kd_scores = kd.detect(x_samples)
```

### `models.py` — Model Definitions & Wrapper

**Architectures**: `MNISTModel`, `CIFARModel`, `SVHNModel`, `ToyModel` (2D binary classification).

**`ModelWrapper`**: Wraps any model and provides:
- `get_logits(x)` — raw model output
- `predict(x)` — class labels (binary: sigmoid threshold; multi-class: argmax)
- `get_features(x)` — list of layer activations (input + all hooks)
- `__call__(x)` — equivalent to `get_logits(x)`

Hooks are auto-registered on `Conv2d`, `Linear`, `BatchNorm`, and `MaxPool2d` layers. Conv layers use Global Average Pooling to keep dimensionality manageable.

**`get_model(dataset_name, model_path, device)`**: Factory function returning a `ModelWrapper`.

```python
from core.models import get_model

model = get_model('mnist', 'data/model_mnist.pth', device='cuda')
preds = model.predict(x)
features = model.get_features(x)
```

### `utils.py` — Utilities

| Function | Description |
|----------|-------------|
| `mle_batch(data, batch, k)` | LID estimation using MLE on k-NN distances. Scipy-based. |
| `kmean_batch(data, batch, k)` | Mean distance to k nearest neighbors. |
| `get_noisy_samples(X, dataset_name, attack_name, std)` | Gaussian noise injection with per-dataset/attack standard deviations. |
| `lid_adv_term(clean_logits, adv_logits, k)` | PyTorch LID loss for CW-LID attack. |

## Usage

```python
from core.data_loaders import get_dataloader
from core.models import get_model
from core.attacks import FGSM
from core.detectors import LIDDetector

# Load
loader = get_dataloader('mnist', batch_size=100, train=False)
model = get_model('mnist', 'data/model_mnist.pth', device='cuda')

# Attack
attack = FGSM(model, eps=0.3)
x, y = next(iter(loader))
x_adv = attack.generate(x.to('cuda'), y.to('cuda'))

# Detect
lid = LIDDetector(model, k=20)
scores = lid.detect(x_adv)
```
