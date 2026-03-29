# Core Module

Central library for adversarial example generation and detection.

## STRUCTURE

```
core/
├── __init__.py       # Package exports (14 symbols)
├── attacks.py        # FGSM, BIM, JSMA, CarliniL2
├── data_loaders.py   # MNIST/CIFAR/SVHN/Toy loaders
├── detectors.py      # LID, KD, KM, TDA detectors
├── models.py         # CNN architectures + ModelWrapper
└── utils.py          # MLE, distance metrics, noise generation
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add attack | `attacks.py` | Inherit `BaseAttack`, implement `generate(x, y)` |
| Add detector | `detectors.py` | Inherit `BaseDetector`, implement `fit()` + `detect()` |
| Add model | `models.py` | Add class, update `get_model()` factory |
| Add dataset loader | `data_loaders.py` | Add case in `get_dataloader()` |
| Math utilities | `utils.py` | `mle_batch()`, `kmean_batch()`, `get_noisy_samples()` |

## CONVENTIONS

- All attacks: `BaseAttack(model_wrapper)` → `generate(x, y)` → tensor output
- All detectors: `BaseDetector(model_wrapper)` → `fit(train_loader)` (optional) → `detect(x)` → scores
- Models wrapped via `ModelWrapper` with hooks on Conv2d/Linear/BatchNorm/MaxPool2d
- Conv layers use Global Average Pooling for feature extraction

## ANTI-PATTERNS

- **NEVER** assume `y_target` is pre-set in attacks
- **ALWAYS** check `ripser` import before using `TDADetector`
- **DO NOT** modify base classes without updating all subclasses
