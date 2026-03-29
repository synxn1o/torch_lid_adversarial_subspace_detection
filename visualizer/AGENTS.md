# Visualizer Module

Comprehensive visualization toolkit for adversarial ML research.

## STRUCTURE

```
visualizer/
├── __init__.py         # Package exports + version info
├── config.py           # Dataset configs, attack params, viz settings
├── data_loaders.py     # Load original/adv/characteristic data
├── main.py             # CLI entry point (`python -m visualizer.main`)
├── utils.py            # Argument parsing, environment setup
├── visualizers.py      # Core classes (1067 lines — LARGEST FILE)
└── outputs/            # Generated plots (adversarial/, model/, detection/, tda/)
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add visualization | `visualizers.py` | Inherit `BaseVisualizer` or extend existing classes |
| Add data source | `data_loaders.py` | Add loader function, update `check_required_files()` |
| CLI args | `utils.py` | Modify `parse_arguments()` |
| Config changes | `config.py` | Update `ATTACKS`, `CHARACTERISTICS`, `MNIST_CONFIG` |

## VISUALIZATION MODES

| Mode | Class | Output |
|------|-------|--------|
| `adversarial` | `AdversarialVisualizer` | Image grids, perturbation analysis |
| `model` | `ModelVisualizer` | Training curves, confusion matrix, ROC |
| `detection` | `DetectionVisualizer` | ROC comparison, 3D features, distributions |
| `tda` | via `DetectionVisualizer` | Persistence diagrams, topological features |

## CONVENTIONS

- Style presets: `presentation` (16x12), `paper` (8x6), `web` (12x8)
- DPI default: 300 (configurable via `--dpi`)
- Sample limit: 1000 (prevents memory issues)
- Bare `except Exception` used extensively — **do not add more**

## ANTI-PATTERNS

- **DO NOT** add bare `except Exception` handlers (34+ already exist)
- **NEVER** load full dataset without sample limiting
- **ALWAYS** call `setup_environment()` before visualization
