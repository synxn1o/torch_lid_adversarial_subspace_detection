# LEGACY CODE — DO NOT USE

**Status**: DISABLED — Original TensorFlow implementation, replaced by PyTorch `core/` module.

## ⚠️ THIS DIRECTORY IS FOR REFERENCE ONLY

This code is the original TensorFlow/Keras implementation from the ICLR 2018 paper. It has been superseded by the PyTorch reimplementation in `core/`, `experiments/`, and `visualizer/`.

## DO NOT

- **DO NOT** import from `.old/` in any active code
- **DO NOT** edit files in this directory
- **DO NOT** use these scripts as examples for new code
- **DO NOT** run `train_model.py`, `craft_adv_examples.py`, or other scripts

## WHY IT EXISTS

- Historical reference for original paper implementation
- Contains TensorFlow-specific patterns no longer applicable
- Test files (`test_*.py`) use custom runners, not pytest

## IF YOU NEED TO REFERENCE

- `attacks.py` / `cw_attacks.py` — Original attack implementations
- `util.py` — Original utility functions (TensorFlow-based)
- `toy_example_old/` — Legacy 2D experiment code

## MIGRATION

All functionality has been migrated to:
- Attacks → `core/attacks.py`
- Detectors → `core/detectors.py`
- Models → `core/models.py`
- Utilities → `core/utils.py`
- Experiments → `experiments/`
