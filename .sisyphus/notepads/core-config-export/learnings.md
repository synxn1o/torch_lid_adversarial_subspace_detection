# Learnings - Core Config Export

## Task Summary
Updated `core/__init__.py` to export config symbols from `core.config`.

## Changes Made
1. Added import: `from .config import DATA_DIR, get_model_path, get_results_dir`
2. Updated `__all__` list to include the three config symbols

## Key Observations
- The `core/__init__.py` file follows a consistent pattern: imports grouped by module, then `__all__` list
- Config symbols are placed after attacks imports but before data_loaders imports
- The `__all__` list maintains alphabetical order within groups
- No module docstring changes were needed

## Verification
- Python syntax check passed
- LSP diagnostics clean
- Import test failed due to missing torch dependency (expected in this environment)

## Conventions
- Keep existing exports intact
- Maintain consistent ordering in `__all__`
- Follow existing import pattern (relative imports with dot notation)