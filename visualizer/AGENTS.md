# Visualizer Module

Comprehensive visualization toolkit for adversarial ML research.

## STRUCTURE

```
visualizer/
├── __init__.py         # Package exports + version info
├── config.py           # PALETTES, dataset configs, viz settings
├── data_loaders.py     # Load original/adv/characteristic data
├── main.py             # CLI entry point (`python -m visualizer.main`)
├── utils.py            # Argument parsing, environment setup
├── visualizers.py      # Core classes (BaseVisualizer, AdversarialVisualizer, ModelVisualizer, DetectionVisualizer)
└── tda_visualizers.py  # TDAVisualizer (active, exported from __init__.py)
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add visualization | `visualizers.py` or `tda_visualizers.py` | Inherit `BaseVisualizer`, implement with `self._palette()` and `title` param |
| Add data source | `data_loaders.py` | Add loader function, update `check_required_files()` |
| CLI args | `utils.py` | Modify `parse_arguments()` |
| Config changes | `config.py` | Update `PALETTES`, `ATTACKS`, `CHARACTERISTICS`, `MNIST_CONFIG` |
| Add palette | `config.py` → `PALETTES` dict | Then use `self._palette("key")` in methods |

## COLOR PALETTES

All methods use palettes via `self._palette(kind)`:

| Kind | Palette | Use for |
|------|---------|---------|
| `"categorical"` | `colorblind` | bar charts, grouped comparisons, clean/adv labels |
| `"sequential"` | `crest` | histograms, line plots, heatmaps, persistence diagrams |
| `"diverging"` | `BrBG` | signed differences, correlation matrices |

Call with `sns.barplot(..., palette=self._palette("categorical"))` or `sns.color_palette(self._palette("sequential"), n)`.

## TITLE SUPPORT

Every public plotting method must accept `title: Optional[str] = None`. Apply with:
```python
if title is None:
    title = 'Default Title'
fig.suptitle(title, fontsize=16)  # or ax.set_title(title) for single-axes plots
```

## CONVENTIONS

- All plotting uses seaborn functions (`sns.histplot`, `sns.barplot`, `sns.lineplot`, `sns.scatterplot`, `sns.heatmap`) — not raw `ax.bar()`, `ax.hist()`, etc.
- Matplotlib is still used for `plt.subplots()`, `fig.savefig()`, `plt.suptitle()`, `plt.tight_layout()`, `ax.imshow()`, and 3D axes
- Style presets: `presentation` (16x12), `paper` (8x6), `web` (12x8)
- DPI default: 300 (configurable via `--dpi`)
- Sample limit: 1000 (prevents memory issues)
- There are two `TDAVisualizer` classes — the one in `visualizers.py` is shadowed/dead code; the one in `tda_visualizers.py` is exported

## ANTI-PATTERNS

- **DO NOT** add bare `except Exception` handlers (34+ already exist)
- **NEVER** load full dataset without sample limiting
- **ALWAYS** call `setup_environment()` before visualization via CLI
- **DO NOT** use raw `ax.bar()` / `ax.hist()` / `ax.plot()` — use seaborn equivalents
- **DO NOT** hardcode colors — use `self._palette()` and `sns.color_palette()`
