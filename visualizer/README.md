# Adversarial ML Visualization Utility

A comprehensive visualization toolkit for adversarial machine learning research, using seaborn-based plotting with standardized color palettes.

## Features

### 1. Adversarial Example Analysis
- **Image Grid Comparison**: Side-by-side comparison of original, adversarial, perturbation heatmap, and difference images
- **Perturbation Analysis**: L2 distance distributions across different attack types
- **Attack Success Metrics**: Success rates, confidence drops, and model performance under attacks

### 2. Model Training & Performance
- **Training Curves**: Loss and accuracy evolution over epochs
- **Confusion Matrix**: Class-wise performance visualization
- **ROC Analysis**: Multi-class receiver operating characteristic curves

### 3. Adversarial Detection
- **ROC Comparison**: Compare detection performance across characteristics (LID, KD, KM)
- **3D Feature Visualization**: 3D scatter plots of characteristic features
- **Probability Distributions**: Separation analysis of adversarial vs normal detection probabilities
- **Metrics Comparison**: Comprehensive comparison of accuracy, precision, recall, and AUC

### 4. Topological Analysis (TDA)
- **Persistence Diagrams**: Visualization of 0 and 1-dimensional topological structures
- **Lifetime Histograms**: Persistence lifetime distributions
- **Bottleneck Distances**: Grouped bar charts per layer or condition
- **Epsilon Sweep**: Bottleneck distance vs perturbation budget (dual y-axis)
- **Classifier Results**: Confusion matrix + ROC curve for persistence image classifiers
- **Clean vs Adversarial Comparison**: Side-by-side TDA comparison

## Color Palettes

All plots use standardized seaborn palettes by plot type:

| Plot Type | Palette | Config Key |
|-----------|---------|------------|
| Categorical (bar, grouped) | `colorblind` | `PALETTES["categorical"]` |
| Sequential (hist, line, heatmap) | `crest` | `PALETTES["sequential"]` |
| Diverging (signed diffs, correlation) | `BrBG` | `PALETTES["diverging"]` |

Defined in `visualizer/config.py` as the `PALETTES` dict. Each visualizer class provides a `self._palette(kind)` helper.

## Title Support

Every public plotting method accepts an optional `title: Optional[str] = None` parameter. When `None`, a sensible default is used. When provided, it overrides the default.

## Usage

### CLI
```bash
# All visualizations for MNIST
python -m visualizer.main --mode all --dataset mnist

# Specific modes
python -m visualizer.main --mode adversarial --dataset mnist --attack fgsm
python -m visualizer.main --mode model --dataset mnist
python -m visualizer.main --mode detection --dataset mnist
python -m visualizer.main --mode tda --dataset mnist
```

### Python API
```python
from visualizer.visualizers import AdversarialVisualizer, ModelVisualizer, DetectionVisualizer
from visualizer.tda_visualizers import TDAVisualizer

# Adversarial
adv_viz = AdversarialVisualizer(dataset='mnist', style='presentation', dpi=300)
adv_viz.create_image_grid_comparison('fgsm', num_samples=16, title='Custom Title')
adv_viz.create_perturbation_analysis(attacks=['fgsm', 'bim-a'])

# Model
model_viz = ModelVisualizer(dataset='mnist')
model_viz.create_training_curves()
model_viz.create_confusion_matrix(title='Custom Confusion Matrix')

# Detection
detect_viz = DetectionVisualizer(dataset='mnist')
detect_viz.create_roc_comparison(attacks=['fgsm', 'bim-a'], characteristics=['lid', 'kd'])
detect_viz.create_metrics_comparison()

# TDA
tda_viz = TDAVisualizer(dataset='mnist')
tda_viz.plot_persistence_diagrams(tda_results, conditions, title='Custom Title')
tda_viz.plot_bottleneck_distances(distances_dict, group_by='layer')
```

## Output Structure

```
visualizer/outputs/
├── adversarial/
│   ├── adversarial_grid_fgsm.png
│   ├── perturbation_analysis.png
│   └── attack_metrics.png
├── model/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── roc_curves.png
├── detection/
│   ├── roc_comparison_all.png
│   ├── 3d_lid_fgsm.png
│   ├── detection_prob_distributions.png
│   └── metrics_comparison.png
└── tda/
    ├── persistence_diagrams.png
    ├── lifetime_histogram.png
    ├── bottleneck_distances.png
    ├── epsilon_sweep.png
    ├── classifier_results.png
    └── tda_comparison_clean_<attack>.png
```

## Style Presets

- **presentation**: Large fonts, high contrast (16x12 figures)
- **paper**: Publication quality (8x6 figures)
- **web**: Optimized for screen (12x8 figures)

Default DPI: 300.

## Architecture

```
visualizer/
├── __init__.py         # Package exports
├── config.py           # PALETTES, dataset configs, viz settings
├── data_loaders.py     # Load original/adv/characteristic data
├── main.py             # CLI entry point
├── utils.py            # Argument parsing, environment setup
├── visualizers.py      # BaseVisualizer, AdversarialVisualizer, ModelVisualizer, DetectionVisualizer
└── tda_visualizers.py  # TDAVisualizer
```

## Dependencies

```bash
pip install matplotlib seaborn scikit-learn torch pandas numpy
# Optional for TDA:
pip install ripser persim
```
