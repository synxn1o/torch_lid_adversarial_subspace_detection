# Adversarial ML Visualization Utility

A comprehensive visualization toolkit for adversarial machine learning research, specifically designed for the MNIST dataset. This utility provides visual analysis across three main categories:

1. **Original vs Adversarial Data**: Image comparisons, perturbation analysis, attack metrics
2. **Model Performance**: Training curves, confusion matrices, ROC analysis
3. **Adversarial Detection**: Multi-characteristic analysis, 3D feature spaces, probability distributions
4. **Topological Analysis**: TDA-based Trojan detection, persistence diagrams, topological feature comparison

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
- **ROC Comparison**: Compare detection performance across characteristics (LID, KD, BU, KM)
- **3D Feature Visualization**: Interactive 3D scatter plots of characteristic features
- **Probability Distributions**: Separation analysis of adversarial vs normal detection probabilities
- **Metrics Comparison**: Comprehensive comparison of accuracy, precision, recall, and AUC

### 4. Topological Analysis (TDA)
- **Persistence Diagrams**: Visualization of 0, 1, and 2-dimensional topological structures
- **Feature Comparison**: Comparison of topological characteristics (max persistence, avg death) across models

## Installation & Dependencies

### Required Packages
```bash
pip install matplotlib seaborn scikit-learn torch pandas numpy ripser
# Optional for interactive plots:
pip install plotly
```

### Required Data
The utility expects data files in the `data/` directory:
- `model_mnist.pth` - Trained model
- `Adv_mnist_{attack}.npy` - Adversarial examples for each attack
- `{characteristic}_mnist_{attack}.npy` - Extracted characteristics

## Usage

### Basic Usage

```bash
# Generate all visualizations for MNIST
python -m visualizer.main --mode all --dataset mnist

# Generate specific visualizations
python -m visualizer.main --mode adversarial --dataset mnist --attack fgsm
python -m visualizer.main --mode model --dataset mnist
python -m visualizer.main --mode detection --dataset mnist --attack all
python -m visualizer.main --mode tda --dataset mnist

# Specify output directory
python -m visualizer.main --mode all --dataset mnist --output-dir ./my_plots

# Change output format and resolution
python -m visualizer.main --mode all --dataset mnist --format pdf --dpi 600
```

### Command Line Arguments

#### Mode Selection
- `--mode {adversarial, model, detection, tda, all, interactive}`
- Default: `all`

#### Data Selection
- `--dataset {mnist, cifar, svhn}` - Dataset name
- `--attack {fgsm, bim-a, bim-b, jsma, cw-l2, cw-lid, all}` - Attack type(s)
- `--characteristics {lid, kd, bu, km, all}` - Characteristic type(s)

#### Output Configuration
- `--output-dir PATH` - Output directory (default: `visualizer/outputs/`)
- `--format {png, pdf, svg, html}` - Output format (default: `png`)
- `--dpi INTEGER` - Resolution (default: `300`)

#### Advanced Options
- `--what TEXT` - Specific visualization names (comma-separated)
- `--sample-limit INTEGER` - Maximum samples to load
- `--no-cache` - Disable caching

### Examples

#### Generate All Visualizations
```bash
python -m visualizer.main --mode all --dataset mnist
```
This generates:
- Image grids for all attacks
- Perturbation analysis
- Attack success metrics  
- Training curves
- Confusion matrix
- ROC curves
- Detection ROC comparisons
- 3D feature plots
- Probability distributions
- Metrics comparison

#### Focus on Specific Attacks
```bash
python -m visualizer.main --mode adversarial --dataset mnist --attack fgsm,bim-a
```

#### Detection Analysis Only
```bash
python -m visualizer.main --mode detection --dataset mnist --characteristics lid,kd --attack all
```

#### TDA Analysis
```bash
# Run TDA detector first
python tda_detector.py --dataset mnist --name my_model

# Visualize results
python -m visualizer.main --mode tda --dataset mnist
```

#### Comparing Multiple Models with TDA
To compare a clean model with a poisoned model:
1. Run TDA on the clean model:
   ```bash
   python tda_detector.py --dataset mnist --name clean
   ```
2. Run TDA on the poisoned model:
   ```bash
   python tda_detector.py --dataset mnist --model_path data/model_mnist_poisoned.pth --name poisoned
   ```
3. Visualize the comparison:
   ```bash
   python -m visualizer.main --mode tda --dataset mnist --characteristics clean,poisoned
   ```

#### Custom Visualizations
```bash
python -m visualizer.main --mode all --dataset mnist --what image_grid,training_curves,roc_comparison
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
└── summary_mnist_all.txt
```

## Python API

You can also use the visualizers programmatically:

```python
from visualizer.visualizers import AdversarialVisualizer, ModelVisualizer, DetectionVisualizer

# Adversarial visualizations
adv_viz = AdversarialVisualizer(dataset='mnist', style='presentation')
adv_viz.create_image_grid_comparison('fgsm', num_samples=16)
adv_viz.create_perturbation_analysis()

# Model visualizations
model_viz = ModelVisualizer(dataset='mnist')
model_viz.create_training_curves()
model_viz.create_confusion_matrix()

# Detection visualizations
detect_viz = DetectionVisualizer(dataset='mnist')
detect_viz.create_roc_comparison(attacks=['fgsm', 'bim-a'], characteristics=['lid', 'kd'])
detect_viz.create_3d_feature_space('fgsm', 'lid')
```

## Data Loading

The utility automatically handles data loading from the existing file structure:

1. **Original Data**: Loads via `util.get_data()` function
2. **Adversarial Examples**: Loads from `data/Adv_{dataset}_{attack}.npy`
3. **Characteristics**: Loads from `data/{char}_{dataset}_{attack}.npy`
4. **Model**: Loads from `data/model_{dataset}.pth`

All data is automatically validated and cleaned (handles Inf/NaN values).

## Visual Style Presets

Three style presets are available:

- **presentation**: Large fonts, high contrast (16x12 figures)
- **paper**: Publication quality (8x6 figures)
- **web**: Optimized for screen (12x8 figures)

```python
viz = AdversarialVisualizer(dataset='mnist', style='paper', dpi=300)
```

## Performance Considerations

- **Sample Limiting**: By default, limits to 1000 samples to avoid memory issues
- **Batch Processing**: Large datasets are processed in batches
- **Caching**: Processed data is cached for faster subsequent runs
- **GPU Support**: Automatically uses GPU if available for model inference

## Troubleshooting

### "File not found" errors
Check data availability:
```python
from visualizer.data_loaders import check_required_files
status = check_required_files('mnist')
print(status)
```

### Missing dependencies
Install required packages:
```bash
pip install matplotlib seaborn scikit-learn torch pandas
```

### Memory issues
Reduce sample limit:
```bash
python -m visualizer.main --mode all --dataset mnist --sample-limit 100
```

## Integration with Existing Scripts

This utility is designed to work seamlessly with:
- `train_model.py` - Generates model files
- `craft_adv_examples.py` - Generates adversarial examples
- `extract_characteristics.py` - Generates characteristic files
- `detect_adv_examples.py` - Generates detection metrics

Run these scripts first to generate the required data, then use the visualization utility to analyze the results.

## License

MIT License - See project repository for details.

## Contact

For questions or issues, please refer to the project documentation.