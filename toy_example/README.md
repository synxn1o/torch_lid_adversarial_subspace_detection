# Toy Example: LID Adversarial Subspace Detection

This directory contains a simplified "toy" implementation of the LID (Local Intrinsic Dimensionality) adversarial detection method. It uses a synthetic 2D dataset (concentric circles) and a small binary neural network to visualize and demonstrate the concepts of adversarial attacks and LID-based detection.

## Workflow Overview

The workflow consists of 5 sequential steps:

1.  **Generate Data**: Create a synthetic 2D dataset.
2.  **Train Model**: Train a simple Neural Network to classify the data.
3.  **Attack**: Generate adversarial examples (FGSM, BIM, JSMA) against the model.
4.  **Extract Features**: Calculate LID scores for clean, noisy, and adversarial samples.
5.  **Detect**: Train a Logistic Regression classifier on the LID features to detect attacks.

## Prerequisites

Ensure you are in the root directory of the project (`/mnt/e/lid_adversarial_subspace_detection/`) and have the required dependencies installed (PyTorch, NumPy, Matplotlib, Scikit-learn).

## Usage Guide

### 1. Generate Dataset
Generates a synthetic dataset of two concentric circles (binary classification).
```bash
python toy_example/generate_dataset.py
```
*   **Output**: `toy_example/data/circle_dataset.pkl`
*   **Plot**: `toy_example/plots/dataset_visualization.png`

### 2. Train Neural Network
Trains a small binary classifier (2 hidden layers) on the generated dataset.
```bash
python toy_example/train_NN.py
```
*   **Output**: `toy_example/models/toy_binary_nn.pth`
*   **Plots**: 
    *   `toy_example/plots/training_history.png` (Loss/Acc curves)
    *   `toy_example/plots/decision_boundary.png` (Visual decision boundary)

### 3. Generate Adversarial Examples
Performs various attacks (FGSM, BIM-A, BIM-B, JSMA) to fool the trained model.
```bash
python toy_example/generate_adversarial.py
```
*   **Output**: `toy_example/results/adversarial_results.pkl`
*   **Plots**: `toy_example/plots/` (Detailed visualizations of attack trajectories and success rates)

### 4. Extract Characteristics (LID)
Extracts Local Intrinsic Dimensionality (LID) features from the intermediate layers of the neural network for clean, noisy, and adversarial samples.
```bash
python toy_example/extract_characteristics_toy.py -a all
```
*   **Arguments**:
    *   `-a`: Attack to process (e.g., `fgsm`, `bim-a`, `all`)
    *   `-k`: Number of nearest neighbors for LID (default: 20)
*   **Output**: `toy_example/data/characteristics/lid_toy_<attack>.npy`

### 5. Detect Adversarial Examples
Trains a Logistic Regression detector using the extracted LID features to distinguish between benign (clean/noisy) and adversarial samples.
```bash
python toy_example/detect_adv_examples_toy.py -a all
```
*   **Arguments**:
    *   `-a`: Attack to evaluate (e.g., `fgsm`, `all`)
*   **Plots**: `toy_example/results/detection_plots/`
    *   `detection_roc_curves.png`: ROC curves for each attack.
    *   `detection_metrics.png`: Accuracy, Precision, Recall, AUC comparisons.
    *   `detection_prob_distributions.png`: Distribution of detector probabilities.

## directory Structure

```
toy_example/
├── data/                   # Generated datasets and extracted features
├── models/                 # Trained model weights
├── plots/                  # Visualizations (Decision boundaries, Attack paths)
├── results/                # Adversarial examples and detection plots
├── generate_dataset.py     # Step 1: Data generation
├── train_NN.py             # Step 2: Model training
├── generate_adversarial.py # Step 3: Attack generation
├── extract_characteristics_toy.py # Step 4: LID feature extraction
├── detect_adv_examples_toy.py     # Step 5: Detection training/eval
└── attacks_toy.py          # Helper: Attack implementations
```