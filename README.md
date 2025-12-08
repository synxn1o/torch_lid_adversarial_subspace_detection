# Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality (PyTorch Implementation).

## Project Overview
This project is a **PyTorch implementation** of the methods described in the ICLR 2018 paper "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality" [^1]. It focuses on detecting adversarial examples by characterizing the local intrinsic dimensionality (LID) of the subspaces where these examples reside. Their Github Repo: [https://github.com/xingjunm/lid_adversarial_subspace_detection](https://github.com/xingjunm/lid_adversarial_subspace_detection)

The codebase allows for:
1.  **Training** Deep Neural Networks on MNIST, CIFAR-10, and SVHN.
2.  **Crafting** adversarial examples using attacks like FGSM, BIM, JSMA, and C&W.
3.  **Extracting** characteristics (LID, Kernel Density, Bayesian Uncertainty) from model layers.
4.  **Detecting** adversarial examples using a logistic regression classifier based on these characteristics.

## Directory Structure
*   **`.old/`**: Contains the original legacy TensorFlow implementation and documentation.
*   **`data/`**: Stores trained models (`.pth`) and generated data (`.npy`).
*   **`train_model.py`**: Script to train base classification models.
*   **`craft_adv_examples.py`**: Generates adversarial samples.
*   **`extract_characteristics.py`**: Extracts detection features (LID, etc.).
*   **`detect_adv_examples.py`**: Trains and evaluates the detector.
*   **`util.py`**: Utility functions for data loading and model definitions.
*   **`attacks.py`**: Implementations of FGSM, BIM, and JSMA attacks.
*   **`cw_attacks.py`**: Implementation of Carlini & Wagner (C&W) attacks.

## Usage Workflow

### 1. Train Base Models
Train a ResNet-like or standard CNN model on the target dataset.
```bash
python train_model.py -d <dataset> -e <epochs> -b <batch_size>
# Example:
python train_model.py -d mnist -e 50 -b 128
```
*   **Datasets**: `mnist`, `cifar`, `svhn`.

### 2. Craft Adversarial Examples
Generate adversarial examples against the trained model.
```bash
python craft_adv_examples.py -d <dataset> -a <attack> -b <batch_size>
# Example:
python craft_adv_examples.py -d mnist -a fgsm -b 100
```
*   **Attacks**: `fgsm`, `bim-a`, `bim-b`, `jsma`, `cw-l2`, `cw-lid`, `all`.

### 3. Extract Characteristics
Extract features like LID from the intermediate layers of the network for clean, noisy, and adversarial samples.
```bash
python extract_characteristics.py -d <dataset> -a <attack> -r <characteristics> -k <k_nearest> -b <batch_size>
# Example:
python extract_characteristics.py -d mnist -a fgsm -r lid -k 20 -b 100
```
*   **Characteristics**: `lid`, `kd`, `bu`.

### 4. Train & Evaluate Detector
Train a Logistic Regression detector on the extracted features.
```bash
python detect_adv_examples.py -d <dataset> -a <attack> -r <characteristics> [-t <test_attack>]
# Example (Train on FGSM, Test on FGSM):
python detect_adv_examples.py -d mnist -a fgsm -r lid

# Example (Train on FGSM, Test on C&W):
python detect_adv_examples.py -d mnist -a fgsm -r lid -t cw-l2
```

## Development Conventions
*   **Framework**: PyTorch (`torch`, `torchvision`).
*   **Data format**:
    *   Models are saved as `.pth` files in `data/`.
    *   Adversarial samples and characteristics are saved as `.npy` files in `data/`.
*   **LID Calculation**: Implemented using SciPy for batch processing or PyTorch for loss terms.
*   **Attacks**: implemented in `attacks.py` and `cw_attacks.py`, utilizing PyTorch gradients.

## Key Libraries
*   `torch`, `torchvision`: Deep learning framework.
*   `numpy`, `scipy`: Math and LID calculation.
*   `sklearn`: Logistic Regression and evaluation metrics.
*   `tqdm`: Progress bars.

---

[^1] Ma, X., Li, B., Wang, Y., Erfani, S. M., Wijewickrema, S., Schoenebeck, G., Song, D., Houle, M. E., & Bailey, J. (2018, January 8). Characterizing adversarial subspaces using local intrinsic dimensionality. ICLR 2018. https://arxiv.org/abs/1801.02613