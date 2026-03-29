"""Unified data loading for all supported datasets.

Provides a single interface for loading MNIST, CIFAR-10, SVHN, and Toy (2D binary)
datasets with consistent normalization (to [-0.5, 0.5]) and optional augmentation.

Functions:
    get_dataloader — Returns a PyTorch DataLoader for the specified dataset
    loader_to_numpy — Converts a DataLoader to (X, y) numpy arrays
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os

def get_dataloader(dataset_name, batch_size=128, train=True, download=True, augmentation=False):
    """
    Unified data loader for all supported datasets.
    """
    if dataset_name in ['mnist', 'cifar', 'svhn']:
        # Base transforms
        base_transform_list = [transforms.ToTensor()]
        
        # Normalization to [-0.5, 0.5]
        if dataset_name == 'cifar':
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        else:
            normalize = transforms.Normalize((0.5,), (1.0,))

        # Augmentation
        transform_list = []
        if train and augmentation:
            transform_list.extend([
                transforms.RandomRotation(20),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                transforms.RandomHorizontalFlip()
            ])
        
        transform = transforms.Compose(transform_list + base_transform_list + [normalize])

        if dataset_name == 'mnist':
            dataset = torchvision.datasets.MNIST(root='./data', train=train, download=download, transform=transform)
        elif dataset_name == 'cifar':
            dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=download, transform=transform)
        elif dataset_name == 'svhn':
            split = 'train' if train else 'test'
            dataset = torchvision.datasets.SVHN(root='./data', split=split, download=download, transform=transform)
            
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)
        return loader

    elif dataset_name == 'toy':
        # For toy dataset, we expect a pickle file
        # We'll use the circle dataset by default if it exists
        data_path = "toy_example/data/circle_dataset.pkl"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Toy dataset not found at {data_path}. Run toy_example/generate_dataset.py first.")
            
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        X = data['X'].astype(np.float32)
        y = data['y']
        # Convert labels from {-1, 1} to {0, 1}
        y = (y == 1).astype(np.int64)
        
        # Split (simple 80/20)
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        
        # Use fixed seed for consistency
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        if train:
            sel_indices = indices[:n_train]
        else:
            sel_indices = indices[n_train:]
            
        X_sel = torch.from_numpy(X[sel_indices])
        y_sel = torch.from_numpy(y[sel_indices])
        
        dataset = TensorDataset(X_sel, y_sel)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
        return loader
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def loader_to_numpy(loader):
    """
    Convert a DataLoader to numpy arrays.
    """
    X = []
    Y = []
    for x, y in loader:
        X.append(x.numpy())
        Y.append(y.numpy())
    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
