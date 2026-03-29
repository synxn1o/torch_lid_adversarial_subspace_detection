"""Mathematical utilities for LID estimation, noise generation, and distance computation.

Provides the core computation functions used by detectors and attacks.

Functions:
    mle_batch — MLE-based LID estimation on k-NN distances (Scipy)
    kmean_batch — Mean distance to k nearest neighbors (Scipy)
    get_noisy_samples — Gaussian noise injection with per-dataset/attack std deviations
    lid_adv_term — Differentiable LID loss for CW-LID attack (PyTorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def mle_batch(data, batch, k):
    """
    LID of a batch of query points X (batch) relative to data.
    Numpy/Scipy implementation with robust error handling.
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    
    def f(v):
        if v[-1] < 1e-9:
            return 0.0
        v = np.maximum(v, 1e-10)
        # Avoid log(1) which is 0 in the denominator
        log_ratios = np.log(v/v[-1])
        sum_log = np.sum(log_ratios)
        if abs(sum_log) < 1e-9:
            return 0.0
        return - k / sum_log
    
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def kmean_batch(data, batch, k):
    """
    Mean distance of batch points to their k nearest neighbors in data.
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: np.mean(v)
    
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def get_noisy_samples(X, dataset_name, attack_name=None, std=None):
    """
    Add Gaussian noise to samples.
    """
    if std is None:
        # Default STDEVS from original code
        STDEVS = {
            'mnist': {'fgsm': 0.264, 'bim-a': 0.111, 'bim-b': 0.184, 'cw-l2': 0.588},
            'cifar': {'fgsm': 0.0504, 'bim-a': 0.0087, 'bim-b': 0.0439, 'cw-l2': 0.015},
            'svhn': {'fgsm': 0.1332, 'bim-a': 0.015, 'bim-b': 0.1024, 'cw-l2': 0.0379},
            'toy': 0.2
        }
        if dataset_name == 'toy':
            std = STDEVS['toy']
        else:
            if attack_name in STDEVS.get(dataset_name, {}):
                std = STDEVS[dataset_name][attack_name]
            else:
                std = STDEVS.get(dataset_name, {}).get('cw-l2', 0.1)
                
    noise = np.random.normal(loc=0, scale=std, size=X.shape)
    
    # Clipping depends on dataset
    if dataset_name == 'toy':
        return X + noise
    else:
        return np.clip(X + noise, -0.5, 0.5).astype(np.float32)

def lid_adv_term(clean_logits, adv_logits, k=20):
    """
    Calculate LID loss term for a minibatch of advs logits relative to clean logits.
    PyTorch implementation for use in CW attack.
    """
    batch_size = clean_logits.size(0)
    c_pred = clean_logits.view(batch_size, -1)
    a_pred = adv_logits.view(batch_size, -1)
    
    r_a = torch.sum(a_pred**2, dim=1).view(-1, 1)
    r_c = torch.sum(c_pred**2, dim=1).view(1, -1)
    
    D = r_a - 2 * torch.matmul(a_pred, c_pred.t()) + r_c
    D1 = torch.sqrt(D + 1e-9)
    
    # k+1 because topk includes self if c_pred == a_pred, 
    # but here they are different. However, to be safe and consistent:
    D2, _ = torch.topk(-D1, k=k+1, sorted=True)
    D3 = -D2[:, 1:]
    
    m = D3 / D3[:, -1].view(-1, 1)
    v_log = torch.sum(torch.log(m + 1e-9), dim=1)
    lids = -k / v_log
    
    return lids
