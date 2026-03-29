#!/usr/bin/env python3
"""
Adversarial attacks adapted for 2D toy dataset.
Implements FGSM, BIM, and JSMA for binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from collections import defaultdict

def fgsm(model, x, y, eps, clip_min=-2.0, clip_max=2.0):
    """
    Fast Gradient Sign Method for 2D toy data.
    
    Args:
        model: PyTorch model
        x: Input tensor (batch, 2)
        y: Target labels (batch,) with values 0 or 1
        eps: Perturbation magnitude
        clip_min, clip_max: Clipping bounds
    
    Returns:
        x_adv: Adversarial examples
    """
    x_adv = x.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(x_adv)
    
    # Loss (binary classification)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs.squeeze(), y.float())
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Apply perturbation
    grad_sign = x_adv.grad.sign()
    x_adv = x_adv + eps * grad_sign
    x_adv = torch.clamp(x_adv, clip_min, clip_max)
    
    return x_adv.detach()

def bim(model, x, y, eps, eps_iter=0.01, nb_iter=50, clip_min=-2.0, clip_max=2.0, mode='last'):
    """
    Basic Iterative Method for 2D toy data.
    
    Args:
        model: PyTorch model
        x: Input tensor (batch, 2)
        y: Target labels (batch,) with values 0 or 1
        eps: Total perturbation budget
        eps_iter: Step size per iteration
        nb_iter: Number of iterations
        mode: 'first' (stop at first misclassification) or 'last' (full iterations)
    
    Returns:
        x_adv: Adversarial examples
    """
    model.eval()
    x_adv = x.clone().detach()
    
    # Define bounds
    x_min = torch.clamp(x - eps, clip_min, clip_max)
    x_max = torch.clamp(x + eps, clip_min, clip_max)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Track successful attacks for 'first' mode
    if mode == 'first':
        adv_results = x.clone().detach()
        done_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    
    curr_x = x.clone().detach()
    
    # Store perturbation history for visualization
    perturbation_history = []
    
    for i in range(nb_iter):
        curr_x.requires_grad_(True)
        outputs = model(curr_x)
        loss = criterion(outputs.squeeze(), y.float())
        
        model.zero_grad()
        loss.backward()
        
        grad_sign = curr_x.grad.sign()
        curr_x = curr_x + eps_iter * grad_sign
        
        # Project to epsilon ball
        curr_x = torch.max(torch.min(curr_x, x_max), x_min)
        curr_x = torch.clamp(curr_x, clip_min, clip_max).detach()
        
        # Store current state for visualization
        perturbation_history.append(curr_x.clone().cpu().numpy())
        
        if mode == 'first':
            # Check misclassifications
            with torch.no_grad():
                # Get predictions as 1D tensor
                preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)
                # Ensure y is also 1D
                y_1d = y.float().view(-1)
                # Now both are [batch], comparison works correctly
                misclassified = (preds != y_1d)
                
                # Update those that are misclassified for the first time
                update_mask = misclassified & (~done_mask)
                if update_mask.any():
                    # Update adv_results only for samples that are misclassified
                    adv_results[update_mask] = curr_x[update_mask]
                    done_mask = done_mask | update_mask
                    
                if done_mask.all():
                    break
    
    if mode == 'first':
        # For those never misclassified, return final iteration
        adv_results[~done_mask] = curr_x[~done_mask]
        return adv_results, np.array(perturbation_history)
    else:
        return curr_x, np.array(perturbation_history)

def jsma(model, x, y, theta=0.1, gamma=0.1, clip_min=-2.0, clip_max=2.0):
    """
    Jacobian-based Saliency Map Attack for 2D toy data.
    Uses single sample processing to avoid complexity.
    
    Args:
        model: PyTorch model
        x: Input tensor (batch, 2)
        y: Target labels (batch,) with values 0 or 1
        theta: Perturbation magnitude per step
        gamma: Max fraction of features to perturb
    
    Returns:
        x_adv: Adversarial examples
    """
    model.eval()
    batch_size = x.size(0)
    x_adv_list = []
    perturbation_history = []
    
    # Process each sample individually
    for i in range(batch_size):
        x_single = x[i:i+1].clone().detach()  # Keep batch dim
        y_single = y[i]
        target_class = 1 - y_single.item()  # Opposite class
        
        history = [x_single.clone().cpu().numpy()[0]]
        
        # Max iterations based on gamma
        max_iters = int(2 * gamma / 2)
        
        for _ in range(max_iters):
            # Check misclassification
            with torch.no_grad():
                output = model(x_single)
                pred = (torch.sigmoid(output) > 0.5).float().item()
                if pred != y_single.item():
                    break
            
            # Compute gradients
            x_single.requires_grad_(True)
            output = model(x_single)
            
            # For binary: we want to maximize the logit for target_class
            # Since output is (1, 1), we need to interpret it
            # Logit for class 1 is output, logit for class 0 is -output
            if target_class == 1:
                logit_target = output[0, 0]
                grad_target = torch.autograd.grad(logit_target, x_single, retain_graph=True)[0][0]
            else:
                logit_target = -output[0, 0]
                grad_target = torch.autograd.grad(logit_target, x_single, retain_graph=True)[0][0]
            
            # Saliency for 2D case
            if theta > 0:
                # Increase features that positively impact target
                mask = grad_target > 0
                saliency = grad_target
            else:
                # Decrease features that negatively impact target
                mask = grad_target < 0
                saliency = -grad_target
            
            # Apply saliency
            if torch.any(mask):
                # Zero out non-masked
                saliency = torch.where(mask, saliency, torch.tensor(-float('inf')).to(x.device))
                best_idx = torch.argmax(saliency)
                
                with torch.no_grad():
                    x_single[0, best_idx] += theta
                    x_single[0] = torch.clamp(x_single[0], clip_min, clip_max)
            
            history.append(x_single.clone().cpu().numpy()[0])
        
        x_adv_list.append(x_single.detach().squeeze(0))
        perturbation_history.append(np.array(history))
    
    return torch.stack(x_adv_list), perturbation_history

def generate_adversarial_examples(model, X, y, attacks=None, eps=0.1, eps_iter=0.01):
    """
    Generate adversarial examples using multiple attacks on toy dataset.
    
    Args:
        model: Trained PyTorch model
        X: Clean data (n_samples, 2)
        y: Labels (n_samples,)
        attacks: List of attack names ['fgsm', 'bim-a', 'bim-b', 'jsma']
        eps: Epsilon for FGSM and BIM
        eps_iter: Step size for BIM
    
    Returns:
        Dictionary with adversarial examples and metadata
    """
    if attacks is None:
        attacks = ['fgsm', 'bim-a', 'bim-b', 'jsma']
    
    model.eval()
    device = next(model.parameters()).device
    
    # Convert to tensors
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    
    results = {
        'clean': X,
        'labels': y,
        'attacks': {}
    }
    
    print("Generating adversarial examples for toy dataset...")
    
    for attack_name in attacks:
        print(f"\n--- {attack_name.upper()} ---")
        
        if attack_name == 'fgsm':
            X_adv = fgsm(model, X_t, y_t, eps=eps)
            results['attacks'][attack_name] = {
                'examples': X_adv.cpu().numpy(),
                'eps': eps
            }
            
        elif attack_name == 'bim-a':
            X_adv, pert_history = bim(model, X_t, y_t, eps=eps, eps_iter=eps_iter, 
                                          nb_iter=50, mode='first')
            results['attacks'][attack_name] = {
                'examples': X_adv.cpu().numpy(),
                'perturbation_history': pert_history,
                'eps': eps,
                'eps_iter': eps_iter
            }
            
        elif attack_name == 'bim-b':
            X_adv, pert_history = bim(model, X_t, y_t, eps=eps, eps_iter=eps_iter, 
                                          nb_iter=50, mode='last')
            results['attacks'][attack_name] = {
                'examples': X_adv.cpu().numpy(),
                'perturbation_history': pert_history,
                'eps': eps,
                'eps_iter': eps_iter
            }
            
        elif attack_name == 'jsma':
            X_adv, pert_history = jsma(model, X_t, y_t, theta=0.1, gamma=0.2)
            results['attacks'][attack_name] = {
                'examples': X_adv.cpu().numpy(),
                'perturbation_history': pert_history,
                'theta': 0.1,
                'gamma': 0.2
            }
        
        # Compute success rate
        with torch.no_grad():
            # X_adv is already a tensor on the correct device
            if isinstance(X_adv, torch.Tensor):
                X_adv_tensor = X_adv
            else:
                X_adv_tensor = torch.FloatTensor(X_adv).to(device)
            outputs = model(X_adv_tensor)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
            success_rate = np.mean(preds != y)
            print(f"Success rate: {success_rate:.2%}")
            results['attacks'][attack_name]['success_rate'] = success_rate
    
    return results

def save_adversarial_results(results, filepath):
    """Save adversarial examples and metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nAdversarial results saved to {filepath}")

def load_adversarial_results(filepath):
    """Load adversarial results."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)