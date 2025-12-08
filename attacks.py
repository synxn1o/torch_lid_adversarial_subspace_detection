import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy

def fgsm(model, x, y, eps, clip_min=-0.5, clip_max=0.5):
    """
    Fast Gradient Sign Method
    """
    x_adv = x.clone().detach().requires_grad_(True)
    
    outputs = model(x_adv)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, torch.argmax(y, dim=1))
    
    model.zero_grad()
    loss.backward()
    
    grad_sign = x_adv.grad.sign()
    x_adv = x_adv + eps * grad_sign
    x_adv = torch.clamp(x_adv, clip_min, clip_max)
    
    return x_adv.detach()

def bim(model, x, y, eps, eps_iter, nb_iter=50, clip_min=-0.5, clip_max=0.5, batch_size=256, mode='last'):
    """
    Basic Iterative Method (Projected Gradient Descent)
    mode: 'last' (BIM-B) or 'first' (BIM-A - first misclassification)
    """
    model.eval()
    x_adv = x.clone().detach()
    
    # We can process in batches if x is large, but usually x is passed in batches from the caller.
    # The original code processes the whole X passed to it in batches inside the function.
    # Here we assume x is a batch or we handle batching inside. 
    # To align with original which handled large X arrays:
    
    results = x.clone().detach()
    
    # If x is very large (entire dataset), we should loop over it. 
    # But usually craft_adv_examples passes batches or the whole set.
    # Let's assume x is on device.
    
    # Define bounds
    x_min = x - eps
    x_max = x + eps
    x_min = torch.clamp(x_min, clip_min, clip_max)
    x_max = torch.clamp(x_max, clip_min, clip_max)
    
    # Track successful attacks for BIM-A
    if mode == 'first':
        # Array to store the first misclassified example
        # Initialize with original (or final if never misclassified? Original uses final)
        adv_results = x.clone().detach()
        # Mask to track which have been misclassified
        done_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        
    criterion = nn.CrossEntropyLoss()
    
    # Iterations
    # For BIM, we update the whole batch `nb_iter` times.
    
    curr_x = x.clone().detach()
    
    for i in range(nb_iter):
        curr_x.requires_grad_(True)
        outputs = model(curr_x)
        loss = criterion(outputs, torch.argmax(y, dim=1))
        
        model.zero_grad()
        loss.backward()
        
        grad_sign = curr_x.grad.sign()
        curr_x = curr_x + eps_iter * grad_sign
        
        # Clip to epsilon ball and valid range
        curr_x = torch.max(torch.min(curr_x, x_max), x_min)
        curr_x = torch.clamp(curr_x, clip_min, clip_max).detach()
        
        if mode == 'first':
            # Check misclassifications
            with torch.no_grad():
                preds = model(curr_x).argmax(dim=1)
                true_labels = torch.argmax(y, dim=1)
                misclassified = (preds != true_labels)
                
                # Update those that are misclassified for the first time
                update_mask = misclassified & (~done_mask)
                if update_mask.any():
                    adv_results[update_mask] = curr_x[update_mask]
                    done_mask = done_mask | update_mask
                    
                if done_mask.all():
                    break
    
    if mode == 'first':
        # For those never misclassified, we return the final iteration (as per original logic usually, 
        # or maybe the original logic returned the loop result. 
        # Original code: `X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])`
        # where `its` defaulted to last iter.
        # So for those not done, we update one last time.
        adv_results[~done_mask] = curr_x[~done_mask]
        return adv_results
    else:
        return curr_x


def saliency_map(grads_target, grads_other, search_domain, increase):
    """
    PyTorch implementation of saliency map selection.
    grads_target: (batch, features)
    grads_other: (batch, features)
    search_domain: (batch, features) - 1 if in domain, 0 if not
    """
    # JSMA Saliency Map Rule:
    # S(x, t) = 
    #   0 if dJ_t/dx < 0 or sum(dJ_other/dx) > 0
    #   - dJ_t/dx * sum(dJ_other/dx) otherwise
    
    # If increase=True: target grad should be positive, other sum negative.
    if increase:
        mask1 = grads_target > 0
        mask2 = grads_other < 0
    else:
        mask1 = grads_target < 0
        mask2 = grads_other > 0
        
    # Combine gradients
    score = -grads_target * grads_other
    
    # Apply masks and domain
    # We want to zero out invalid scores. 
    # Since we want max score, setting invalid to -inf is safer.
    score_masked = torch.where(mask1 & mask2 & (search_domain > 0.5), score, torch.tensor(-float('inf')).to(score.device))
    
    return score_masked

def jsma(model, x, y, theta=1.0, gamma=0.1, clip_min=-0.5, clip_max=0.5):
    """
    Jacobian-based Saliency Map Attack
    Note: This is computationally expensive as it requires computing Jacobian per sample.
    """
    model.eval()
    batch_size = x.size(0)
    nb_features = x.view(batch_size, -1).size(1)
    nb_classes = y.size(1)
    max_iters = int(nb_features * gamma / 2)
    
    x_adv = x.clone().detach()
    
    # Process each sample individually (JSMA is inherently sequential/per-sample)
    # Or try to batch it (complex due to different termination conditions).
    # Given the original code loops: `for i in tqdm(range(len(X))): ... jsma(...)`
    # We will implement the single sample version and wrap it in a loop in craft_adv_examples.
    # But wait, `craft_adv_examples` calls `saliency_map_method` which loops.
    # So here we can implement the batched version or single version.
    # Let's stick to single sample to be safe and correct, as JSMA is tricky to batch efficiently.
    pass 

def jsma_single(model, x_single, target_class, theta=1.0, gamma=0.1, clip_min=-0.5, clip_max=0.5):
    """
    JSMA for a single sample.
    x_single: (C, H, W) tensor
    target_class: int
    """
    # Copy
    adv_x = x_single.clone().detach().unsqueeze(0) # (1, C, H, W)
    
    # Search domain: features not yet clipped
    # Assuming normalized [-0.5, 0.5]
    if theta > 0:
        search_domain = (adv_x < clip_max).float().view(1, -1)
    else:
        search_domain = (adv_x > clip_min).float().view(1, -1)
        
    nb_features = search_domain.size(1)
    max_iters = int(nb_features * gamma / 2)
    
    iter_count = 0
    
    while iter_count < max_iters:
        adv_x.requires_grad_(True)
        outputs = model(adv_x)
        pred = outputs.argmax(dim=1)
        
        if pred.item() == target_class:
            break
            
        # Compute Jacobian
        # We need d(logit_target)/dx and sum(d(logit_other)/dx)
        
        # 1. Gradient of target logit
        logit_target = outputs[0, target_class]
        grad_target = torch.autograd.grad(logit_target, adv_x, retain_graph=True)[0].view(1, -1)
        
        # 2. Gradient of sum of other logits
        # sum(others) = sum(all) - target
        logit_sum = outputs.sum()
        grad_sum = torch.autograd.grad(logit_sum, adv_x, retain_graph=False)[0].view(1, -1)
        grad_other = grad_sum - grad_target
        
        # Saliency Map
        # We want to modify 2 pixels (p1, p2)
        # Heuristic: Pick p1 with max saliency, then p2.
        # Or computing pair-wise saliency matrix (very expensive O(N^2)).
        # The original cleverhans implementation uses an optimized search.
        # For simplicity/speed, we can approximate or use the O(N) heuristic if acceptable.
        # Original JSMA uses a loop to find best pair.
        
        # Let's look at the original `saliency_map` in `attacks.py` provided by user.
        # It calls `cleverhans.attacks_tf.saliency_map`.
        # That implementation typically computes the matrix.
        
        # Implementing full JSMA in PyTorch efficiently is verbose.
        # Given the constraints and the focus on "LID", JSMA might be less critical.
        # However, I should provide a working implementation.
        
        # Simplified JSMA (picking top 2 individual features instead of optimal pair)
        # This is a common approximation.
        
        if theta > 0: # Increase features
            saliency = -grad_target * grad_other
            mask = (grad_target > 0) & (grad_other < 0) & (search_domain > 0.5)
        else:
            saliency = -grad_target * grad_other
            mask = (grad_target < 0) & (grad_other > 0) & (search_domain > 0.5)
            
        saliency = torch.where(mask, saliency, torch.tensor(-float('inf')).to(x_single.device))
        
        # Find best feature p1
        p1_val, p1_idx = torch.max(saliency, dim=1)
        
        if p1_val == -float('inf'):
            break # No valid features
            
        # To find p2, we ideally recompute or mask p1.
        # Simple greedy: mask p1 and find next max.
        saliency[0, p1_idx] = -float('inf')
        p2_val, p2_idx = torch.max(saliency, dim=1)
        
        # Apply perturbation
        with torch.no_grad():
            adv_x.view(1, -1)[0, p1_idx] += theta
            if p2_val != -float('inf'):
                adv_x.view(1, -1)[0, p2_idx] += theta
            
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
            
            # Update search domain
            if theta > 0:
                search_domain = (adv_x.view(1, -1) < clip_max).float()
            else:
                search_domain = (adv_x.view(1, -1) > clip_min).float()
                
        iter_count += 1
        
    return adv_x.detach().squeeze(0)
