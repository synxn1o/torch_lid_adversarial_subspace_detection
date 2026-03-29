"""Adversarial attack implementations for adversarial example generation.

Provides gradient-based (FGSM, BIM), saliency-based (JSMA), and
optimization-based (CarliniL2) attacks. All attacks inherit from BaseAttack
and operate on ModelWrapper instances.

Classes:
    BaseAttack — Abstract base class with binary/multi-class loss handling
    FGSM — Fast Gradient Sign Method (single-step)
    BIM — Basic Iterative Method (modes: 'first', 'last')
    JSMA — Jacobian-based Saliency Map Attack (slow, sample-by-sample)
    CarliniL2 — Carlini & Wagner L2 optimization attack (supports CW-LID)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from .utils import lid_adv_term

class BaseAttack:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.device = model_wrapper.device
        self.is_binary = model_wrapper.is_binary

    def _get_loss(self, logits, y):
        if self.is_binary:
            # y should be (batch,) or (batch, 1)
            return nn.BCEWithLogitsLoss()(logits.squeeze(), y.float())
        else:
            # y should be (batch,) indices or (batch, classes) one-hot
            if y.dim() > 1:
                y = torch.argmax(y, dim=1)
            return nn.CrossEntropyLoss()(logits, y)

    def generate(self, x, y):
        raise NotImplementedError

class FGSM(BaseAttack):
    def __init__(self, model_wrapper, eps=0.1, clip_min=-0.5, clip_max=0.5):
        super().__init__(model_wrapper)
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(self, x, y):
        x_adv = x.clone().detach().to(self.device).requires_grad_(True)
        y = y.to(self.device)
        
        logits = self.model_wrapper.get_logits(x_adv)
        loss = self._get_loss(logits, y)
        
        self.model_wrapper.model.zero_grad()
        loss.backward()
        
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + self.eps * grad_sign
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach()

class BIM(BaseAttack):
    def __init__(self, model_wrapper, eps=0.1, eps_iter=0.01, nb_iter=50, 
                 clip_min=-0.5, clip_max=0.5, mode='last'):
        super().__init__(model_wrapper)
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mode = mode # 'first' or 'last'

    def generate(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        
        x_min = torch.clamp(x - self.eps, self.clip_min, self.clip_max)
        x_max = torch.clamp(x + self.eps, self.clip_min, self.clip_max)
        
        adv_results = x.clone().detach()
        done_mask = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
        
        curr_x = x.clone().detach()
        
        for i in range(self.nb_iter):
            curr_x.requires_grad_(True)
            logits = self.model_wrapper.get_logits(curr_x)
            loss = self._get_loss(logits, y)
            
            self.model_wrapper.model.zero_grad()
            loss.backward()
            
            grad_sign = curr_x.grad.sign()
            curr_x = curr_x + self.eps_iter * grad_sign
            
            curr_x = torch.max(torch.min(curr_x, x_max), x_min)
            curr_x = torch.clamp(curr_x, self.clip_min, self.clip_max).detach()
            
            if self.mode == 'first':
                with torch.no_grad():
                    preds = self.model_wrapper.predict(curr_x)
                    if self.is_binary:
                        y_np = y.cpu().numpy()
                    else:
                        if y.dim() > 1:
                            y_np = torch.argmax(y, dim=1).cpu().numpy()
                        else:
                            y_np = y.cpu().numpy()
                            
                    misclassified = torch.from_numpy(preds != y_np).to(self.device)
                    update_mask = misclassified & (~done_mask)
                    if update_mask.any():
                        adv_results[update_mask] = curr_x[update_mask]
                        done_mask = done_mask | update_mask
                    
                    if done_mask.all():
                        break
        
        if self.mode == 'first':
            adv_results[~done_mask] = curr_x[~done_mask]
            return adv_results
        else:
            return curr_x

class JSMA(BaseAttack):
    def __init__(self, model_wrapper, theta=1.0, gamma=0.1, clip_min=-0.5, clip_max=0.5):
        super().__init__(model_wrapper)
        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(self, x, y):
        # JSMA is slow, we process sample by sample
        x_adv_list = []
        for i in tqdm(range(len(x)), desc="JSMA"):
            x_single = x[i:i+1]
            y_single = y[i]
            x_adv_list.append(self._generate_single(x_single, y_single))
        return torch.cat(x_adv_list, dim=0)

    def _generate_single(self, x, y):
        adv_x = x.clone().detach().to(self.device)
        target_class = None
        
        if self.is_binary:
            target_class = 1 - y.item()
        else:
            # For multi-class, pick a target class (e.g. next class)
            current_class = y.item() if y.dim() == 0 else torch.argmax(y).item()
            target_class = (current_class + 1) % 10 # Assuming 10 classes
            
        nb_features = x.view(1, -1).size(1)
        max_iters = int(nb_features * self.gamma / 2)
        
        if self.theta > 0:
            search_domain = (adv_x < self.clip_max).float().view(1, -1)
        else:
            search_domain = (adv_x > self.clip_min).float().view(1, -1)
            
        for _ in range(max_iters):
            adv_x.requires_grad_(True)
            logits = self.model_wrapper.get_logits(adv_x)
            pred = self.model_wrapper.predict(adv_x)[0]
            
            if pred == target_class:
                break
                
            if self.is_binary:
                # Binary case: logit for class 1 is output, for class 0 is -output
                logit = logits[0, 0] if target_class == 1 else -logits[0, 0]
                grad_target = torch.autograd.grad(logit, adv_x, retain_graph=True)[0].view(1, -1)
                # For binary, "other" is just the opposite
                grad_other = -grad_target
            else:
                logit_target = logits[0, target_class]
                grad_target = torch.autograd.grad(logit_target, adv_x, retain_graph=True)[0].view(1, -1)
                logit_sum = logits.sum()
                grad_sum = torch.autograd.grad(logit_sum, adv_x)[0].view(1, -1)
                grad_other = grad_sum - grad_target
                
            if self.theta > 0:
                mask = (grad_target > 0) & (grad_other < 0) & (search_domain > 0.5)
            else:
                mask = (grad_target < 0) & (grad_other > 0) & (search_domain > 0.5)
                
            saliency = -grad_target * grad_other
            saliency = torch.where(mask, saliency, torch.tensor(-float('inf')).to(self.device))
            
            p1_val, p1_idx = torch.max(saliency, dim=1)
            if p1_val == -float('inf'): break
            
            saliency[0, p1_idx] = -float('inf')
            p2_val, p2_idx = torch.max(saliency, dim=1)
            
            with torch.no_grad():
                adv_x.view(1, -1)[0, p1_idx] += self.theta
                if p2_val != -float('inf'):
                    adv_x.view(1, -1)[0, p2_idx] += self.theta
                adv_x = torch.clamp(adv_x, self.clip_min, self.clip_max)
                
                if self.theta > 0:
                    search_domain = (adv_x.view(1, -1) < self.clip_max).float()
                else:
                    search_domain = (adv_x.view(1, -1) > self.clip_min).float()
                    
        return adv_x.detach()

class CarliniL2(BaseAttack):
    def __init__(self, model_wrapper, confidence=0, targeted=False, learning_rate=1e-2,
                 binary_search_steps=9, max_iterations=1000, initial_const=1e-3, use_lid=False):
        super().__init__(model_wrapper)
        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.initial_const = initial_const
        self.use_lid = use_lid

    def generate(self, x, y):
        # Implementation of CW-L2 / CW-LID
        # Simplified for brevity but following the logic in cw_attacks.py
        x = x.to(self.device)
        y = y.to(self.device)
        batch_size = x.size(0)
        
        # Determine targets
        if self.targeted:
            y_target = y # Assumed already target
        else:
            # Untargeted: y is true label, we want to move away from it
            if self.is_binary:
                y_target = 1 - y
            else:
                # For multi-class, we need one-hot for the logic
                if y.dim() == 1:
                    y_onehot = torch.zeros(batch_size, 10, device=self.device)
                    y_onehot.scatter_(1, y.view(-1, 1), 1)
                    y = y_onehot
                # Pick a target class (randomly)
                y_target = torch.zeros_like(y)
                for i in range(batch_size):
                    curr = torch.argmax(y[i]).item()
                    t = np.random.choice([c for c in range(10) if c != curr])
                    y_target[i, t] = 1

        # tanh space conversion
        # Assuming x in [-0.5, 0.5]
        x_tanh = torch.atanh(x * 1.999999).detach()
        
        lower_bound = torch.zeros(batch_size, device=self.device)
        const = torch.ones(batch_size, device=self.device) * self.initial_const
        upper_bound = torch.ones(batch_size, device=self.device) * 1e10
        
        o_bestl2 = torch.ones(batch_size, device=self.device) * 1e10
        o_bestattack = x.clone()
        
        c_logits = None
        if self.use_lid:
            with torch.no_grad():
                c_logits = self.model_wrapper.get_logits(x)

        for outer_step in range(self.binary_search_steps):
            modifier = torch.zeros_like(x, requires_grad=True, device=self.device)
            optimizer = optim.Adam([modifier], lr=self.learning_rate)
            
            bestl2 = torch.ones(batch_size, device=self.device) * 1e10
            bestscore = torch.ones(batch_size, device=self.device, dtype=torch.long) - 1
            
            for iteration in range(self.max_iterations):
                optimizer.zero_grad()
                newimg = torch.tanh(modifier + x_tanh) / 2
                output = self.model_wrapper.get_logits(newimg)
                
                l2dist = torch.sum((newimg - x)**2, dim=list(range(1, x.dim())))
                
                if self.is_binary:
                    # Binary CW loss
                    # target 1: want output > 0. target 0: want output < 0.
                    # loss = max(0, confidence + (other - target))
                    if self.targeted:
                        # target is y_target
                        loss1 = torch.where(y_target == 1, 
                                            torch.clamp(-output.squeeze() + self.confidence, min=0.0),
                                            torch.clamp(output.squeeze() + self.confidence, min=0.0))
                    else:
                        # move away from y
                        loss1 = torch.where(y == 1,
                                            torch.clamp(output.squeeze() + self.confidence, min=0.0),
                                            torch.clamp(-output.squeeze() + self.confidence, min=0.0))
                else:
                    real = torch.sum(y_target * output, dim=1)
                    other = torch.max((1 - y_target) * output - (y_target * 10000), dim=1)[0]
                    loss1 = torch.clamp(other - real + self.confidence, min=0.0)
                
                loss_total = torch.sum(l2dist + const * loss1)
                if self.use_lid:
                    loss_lid = lid_adv_term(c_logits, output, k=20)
                    loss_total += torch.sum(const * loss_lid)
                
                loss_total.backward()
                optimizer.step()
                
                with torch.no_grad():
                    preds = self.model_wrapper.predict(newimg)
                    y_target_np = y_target.cpu().numpy()
                    if not self.is_binary and y_target.dim() > 1:
                        y_target_np = np.argmax(y_target_np, axis=1)
                    
                    success = (preds == y_target_np)
                    mask = (l2dist < bestl2) & torch.from_numpy(success).to(self.device)
                    bestl2[mask] = l2dist[mask]
                    bestscore[mask] = torch.from_numpy(preds[mask]).to(self.device)
                    
                    mask_global = (l2dist < o_bestl2) & torch.from_numpy(success).to(self.device)
                    o_bestl2[mask_global] = l2dist[mask_global]
                    o_bestattack[mask_global] = newimg[mask_global]
            
            # Binary search step
            with torch.no_grad():
                y_target_np = y_target.cpu().numpy()
                if not self.is_binary and y_target.dim() > 1:
                    y_target_np = np.argmax(y_target_np, axis=1)
                
                success = (bestscore.cpu().numpy() == y_target_np) & (bestscore.cpu().numpy() != -1)
                success_t = torch.from_numpy(success).to(self.device)
                
                lower_bound = torch.where(success_t, lower_bound, torch.max(lower_bound, const))
                upper_bound = torch.where(success_t, torch.min(upper_bound, const), upper_bound)
                const = (lower_bound + upper_bound) / 2
                mask_not_found = (~success_t) & (upper_bound >= 1e9)
                const[mask_not_found] *= 10
                
        return o_bestattack
