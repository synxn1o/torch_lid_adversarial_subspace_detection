import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from util import lid_adv_term

class CarliniL2:
    def __init__(self, model, image_size, num_channels, num_labels, batch_size=100,
                 confidence=0, targeted=True, learning_rate=1e-2,
                 binary_search_steps=9, max_iterations=1000,
                 abort_early=True, initial_const=1e-3):
        self.model = model
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.repeat = binary_search_steps >= 10

    def attack(self, x, y):
        # x: (N, C, H, W), y: (N, num_labels)
        device = x.device
        nb_classes = y.size(1)
        
        # Determine targets
        y_target = y.clone()
        if self.targeted:
            # Randomly select a different target class for each sample
            # Current class
            current_classes = torch.argmax(y, dim=1)
            for i in range(len(y)):
                target = np.random.choice([c for c in range(nb_classes) if c != current_classes[i].item()])
                y_target[i] = 0
                y_target[i][target] = 1
        
        x_adv = x.clone()
        
        # Process in batches
        for i in tqdm(range(0, x.size(0), self.batch_size), desc="CW-L2 Attack"):
            start = i
            end = min(i + self.batch_size, x.size(0))
            batch_x = x[start:end]
            batch_y = y_target[start:end]
            
            adv_batch = self.attack_batch(batch_x, batch_y)
            x_adv[start:end] = adv_batch
            
        return x_adv

    def attack_batch(self, imgs, labs):
        device = imgs.device
        batch_size = imgs.size(0)
        
        # Convert to tanh-space
        # imgs are in [-0.5, 0.5]. tanh(w)/2 is in [-0.5, 0.5].
        # w = arctanh(2 * imgs). 
        # Clip slightly to avoid inf
        imgs_tanh = torch.atanh(imgs * 1.999999).detach()
        
        # Bounds for constant c
        lower_bound = torch.zeros(batch_size, device=device)
        const = torch.ones(batch_size, device=device) * self.initial_const
        upper_bound = torch.ones(batch_size, device=device) * 1e10
        
        # Best results tracking
        o_bestl2 = torch.ones(batch_size, device=device) * 1e10
        o_bestscore = torch.ones(batch_size, device=device) * -1
        o_bestattack = imgs.clone()
        
        # Binary search steps
        for outer_step in range(self.binary_search_steps):
            # Modifier variable to optimize
            modifier = torch.zeros_like(imgs, requires_grad=True, device=device)
            optimizer = optim.Adam([modifier], lr=self.learning_rate)
            
            bestl2 = torch.ones(batch_size, device=device) * 1e10
            bestscore = torch.ones(batch_size, device=device) * -1
            
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.binary_search_steps - 1:
                const = upper_bound
                
            prev = 1e6
            
            for iteration in range(self.max_iterations):
                optimizer.zero_grad()
                
                # newimg = tanh(modifier + imgs_tanh) / 2
                newimg = torch.tanh(modifier + imgs_tanh) / 2
                
                # Output logits
                output = self.model(newimg)
                
                # L2 distance loss
                # Sum over all pixels: sum((new-old)^2)
                l2dist = torch.sum((newimg - imgs)**2, dim=[1, 2, 3])
                
                # f function loss
                real = torch.sum(labs * output, dim=1)
                # max(others)
                # We want max of items NOT in target class.
                # subtraction trick: max(output - 10000*target)
                other = torch.max((1 - labs) * output - (labs * 10000), dim=1)[0]
                
                if self.targeted:
                    # minimize other - real (make real > other)
                    loss1 = torch.clamp(other - real + self.confidence, min=0.0)
                else:
                    # minimize real - other (make real < other)
                    loss1 = torch.clamp(real - other + self.confidence, min=0.0)
                
                loss_total = torch.sum(l2dist + const * loss1)
                
                loss_total.backward()
                optimizer.step()
                
                # Tracking
                l = loss_total.item()
                if self.abort_early and iteration % (self.max_iterations // 10) == 0:
                    if l > prev * 0.9999:
                        break
                    prev = l
                
                # Update best results
                with torch.no_grad():
                    preds = output.argmax(dim=1)
                    target_labels = labs.argmax(dim=1)
                    
                    # Success condition
                    if self.targeted:
                        success = (preds == target_labels)
                    else:
                        success = (preds != target_labels)
                    
                    # Update local best
                    mask = (l2dist < bestl2) & success
                    bestl2[mask] = l2dist[mask]
                    bestscore[mask] = preds[mask].float()
                    
                    # Update global best
                    mask = (l2dist < o_bestl2) & success
                    o_bestl2[mask] = l2dist[mask]
                    o_bestscore[mask] = preds[mask].float()
                    o_bestattack[mask] = newimg[mask]
            
            # Binary search step
            with torch.no_grad():
                target_labels = labs.argmax(dim=1)
                # Check if successful for this constant
                # We use bestscore != -1 as indicator of finding ANY solution in this step
                # But we essentially want to know if we found a solution for the CURRENT const
                # The logic in original code uses compare(bestscore[e], target)
                
                if self.targeted:
                    success = (bestscore == target_labels) & (bestscore != -1)
                else:
                    success = (bestscore != target_labels) & (bestscore != -1)
                
                # Adjust const
                # Success -> Decrease const (want smaller distance)
                # Failure -> Increase const (need more weight on classification)
                
                lower_bound = torch.where(success, lower_bound, torch.max(lower_bound, const))
                upper_bound = torch.where(success, torch.min(upper_bound, const), upper_bound)
                
                const_new = (lower_bound + upper_bound) / 2
                
                # If failure and upper_bound is still 1e10 (never found solution), multiply by 10
                mask_not_found = (~success) & (upper_bound >= 1e9)
                const_new[mask_not_found] = const[mask_not_found] * 10
                
                const = const_new
                
        return o_bestattack

class CarliniLID(CarliniL2):
    def attack_batch(self, imgs, labs):
        # Override to add LID loss
        device = imgs.device
        batch_size = imgs.size(0)
        
        imgs_tanh = torch.atanh(imgs * 1.999999).detach()
        
        lower_bound = torch.zeros(batch_size, device=device)
        const = torch.ones(batch_size, device=device) * self.initial_const
        upper_bound = torch.ones(batch_size, device=device) * 1e10
        
        o_bestl2 = torch.ones(batch_size, device=device) * 1e10
        o_bestscore = torch.ones(batch_size, device=device) * -1
        o_bestattack = imgs.clone()
        
        # Precompute clean logits for LID term
        with torch.no_grad():
            c_logits = self.model(imgs)
            
        for outer_step in range(self.binary_search_steps):
            modifier = torch.zeros_like(imgs, requires_grad=True, device=device)
            optimizer = optim.Adam([modifier], lr=self.learning_rate)
            
            bestl2 = torch.ones(batch_size, device=device) * 1e10
            bestscore = torch.ones(batch_size, device=device) * -1
            
            if self.repeat and outer_step == self.binary_search_steps - 1:
                const = upper_bound
                
            prev = 1e6
            
            for iteration in range(self.max_iterations):
                optimizer.zero_grad()
                newimg = torch.tanh(modifier + imgs_tanh) / 2
                output = self.model(newimg)
                
                l2dist = torch.sum((newimg - imgs)**2, dim=[1, 2, 3])
                
                real = torch.sum(labs * output, dim=1)
                other = torch.max((1 - labs) * output - (labs * 10000), dim=1)[0]
                
                if self.targeted:
                    loss1 = torch.clamp(other - real + self.confidence, min=0.0)
                else:
                    loss1 = torch.clamp(real - other + self.confidence, min=0.0)
                
                # LID Term
                # Note: lid_adv_term returns (batch_size,) tensor of LID values
                # We want to minimize LID? Original paper: "LID is high for adv".
                # The code in cw_attacks.py does: `self.loss = self.loss1 + self.loss2`
                # where `self.loss1 = tf.reduce_sum(self.const * (loss1 + loss_lid))`
                # So it minimizes (ClassLoss + LID).
                # This implies they want the adversarial example to have LOW LID?
                # Or they want to evade the detector?
                # If detector rejects high LID, then we want to minimize LID. Yes.
                
                loss_lid = lid_adv_term(c_logits, output, batch_size=batch_size)
                
                loss_total = torch.sum(l2dist + const * (loss1 + loss_lid))
                
                loss_total.backward()
                optimizer.step()
                
                l = loss_total.item()
                if self.abort_early and iteration % (self.max_iterations // 10) == 0:
                    if l > prev * 0.9999:
                        break
                    prev = l
                
                with torch.no_grad():
                    preds = output.argmax(dim=1)
                    target_labels = labs.argmax(dim=1)
                    
                    if self.targeted:
                        success = (preds == target_labels)
                    else:
                        success = (preds != target_labels)
                    
                    mask = (l2dist < bestl2) & success
                    bestl2[mask] = l2dist[mask]
                    bestscore[mask] = preds[mask].float()
                    
                    mask = (l2dist < o_bestl2) & success
                    o_bestl2[mask] = l2dist[mask]
                    o_bestscore[mask] = preds[mask].float()
                    o_bestattack[mask] = newimg[mask]
            
            # Binary Search Logic (Same as above)
            with torch.no_grad():
                target_labels = labs.argmax(dim=1)
                if self.targeted:
                    success = (bestscore == target_labels) & (bestscore != -1)
                else:
                    success = (bestscore != target_labels) & (bestscore != -1)
                
                lower_bound = torch.where(success, lower_bound, torch.max(lower_bound, const))
                upper_bound = torch.where(success, torch.min(upper_bound, const), upper_bound)
                const_new = (lower_bound + upper_bound) / 2
                mask_not_found = (~success) & (upper_bound >= 1e9)
                const_new[mask_not_found] = const[mask_not_found] * 10
                const = const_new
                
        return o_bestattack
