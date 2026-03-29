import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from util import get_data, get_model
from attacks import fgsm, bim, jsma_single
from cw_attacks import CarliniL2, CarliniLID

# Attack Parameters
ATTACK_PARAMS = {
    'mnist': {'eps': 0.40, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010}
}
PATH_DATA = "data/"

def craft_one_type(dataset, attack, batch_size, device):
    print(f"Crafting {attack} examples for {dataset}...")
    
    # Load model
    model = get_model(dataset).to(device)
    model_path = os.path.join(PATH_DATA, f"model_{dataset}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    _, test_loader = get_data(dataset, batch_size=batch_size, augmentation=False)
    
    adv_samples = []
    
    # For C&W, we might want to limit samples if it's too slow, but original runs on full test set (or subsets).
    # "svhn has 26032 test images... batch_size for cw-l2 should be 16"
    
    for inputs, labels in tqdm(test_loader, desc=f"Generating {attack}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # One-hot labels for attacks that expect them (CW, BIM implementation uses argmax internally usually)
        # My BIM/FGSM implementation uses standard CrossEntropy which expects indices, but let's check.
        # attacks.py: criterion(outputs, torch.argmax(y, dim=1)) -> expects one-hot Y.
        # But DataLoader returns indices.
        # So I need to one-hot encode labels.
        
        nb_classes = 10
        labels_onehot = torch.zeros(labels.size(0), nb_classes, device=device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        if attack == 'fgsm':
            x_adv = fgsm(model, inputs, labels_onehot, eps=ATTACK_PARAMS[dataset]['eps'])
            
        elif attack == 'bim-a':
            x_adv = bim(model, inputs, labels_onehot, eps=ATTACK_PARAMS[dataset]['eps'],
                        eps_iter=ATTACK_PARAMS[dataset]['eps_iter'], mode='first')
            
        elif attack == 'bim-b':
            x_adv = bim(model, inputs, labels_onehot, eps=ATTACK_PARAMS[dataset]['eps'],
                        eps_iter=ATTACK_PARAMS[dataset]['eps_iter'], mode='last')
            
        elif attack == 'jsma':
            # JSMA is slow, usually done one by one
            x_adv = []
            for i in range(inputs.size(0)):
                # Target: random other class
                curr_y = labels[i].item()
                target = np.random.choice([c for c in range(nb_classes) if c != curr_y])
                
                res = jsma_single(model, inputs[i], target_class=target, theta=1.0, gamma=0.1)
                x_adv.append(res)
            x_adv = torch.stack(x_adv)
            
        elif attack == 'cw-l2':
            cw = CarliniL2(model, image_size=inputs.size(2), num_channels=inputs.size(1), num_labels=nb_classes,
                           batch_size=batch_size, targeted=True)
            # attack_batch expects batch
            x_adv = cw.attack_batch(inputs, labels_onehot)
            
        elif attack == 'cw-lid':
            cw = CarliniLID(model, image_size=inputs.size(2), num_channels=inputs.size(1), num_labels=nb_classes,
                           batch_size=batch_size, targeted=True)
            x_adv = cw.attack_batch(inputs, labels_onehot)
            
        else:
            raise ValueError(f"Unknown attack: {attack}")
            
        adv_samples.append(x_adv.cpu().numpy())
    
    # Concatenate and save
    adv_samples = np.concatenate(adv_samples, axis=0)
    save_path = os.path.join(PATH_DATA, f"Adv_{dataset}_{attack}.npy")
    np.save(save_path, adv_samples)
    print(f"Saved adversarial examples to {save_path}")
    
    # Evaluate
    # We need to reload data as numpy or use loader again?
    # Quick eval using the generated array and iterating loader again?
    # Or just construct tensor from numpy.
    
    # Let's verify accuracy
    print("Evaluating...")
    test_total = 0
    test_correct = 0
    
    # We need to iterate loader again to get labels
    # Optimization: we could have stored labels.
    
    # Re-instantiate loader
    _, test_loader_eval = get_data(dataset, batch_size=batch_size, augmentation=False)
    
    adv_tensor = torch.from_numpy(adv_samples)
    ptr = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader_eval:
            batch_len = inputs.size(0)
            if ptr + batch_len > len(adv_tensor):
                break
                
            batch_adv = adv_tensor[ptr : ptr+batch_len].to(device)
            labels = labels.to(device)
            
            outputs = model(batch_adv)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            ptr += batch_len
            
    acc = 100. * test_correct / test_total
    print(f"Model accuracy on {attack} adversarial examples: {acc:.2f}%")
    
    # L2 diff
    # Clean data is not stored here easily without re-loading. 
    # But original code computed it.
    # We can skip for now or re-load.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str, choices=['mnist', 'cifar', 'svhn'])
    parser.add_argument('-a', '--attack', required=True, type=str, choices=['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'cw-lid', 'all'])
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.attack == 'all':
        attacks = ['fgsm', 'bim-a', 'bim-b', 'cw-l2'] # JSMA omitted for time
        for att in attacks:
            craft_one_type(args.dataset, att, args.batch_size, device)
    else:
        craft_one_type(args.dataset, args.attack, args.batch_size, device)

if __name__ == "__main__":
    main()
