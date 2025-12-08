import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from util import get_data, get_model, FeatureExtractor, mle_batch, CLIP_MIN, CLIP_MAX
import scipy.io as sio

PATH_DATA = "data/"

# Standard deviations for noisy samples (from original code)
STDEVS = {
    'mnist': {'fgsm': 0.264, 'bim-a': 0.111, 'bim-b': 0.184, 'cw-l2': 0.588},
    'cifar': {'fgsm': 0.0504, 'bim-a': 0.0087, 'bim-b': 0.0439, 'cw-l2': 0.015},
    'svhn': {'fgsm': 0.1332, 'bim-a': 0.015, 'bim-b': 0.1024, 'cw-l2': 0.0379}
}

# Bandwidths for KDE (from original code)
BANDWIDTHS = {'mnist': 3.7926, 'cifar': 0.26, 'svhn': 1.00}

def get_noisy_samples(X_test, dataset, attack):
    """
    Add Gaussian noise to X_test
    """
    # If attack not in dictionary (e.g. cw-lid), use a default or cw-l2
    if attack not in STDEVS[dataset]:
        std = STDEVS[dataset]['cw-l2'] # Default
    else:
        std = STDEVS[dataset][attack]
        
    noise = np.random.normal(loc=0, scale=std, size=X_test.shape)
    X_noisy = np.clip(X_test + noise, CLIP_MIN, CLIP_MAX)
    return X_noisy.astype(np.float32)

def get_deep_representations(model, loader, device):
    """
    Get output of penultimate layer (before logits)
    """
    model.eval()
    features = []
    # We need to define what is "deep representation". 
    # Usually the input to the final Linear layer.
    # We can use a hook on the input of the last fc layer, or output of second to last.
    # For simplicity, let's hook the input of the last layer.
    
    last_layer = list(model.children())[-1]
    # If model is defined as in util.py, last layer is 'fc2' (MNIST) or 'fc3' (CIFAR/SVHN).
    # Let's try to identify it dynamically or hardcode based on model type.
    
    # Generic approach: register hook on final linear layer's input.
    
    deep_feats = []
    def hook(module, input, output):
        deep_feats.append(input[0].detach().cpu().numpy())
        
    # Find last Linear layer
    modules = list(model.named_modules())
    last_linear_name = None
    last_linear_module = None
    for name, m in reversed(modules):
        if isinstance(m, nn.Linear):
            last_linear_name = name
            last_linear_module = m
            break
            
    handle = last_linear_module.register_forward_hook(hook)
    
    for inputs, _ in loader:
        inputs = inputs.to(device)
        deep_feats[:] = [] # Clear buffer
        _ = model(inputs)
        # deep_feats now has [batch_size, features]
        features.append(np.concatenate(deep_feats, axis=0))
        
    handle.remove()
    return np.concatenate(features, axis=0)

def get_mc_predictions(model, inputs_tensor, nb_iter=50, batch_size=256):
    """
    Get Monte Carlo predictions (enable dropout)
    """
    device = inputs_tensor.device
    model.train() # Enable dropout
    
    # We need to ensure ONLY dropout is active, but Batch Norm statistics should ideally be frozen (eval mode).
    # PyTorch 'train()' enables both. 
    # To do MC Dropout correctly: model.eval(), then manually set dropout layers to train.
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            
    preds = []
    n_batches = int(np.ceil(inputs_tensor.size(0) / batch_size))
    
    for i in range(nb_iter):
        iter_preds = []
        for b in range(n_batches):
            batch = inputs_tensor[b*batch_size : (b+1)*batch_size]
            with torch.no_grad():
                output = model(batch)
            iter_preds.append(output.cpu().numpy())
        preds.append(np.concatenate(iter_preds, axis=0))
        
    return np.array(preds) # (nb_iter, N, classes)

def get_lids_random_batch(model, X, X_noisy, X_adv, dataset, k=20, batch_size=100, device='cpu'):
    """
    Extract LID characteristics
    """
    # Convert to loaders or process in batches
    # We need to extract features for all 3 sets.
    # Use FeatureExtractor
    
    extractor = FeatureExtractor(model)
    model.eval()
    
    # Helper to get features for a batch
    def get_batch_features(batch_x):
        batch_x = torch.from_numpy(batch_x).to(device)
        feats = extractor(batch_x) # List of tensors
        return [f.detach().cpu().numpy() for f in feats]

    n_batches = int(np.ceil(X.shape[0] / batch_size))
    
    lids = []
    lids_noisy = []
    lids_adv = []
    
    for i in tqdm(range(n_batches), desc="LID Extraction"):
        start = i * batch_size
        end = min((i + 1) * batch_size, X.shape[0])
        
        batch_X = X[start:end]
        batch_noisy = X_noisy[start:end]
        batch_adv = X_adv[start:end]
        
        # Get features
        # Note: original code calculates LID for a batch relative to ITSELF (random batch).
        # "LID estimated by k close neighbours in the random batch it lies in."
        
        # We process Clean, Noisy, Adv separately?
        # The code computes:
        # lid_batch[:, i] = mle_batch(X_act, X_act, k=k)
        # lid_batch_adv[:, i] = mle_batch(X_act, X_adv_act, k=k) ??
        # Wait, the original code:
        # lid_batch_adv[:, i] = mle_batch(X_act, X_adv_act, k=k) 
        # It computes LID of Adv samples relative to Clean samples in the batch?
        # Yes: `mle_batch(data, batch)` -> data is reference, batch is query.
        # So reference is always X_act (clean).
        
        clean_feats = get_batch_features(batch_X)
        noisy_feats = get_batch_features(batch_noisy)
        adv_feats = get_batch_features(batch_adv)
        
        # Number of layers
        lid_dim = len(clean_feats)
        
        b_lids = np.zeros((len(batch_X), lid_dim))
        b_lids_noisy = np.zeros((len(batch_X), lid_dim))
        b_lids_adv = np.zeros((len(batch_X), lid_dim))
        
        for l in range(lid_dim):
            f_clean = clean_feats[l].reshape(len(batch_X), -1)
            f_noisy = noisy_feats[l].reshape(len(batch_X), -1)
            f_adv = adv_feats[l].reshape(len(batch_X), -1)
            
            # LID of clean relative to clean
            b_lids[:, l] = mle_batch(f_clean, f_clean, k=k)
            
            # LID of noisy relative to clean
            b_lids_noisy[:, l] = mle_batch(f_clean, f_noisy, k=k)
            
            # LID of adv relative to clean
            b_lids_adv[:, l] = mle_batch(f_clean, f_adv, k=k)
            
        lids.append(b_lids)
        lids_noisy.append(b_lids_noisy)
        lids_adv.append(b_lids_adv)
        
    lids = np.concatenate(lids, axis=0)
    lids_noisy = np.concatenate(lids_noisy, axis=0)
    lids_adv = np.concatenate(lids_adv, axis=0)
    
    extractor.close()
    return lids, lids_noisy, lids_adv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str)
    parser.add_argument('-a', '--attack', required=True, type=str)
    parser.add_argument('-r', '--characteristic', required=True, type=str, choices=['kd', 'bu', 'lid', 'all'])
    parser.add_argument('-k', '--k_nearest', default=20, type=int)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = get_model(args.dataset).to(device)
    model.load_state_dict(torch.load(os.path.join(PATH_DATA, f"model_{args.dataset}.pth"), map_location=device))
    model.eval()
    
    # Load Data
    # We need X_train (for KDE), X_test, Y_test
    # get_data returns loaders. We can extract arrays.
    train_loader, test_loader = get_data(args.dataset, batch_size=args.batch_size, augmentation=False)
    
    # Helper to get all data from loader
    def loader_to_numpy(loader):
        X = []
        Y = []
        for x, y in loader:
            X.append(x.numpy())
            Y.append(y.numpy()) # Indices
        return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)

    print("Loading data...")
    X_test, Y_test_indices = loader_to_numpy(test_loader)
    
    # Load Adversarial
    adv_path = os.path.join(PATH_DATA, f"Adv_{args.dataset}_{args.attack}.npy")
    if not os.path.exists(adv_path):
        raise FileNotFoundError(f"Adversarial examples not found at {adv_path}")
    X_adv = np.load(adv_path)
    
    # Generate/Load Noisy
    noisy_path = os.path.join(PATH_DATA, f"Noisy_{args.dataset}_{args.attack}.npy")
    if os.path.exists(noisy_path):
        X_noisy = np.load(noisy_path)
    else:
        print("Generating noisy samples...")
        X_noisy = get_noisy_samples(X_test, args.dataset, args.attack)
        np.save(noisy_path, X_noisy)
        
    # Truncate to match lengths (if adv generated on subset, though craft_adv uses full test)
    min_len = min(len(X_test), len(X_adv), len(X_noisy))
    X_test = X_test[:min_len]
    X_adv = X_adv[:min_len]
    X_noisy = X_noisy[:min_len]
    Y_test_indices = Y_test_indices[:min_len]
    
    # Only use correctly classified clean samples
    print("Filtering correctly classified samples...")
    
    # We need to batch predict
    def predict_batch(X):
        model.eval()
        preds = []
        bs = args.batch_size
        for i in range(0, len(X), bs):
            bx = torch.from_numpy(X[i:i+bs]).to(device)
            with torch.no_grad():
                out = model(bx)
            preds.append(out.argmax(1).cpu().numpy())
        return np.concatenate(preds)

    preds = predict_batch(X_test)
    correct_idxs = np.where(preds == Y_test_indices)[0]
    
    X_test = X_test[correct_idxs]
    X_adv = X_adv[correct_idxs]
    X_noisy = X_noisy[correct_idxs]
    Y_test_indices = Y_test_indices[correct_idxs]
    
    print(f"Using {len(X_test)} correctly classified samples.")
    
    def merge_and_save(pos, neg, name):
        # pos: adv, neg: normal + noisy
        X_pos = pos
        X_neg = neg
        X = np.concatenate((X_pos, X_neg), axis=0)
        y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))
        
        data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        save_name = os.path.join(PATH_DATA, f"{name}_{args.dataset}_{args.attack}.npy")
        np.save(save_name, data)
        print(f"Saved to {save_name}")

    # Characteristics
    if args.characteristic in ['lid', 'all']:
        lids_normal, lids_noisy, lids_adv = get_lids_random_batch(model, X_test, X_noisy, X_adv, 
                                                                  args.dataset, k=args.k_nearest, 
                                                                  batch_size=args.batch_size, device=device)
        
        # Merge: Pos=Adv, Neg=Normal+Noisy
        lids_pos = lids_adv
        lids_neg = np.concatenate((lids_normal, lids_noisy), axis=0)
        merge_and_save(lids_pos, lids_neg, 'lid')

    if args.characteristic in ['kd', 'all']:
        # Needs X_train for fitting KDE
        print("Loading training data for KDE...")
        X_train, Y_train_indices = loader_to_numpy(train_loader)
        
        # Extract features
        # To save time, maybe subsample X_train? Original code uses full.
        # But feature extraction on 60k images takes time.
        
        # Wrapper to get deep feats for numpy array
        def get_feats_numpy(X):
            ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.zeros(len(X))) # Dummy y
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)
            return get_deep_representations(model, dl, device)

        print("Extracting features...")
        feats_train = get_feats_numpy(X_train)
        feats_test = get_feats_numpy(X_test)
        feats_noisy = get_feats_numpy(X_noisy)
        feats_adv = get_feats_numpy(X_adv)
        
        # Train KDE per class
        kdes = {}
        for i in range(10):
            class_subset = feats_train[Y_train_indices == i]
            kdes[i] = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTHS[args.dataset]).fit(class_subset)
            
        # Score
        def score_samples(kdes, feats, preds):
            densities = []
            for i in range(len(feats)):
                label = preds[i]
                densities.append(kdes[label].score_samples(feats[i].reshape(1, -1))[0])
            return np.array(densities).reshape(-1, 1)

        # We need predictions for test samples to know which KDE to use
        # Original code uses model predictions.
        preds_test = predict_batch(X_test)
        preds_noisy = predict_batch(X_noisy)
        preds_adv = predict_batch(X_adv)
        
        densities_normal = score_samples(kdes, feats_test, preds_test)
        densities_noisy = score_samples(kdes, feats_noisy, preds_noisy)
        densities_adv = score_samples(kdes, feats_adv, preds_adv)
        
        merge_and_save(densities_adv, np.concatenate((densities_normal, densities_noisy), axis=0), 'kd')

    if args.characteristic in ['bu', 'all']:
        print("Calculating Bayesian Uncertainty...")
        # Variance of predictions
        # get_mc_predictions returns (iter, N, classes)
        # We want mean variance?
        # Original code: .var(axis=0).mean(axis=1) -> Variance across iterations, then mean across classes?
        
        def get_bu_values(X):
            X_tensor = torch.from_numpy(X).to(device)
            preds = get_mc_predictions(model, X_tensor, nb_iter=50, batch_size=args.batch_size)
            return preds.var(axis=0).mean(axis=1).reshape(-1, 1)
        
        bu_normal = get_bu_values(X_test)
        bu_noisy = get_bu_values(X_noisy)
        bu_adv = get_bu_values(X_adv)
        
        merge_and_save(bu_adv, np.concatenate((bu_normal, bu_noisy), axis=0), 'bu')

if __name__ == "__main__":
    main()
