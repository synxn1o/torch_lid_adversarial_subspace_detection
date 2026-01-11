import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity

# Add parent directory to path to import util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from util import mle_batch # Removed to use local robust implementation

# Add current directory to path to import train_NN
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_NN import BinaryNN, load_toy_data, prepare_data

def mle_batch(data, batch, k):
    """
    LID of a batch of query points X (batch) relative to data.
    Numpy/Scipy implementation with robust error handling.
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    
    def f(v):
        # v contains distances to k nearest neighbors
        # v[-1] is the max distance among neighbors (distance to k-th neighbor)
        if v[-1] < 1e-9:
            return 0.0 # Collapsed neighborhood
        
        # Avoid log(0) by clipping
        v = np.maximum(v, 1e-10)
        
        # Calculate LID: -k / sum(log(r_i/r_k))
        return - k / np.sum(np.log(v/v[-1]))
    
    # Calculate distances
    a = cdist(batch, data)
    
    # Sort and take k nearest neighbors
    # Note: We skip index 0 if it's 0 (distance to self), but cdist returns all.
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    
    # Apply LID calculation
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def get_noisy_samples(X, std=0.1):
    """Add Gaussian noise to samples."""
    noise = np.random.normal(loc=0, scale=std, size=X.shape)
    return X + noise

class FeatureExtractor(nn.Module):
    """Extract activations from intermediate layers."""
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.features = []
        
    def forward(self, x):
        self.features = []
        
        # Manually forward through BinaryNN to get features
        # x -> fc1 -> relu -> fc2 -> relu -> fc3
        
        # Input layer (LID often includes input)
        self.features.append(x.detach().cpu().numpy())
        
        # Layer 1
        out1 = self.model.fc1(x)
        act1 = self.model.relu(out1)
        self.features.append(act1.detach().cpu().numpy())
        
        # Layer 2
        out2 = self.model.fc2(act1)
        act2 = self.model.relu(out2)
        self.features.append(act2.detach().cpu().numpy())
        
        # Layer 3 (Logits)
        out3 = self.model.fc3(act2)
        self.features.append(out3.detach().cpu().numpy())
        
        return self.features

def get_lids_random_batch(model, X, X_noisy, X_adv, k=20, batch_size=100, device='cpu'):
    """Extract LID characteristics."""
    extractor = FeatureExtractor(model)
    model.eval()
    
    n_batches = int(np.ceil(X.shape[0] / batch_size))
    
    lids = []
    lids_noisy = []
    lids_adv = []
    
    for i in tqdm(range(n_batches), desc="LID Extraction"):
        start = i * batch_size
        end = min((i + 1) * batch_size, X.shape[0])
        
        # Get batches
        b_X = torch.FloatTensor(X[start:end]).to(device)
        b_noisy = torch.FloatTensor(X_noisy[start:end]).to(device)
        b_adv = torch.FloatTensor(X_adv[start:end]).to(device)
        
        # Get features
        # features list: [input, layer1, layer2, logits]
        clean_feats = extractor(b_X)
        noisy_feats = extractor(b_noisy)
        adv_feats = extractor(b_adv)
        
        lid_dim = len(clean_feats)
        curr_batch_size = len(b_X)
        
        b_lids = np.zeros((curr_batch_size, lid_dim))
        b_lids_noisy = np.zeros((curr_batch_size, lid_dim))
        b_lids_adv = np.zeros((curr_batch_size, lid_dim))
        
        for l in range(lid_dim):
            f_clean = clean_feats[l].reshape(curr_batch_size, -1)
            f_noisy = noisy_feats[l].reshape(curr_batch_size, -1)
            f_adv = adv_feats[l].reshape(curr_batch_size, -1)
            
            # MLE batch estimation
            # LID of X relative to X
            b_lids[:, l] = mle_batch(f_clean, f_clean, k=k)
            # LID of Noisy relative to X
            b_lids_noisy[:, l] = mle_batch(f_clean, f_noisy, k=k)
            # LID of Adv relative to X
            b_lids_adv[:, l] = mle_batch(f_clean, f_adv, k=k)
            
        lids.append(b_lids)
        lids_noisy.append(b_lids_noisy)
        lids_adv.append(b_lids_adv)
        
    lids = np.concatenate(lids, axis=0)
    lids_noisy = np.concatenate(lids_noisy, axis=0)
    lids_adv = np.concatenate(lids_adv, axis=0)
    
    return lids, lids_noisy, lids_adv

def get_kd(model, X_train, y_train, X, X_noisy, X_adv, batch_size=100, device='cpu'):
    """Extract Kernel Density characteristics."""
    extractor = FeatureExtractor(model)
    model.eval()
    
    # 1. Extract features for training set
    print("  Extracting training features for KDE...")
    train_feats_by_layer = []
    
    # Process training data in batches
    n_train_batches = int(np.ceil(len(X_train) / batch_size))
    for i in range(n_train_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_train))
        b_X = torch.FloatTensor(X_train[start:end]).to(device)
        feats = extractor(b_X)
        
        if i == 0:
            for _ in feats:
                train_feats_by_layer.append([])
        
        for l, f in enumerate(feats):
            train_feats_by_layer[l].append(f)
            
    # Concatenate training features
    for l in range(len(train_feats_by_layer)):
        train_feats_by_layer[l] = np.concatenate(train_feats_by_layer[l], axis=0)
        
    # 2. Fit KDEs per class per layer
    print("  Fitting KDEs...")
    kdes = {} # kdes[layer][class_label]
    num_layers = len(train_feats_by_layer)
    classes = np.unique(y_train)
    
    for l in range(num_layers):
        kdes[l] = {}
        for c in classes:
            # Filter training features for this class
            class_data = train_feats_by_layer[l][y_train == c]
            
            # Fit KDE
            # Bandwidth selection is tricky. Using a heuristic or fixed value.
            # Original code uses fixed bandwidths. We'll use 1.0 or heuristic.
            # Simple heuristic: std dev of data * (4/3n)^1/5 (Scott's Rule)
            # Or just fix it to something small like 0.1 or 1.0 depending on scale.
            # Since our data is small scale (-1 to 1 mostly), maybe 0.2?
            # Let's try to estimate or pick a safe one.
            bw = 0.2 
            kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(class_data)
            kdes[l][c] = kde
            
    # 3. Score samples
    def score_batch(batch_X, batch_y_pred):
        # batch_y_pred is needed to know which class KDE to score against
        scores = np.zeros((len(batch_X), num_layers))
        
        b_tensor = torch.FloatTensor(batch_X).to(device)
        feats = extractor(b_tensor)
        
        for l in range(num_layers):
            f_flat = feats[l].reshape(len(batch_X), -1)
            
            for i in range(len(batch_X)):
                pred_c = batch_y_pred[i]
                # Score against predicted class
                # score_samples returns log-density
                scores[i, l] = kdes[l][pred_c].score_samples(f_flat[i].reshape(1, -1))[0]
        return scores

    # We need predictions for the samples to decide which KDE to check against
    # (Assuming we trust the model's prediction, or we check against the predicted class)
    # The original paper checks against the predicted class.
    
    def get_preds(batch_X):
        b_tensor = torch.FloatTensor(batch_X).to(device)
        logits = model(b_tensor)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().flatten()
        return preds

    print("  Scoring samples...")
    # Helper to process a whole set
    def process_set(data_X):
        n_batches = int(np.ceil(len(data_X) / batch_size))
        all_scores = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(data_X))
            b_X = data_X[start:end]
            
            preds = get_preds(b_X)
            scores = score_batch(b_X, preds)
            all_scores.append(scores)
        return np.concatenate(all_scores, axis=0)

    kd_clean = process_set(X)
    kd_noisy = process_set(X_noisy)
    kd_adv = process_set(X_adv)
    
    return kd_clean, kd_noisy, kd_adv

def get_km(model, X_train, y_train, X, X_noisy, X_adv, k=20, batch_size=100, device='cpu'):
    """Extract K-Means (Distance to k-nearest in same class) characteristics."""
    extractor = FeatureExtractor(model)
    model.eval()
    
    # 1. Extract training features
    print("  Extracting training features for KM...")
    train_feats_by_layer = []
    n_train_batches = int(np.ceil(len(X_train) / batch_size))
    for i in range(n_train_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_train))
        b_X = torch.FloatTensor(X_train[start:end]).to(device)
        feats = extractor(b_X)
        if i == 0:
            for _ in feats: train_feats_by_layer.append([])
        for l, f in enumerate(feats):
            train_feats_by_layer[l].append(f)
            
    for l in range(len(train_feats_by_layer)):
        train_feats_by_layer[l] = np.concatenate(train_feats_by_layer[l], axis=0)
        
    num_layers = len(train_feats_by_layer)
    
    # 2. Score samples
    def score_batch(batch_X, batch_y_pred):
        scores = np.zeros((len(batch_X), num_layers))
        b_tensor = torch.FloatTensor(batch_X).to(device)
        feats = extractor(b_tensor)
        
        for l in range(num_layers):
            f_flat = feats[l].reshape(len(batch_X), -1)
            train_f_flat = train_feats_by_layer[l].reshape(len(X_train), -1)
            
            # For each sample, compute distance to k nearest neighbors in the PREDICTED class
            # This is slow if done one by one.
            # Optimization: Pre-compute distance matrix? No, too big.
            # We can separate training data by class.
            
            # Split train data by class
            train_f_c0 = train_f_flat[y_train == 0]
            train_f_c1 = train_f_flat[y_train == 1]
            
            for i in range(len(batch_X)):
                pred_c = batch_y_pred[i]
                target_train = train_f_c1 if pred_c == 1 else train_f_c0
                
                # Compute distances to target_train
                # Using cdist for single point
                dists = cdist(f_flat[i].reshape(1, -1), target_train)[0]
                
                # Mean distance to k nearest
                k_actual = min(k, len(dists))
                dists.sort()
                scores[i, l] = np.mean(dists[:k_actual])
                
        return scores

    def get_preds(batch_X):
        b_tensor = torch.FloatTensor(batch_X).to(device)
        logits = model(b_tensor)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().flatten()
        return preds

    print("  Calculaing KM distances...")
    def process_set(data_X):
        n_batches = int(np.ceil(len(data_X) / batch_size))
        all_scores = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(data_X))
            b_X = data_X[start:end]
            preds = get_preds(b_X)
            scores = score_batch(b_X, preds)
            all_scores.append(scores)
        return np.concatenate(all_scores, axis=0)

    km_clean = process_set(X)
    km_noisy = process_set(X_noisy)
    km_adv = process_set(X_adv)
    
    return km_clean, km_noisy, km_adv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attack', default='all', type=str, help='Attack to process (fgsm, bim-a, etc. or all)')
    parser.add_argument('-r', '--characteristic', default='lid', type=str, choices=['lid', 'kd', 'km', 'all'])
    parser.add_argument('-k', '--k_nearest', default=20, type=int)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    args = parser.parse_args()
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data/circle_dataset.pkl")
    model_path = os.path.join(base_dir, "models/toy_binary_nn.pth")
    adv_results_path = os.path.join(base_dir, "results/adversarial_results.pkl")
    output_dir = os.path.join(base_dir, "data/characteristics")
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    print("Loading model...")
    model = BinaryNN(input_dim=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Load Data (Needed for KD/KM)
    X_train, y_train = None, None
    if args.characteristic in ['kd', 'km', 'all']:
        print("Loading training data for KD/KM...")
        # We use load_toy_data and prepare_data to get the exact training split used
        X_all, _, y_binary = load_toy_data(data_path)
        # Note: We must ensure this split matches what the model was trained on.
        # Since random_state is fixed (42) in train_NN.py, it should match.
        X_train, _, _, y_train, _, _ = prepare_data(X_all, y_binary)
        print(f"  Training data shape: {X_train.shape}")
    
    # 3. Load Adversarial Results
    print(f"Loading adversarial results from {adv_results_path}...")
    if not os.path.exists(adv_results_path):
        raise FileNotFoundError("Adversarial results not found. Run generate_adversarial.py first.")
        
    with open(adv_results_path, 'rb') as f:
        adv_results = pickle.load(f)
        
    X_clean = adv_results['clean']
    y_labels = adv_results['labels']
    attack_dict = adv_results['attacks']
    
    # Select attacks
    if args.attack == 'all':
        attacks_to_process = list(attack_dict.keys())
    else:
        if args.attack not in attack_dict:
            raise ValueError(f"Attack {args.attack} not found in results.")
        attacks_to_process = [args.attack]
        
    # Generate Noisy Samples (once)
    print("Generating noisy samples...")
    X_noisy = get_noisy_samples(X_clean, std=0.2)
    
    # 4. Process each attack
    for attack_name in attacks_to_process:
        print(f"\nProcessing {attack_name}...")
        
        X_adv = attack_dict[attack_name]['examples']
        
        # Filter correctly classified
        with torch.no_grad():
            clean_tensor = torch.FloatTensor(X_clean).to(device)
            logits = model(clean_tensor)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().flatten()
            correct_mask = (preds == y_labels)
            
        print(f"  Correctly classified samples: {np.sum(correct_mask)}/{len(X_clean)}")
        
        X_clean_filt = X_clean[correct_mask]
        X_noisy_filt = X_noisy[correct_mask]
        X_adv_filt = X_adv[correct_mask]
        # y_filt = y_labels[correct_mask] # Not used currently
        
        # Define tasks
        tasks = []
        if args.characteristic == 'all':
            tasks = ['lid', 'kd', 'km']
        else:
            tasks = [args.characteristic]
            
        for char_type in tasks:
            print(f"  Extracting {char_type.upper()}...")
            
            if char_type == 'lid':
                f_clean, f_noisy, f_adv = get_lids_random_batch(
                    model, X_clean_filt, X_noisy_filt, X_adv_filt,
                    k=args.k_nearest, batch_size=args.batch_size, device=device
                )
            elif char_type == 'kd':
                f_clean, f_noisy, f_adv = get_kd(
                    model, X_train, y_train, X_clean_filt, X_noisy_filt, X_adv_filt,
                    batch_size=args.batch_size, device=device
                )
            elif char_type == 'km':
                f_clean, f_noisy, f_adv = get_km(
                    model, X_train, y_train, X_clean_filt, X_noisy_filt, X_adv_filt,
                    k=args.k_nearest, batch_size=args.batch_size, device=device
                )
                
            # Save results
            # Positive samples (Adversarial)
            X_pos = f_adv
            y_pos = np.ones((len(X_pos), 1))
            
            # Negative samples (Clean + Noisy)
            X_neg = np.concatenate((f_clean, f_noisy), axis=0)
            y_neg = np.zeros((len(X_neg), 1))
            
            # Combine
            X_combined = np.concatenate((X_pos, X_neg), axis=0)
            y_combined = np.concatenate((y_pos, y_neg), axis=0)
            
            data_to_save = np.concatenate((X_combined, y_combined), axis=1)
            
            save_path = os.path.join(output_dir, f"{char_type}_toy_{attack_name}.npy")
            np.save(save_path, data_to_save)
            print(f"  Saved {char_type.upper()} characteristics to {save_path}")

if __name__ == "__main__":
    main()
