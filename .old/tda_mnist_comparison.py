import torch
import numpy as np
import os
import json
import argparse
from util import get_model, get_data
from tda_utils import (
    get_neuron_activations, 
    compute_correlation_matrix, 
    compute_persistence_diagrams, 
    extract_topological_features
)
from torch.utils.data import DataLoader, TensorDataset

PATH_DATA = "data/"
PATH_TDA = "data/tda/"

def run_tda_on_data(model, data_loader, device, name, dataset_name):
    print(f"Running TDA for {name}...")
    activations = get_neuron_activations(model, data_loader, device)
    print(f"  Activations shape: {activations.shape}")
    
    corr_matrix = compute_correlation_matrix(activations)
    print(f"  Correlation matrix computed.")
    
    rips_output = compute_persistence_diagrams(corr_matrix, maxdim=1)
    dgms = rips_output['dgms']
    print(f"  Persistence diagrams computed.")
    
    features = extract_topological_features(dgms)
    
    save_path = os.path.join(PATH_TDA, f"{name}_{dataset_name}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    serializable_dgms = [d.tolist() for d in dgms]
    output = {
        'features': features,
        'diagrams': serializable_dgms,
        'correlation_matrix': corr_matrix.tolist()
    }
    with open(save_path, 'w') as f:
        json.dump(output, f)
    print(f"  Results saved to {save_path}")
    return features, dgms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mnist', type=str)
    parser.add_argument('-a', '--attack', default='fgsm', type=str)
    parser.add_argument('-n', '--num_samples', default=500, type=int)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = get_model(args.dataset).to(device)
    model_path = os.path.join(PATH_DATA, f"model_{args.dataset}.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}")
    model.eval()
    
    # 2. Load Clean Data
    _, test_loader = get_data(args.dataset, batch_size=args.num_samples)
    clean_inputs, clean_labels = next(iter(test_loader))
    clean_loader = [(clean_inputs, clean_labels)]
    
    # 3. Load Adversarial Data
    adv_path = os.path.join(PATH_DATA, f"Adv_{args.dataset}_{args.attack}.npy")
    if os.path.exists(adv_path):
        adv_data = np.load(adv_path)
        # Take the same number of samples
        adv_inputs = torch.from_numpy(adv_data[:args.num_samples]).float()
        # We use the same labels for simplicity in the loader, though TDA doesn't use labels
        adv_loader = [(adv_inputs, clean_labels[:args.num_samples])]
        print(f"Loaded adversarial data from {adv_path}")
    else:
        print(f"Error: Adversarial data not found at {adv_path}")
        return

    # 4. Run TDA on both
    run_tda_on_data(model, clean_loader, device, "clean", args.dataset)
    run_tda_on_data(model, adv_loader, device, args.attack, args.dataset)
    
    print("\nTDA Comparison completed successfully.")

if __name__ == "__main__":
    main()
