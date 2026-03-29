import argparse
import os
import torch
import numpy as np
import json
from util import get_data, get_model
from tda_utils import (
    get_neuron_activations, 
    compute_correlation_matrix, 
    compute_persistence_diagrams, 
    extract_topological_features
)

PATH_DATA = "data/"
PATH_TDA = "data/tda/"

def run_tda_pipeline(model, loader, device, save_path=None):
    """
    Run the full TDA pipeline for a given model and data loader.
    """
    print("Extracting neuron activations...")
    activations = get_neuron_activations(model, loader, device)
    print(f"Activations shape: {activations.shape}")
    
    print("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(activations)
    
    print("Computing persistence diagrams (this may take a while)...")
    rips_output = compute_persistence_diagrams(corr_matrix, maxdim=2)
    dgms = rips_output['dgms']
    
    print("Extracting topological features...")
    features = extract_topological_features(dgms)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save diagrams and features
        # We convert diagrams to list for JSON serialization
        serializable_dgms = [d.tolist() for d in dgms]
        output = {
            'features': features,
            'diagrams': serializable_dgms
        }
        with open(save_path, 'w') as f:
            json.dump(output, f)
        print(f"Results saved to {save_path}")
        
    return features, dgms

def main():
    parser = argparse.ArgumentParser(description="TDA-based Trojan Detector")
    parser.add_argument('-d', '--dataset', default='mnist', type=str)
    parser.add_argument('-m', '--model_path', type=str, help="Path to the model to analyze")
    parser.add_argument('-n', '--num_samples', default=500, type=int, help="Number of samples to use for activations")
    parser.add_argument('--name', type=str, default='model_analysis', help="Name for the output file")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = get_model(args.dataset).to(device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        default_model_path = os.path.join(PATH_DATA, f"model_{args.dataset}.pth")
        if os.path.exists(default_model_path):
            model.load_state_dict(torch.load(default_model_path, map_location=device))
        else:
            print(f"Warning: No model found at {default_model_path}. Using uninitialized model.")
    
    # Load Data
    _, test_loader = get_data(args.dataset, batch_size=args.num_samples)
    # Get a single batch of samples
    data_iter = iter(test_loader)
    inputs, labels = next(data_iter)
    # Create a small loader for this batch
    small_loader = [(inputs, labels)]
    
    save_path = os.path.join(PATH_TDA, f"{args.name}_{args.dataset}.json")
    features, dgms = run_tda_pipeline(model, small_loader, device, save_path=save_path)
    
    print("\nTopological Features Summary:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
