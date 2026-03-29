import torch
import torch.nn as nn
import numpy as np
from ripser import ripser
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional

def get_neuron_activations(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> np.ndarray:
    """
    Extract activations for all neurons in the network.
    Uses Global Average Pooling for convolutional layers to keep the number of neurons manageable.
    """
    model.eval()
    activations = []
    
    def get_hook(name, layer_type):
        def hook(module, input, output):
            # output shape: [batch, channels, h, w] for conv, [batch, features] for linear
            if layer_type == 'conv':
                # Global Average Pooling: [batch, channels, h, w] -> [batch, channels]
                act = torch.mean(output, dim=(2, 3))
            else:
                act = output
            activations_dict[name].append(act.detach().cpu().numpy())
        return hook

    activations_dict = {}
    hooks = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            activations_dict[name] = []
            hooks.append(module.register_forward_hook(get_hook(name, 'conv')))
        elif isinstance(module, nn.Linear):
            activations_dict[name] = []
            hooks.append(module.register_forward_hook(get_hook(name, 'linear')))

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            model(inputs)

    for h in hooks:
        h.remove()

    # Concatenate all layer activations into a single matrix [num_samples, total_neurons]
    all_layer_acts = []
    for name in activations_dict:
        layer_act = np.concatenate(activations_dict[name], axis=0)
        all_layer_acts.append(layer_act)
    
    return np.concatenate(all_layer_acts, axis=1)

def compute_correlation_matrix(activations: np.ndarray) -> np.ndarray:
    """
    Compute the Pearson correlation matrix between neurons.
    activations: [n_samples, m_neurons]
    Returns: [m_neurons, m_neurons]
    """
    # np.corrcoef expects [m_features, n_observations]
    return np.corrcoef(activations.T)

def compute_persistence_diagrams(correlation_matrix: np.ndarray, maxdim: int = 2) -> Dict:
    """
    Compute Vietoris-Rips filtration and persistence diagrams.
    correlation_matrix: [m, m]
    Returns: Ripser output dictionary
    """
    # Distance matrix D = 1 - Correlation
    # Ensure we handle potential NaN from zero variance neurons
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    distance_matrix = 1.0 - correlation_matrix
    # Ensure diagonal is exactly 0
    np.fill_diagonal(distance_matrix, 0)
    # Ensure symmetry and non-negativity
    distance_matrix = np.maximum(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    
    return ripser(distance_matrix, distance_matrix=True, maxdim=maxdim)

def extract_topological_features(dgms: List[np.ndarray]) -> Dict[str, float]:
    """
    Extract features from persistence diagrams.
    dgms: List of diagrams (one per dimension)
    """
    features = {}
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            features[f'dim{dim}_max_persistence'] = 0.0
            features[f'dim{dim}_avg_midlife'] = 0.0
            features[f'dim{dim}_avg_death'] = 0.0
            continue
            
        # Filter out infinite death times if any (usually only in dim 0)
        finite_dgm = dgm[np.isfinite(dgm[:, 1])]
        if len(finite_dgm) == 0:
            features[f'dim{dim}_max_persistence'] = 0.0
            features[f'dim{dim}_avg_midlife'] = 0.0
            features[f'dim{dim}_avg_death'] = 0.0
            continue

        births = finite_dgm[:, 0]
        deaths = finite_dgm[:, 1]
        persistences = deaths - births
        
        features[f'dim{dim}_max_persistence'] = float(np.max(persistences))
        features[f'dim{dim}_avg_midlife'] = float(np.mean((births + deaths) / 2.0))
        features[f'dim{dim}_avg_death'] = float(np.mean(deaths))
        features[f'dim{dim}_num_points'] = float(len(finite_dgm))
        
    return features
