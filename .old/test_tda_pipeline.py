import torch
import torch.nn as nn
import numpy as np
import os
import json
from tda_utils import (
    get_neuron_activations, 
    compute_correlation_matrix, 
    compute_persistence_diagrams, 
    extract_topological_features
)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.fc1 = nn.Linear(4 * 26 * 26, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def test_tda_pipeline():
    print("Starting TDA Pipeline Test...")
    
    # 1. Setup dummy model and data
    device = torch.device("cpu")
    model = SimpleModel().to(device)
    
    # 10 samples of 28x28 images
    dummy_inputs = torch.randn(10, 1, 28, 28)
    dummy_labels = torch.zeros(10)
    loader = [(dummy_inputs, dummy_labels)]
    
    # 2. Test Activation Extraction
    print("Testing activation extraction...")
    activations = get_neuron_activations(model, loader, device)
    # conv1 has 4 filters (GAP -> 4 neurons), fc1 has 10 neurons. Total = 14 neurons.
    print(f"Activations shape: {activations.shape}")
    assert activations.shape == (10, 14), f"Expected (10, 14), got {activations.shape}"
    
    # 3. Test Correlation Matrix
    print("Testing correlation matrix computation...")
    corr_matrix = compute_correlation_matrix(activations)
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    assert corr_matrix.shape == (14, 14), f"Expected (14, 14), got {corr_matrix.shape}"
    
    # 4. Test Persistence Diagrams
    print("Testing persistence diagrams (Ripser)...")
    rips_output = compute_persistence_diagrams(corr_matrix, maxdim=1)
    dgms = rips_output['dgms']
    print(f"Number of diagrams: {len(dgms)}")
    assert len(dgms) >= 2, "Expected at least H0 and H1 diagrams"
    
    # 5. Test Feature Extraction
    print("Testing feature extraction...")
    features = extract_topological_features(dgms)
    print("Extracted features:", features.keys())
    assert 'dim0_max_persistence' in features
    assert 'dim1_max_persistence' in features
    
    # 6. Test JSON serialization (as used in tda_detector.py)
    print("Testing serialization...")
    serializable_dgms = [d.tolist() for d in dgms]
    output = {'features': features, 'diagrams': serializable_dgms}
    json_str = json.dumps(output)
    assert len(json_str) > 0
    
    print("\nAll TDA Pipeline tests passed!")

if __name__ == "__main__":
    test_tda_pipeline()
