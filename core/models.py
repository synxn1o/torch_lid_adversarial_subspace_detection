"""CNN model architectures and feature extraction wrapper.

Defines network architectures for MNIST (1-channel), CIFAR-10 (3-channel),
SVHN (3-channel), and Toy (2D binary). ModelWrapper provides a unified
interface for logits, predictions, and intermediate layer activations.

Classes:
    MNISTModel — CNN for 28x28 grayscale input, 10 classes
    CIFARModel — Deeper CNN for 32x32 RGB input, 10 classes
    SVHNModel — CNN for 32x32 RGB input, 10 classes
    ToyModel — Small MLP for 2D binary classification
    ModelWrapper — Wraps any model with feature extraction hooks

Functions:
    get_model — Factory: creates ModelWrapper with the correct architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- Model Definitions ---

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=0) 
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.bn7(x)
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.bn8(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        return x

class SVHNModel(nn.Module):
    def __init__(self):
        super(SVHNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.bn4(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        return x

class ToyModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Model Wrapper ---

class ModelWrapper:
    def __init__(self, model, dataset_name, device='cpu'):
        self.model = model.to(device)
        self.dataset_name = dataset_name
        self.device = device
        self.is_binary = (dataset_name == 'toy')
        self.model.eval()
        
        self.features = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            out = output
            if out.dim() > 2:
                # Use Global Average Pooling for convolutional layers to keep dimensionality manageable
                if isinstance(module, nn.Conv2d):
                    out = torch.mean(output, dim=(2, 3))
                else:
                    out = out.view(out.size(0), -1)
            self.features.append(out)

        # Register hooks for all relevant layers
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0: # Leaf module
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d, nn.MaxPool2d)):
                    self.hooks.append(module.register_forward_hook(hook_fn))

    def get_logits(self, x):
        if isinstance(x, torch.Tensor):
            x = x.to(self.device).float()
        else:
            x = torch.from_numpy(x).to(self.device).float()
        return self.model(x)

    def predict(self, x):
        logits = self.get_logits(x)
        if self.is_binary:
            return (torch.sigmoid(logits) > 0.5).long().cpu().numpy().flatten()
        else:
            return torch.argmax(logits, dim=1).cpu().numpy()

    def get_features(self, x):
        self.features = [] # Reset
        logits = self.get_logits(x)
        # Include input as a feature?
        # Original code often includes input.
        # Let's prepend it.
        if isinstance(x, torch.Tensor):
            inp = x.view(x.size(0), -1).detach()
        else:
            inp = torch.from_numpy(x).view(len(x), -1).to(self.device).detach()
            
        return [inp] + [f.detach() for f in self.features]

    def __call__(self, x):
        return self.get_logits(x)

def get_model(dataset_name, model_path=None, device='cpu'):
    if dataset_name == 'mnist':
        model = MNISTModel()
    elif dataset_name == 'cifar':
        model = CIFARModel()
    elif dataset_name == 'svhn':
        model = SVHNModel()
    elif dataset_name == 'toy':
        model = ToyModel()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    return ModelWrapper(model, dataset_name, device)
