import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.spatial.distance import cdist
import os

# Constants
CLIP_MIN = -0.5
CLIP_MAX = 0.5
PATH_DATA = "data/"

def get_data(dataset='mnist', batch_size=128, download=True, augmentation=False):
    """
    Data loading with normalization to [-0.5, 0.5]
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], "dataset parameter must be either 'mnist', 'cifar', or 'svhn'"

    # Base transforms
    base_transform_list = [
        transforms.ToTensor(),
    ]
    
    # Normalization
    if dataset == 'cifar':
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    else:
        normalize = transforms.Normalize((0.5,), (1.0,))

    # Augmentation
    train_transform_list = []
    if augmentation:
        train_transform_list.extend([
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.RandomHorizontalFlip()
        ])
    
    # Combine
    train_transform = transforms.Compose(train_transform_list + base_transform_list + [normalize])
    test_transform = transforms.Compose(base_transform_list + [normalize])

    if dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=download, transform=train_transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=download, transform=test_transform)
    elif dataset == 'cifar':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=test_transform)
    elif dataset == 'svhn':
        train_set = torchvision.datasets.SVHN(root='./data', split='train', download=download, transform=train_transform)
        test_set = torchvision.datasets.SVHN(root='./data', split='test', download=download, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=0) 
        self.bn1 = nn.BatchNorm2d(64)
        # Conv2D(64, (3, 3))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        # MaxPooling2D(pool_size=(2, 2))
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout(0.5)
        self.dropout1 = nn.Dropout(0.5)
        # Flatten handled in forward
        # Dense(128)
        # Input size calculation: 28 ->(conv1)-> 26 ->(conv2)-> 24 ->(pool)-> 12
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        # Dense(10)
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
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(0.5)
        
        # Flatten: 32 -> 16 -> 8 -> 4. 128 * 4 * 4
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
        # Conv2D(64, (3, 3), padding='valid')
        self.conv1 = nn.Conv2d(3, 64, 3, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        
        # Flatten. 32 -> 30 -> 28 -> 14. 64 * 14 * 14
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

def get_model(dataset='mnist'):
    if dataset == 'mnist':
        return MNISTModel()
    elif dataset == 'cifar':
        return CIFARModel()
    elif dataset == 'svhn':
        return SVHNModel()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# Helper to get intermediate layer activations
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.features = []
        self.hooks = []
        
        # Register hooks for all conv and linear layers (or generally any layer with weights)
        # The original Keras code used `model.layers` which gives almost everything.
        # We will target Conv2d, Linear, and BatchNorm2d? Or just inputs/outputs of blocks?
        # The original code: [layer.output for layer in model.layers]
        # This implies EVERY layer.
        
        def hook_fn(module, input, output):
            # Flatten if necessary, as LID expects (batch, dim)
            out = output
            if out.dim() > 2:
                out = out.view(out.size(0), -1)
            self.features.append(out)

        for name, module in self.model.named_modules():
            # We want to capture output of "layers". 
            # In PyTorch, a Sequential or Model is also a module. We generally want leaf modules.
            if len(list(module.children())) == 0: 
                # It's a leaf module (Conv, Linear, ReLU, etc.)
                # Note: The original Keras model had explicit Activation layers. 
                # Our PyTorch model has F.relu inside forward.
                # To match exactly, we might need to change the model definition to use nn.ReLU layers 
                # or just hook the main layers. 
                # For LID, usually dense representations are most important. 
                # Let's hook Conv, Linear, and BN.
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d, nn.MaxPool2d, nn.Dropout)):
                    self.hooks.append(module.register_forward_hook(hook_fn))

    def forward(self, x):
        self.features = [] # Reset
        # Also include input? Original code: acts = [model.layers[0].input] + ...
        if x.dim() > 2:
            self.features.append(x.view(x.size(0), -1))
        else:
            self.features.append(x)
            
        self.model(x) # Run forward, hooks will populate self.features
        return self.features
    
    def close(self):
        for hook in self.hooks:
            hook.remove()

def mle_batch(data, batch, k):
    """
    LID of a batch of query points X (batch) relative to data.
    Numpy/Scipy implementation.
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def mle_single(data, x, k=20):
    """
    LID of a single query point x.
    """
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]

def lid_term(logits, batch_size=100):
    """
    Calculate LID loss term for a minibatch of logits using PyTorch.
    """
    y_pred = logits
    
    # Calculate pairwise distance matrix
    # D_ij = ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
    r = torch.sum(y_pred**2, dim=1).view(-1, 1)
    D = r - 2 * torch.matmul(y_pred, y_pred.t()) + r.t()
    
    # Sqrt to get Euclidean distance
    D1 = torch.sqrt(D + 1e-9)
    
    # Find k nearest neighbors (k=21 because topk includes self)
    # We want smallest distances, so we use topk on negative distance
    D2, _ = torch.topk(-D1, k=21, sorted=True)
    D3 = -D2[:, 1:] # Exclude self (0 distance)
    
    # D3[:, -1] is the distance to the k-th neighbor (max distance in the neighborhood)
    m = D3 / D3[:, -1].view(-1, 1)
    
    v_log = torch.sum(torch.log(m + 1e-9), dim=1)
    lids = -20 / v_log
    
    return lids

def lid_adv_term(clean_logits, adv_logits, batch_size=100):
    """
    Calculate LID loss term for a minibatch of advs logits relative to clean logits.
    """
    c_pred = clean_logits.view(batch_size, -1)
    a_pred = adv_logits.view(batch_size, -1)
    
    r_a = torch.sum(a_pred**2, dim=1).view(-1, 1)
    r_c = torch.sum(c_pred**2, dim=1).view(1, -1)
    
    # D[i, j] = dist(adv[i], clean[j])
    D = r_a - 2 * torch.matmul(a_pred, c_pred.t()) + r_c
    
    D1 = torch.sqrt(D + 1e-9)
    D2, _ = torch.topk(-D1, k=21, sorted=True)
    D3 = -D2[:, 1:]
    
    m = D3 / D3[:, -1].view(-1, 1)
    v_log = torch.sum(torch.log(m + 1e-9), dim=1)
    lids = -20 / v_log
    
    # Batch normalize lids (optional, present in original code)
    lids = F.normalize(lids.view(-1, 1), p=2, dim=0).squeeze()
    
    return lids
