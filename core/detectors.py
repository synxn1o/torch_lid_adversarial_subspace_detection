"""Adversarial example detectors using statistical and topological features.

Provides four detector types that characterize adversarial examples through
different statistical measures of model layer activations.

Classes:
    BaseDetector — Abstract base with fit() and detect() interface
    LIDDetector — Local Intrinsic Dimensionality via MLE on k-NN distances
    KDDetector — Kernel Density Estimation per predicted class (penultimate layer)
    KMDetector — Mean k-NN distance to class training centroids
    TDADetector — Topological Data Analysis via persistent homology on neuron correlations
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from .utils import mle_batch, kmean_batch
from .models import ModelWrapper

class BaseDetector:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.device = model_wrapper.device

    def fit(self, train_loader):
        pass

    def detect(self, x):
        raise NotImplementedError

class LIDDetector(BaseDetector):
    def __init__(self, model_wrapper, k=20, batch_size=100):
        super().__init__(model_wrapper)
        self.k = k
        self.batch_size = batch_size

    def detect(self, x, x_clean_ref=None):
        """
        x: samples to detect
        x_clean_ref: reference clean samples (if None, use x itself as in random batch LID)
        """
        if isinstance(x, torch.Tensor):
            x = x.float()
        if x_clean_ref is not None and isinstance(x_clean_ref, torch.Tensor):
            x_clean_ref = x_clean_ref.float()
            
        # Extract features for all layers
        features = self.model_wrapper.get_features(x)
        num_layers = len(features)
        num_samples = len(x)
        
        if x_clean_ref is not None:
            ref_features = self.model_wrapper.get_features(x_clean_ref)
        else:
            ref_features = features
            
        lids = np.zeros((num_samples, num_layers))
        
        for l in range(num_layers):
            f = features[l].cpu().numpy()
            f_ref = ref_features[l].cpu().numpy()
            lids[:, l] = mle_batch(f_ref, f, k=self.k)
            
        return lids

class KDDetector(BaseDetector):
    def __init__(self, model_wrapper, bandwidth=None):
        super().__init__(model_wrapper)
        self.bandwidth = bandwidth
        self.kdes = {} # kdes[layer][class]

    def fit(self, train_loader):
        print("Fitting KDE detector...")
        # Extract training features in batches to avoid OOM
        all_features = []
        Y_train = []
        
        for x, y in tqdm(train_loader, desc="Extracting train features"):
            batch_features = self.model_wrapper.get_features(x)
            if not all_features:
                all_features = [[] for _ in range(len(batch_features))]
            for l, f in enumerate(batch_features):
                all_features[l].append(f.cpu().numpy())
            Y_train.append(y.numpy())
            
        Y_train = np.concatenate(Y_train, axis=0)
        num_layers = len(all_features)
        classes = np.unique(Y_train)
        
        # Default bandwidths from original code
        BANDWIDTHS = {'mnist': 3.7926, 'cifar': 0.26, 'svhn': 1.00, 'toy': 0.2}
        bw = self.bandwidth or BANDWIDTHS.get(self.model_wrapper.dataset_name, 0.1)
        
        # Use only the penultimate layer for KD (as in original code)
        l = num_layers - 2
        self.kdes[l] = {}
        f_layer = np.concatenate(all_features[l], axis=0)
        for c in classes:
            class_subset = f_layer[Y_train == c]
            self.kdes[l][c] = KernelDensity(kernel='gaussian', bandwidth=bw).fit(class_subset)

    def detect(self, x):
        features = self.model_wrapper.get_features(x)
        num_layers = len(features)
        num_samples = len(x)
        preds = self.model_wrapper.predict(x)
        
        # Only score the penultimate layer
        l = num_layers - 2
        densities = np.zeros((num_samples, 1))
        f_layer = features[l].cpu().numpy()
        for i in range(num_samples):
            c = preds[i]
            densities[i, 0] = self.kdes[l][c].score_samples(f_layer[i].reshape(1, -1))[0]
        return densities

class KMDetector(BaseDetector):
    def __init__(self, model_wrapper, k=20):
        super().__init__(model_wrapper)
        self.k = k
        self.train_features = {} # train_features[layer][class]

    def fit(self, train_loader):
        print("Fitting KMeans detector...")
        all_features = []
        Y_train = []
        
        for x, y in tqdm(train_loader, desc="Extracting train features"):
            batch_features = self.model_wrapper.get_features(x)
            if not all_features:
                all_features = [[] for _ in range(len(batch_features))]
            for l, f in enumerate(batch_features):
                all_features[l].append(f.cpu().numpy())
            Y_train.append(y.numpy())
            
        Y_train = np.concatenate(Y_train, axis=0)
        num_layers = len(all_features)
        classes = np.unique(Y_train)
        
        # Use only the penultimate layer for KM
        l = num_layers - 2
        self.train_features[l] = {}
        f_layer = np.concatenate(all_features[l], axis=0)
        for c in classes:
            self.train_features[l][c] = f_layer[Y_train == c]

    def detect(self, x):
        features = self.model_wrapper.get_features(x)
        num_layers = len(features)
        num_samples = len(x)
        preds = self.model_wrapper.predict(x)
        
        l = num_layers - 2
        kms = np.zeros((num_samples, 1))
        f_layer = features[l].cpu().numpy()
        for i in range(num_samples):
            c = preds[i]
            ref_f = self.train_features[l][c]
            kms[i, 0] = kmean_batch(ref_f, f_layer[i:i+1], k=self.k)[0]
        return kms

class TDADetector(BaseDetector):
    def __init__(self, model_wrapper, maxdim=1):
        super().__init__(model_wrapper)
        self.maxdim = maxdim
        try:
            from ripser import ripser
            self.ripser = ripser
        except ImportError:
            print("Warning: ripser not installed. TDADetector will not work.")

    def detect(self, x):
        # TDA usually works on a batch of activations to find topological patterns
        # Extract activations for all neurons
        
        activations = self._get_neuron_activations(x)
        corr_matrix = np.corrcoef(activations.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        dist_matrix = 1.0 - corr_matrix
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        
        rips_output = self.ripser(dist_matrix, distance_matrix=True, maxdim=self.maxdim)
        dgms = rips_output['dgms']
        
        # Return both features and raw TDA data for visualization
        features = self._extract_topological_features(dgms)
        
        # Convert diagrams to list for JSON serialization
        serializable_dgms = [d.tolist() for d in dgms]
        
        return {
            'features': features,
            'diagrams': serializable_dgms,
            'correlation_matrix': corr_matrix.tolist()
        }

    def _get_neuron_activations(self, x):
        # Similar to ModelWrapper.get_features but with GAP for conv layers
        # to keep neuron count manageable for TDA
        activations = []
        def hook_fn(module, input, output):
            if isinstance(module, torch.nn.Conv2d):
                act = torch.mean(output, dim=(2, 3))
            else:
                act = output
            activations.append(act.detach().cpu().numpy())

        hooks = []
        for name, module in self.model_wrapper.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            self.model_wrapper.model(x.to(self.device))

        for h in hooks: h.remove()
        
        # activations is a list of [batch, neurons] for each layer
        # We want to concatenate neurons across layers
        # But wait, TDA usually wants [samples, total_neurons]
        # The hooks were called once for the whole batch x.
        # So activations[i] is [batch, neurons_i]
        return np.concatenate(activations, axis=1)

    def _extract_topological_features(self, dgms):
        features = {}
        for dim, dgm in enumerate(dgms):
            if len(dgm) == 0:
                features[f'dim{dim}_max_persistence'] = 0.0
                continue
            finite_dgm = dgm[np.isfinite(dgm[:, 1])]
            if len(finite_dgm) == 0:
                features[f'dim{dim}_max_persistence'] = 0.0
                continue
            persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
            features[f'dim{dim}_max_persistence'] = float(np.max(persistences))
            features[f'dim{dim}_num_points'] = float(len(finite_dgm))
        return features
