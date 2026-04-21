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

import gc
import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
        BANDWIDTHS = {'mnist': 3.7926, 'cifar': 0.26, 'cifar100': 0.25, 'svhn': 1.00, 'fashion_mnist': 3.5, 'toy': 0.2}
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
                features[f'dim{dim}_num_points'] = 0.0
                features[f'dim{dim}_avg_birth'] = 0.0
                features[f'dim{dim}_avg_death'] = 0.0
                features[f'dim{dim}_avg_persistence'] = 0.0
                continue
            finite_dgm = dgm[np.isfinite(dgm[:, 1])]
            if len(finite_dgm) == 0:
                features[f'dim{dim}_max_persistence'] = 0.0
                features[f'dim{dim}_num_points'] = 0.0
                features[f'dim{dim}_avg_birth'] = 0.0
                features[f'dim{dim}_avg_death'] = 0.0
                features[f'dim{dim}_avg_persistence'] = 0.0
                continue
            persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
            features[f'dim{dim}_max_persistence'] = float(np.max(persistences))
            features[f'dim{dim}_num_points'] = float(len(finite_dgm))
            features[f'dim{dim}_avg_birth'] = float(np.mean(finite_dgm[:, 0]))
            features[f'dim{dim}_avg_death'] = float(np.mean(finite_dgm[:, 1]))
            features[f'dim{dim}_avg_persistence'] = float(np.mean(persistences))
        return features


class PersistenceImageDetector(BaseDetector):
    """Adversarial detector using persistence images from a target layer.

    Groups samples into point clouds, computes persistence diagrams,
    converts to persistence images, and classifies using an SVC.

    Args:
        model_wrapper: ModelWrapper instance
        layer_name: str, target layer for activation extraction (default: auto-detect last hidden)
        group_size: int, samples per point cloud group (default: 25)
        pixel_size: float, persistence image pixel size (default: 0.15)
        spread: float, persistence image Gaussian spread (default: 0.6)
        maxdim: int, maximum homology dimension (default: 1)

    Usage:
        detector = PersistenceImageDetector(model, layer_name='fc1', group_size=25)
        detector.fit(train_loader)  # builds persistence images from training data
        scores = detector.detect(x)  # returns classification scores
    """

    def __init__(self, model_wrapper, layer_name=None, group_size=25,
                 pixel_size=0.15, spread=0.6, maxdim=1):
        super().__init__(model_wrapper)
        self.group_size = group_size
        self.pixel_size = pixel_size
        self.spread = spread
        self.maxdim = maxdim
        self.classifier = None
        self.birth_range = None
        self.pers_range = None

        if layer_name is None:
            self.layer_name = self._auto_detect_layer()
        else:
            self.layer_name = layer_name

        # Check ripser availability
        try:
            from ripser import ripser
            self._ripser = ripser
        except ImportError:
            print("Warning: ripser not installed. PersistenceImageDetector will not work.")
            self._ripser = None

    def _auto_detect_layer(self):
        """Auto-detect the last hidden layer name from model architecture."""
        linear_layers = []
        for name, module in self.model_wrapper.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append(name)
        if len(linear_layers) >= 2:
            return linear_layers[-2]  # penultimate linear layer
        elif linear_layers:
            return linear_layers[-1]
        return None

    def _extract_activations(self, X, batch_size=200):
        """Extract activations from the target layer."""
        from .tda_utils import extract_layer_activations
        return extract_layer_activations(self.model_wrapper, X, self.layer_name, batch_size)

    def _split_into_groups(self, data):
        """Split (N, D) array into list of (group_size, D) arrays."""
        groups = []
        for i in range(0, len(data), self.group_size):
            chunk = data[i:i + self.group_size]
            if len(chunk) >= 10:
                groups.append(chunk)
        return groups

    def _group_to_diagrams(self, group):
        """Convert an activation group to persistence diagrams."""
        from .tda_utils import compute_correlation_distance
        dist_matrix = compute_correlation_distance(group)
        result = self._ripser(dist_matrix, distance_matrix=True, maxdim=self.maxdim)
        return result['dgms']

    def _compute_global_bounds(self, all_groups):
        """Compute global min/max for births and persistences."""
        all_births = []
        all_pers = []
        for groups in all_groups:
            for group in groups:
                diagrams = self._group_to_diagrams(group)
                for dg in diagrams:
                    dg = np.array(dg)
                    if len(dg) == 0:
                        continue
                    finite = np.isfinite(dg[:, 1])
                    if finite.any():
                        all_births.append(dg[finite, 0])
                        all_pers.append(dg[finite, 1] - dg[finite, 0])
        if not all_births:
            return (0, 1, 0, 1)
        all_births = np.concatenate(all_births)
        all_pers = np.concatenate(all_pers)
        margin = self.spread * 3
        return (all_births.min() - margin, all_births.max() + margin,
                max(0, all_pers.min()) - margin, all_pers.max() + margin)

    def _build_features(self, activations):
        """Convert activations to persistence image feature vectors."""
        from .tda_utils import persistence_diagrams_to_images

        groups = self._split_into_groups(activations)
        features = []
        for group in groups:
            diagrams = self._group_to_diagrams(group)
            images = persistence_diagrams_to_images(
                diagrams, pixel_size=self.pixel_size, spread=self.spread,
                birth_range=self.birth_range, pers_range=self.pers_range
            )
            feat = np.concatenate(images) if images else np.array([])
            if len(feat) > 0:
                features.append(feat)
        return np.array(features) if features else np.empty((0, 0))

    def fit(self, train_loader=None, X_clean=None, X_adv=None):
        """Fit the detector on clean and adversarial activations.

        Either provide train_loader (for unsupervised) or X_clean + X_adv (supervised).
        For supervised: extracts activations, groups, computes persistence images, trains SVC.

        Args:
            train_loader: optional DataLoader (not used for supervised mode)
            X_clean: numpy array of clean samples
            X_adv: numpy array of adversarial samples
        """
        if self._ripser is None:
            raise RuntimeError("ripser is required but not installed.")

        if X_clean is not None and X_adv is not None:
            # Supervised mode
            act_clean = self._extract_activations(X_clean)
            act_adv = self._extract_activations(X_adv)

            # Compute global bounds
            groups_clean = self._split_into_groups(act_clean)
            groups_adv = self._split_into_groups(act_adv)
            b_min, b_max, p_min, p_max = self._compute_global_bounds([groups_clean, groups_adv])
            self.birth_range = (b_min, b_max)
            self.pers_range = (p_min, p_max)

            # Build features
            feat_clean = self._build_features(act_clean)
            feat_adv = self._build_features(act_adv)

            if feat_clean.size == 0 or feat_adv.size == 0:
                print("Warning: No features generated. Cannot fit classifier.")
                return

            X_train = np.vstack([feat_clean, feat_adv])
            y_train = np.concatenate([np.zeros(len(feat_clean)), np.ones(len(feat_adv))])

            # Train SVC
            self.classifier = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel='rbf', probability=True, class_weight='balanced'))
            ])
            self.classifier.fit(X_train, y_train)
            print(f"PersistenceImageDetector fitted: {len(feat_clean)} clean, {len(feat_adv)} adv groups")

        elif train_loader is not None:
            # Unsupervised: extract activations from training data
            X_list = []
            for x, y in train_loader:
                X_list.append(x.numpy())
            X_all = np.concatenate(X_list, axis=0)
            act_all = self._extract_activations(X_all)
            groups = self._split_into_groups(act_all)
            b_min, b_max, p_min, p_max = self._compute_global_bounds([groups])
            self.birth_range = (b_min, b_max)
            self.pers_range = (p_min, p_max)
            print(f"PersistenceImageDetector bounds set from training data")
        else:
            raise ValueError("Provide either train_loader or both X_clean and X_adv")

    def detect(self, x):
        """Detect adversarial samples.

        Args:
            x: tensor or numpy array of samples

        Returns:
            dict with 'predictions' (0=clean, 1=adversarial), 'scores' (probability),
            'persistence_images' (for visualization)
        """
        if self._ripser is None:
            raise RuntimeError("ripser is required but not installed.")

        act = self._extract_activations(x)
        feat = self._build_features(act)

        if self.classifier is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        if feat.size == 0:
            return {'predictions': np.array([]), 'scores': np.array([]), 'persistence_images': []}

        predictions = self.classifier.predict(feat)
        scores = self.classifier.predict_proba(feat)[:, 1]

        return {
            'predictions': predictions.astype(int),
            'scores': scores,
            'persistence_images': feat
        }
