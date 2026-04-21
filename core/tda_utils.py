"""Topological data analysis utilities for adversarial detection.

Pure functions operating on numpy arrays. No torch dependency in TDA functions.

Functions:
    extract_layer_activations — Extract activations from a specific model layer via hooks
    compute_correlation_distance — Activation matrix → correlation → distance matrix
    run_tda — Distance matrix → ripser → persistence diagrams + topological features
    bottleneck_distance — Compute bottleneck distance between two persistence diagrams
    persistence_diagrams_to_images — Convert persistence diagrams to persistence images
    extract_topological_features — Extract scalar features from persistence diagrams
"""

import gc
import numpy as np
import torch


def extract_layer_activations(model_wrapper, X, layer_name=None, batch_size=200):
    """Extract activations from a named layer using forward hooks.

    For Conv2d layers: apply Global Average Pooling (mean over H,W).
    For Linear layers: use output directly.
    If layer_name is None, extracts from all Conv2d/Linear layers and concatenates.

    Args:
        model_wrapper: ModelWrapper instance
        X: numpy array [N, C, H, W]
        layer_name: str, e.g. 'fc1', 'conv2'. If None, extract from all layers.
        batch_size: int

    Returns:
        numpy array [N, layer_dim]
    """
    model = model_wrapper.model
    device = model_wrapper.device
    activations = []

    def hook_fn(module, input, output):
        out = output.detach().cpu()
        if out.dim() > 2 and isinstance(module, torch.nn.Conv2d):
            out = torch.mean(out, dim=(2, 3))
        elif out.dim() > 2:
            out = out.view(out.size(0), -1)
        activations.append(out)

    hooks = []
    if layer_name is not None:
        target_module = getattr(model, layer_name)
        hooks.append(target_module.register_forward_hook(hook_fn))
    else:
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            end = min(i + batch_size, len(X))
            batch = torch.from_numpy(X[i:end]).to(device).float()
            _ = model(batch)
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    if layer_name is not None:
        result = torch.cat(activations, dim=0).numpy()
    else:
        result = torch.cat(activations, dim=1).numpy() if len(activations) > 1 else activations[0].numpy()

    gc.collect()
    return result


def compute_correlation_distance(activations):
    """Compute correlation-based distance matrix from activation matrix.

    Args:
        activations: numpy array [N_samples, N_features]

    Returns:
        distance matrix [N_features, N_features] (1 - |correlation|, symmetrized, zero diagonal)
    """
    corr_matrix = np.corrcoef(activations.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    dist_matrix = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, None)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    return dist_matrix


def run_tda(distance_matrix, maxdim=1):
    """Run ripser persistent homology on a distance matrix.

    Args:
        distance_matrix: numpy array [N, N], symmetric with zero diagonal
        maxdim: int, maximum homology dimension

    Returns:
        dict with keys: 'features' (dict), 'diagrams' (list of arrays), 'distance_matrix' (array)
    """
    try:
        from ripser import ripser
    except ImportError:
        raise ImportError("ripser is required for TDA. Install with: pip install ripser")

    rips_output = ripser(distance_matrix, distance_matrix=True, maxdim=maxdim)
    dgms = rips_output['dgms']

    features = extract_topological_features(dgms)
    serializable_dgms = [d.tolist() for d in dgms]

    return {
        'features': features,
        'diagrams': serializable_dgms,
        'distance_matrix': distance_matrix.tolist()
    }


def bottleneck_distance(dgm1, dgm2):
    """Compute bottleneck distance between two persistence diagrams.

    Args:
        dgm1, dgm2: numpy arrays of shape [n_points, 2] (birth, death)

    Returns:
        float, bottleneck distance
    """
    try:
        from persim import bottleneck
    except ImportError:
        raise ImportError("persim is required for bottleneck distance. Install with: pip install persim")

    dgm1 = np.array(dgm1)
    dgm2 = np.array(dgm2)

    if len(dgm1) == 0 or len(dgm2) == 0:
        return float('inf')

    finite1 = dgm1[np.isfinite(dgm1[:, 1])]
    finite2 = dgm2[np.isfinite(dgm2[:, 1])]

    if len(finite1) == 0 or len(finite2) == 0:
        return float('inf')

    return float(bottleneck(finite1, finite2))


def persistence_diagrams_to_images(diagrams, pixel_size=0.15, spread=0.6,
                                    birth_range=None, pers_range=None):
    """Convert persistence diagrams to persistence images.

    Args:
        diagrams: list of numpy arrays, each [n_points, 2]
        pixel_size, spread: persim parameters
        birth_range, pers_range: optional global bounds (auto-compute if None)

    Returns:
        list of flattened numpy arrays (persistence images)
    """
    # Auto-compute global bounds if not provided
    if birth_range is None or pers_range is None:
        all_births = []
        all_pers = []
        for dgm in diagrams:
            dgm = np.array(dgm)
            if len(dgm) == 0:
                continue
            finite = np.isfinite(dgm[:, 1])
            if finite.any():
                all_births.append(dgm[finite, 0])
                all_pers.append(dgm[finite, 1] - dgm[finite, 0])

        if all_births:
            all_births = np.concatenate(all_births)
            all_pers = np.concatenate(all_pers)
            margin = spread * 3
            if birth_range is None:
                birth_range = (all_births.min() - margin, all_births.max() + margin)
            if pers_range is None:
                pers_range = (max(0, all_pers.min()) - margin, all_pers.max() + margin)
        else:
            if birth_range is None:
                birth_range = (0.0, 1.0)
            if pers_range is None:
                pers_range = (0.0, 1.0)

    images = []
    for dgm in diagrams:
        dgm = np.array(dgm)
        if len(dgm) == 0:
            img = _empty_persistence_image(birth_range, pers_range, pixel_size)
            images.append(img)
            continue

        finite_mask = np.isfinite(dgm[:, 1])
        diag_clean = dgm[finite_mask]

        if len(diag_clean) == 0:
            img = _empty_persistence_image(birth_range, pers_range, pixel_size)
            images.append(img)
            continue

        img = _manual_persistence_image(diag_clean, pixel_size, spread,
                                         birth_range, pers_range)
        images.append(img)

    return images


def _empty_persistence_image(birth_range, pers_range, pixel_size):
    """Return a zero-filled persistence image for empty diagrams."""
    b_min, b_max = birth_range
    p_min, p_max = pers_range
    bx = np.arange(b_min, b_max + pixel_size, pixel_size)
    py = np.arange(p_min, p_max + pixel_size, pixel_size)
    return np.zeros(len(bx) * len(py))


def _manual_persistence_image(diagram, pixel_size, spread, birth_range, pers_range):
    """Manual persistence image implementation per Adams et al. 2017.

    Args:
        diagram: numpy array [n_points, 2], filtered to finite points
        pixel_size: float
        spread: float, Gaussian spread
        birth_range: tuple (min, max)
        pers_range: tuple (min, max)

    Returns:
        flattened numpy array
    """
    births = diagram[:, 0]
    persistences = diagram[:, 1] - diagram[:, 0]
    weights = persistences

    b_min, b_max = birth_range
    p_min, p_max = pers_range
    bx = np.arange(b_min, b_max + pixel_size, pixel_size)
    py = np.arange(p_min, p_max + pixel_size, pixel_size)
    BX, PY = np.meshgrid(bx, py)

    image = np.zeros_like(BX)
    for i in range(len(diagram)):
        b, p = births[i], persistences[i]
        w = weights[i]
        gaussian = np.exp(-((BX - b) ** 2 + (PY - p) ** 2) / (2 * spread ** 2))
        image += w * gaussian

    return image.flatten()


def extract_topological_features(dgms):
    """Extract scalar features from persistence diagrams.

    Mirrors TDADetector._extract_topological_features.
    Returns dict with keys: dim{d}_max_persistence, dim{d}_num_points,
                            dim{d}_avg_birth, dim{d}_avg_death, dim{d}_avg_persistence

    Args:
        dgms: list of numpy arrays, each [n_points, 2] (birth, death)

    Returns:
        dict of feature_name -> float
    """
    features = {}
    for dim, dgm in enumerate(dgms):
        dgm = np.array(dgm)
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
