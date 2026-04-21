"""Topological Data Analysis visualizers for adversarial detection results.

Classes:
    TDAVisualizer — Plots for persistence diagrams, lifetime histograms,
                    bottleneck distances, epsilon sweeps, and classifier results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from visualizer.visualizers import BaseVisualizer
from visualizer.config import get_output_path, PLOT_STYLES


class TDAVisualizer(BaseVisualizer):
    """Visualizer for TDA-based adversarial detection results.

    Plots:
        - persistence_diagrams: grid of persistence diagrams per condition
        - persistence_lifetime_histogram: lifetime distributions overlaid
        - bottleneck_distances: grouped bar chart per layer or condition
        - epsilon_sweep: bottleneck distance vs epsilon (dual y-axis)
        - classifier_results: confusion matrix + ROC curve
        - per_layer_comparison: bottleneck distance across layers
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_persistence_diagrams(self, tda_results, conditions, save=True):
        """Plot persistence diagrams for multiple conditions.

        Args:
            tda_results: dict {condition_name: {'diagrams': list_of_arrays}}
            conditions: list of condition names to plot
            save: bool, save to file

        Returns:
            matplotlib Figure or None
        """
        n_conds = len(conditions)
        fig, axes = plt.subplots(n_conds, 2, figsize=(10, 4 * n_conds))
        if n_conds == 1:
            axes = axes.reshape(1, -1)

        dim_colors = {0: '#3498db', 1: '#e74c3c'}
        dim_labels = {0: 'H0 (Connected)', 1: 'H1 (Loops)'}

        for row, cond in enumerate(conditions):
            dgms = tda_results[cond]['diagrams']
            for dim in range(min(2, len(dgms))):
                ax = axes[row, dim]
                dgm = np.array(dgms[dim])
                if len(dgm) == 0:
                    ax.set_title(f'{cond} — dim{dim} (empty)')
                    continue

                finite = dgm[np.isfinite(dgm[:, 1])]
                inf_pts = dgm[~np.isfinite(dgm[:, 1])]

                if len(finite) > 0:
                    ax.scatter(finite[:, 0], finite[:, 1],
                              c=dim_colors.get(dim, 'gray'), alpha=0.6, s=20,
                              label=f'{len(finite)} points')

                if len(inf_pts) > 0:
                    max_finite = finite[:, 1].max() if len(finite) > 0 else 1.0
                    y_inf = max_finite * 1.2
                    ax.scatter(inf_pts[:, 0], np.full(len(inf_pts), y_inf),
                              c=dim_colors.get(dim, 'gray'), alpha=0.6, s=20,
                              marker='^', label=f'{len(inf_pts)} ∞')

                all_pts = dgm[np.isfinite(dgm[:, 0])]
                if len(all_pts) > 0:
                    lims = [min(all_pts[:, 0].min(), all_pts[:, 1].min() if len(finite) > 0 else all_pts[:, 0].min()),
                            max(all_pts[:, 0].max(), y_inf if len(inf_pts) > 0 else (finite[:, 1].max() if len(finite) > 0 else 1))]
                    margin = (lims[1] - lims[0]) * 0.1
                    ax.plot([lims[0] - margin, lims[1] + margin],
                           [lims[0] - margin, lims[1] + margin],
                           'k--', alpha=0.3, linewidth=0.8)

                ax.set_xlabel('Birth')
                ax.set_ylabel('Death')
                ax.set_title(f'{cond} — {dim_labels.get(dim, f"dim{dim}")}')
                ax.legend(fontsize=8)

        plt.suptitle('Persistence Diagrams', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            self.save_figure(fig, 'persistence_diagrams.png', 'tda')
        return fig

    def plot_persistence_lifetime_histogram(self, tda_results, conditions, save=True):
        """Plot overlaid lifetime histograms.

        Args:
            tda_results: dict {condition_name: {'diagrams': list_of_arrays}}
            conditions: list of condition names
            save: bool

        Returns:
            matplotlib Figure or None
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

        for dim in range(2):
            ax = axes[dim]
            for cond, color in zip(conditions, colors):
                dgms = tda_results[cond]['diagrams']
                if dim >= len(dgms):
                    continue
                dgm = np.array(dgms[dim])
                if len(dgm) == 0:
                    continue
                finite = dgm[np.isfinite(dgm[:, 1])]
                if len(finite) == 0:
                    continue
                lifetimes = finite[:, 1] - finite[:, 0]
                lifetimes = lifetimes[lifetimes > 0]
                if len(lifetimes) > 0:
                    ax.hist(lifetimes, bins=30, alpha=0.5, color=color,
                           label=cond, edgecolor='white', density=True)

            ax.set_xlabel('Lifetime (Death − Birth)')
            ax.set_ylabel('Density')
            ax.set_title(f'H{dim} Lifetime Distribution')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Persistence Lifetime Histograms', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            self.save_figure(fig, 'lifetime_histogram.png', 'tda')
        return fig

    def plot_bottleneck_distances(self, distances_dict, group_by='layer', save=True):
        """Plot grouped bar chart of bottleneck distances.

        Args:
            distances_dict: dict with structure {group_name: {comparison: distance}}
                e.g. {'conv1': {'clean_vs_bim': 0.5, 'clean_vs_cw': 0.3}, ...}
            group_by: 'layer' or 'condition'
            save: bool

        Returns:
            matplotlib Figure or None
        """
        groups = list(distances_dict.keys())
        comparisons = list(distances_dict[groups[0]].keys())

        x = np.arange(len(groups))
        n_comp = len(comparisons)
        width = 0.8 / n_comp
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 2), 6))

        for i, comp in enumerate(comparisons):
            vals = []
            for g in groups:
                v = distances_dict[g].get(comp, 0.0)
                vals.append(v if np.isfinite(v) else 0.0)
            bars = ax.bar(x + i * width - (n_comp - 1) * width / 2, vals,
                         width, label=comp, color=colors[i % len(colors)],
                         edgecolor='white')
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                           bar.get_height() + 0.003,
                           f'{v:.3f}', ha='center', va='bottom', fontsize=7)

        xlabel = 'Layer' if group_by == 'layer' else 'Condition'
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Bottleneck Distance', fontsize=12)
        ax.set_title(f'Bottleneck Distances by {xlabel}', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save:
            self.save_figure(fig, 'bottleneck_distances.png', 'tda')
        return fig

    def plot_epsilon_sweep(self, sweep_results, save=True):
        """Plot bottleneck distance vs epsilon with dual y-axis.

        Args:
            sweep_results: dict with keys:
                'epsilons': list of float
                'bottleneck_distances': list of float (dim1)
                'accuracy': list of float (classifier accuracy at each epsilon)
            save: bool

        Returns:
            matplotlib Figure or None
        """
        epsilons = sweep_results['epsilons']
        distances = sweep_results['bottleneck_distances']
        accuracies = sweep_results.get('accuracy', None)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = '#3498db'
        ax1.set_xlabel('Epsilon (perturbation budget)', fontsize=12)
        ax1.set_ylabel('Bottleneck Distance (dim1)', color=color1, fontsize=12)
        ax1.plot(epsilons, distances, 'o-', color=color1, linewidth=2, markersize=6)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        if accuracies is not None:
            ax2 = ax1.twinx()
            color2 = '#e74c3c'
            ax2.set_ylabel('Classifier Accuracy', color=color2, fontsize=12)
            ax2.plot(epsilons, accuracies, 's--', color=color2, linewidth=2, markersize=6)
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim([0, 1.05])

        plt.title('Epsilon Sweep: Bottleneck Distance vs Perturbation Budget',
                  fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save:
            self.save_figure(fig, 'epsilon_sweep.png', 'tda')
        return fig

    def plot_classifier_results(self, metrics, y_test, y_prob, save=True):
        """Plot confusion matrix heatmap + ROC curve side by side.

        Args:
            metrics: dict with 'accuracy', 'f1', 'auc', 'cv_accuracy_mean', 'cv_accuracy_std'
            y_test: true labels (binary)
            y_prob: predicted probabilities for positive class
            save: bool

        Returns:
            matplotlib Figure or None
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion matrix
        ax = axes[0]
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Clean', 'Perturbed'],
                    yticklabels=['Clean', 'Perturbed'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        # ROC curve
        ax = axes[1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_val:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f'Persistence Image Classifier — '
            f'Acc={metrics.get("accuracy", 0):.3f}  '
            f'F1={metrics.get("f1", 0):.3f}  '
            f'AUC={metrics.get("auc", 0):.3f}  '
            f'(CV={metrics.get("cv_accuracy_mean", 0):.3f}'
            f'±{metrics.get("cv_accuracy_std", 0):.3f})',
            fontsize=12, fontweight='bold', y=1.02
        )

        plt.tight_layout()

        if save:
            self.save_figure(fig, 'classifier_results.png', 'tda')
        return fig

    def plot_per_layer_comparison(self, results, save=True):
        """Plot bottleneck distance across layers as grouped bar chart.

        Args:
            results: dict {layer_name: {comparison: distance}}
                e.g. {'conv1': {'clean_vs_bim': 0.5, ...}, 'conv2': {...}, ...}
            save: bool

        Returns:
            matplotlib Figure or None
        """
        return self.plot_bottleneck_distances(results, group_by='layer', save=save)
