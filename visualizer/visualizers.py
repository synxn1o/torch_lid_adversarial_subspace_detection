"""
Visualizer classes for different types of adversarial ML analysis
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from visualizer.config import (
    get_output_path,
    PLOT_STYLES,
    PALETTES,
    VISUALIZATION_CONFIG,
    get_dataset_config,
    RESULTS_DIR
)
from visualizer.data_loaders import (
    load_original_data,
    load_adversarial_data,
    load_characteristics_data,
    load_model_predictions,
    load_training_metrics,
    load_all_characteristics
)


class BaseVisualizer:
    """Base class for all visualizers"""
    
    def __init__(self, dataset: str = "mnist", style: str = "presentation", 
                 output_dir: Optional[str] = None, dpi: int = 300):
        self.dataset = dataset
        self.style = style
        self.dpi = dpi
        self.output_dir = Path(output_dir) if output_dir else None
        self.config = get_dataset_config(dataset)
        
        # Setup plot style
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib/seaborn style"""
        style_config = PLOT_STYLES.get(self.style, PLOT_STYLES["presentation"])
        
        plt.rcParams.update({
            'figure.figsize': style_config["figsize"],
            'font.size': style_config["font_size"],
            'axes.titlesize': style_config["font_size"] + 2,
            'axes.labelsize': style_config["font_size"],
            'xtick.labelsize': style_config["font_size"] - 2,
            'ytick.labelsize': style_config["font_size"] - 2,
            'legend.fontsize': style_config["font_size"] - 1,
            'lines.linewidth': style_config["line_width"],
            'lines.markersize': style_config["marker_size"],
        })
        
        sns.set_palette("colorblind")

    def _palette(self, kind: str = "categorical") -> str:
        """Return palette name for a plot type ('categorical', 'sequential', 'diverging')."""
        return PALETTES[kind]

    def _apply_title(self, fig_or_ax, title: Optional[str], default: str):
        """Apply title to a figure (suptitle) or axes (set_title)."""
        if title is None:
            title = default
        if hasattr(fig_or_ax, 'suptitle'):
            fig_or_ax.suptitle(title, fontsize=16)
        else:
            fig_or_ax.set_title(title)
    
    def save_figure(self, fig, filename: str, subdir: Optional[str] = None):
        """Save figure to output directory"""
        cat = subdir if subdir else "general"
        output_path = get_output_path(cat, filename, create_dir=True, base_dir=self.output_dir)
        
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)


class AdversarialVisualizer(BaseVisualizer):
    """Visualizer for original vs adversarial data analysis"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def create_image_grid_comparison(self, attack: str, num_samples: int = 16,
                                   save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Create grid comparison: Original vs Adversarial vs Perturbation vs Difference
        """
        try:
            # Load data
            original_data, original_labels = load_original_data(self.dataset, use_test_set=True)
            adversarial_data = load_adversarial_data(self.dataset, attack)
            
            # Limit samples
            num_samples = min(num_samples, len(original_data), len(adversarial_data))
            
            # Convert to numpy for easier handling
            if isinstance(original_data, torch.Tensor):
                original_data = original_data[:num_samples].numpy()
                original_labels = original_labels[:num_samples].numpy()
            else:
                original_data = original_data[:num_samples]
                original_labels = original_labels[:num_samples]
            
            adversarial_data = adversarial_data[:num_samples]
            
            # Special handling for toy dataset (2D points)
            if self.dataset == 'toy':
                return self._create_toy_scatter_comparison(attack, original_data, adversarial_data, original_labels, save, title)

            # Calculate perturbations
            perturbations = np.abs(adversarial_data - original_data)
            differences = adversarial_data - original_data
            
            # Create grid
            rows = min(4, int(np.ceil(num_samples / 4)))
            cols = 4
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
            
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(rows):
                # Original
                ax = axes[i, 0]
                img = original_data[i].squeeze()
                ax.imshow(img, cmap='gray' if self.dataset == 'mnist' else None)
                ax.set_title(f'Original\nLabel: {original_labels[i]}')
                ax.axis('off')
                
                # Adversarial
                ax = axes[i, 1]
                img = adversarial_data[i].squeeze()
                ax.imshow(img, cmap='gray' if self.dataset == 'mnist' else None)
                ax.set_title(f'Adversarial\nAttack: {attack}')
                ax.axis('off')
                
                # Perturbation (heatmap)
                ax = axes[i, 2]
                img = perturbations[i].squeeze()
                im = ax.imshow(img, cmap=self._palette("sequential"))
                ax.set_title(f'Perturbation\n|Δ| = {np.mean(img):.3f}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, shrink=0.6)
                
                # Difference
                ax = axes[i, 3]
                img = differences[i].squeeze()
                im = ax.imshow(img, cmap=self._palette("diverging"), vmin=-0.5, vmax=0.5)
                ax.set_title(f'Difference\n±{np.max(np.abs(img)):.3f}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, shrink=0.6)
            
            if title is None:
                title = f'{self.dataset.upper()}: Original vs {attack.upper()} Adversarial Examples'
            plt.suptitle(title, fontsize=16, y=1.02)
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, f"adversarial_grid_{attack}.png", "adversarial")
            
            return fig
            
        except Exception as e:
            print(f"Error creating image grid: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_toy_scatter_comparison(self, attack, original, adversarial, labels, save, title: Optional[str] = None):
        """Create scatter plot comparison for 2D toy data"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        palette = self._palette("diverging")

        # Original
        sns.scatterplot(x=original[:, 0], y=original[:, 1], hue=labels,
                        palette=palette, alpha=0.6, ax=axes[0], legend=False)
        axes[0].set_title('Original Data')
        axes[0].grid(True, alpha=0.3)

        # Adversarial
        sns.scatterplot(x=adversarial[:, 0], y=adversarial[:, 1], hue=labels,
                        palette=palette, alpha=0.6, ax=axes[1], legend=False)
        # Draw arrows from original to adversarial
        for i in range(min(50, len(original))):
            axes[1].arrow(original[i, 0], original[i, 1],
                         adversarial[i, 0] - original[i, 0],
                         adversarial[i, 1] - original[i, 1],
                         head_width=0.05, head_length=0.1, fc='k', ec='k', alpha=0.3)

        axes[1].set_title(f'Adversarial Data ({attack})')
        axes[1].grid(True, alpha=0.3)

        if title is None:
            title = f'Toy Dataset: {attack.upper()} Attack Visualization'
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save:
            self.save_figure(fig, f"adversarial_scatter_{attack}.png", "adversarial")
        return fig
    
    def create_perturbation_analysis(self, attacks: Optional[List[str]] = None,
                                   save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Analyze perturbation magnitudes across attacks
        """
        if attacks is None:
            attacks = ["fgsm", "bim-a", "bim-b", "jsma"]
        
        try:
            original_data, _ = load_original_data(self.dataset, use_test_set=True)
            if isinstance(original_data, torch.Tensor):
                original_data = original_data.numpy()
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            palette_colors = sns.color_palette(self._palette("sequential"), n_colors=len(attacks))

            for idx, attack in enumerate(attacks):
                try:
                    adv_data = load_adversarial_data(self.dataset, attack)
                    # Limit adversarial data to match original data length
                    min_samples = min(len(original_data), len(adv_data))
                    perturbations = np.abs(adv_data[:min_samples] - original_data[:min_samples])
                    distances = np.linalg.norm(perturbations.reshape(perturbations.shape[0], -1), axis=1)

                    ax = axes[idx]
                    sns.histplot(distances, bins=30, alpha=0.7, color=palette_colors[idx],
                                edgecolor='black', ax=ax)
                    ax.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {np.mean(distances):.3f}')
                    ax.axvline(np.median(distances), color='blue', linestyle='--', linewidth=2,
                              label=f'Median: {np.median(distances):.3f}')
                    ax.set_xlabel('L2 Perturbation Magnitude')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{attack.upper()} Perturbation Distribution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                except Exception:
                    axes[idx].text(0.5, 0.5, f'{attack.upper()}\nData not available', 
                                 ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].set_xticks([])
                    axes[idx].set_yticks([])
            
            if title is None:
                title = 'Perturbation Magnitude Analysis Across Attacks'
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, "perturbation_analysis.png", "adversarial")
            
            return fig
            
        except Exception as e:
            print(f"Error creating perturbation analysis: {e}")
            return None
    
    def create_attack_success_metrics(self, attacks: Optional[List[str]] = None,
                                    save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Analyze attack success rates and metrics
        """
        if attacks is None:
            attacks = ["fgsm", "bim-a", "bim-b", "jsma"]
        
        try:
            # Load model predictions
            all_metrics = {}
            
            for attack in attacks:
                try:
                    preds, probs, true_labels = load_model_predictions(
                        self.dataset, "adversarial", attack
                    )
                    
                    success_rate = np.mean(preds != true_labels)
                    confidence_drop = np.mean(
                        np.max(probs, axis=1) - np.max(
                            load_model_predictions(self.dataset, "original")[1], axis=1
                        )[:len(probs)]
                    )
                    
                    all_metrics[attack] = {
                        "success_rate": success_rate,
                        "confidence_drop": confidence_drop,
                        "mean_confidence": np.mean(np.max(probs, axis=1))
                    }
                except Exception:
                    continue
            
            if not all_metrics:
                print("No attack metrics available")
                return None
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Success rates
            attacks_list = list(all_metrics.keys())
            success_rates = [all_metrics[a]["success_rate"] for a in attacks_list]

            sns.barplot(x=attacks_list, y=success_rates,
                        palette=self._palette("categorical"), ax=axes[0])
            axes[0].set_xticklabels(attacks_list, rotation=45)
            axes[0].set_ylabel('Success Rate')
            axes[0].set_title('Attack Success Rates')
            axes[0].set_ylim(0, 1)
            axes[0].grid(True, alpha=0.3, axis='y')

            for i, rate in enumerate(success_rates):
                axes[0].text(i, rate + 0.01, f'{rate:.1%}', ha='center', va='bottom')

            # Confidence drops
            confidence_drops = [all_metrics[a]["confidence_drop"] for a in attacks_list]
            sns.barplot(x=attacks_list, y=confidence_drops,
                        palette=self._palette("categorical"), ax=axes[1])
            axes[1].set_xticklabels(attacks_list, rotation=45)
            axes[1].set_ylabel('Confidence Drop')
            axes[1].set_title('Model Confidence Reduction')
            axes[1].grid(True, alpha=0.3, axis='y')

            # Mean confidence
            mean_conf = [all_metrics[a]["mean_confidence"] for a in attacks_list]
            sns.barplot(x=attacks_list, y=mean_conf,
                        palette=self._palette("categorical"), ax=axes[2])
            axes[2].set_xticklabels(attacks_list, rotation=45)
            axes[2].set_ylabel('Mean Confidence')
            axes[2].set_title('Adversarial Prediction Confidence')
            axes[2].grid(True, alpha=0.3, axis='y')
            
            if title is None:
                title = 'Attack Success and Confidence Analysis'
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, "attack_metrics.png", "adversarial")
            
            return fig
            
        except Exception as e:
            print(f"Error creating attack metrics: {e}")
            return None


class ModelVisualizer(BaseVisualizer):
    """Visualizer for model training and performance analysis"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def create_training_curves(self, save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Create training/validation loss and accuracy curves
        """
        try:
            metrics = load_training_metrics(self.dataset)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            palette = self._palette("sequential")

            # Loss curves
            sns.lineplot(x=metrics["epochs"], y=metrics["train_loss"],
                         color=sns.color_palette(palette, 3)[0],
                         linewidth=2, label='Training Loss', ax=axes[0])
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss Curve')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Accuracy curves
            colors = sns.color_palette(palette, 3)
            sns.lineplot(x=metrics["epochs"], y=metrics["train_acc"],
                         color=colors[0], linewidth=2, label='Training Accuracy', ax=axes[1])
            sns.lineplot(x=metrics["epochs"], y=metrics["val_acc"],
                         color=colors[1], linewidth=2, linestyle='--',
                         label='Validation Accuracy', ax=axes[1])
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Accuracy Evolution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Final performance summary
            test_acc = metrics["test_accuracy"]
            sns.barplot(x=['Test Accuracy'], y=[test_acc * 100],
                        palette=self._palette("categorical"), ax=axes[2])
            axes[2].set_ylabel('Accuracy (%)')
            axes[2].set_title(f'Model Performance\nTest Accuracy: {test_acc:.2%}')
            axes[2].set_ylim(0, 100)
            axes[2].grid(True, alpha=0.3, axis='y')
            
            if title is None:
                title = f'{self.dataset.upper()} Model Training Analysis'
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, "training_curves.png", "model")
            
            return fig
            
        except Exception as e:
            print(f"Error creating training curves: {e}")
            return None
    
    def create_confusion_matrix(self, save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Create confusion matrix visualization
        """
        try:
            metrics = load_training_metrics(self.dataset)
            cm = metrics["confusion_matrix"]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            labels = [str(i) for i in range(cm.shape[0])]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=self._palette("sequential"),
                       xticklabels=labels, yticklabels=labels, ax=ax)
            
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            if title is None:
                title = f'Confusion Matrix\nOverall Accuracy: {metrics["test_accuracy"]:.2%}'
            ax.set_title(title)
            
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, "confusion_matrix.png", "model")
            
            return fig
            
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            return None
    
    def create_roc_analysis(self, save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Create ROC curves for multi-class classification
        """
        try:
            metrics = load_training_metrics(self.dataset)

            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot ROC for first 3 classes
            colors = sns.color_palette(self._palette("categorical"), 3)
            for i, class_idx in enumerate([0, 1, 2]):
                if class_idx in metrics["fpr"]:
                    fpr = metrics["fpr"][class_idx]
                    tpr = metrics["tpr"][class_idx]
                    roc_auc = metrics["roc_auc"][class_idx]

                    sns.lineplot(x=fpr, y=tpr, color=colors[i], linewidth=2,
                                 label=f'Class {class_idx} (AUC = {roc_auc:.3f})', ax=ax)

            ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            if title is None:
                title = 'ROC Curves (First 3 Classes)'
            ax.set_title(title)
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, "roc_curves.png", "model")
            
            return fig
            
        except Exception as e:
            print(f"Error creating ROC curves: {e}")
            return None


class DetectionVisualizer(BaseVisualizer):
    """Visualizer for adversarial detection analysis"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def create_roc_comparison(self, attacks: Optional[List[str]] = None,
                            characteristics: Optional[List[str]] = None,
                            save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Compare ROC curves across characteristics and attacks
        """
        if attacks is None:
            attacks = ["fgsm", "bim-a", "bim-b", "jsma"]
        if characteristics is None:
            characteristics = ["lid", "kd", "km"]
        
        try:
            all_data = load_all_characteristics(self.dataset, attacks, characteristics)
            
            num_attacks = len(attacks)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx, attack in enumerate(attacks):
                if attack not in all_data or not all_data[attack]:
                    axes[idx].text(0.5, 0.5, f'{attack.upper()}\nNo Data', 
                                 ha='center', va='center', transform=axes[idx].transAxes,
                                 fontsize=14)
                    axes[idx].set_xticks([])
                    axes[idx].set_yticks([])
                    continue
                
                ax = axes[idx]
                colors = sns.color_palette(self._palette("categorical"), len(characteristics))

                for char_idx, characteristic in enumerate(characteristics):
                    if characteristic not in all_data[attack]:
                        continue

                    X, y = all_data[attack][characteristic]

                    # Train simple detector
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.model_selection import train_test_split

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )

                    scaler = MinMaxScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    lr = LogisticRegression(max_iter=1000)
                    lr.fit(X_train_scaled, y_train)

                    y_probs = lr.predict_proba(X_test_scaled)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    roc_auc = auc(fpr, tpr)

                    sns.lineplot(x=fpr, y=tpr, color=colors[char_idx], linewidth=2,
                                 label=f'{characteristic.upper()} (AUC = {roc_auc:.3f})', ax=ax)
                
                ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{attack.upper()} - ROC Comparison')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(num_attacks, 4):
                axes[i].set_visible(False)
            
            if title is None:
                title = 'Adversarial Detector ROC Curves Comparison'
            plt.suptitle(title, fontsize=16, y=1.02)
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, "roc_comparison_all.png", "detection")
            
            return fig
            
        except Exception as e:
            print(f"Error creating ROC comparison: {e}")
            return None
    
    def create_3d_feature_space(self, attack: str = "fgsm",
                              characteristic: str = "lid",
                              save: bool = True,
                              title: Optional[str] = None) -> Optional[Figure]:
        """
        Create 3D scatter plot of feature space
        """
        try:
            X, y = load_characteristics_data(self.dataset, characteristic, attack)

            if X.shape[1] < 3:
                print(f"Not enough features for 3D plot (only {X.shape[1]} available)")
                return None

            # Sample to avoid overcrowding
            max_points = 500
            if len(X) > max_points:
                indices = np.random.choice(len(X), max_points, replace=False)
                X = X[indices]
                y = y[indices]

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')

            # Separate clean and adversarial
            clean_mask = y == 0
            adv_mask = y == 1
            cat_colors = sns.color_palette(self._palette("categorical"), 2)

            if np.any(clean_mask):
                ax.scatter(X[clean_mask, 0], X[clean_mask, 1], X[clean_mask, 2],
                          c=[cat_colors[0]], label='Normal', alpha=0.6, s=20)

            if np.any(adv_mask):
                ax.scatter(X[adv_mask, 0], X[adv_mask, 1], X[adv_mask, 2],
                          c=[cat_colors[1]], label='Adversarial', alpha=0.6, s=20)

            ax.set_xlabel(f'{characteristic.upper()} Feature 1')
            ax.set_ylabel(f'{characteristic.upper()} Feature 2')
            ax.set_zlabel(f'{characteristic.upper()} Feature 3')
            if title is None:
                title = f'3D Feature Space: {characteristic.upper()} on {attack.upper()}'
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, f"3d_{characteristic}_{attack}.png", "detection")
            
            return fig
            
        except Exception as e:
            print(f"Error creating 3D feature plot: {e}")
            return None
    
    def create_probability_distributions(self, attacks: Optional[List[str]] = None,
                                       characteristics: Optional[List[str]] = None,
                                       save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Create probability distribution histograms for detection outputs
        """
        if attacks is None:
            attacks = ["fgsm", "bim-a", "bim-b", "jsma"]
        if characteristics is None:
            characteristics = ["lid", "kd", "km"]
        
        try:
            all_data = load_all_characteristics(self.dataset, attacks, characteristics)
            
            num_plots = len(attacks) * len(characteristics)
            if num_plots == 0:
                return None
            
            fig, axes = plt.subplots(len(attacks), len(characteristics), 
                                   figsize=(6*len(characteristics), 5*len(attacks)))
            
            if num_plots == 1:
                axes = np.array([[axes]])
            elif len(attacks) == 1:
                axes = axes.reshape(1, -1)
            elif len(characteristics) == 1:
                axes = axes.reshape(-1, 1)
            
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import MinMaxScaler
            
            for i, attack in enumerate(attacks):
                for j, char in enumerate(characteristics):
                    if attack not in all_data or char not in all_data[attack]:
                        axes[i, j].text(0.5, 0.5, 'No Data', ha='center', va='center',
                                      transform=axes[i, j].transAxes)
                        continue
                    
                    X, y = all_data[attack][char]
                    
                    # Train detector
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    scaler = MinMaxScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    lr = LogisticRegression(max_iter=1000)
                    lr.fit(X_train_scaled, y_train)
                    
                    y_probs = lr.predict_proba(X_test_scaled)[:, 1]
                    
                    # Plot distributions
                    clean_probs = y_probs[y_test == 0]
                    adv_probs = y_probs[y_test == 1]
                    dist_colors = sns.color_palette(self._palette("sequential"), 2)

                    if len(clean_probs) > 0:
                        sns.histplot(clean_probs, bins=30, alpha=0.6,
                                     label='Normal', color=dist_colors[0],
                                     stat='density', ax=axes[i, j])
                    if len(adv_probs) > 0:
                        sns.histplot(adv_probs, bins=30, alpha=0.6,
                                     label='Adversarial', color=dist_colors[1],
                                     stat='density', ax=axes[i, j])
                    
                    axes[i, j].set_title(f'{attack.upper()} - {char.upper()}')
                    axes[i, j].set_xlabel('Detection Probability')
                    axes[i, j].set_ylabel('Density')
                    axes[i, j].legend()
                    axes[i, j].grid(True, alpha=0.3)
            
            if title is None:
                title = 'Detection Probability Distributions'
            plt.suptitle(title, fontsize=16, y=1.00)
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, "detection_prob_distributions.png", "detection")
            
            return fig
            
        except Exception as e:
            print(f"Error creating probability distributions: {e}")
            return None
    
    def create_metrics_comparison(self, attacks: Optional[List[str]] = None,
                                 characteristics: Optional[List[str]] = None,
                                 save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Create comprehensive metrics comparison bar charts
        """
        if attacks is None:
            attacks = ["fgsm", "bim-a", "bim-b", "jsma"]
        if characteristics is None:
            characteristics = ["lid", "kd", "km"]
        
        try:
            all_data = load_all_characteristics(self.dataset, attacks, characteristics)
            
            metrics_data = []
            
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import MinMaxScaler
            
            for attack in attacks:
                for char in characteristics:
                    if attack not in all_data or char not in all_data[attack]:
                        continue
                    
                    X, y = all_data[attack][char]
                    
                    # Train detector
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    scaler = MinMaxScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    lr = LogisticRegression(max_iter=1000)
                    lr.fit(X_train_scaled, y_train)
                    
                    y_pred = lr.predict(X_test_scaled)
                    y_probs = lr.predict_proba(X_test_scaled)[:, 1]
                    
                    # Calculate metrics
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    auc_score = auc(fpr, tpr)
                    
                    metrics_data.append({
                        'attack': attack,
                        'characteristic': char,
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'auc': auc_score
                    })
            
            if not metrics_data:
                return None
            
            df = pd.DataFrame(metrics_data)
            
            # Create 2x2 grid of metrics
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'auc']
            titles = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']
            
            for idx, (metric, subplot_title) in enumerate(zip(metrics_to_plot, titles)):
                ax = axes[idx // 2, idx % 2]

                sns.barplot(data=df, x='attack', y=metric, hue='characteristic',
                            palette=self._palette("categorical"), ax=ax)

                ax.set_title(subplot_title)
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_xlabel('Attack')
                ax.legend(title='Characteristic', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylim(0, 1.05)
            
            if title is None:
                title = 'Detection Metrics Comparison Across Attacks and Characteristics'
            plt.suptitle(title, fontsize=16, y=1.00)
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, "metrics_comparison.png", "detection")
            
            return fig
            
        except Exception as e:
            print(f"Error creating metrics comparison: {e}")
            return None


class TDAVisualizer(BaseVisualizer):
    """Visualizer for Topological Data Analysis (TDA) results"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _get_tda_path(self, attack: str) -> Path:
        from visualizer.config import ANALYSIS_DIR
        return ANALYSIS_DIR / self.dataset / f"tda_{self.dataset}_{attack}.json"
    
    def create_persistence_diagram(self, attack: str, save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Plot persistence diagrams for a given TDA result
        """
        try:
            import json

            tda_path = self._get_tda_path(attack)
            if not tda_path.exists():
                tda_path = RESULTS_DIR / "tda" / f"tda_{self.dataset}.json"
            if not tda_path.exists():
                tda_path = RESULTS_DIR / "tda" / f"{attack}_{self.dataset}.json"
            if not tda_path.exists():
                print(f"TDA data not found at {tda_path}")
                return None
                
            with open(tda_path, 'r') as f:
                data = json.load(f)
            
            diagrams = data['diagrams']
            
            fig, axes = plt.subplots(1, len(diagrams), figsize=(6 * len(diagrams), 5))
            if len(diagrams) == 1:
                axes = [axes]
            
            for dim, dgm in enumerate(diagrams):
                ax = axes[dim]
                dgm = np.array(dgm)
                
                if len(dgm) > 0:
                    finite_mask = np.isfinite(dgm[:, 1])
                    finite_dgm = dgm[finite_mask]
                    
                    ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], s=25, c=f'C{dim}', 
                              alpha=0.6, label=f'H{dim}')
                    
                    max_val = np.max(finite_dgm) if len(finite_dgm) > 0 else 1.0
                    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4)
                    
                    inf_points = dgm[~finite_mask]
                    if len(inf_points) > 0:
                        ax.scatter(inf_points[:, 0], [max_val * 1.1] * len(inf_points), 
                                  marker='x', c='red', s=50, label='Infinite')
                
                ax.set_title(f'Dimension {dim} Persistence')
                ax.set_xlabel('Birth')
                ax.set_ylabel('Death')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            if title is None:
                title = f'Persistence Diagrams: {attack.upper()}'
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, f"persistence_{attack}.png", "tda")
            
            return fig
            
        except Exception as e:
            print(f"Error creating persistence diagram: {e}")
            return None

    def create_feature_comparison(self, attacks: List[str], save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Compare topological features across different attacks
        """
        try:
            import json

            all_features = []
            for attack in attacks:
                tda_path = self._get_tda_path(attack)
                if not tda_path.exists():
                    tda_path = RESULTS_DIR / "tda" / f"{attack}_{self.dataset}.json"
                if tda_path.exists():
                    with open(tda_path, 'r') as f:
                        data = json.load(f)
                        feat = data['features']
                        feat['attack_name'] = attack
                        all_features.append(feat)
            
            if not all_features:
                print("No TDA features found for comparison")
                return None
                
            df = pd.DataFrame(all_features)
            
            plot_features = [f for f in df.columns if 'max_persistence' in f or 'avg_death' in f]
            
            fig, axes = plt.subplots(1, len(plot_features), figsize=(5 * len(plot_features), 6))
            if len(plot_features) == 1:
                axes = [axes]
                
            for i, feat in enumerate(plot_features):
                ax = axes[i]
                sns.barplot(x='attack_name', y=feat, data=df, ax=ax)
                ax.set_title(feat.replace('_', ' ').title())
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                ax.set_xlabel('')
                ax.grid(True, alpha=0.3, axis='y')
            
            if title is None:
                title = 'Topological Feature Comparison'
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, "tda_feature_comparison.png", "tda")
            
            return fig
            
        except Exception as e:
            print(f"Error creating TDA feature comparison: {e}")
            return None

    def create_clean_vs_adversarial_comparison(self, attack: str, save: bool = True, title: Optional[str] = None) -> Optional[Figure]:
        """
        Create a side-by-side comparison of clean vs adversarial TDA results
        """
        try:
            import json

            clean_path = self._get_tda_path("clean")
            adv_path = self._get_tda_path(attack)
            if not clean_path.exists():
                clean_path = RESULTS_DIR / "tda" / f"clean_{self.dataset}.json"
            if not adv_path.exists():
                adv_path = RESULTS_DIR / "tda" / f"{attack}_{self.dataset}.json"
            
            if not clean_path.exists() or not adv_path.exists():
                print(f"TDA data not found. Clean: {clean_path.exists()}, Adv: {adv_path.exists()}")
                return None
                
            with open(clean_path, 'r') as f:
                clean_data = json.load(f)
            with open(adv_path, 'r') as f:
                adv_data = json.load(f)
            
            clean_dgms = clean_data['diagrams']
            adv_dgms = adv_data['diagrams']
            
            # We'll plot H1 for both
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Clean H1 Persistence
            ax = axes[0, 0]
            dgm = np.array(clean_dgms[1]) if len(clean_dgms) > 1 else np.array([])
            if len(dgm) > 0:
                finite_dgm = dgm[np.isfinite(dgm[:, 1])]
                ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], s=30, c='blue', alpha=0.6, label='Clean H1')
                max_val = np.max(finite_dgm) if len(finite_dgm) > 0 else 1.0
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4)
            ax.set_title('Clean Data Persistence (H1)')
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Adversarial H1 Persistence
            ax = axes[0, 1]
            dgm = np.array(adv_dgms[1]) if len(adv_dgms) > 1 else np.array([])
            if len(dgm) > 0:
                finite_dgm = dgm[np.isfinite(dgm[:, 1])]
                ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], s=30, c='red', alpha=0.6, label=f'{attack.upper()} H1')
                max_val = np.max(finite_dgm) if len(finite_dgm) > 0 else 1.0
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4)
            ax.set_title(f'Adversarial ({attack.upper()}) Persistence (H1)')
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. Feature Comparison Bar Chart
            ax = axes[1, 0]
            clean_feat = clean_data['features']
            adv_feat = adv_data['features']
            
            labels = ['Max Persistence', 'Avg Death', 'Num Points']
            clean_vals = [clean_feat['dim1_max_persistence'], clean_feat['dim1_avg_death'], clean_feat['dim1_num_points']]
            adv_vals = [adv_feat['dim1_max_persistence'], adv_feat['dim1_avg_death'], adv_feat['dim1_num_points']]
            
            # Normalize for visualization if needed, but here we just plot
            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width/2, clean_vals, width, label='Clean', color='blue', alpha=0.7)
            ax.bar(x + width/2, adv_vals, width, label='Adversarial', color='red', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title('H1 Feature Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 4. Persistence Histogram
            ax = axes[1, 1]
            if len(clean_dgms) > 1 and len(adv_dgms) > 1:
                c_dgm = np.array(clean_dgms[1])
                a_dgm = np.array(adv_dgms[1])
                c_pers = c_dgm[np.isfinite(c_dgm[:, 1]), 1] - c_dgm[np.isfinite(c_dgm[:, 1]), 0]
                a_pers = a_dgm[np.isfinite(a_dgm[:, 1]), 1] - a_dgm[np.isfinite(a_dgm[:, 1]), 0]
                
                ax.hist(c_pers, bins=20, alpha=0.5, label='Clean', color='blue', density=True)
                ax.hist(a_pers, bins=20, alpha=0.5, label='Adversarial', color='red', density=True)
                ax.set_title('H1 Persistence Distribution')
                ax.set_xlabel('Persistence (Death - Birth)')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            if title is None:
                title = f'TDA Comparison: Clean vs {attack.upper()} Adversarial ({self.dataset.upper()})'
            plt.suptitle(title, fontsize=16)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            
            if save:
                self.save_figure(fig, f"tda_comparison_clean_{attack}.png", "tda")
            
            return fig
            
        except Exception as e:
            print(f"Error creating TDA clean vs adversarial comparison: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_correlation_matrix_plot(self, attack: str, save: bool = True) -> Optional[Figure]:
        """
        Visualize the neuron correlation matrix
        """
        try:
            import json

            tda_path = self._get_tda_path(attack)
            if not tda_path.exists():
                tda_path = RESULTS_DIR / "tda" / f"tda_{self.dataset}.json"
            if not tda_path.exists():
                tda_path = RESULTS_DIR / "tda" / f"{attack}_{self.dataset}.json"
            if not tda_path.exists():
                print(f"TDA data not found at {tda_path}")
                return None
                
            with open(tda_path, 'r') as f:
                data = json.load(f)
            
            if 'correlation_matrix' not in data:
                print(f"Correlation matrix not found in {tda_path}")
                return None
                
            corr_matrix = np.array(data['correlation_matrix'])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            
            ax.set_title(f'Neuron Correlation Matrix: {attack.upper()}')
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Neuron Index')
            
            plt.tight_layout()
            
            if save:
                self.save_figure(fig, f"correlation_matrix_{attack}.png", "tda")
            
            return fig
            
        except Exception as e:
            print(f"Error creating correlation matrix plot: {e}")
            return None
