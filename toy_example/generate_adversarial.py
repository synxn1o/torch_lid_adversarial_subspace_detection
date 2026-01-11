#!/usr/bin/env python3
"""
Generate adversarial examples for toy dataset and visualize the results.
This script creates attack visualizations and comparisons.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PatchCollection

# Import our custom modules
from attacks_toy import generate_adversarial_examples, save_adversarial_results, load_adversarial_results
from generate_dataset import load_dataset

def load_model(model_path, device):
    """Load the trained binary neural network."""
    from train_NN import BinaryNN
    
    # Load training results to get model architecture info
    results_path = "toy_example/models/training_results.pkl"
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded training results (test accuracy: {results['test_accuracy']:.4f})")
    
    # Initialize model
    model = BinaryNN(input_dim=2)
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model not found at {model_path}")
        return None
    
    return model

def visualize_attack_process(results, attack_name, model, n_samples=5, save_path=None):
    """
    Visualize the attack process step-by-step for BIM/JSMA.
    
    Args:
        results: Adversarial results dictionary
        attack_name: Name of attack to visualize
        model: The PyTorch model for drawing decision boundaries
        n_samples: Number of samples to show
        save_path: Path to save the visualization
    """
    if attack_name not in results['attacks']:
        print(f"Attack {attack_name} not found in results")
        return
    
    attack_data = results['attacks'][attack_name]
    
    if 'perturbation_history' not in attack_data:
        print(f"No perturbation history for {attack_name}")
        return
    
    pert_history_raw = attack_data['perturbation_history']
    labels = results['labels']
    
    # Handle different formats: BIM returns (n_steps, n_samples, 2), JSMA returns list of arrays
    if isinstance(pert_history_raw, list):
        # JSMA format: list of (n_steps, 2) arrays
        pert_history = np.array(pert_history_raw)  # (n_samples, n_steps, 2)
        pert_history = np.transpose(pert_history, (1, 0, 2))  # (n_steps, n_samples, 2)
    else:
        # BIM format: already (n_steps, n_samples, 2)
        pert_history = pert_history_raw
    
    # Select samples
    n_samples = min(n_samples, pert_history.shape[1])
    indices = np.random.choice(pert_history.shape[1], n_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    
    # Color scheme
    clean_color = 'green'
    adv_color = 'red'
    path_cmap = 'viridis'
    
    for idx, sample_idx in enumerate(indices):
        ax = axes[idx]
        
        # Get trajectory
        trajectory = pert_history[:, sample_idx, :]  # (n_steps, 2)
        clean_point = results['clean'][sample_idx]
        final_adv = trajectory[-1]
        label = labels[sample_idx]
        
        # Plot trajectory
        if len(trajectory) > 1:
            # Color by step
            colors = np.arange(len(trajectory))
            scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], 
                               c=colors, cmap=path_cmap, s=30, alpha=0.6, zorder=2)
            
            # Add arrows to show direction
            for i in range(len(trajectory)-1):
                arrow = FancyArrowPatch(
                    (trajectory[i, 0], trajectory[i, 1]),
                    (trajectory[i+1, 0], trajectory[i+1, 1]),
                    arrowstyle='-|>', mutation_scale=10,
                    color='black', alpha=0.4, zorder=3
                )
                ax.add_patch(arrow)
        
        # Plot start and end points
        ax.scatter(clean_point[0], clean_point[1], 
                  c=clean_color, s=100, marker='o', edgecolors='black', 
                  linewidth=2, label=f'Clean (C{label})', zorder=4)
        ax.scatter(final_adv[0], final_adv[1], 
                  c=adv_color, s=100, marker='s', edgecolors='black', 
                  linewidth=2, label=f'Adv (C{1-label})', zorder=4)
        
        # Draw decision boundary
        x_range = np.linspace(-2, 2, 100)
        y_range = np.linspace(-2, 2, 100)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        with torch.no_grad():
            grid_tensor = torch.FloatTensor(grid_points).to(next(model.parameters()).device)
            outputs = model(grid_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().reshape(xx.shape)
        
        ax.contour(xx, yy, probs, levels=[0.5], colors='black', linestyles='--', alpha=0.6)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sample {sample_idx} - Label: C{label}')
        ax.legend()
    
    plt.suptitle(f'Attack Process: {attack_name.upper()}\nTrajectory from Clean to Adversarial', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attack process visualization saved to {save_path}")
    
    plt.show()

def compare_clean_vs_adv(results, attack_name, model, n_samples=20, save_path=None):
    """
    Compare clean vs adversarial examples.
    
    Args:
        results: Adversarial results dictionary
        attack_name: Name of attack to visualize
        model: The PyTorch model for drawing decision boundaries
        n_samples: Number of samples to show
        save_path: Path to save the visualization
    """
    if attack_name not in results['attacks']:
        print(f"Attack {attack_name} not found in results")
        return
    
    X_clean = results['clean']
    y = results['labels']
    X_adv = results['attacks'][attack_name]['examples']
    
    # Select samples
    n_samples = min(n_samples, len(X_clean))
    indices = np.random.choice(len(X_clean), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Scatter comparison
    ax1 = axes[0, 0]
    c1_mask = y == 0
    c2_mask = y == 1
    
    # Plot clean points
    ax1.scatter(X_clean[c1_mask, 0], X_clean[c1_mask, 1], 
               c='lightgreen', alpha=0.5, s=40, label='C1 Clean', edgecolors='black', linewidth=0.5)
    ax1.scatter(X_clean[c2_mask, 0], X_clean[c2_mask, 1], 
               c='lightblue', alpha=0.5, s=40, label='C2 Clean', edgecolors='black', linewidth=0.5)
    
    # Plot adversarial points
    adv_c1_mask = (y == 0) & (c1_mask)  # C1 that might be misclassified
    adv_c2_mask = (y == 1) & (c2_mask)  # C2 that might be misclassified
    
    ax1.scatter(X_adv[c1_mask, 0], X_adv[c1_mask, 1], 
               c='red', alpha=0.7, s=40, marker='s', label='C1 Adversarial', edgecolors='black', linewidth=1)
    ax1.scatter(X_adv[c2_mask, 0], X_adv[c2_mask, 1], 
               c='blue', alpha=0.7, s=40, marker='s', label='C2 Adversarial', edgecolors='black', linewidth=1)
    
    # Draw arrows for perturbations
    for i in indices:
        if c1_mask[i] or c2_mask[i]:
            ax1.annotate('', 
                        xy=(X_adv[i, 0], X_adv[i, 1]),
                        xytext=(X_clean[i, 0], X_clean[i, 1]),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.4, lw=1))
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Clean vs Adversarial Examples')
    ax1.legend()
    
    # Plot 2: Distribution of perturbations
    ax2 = axes[0, 1]
    perturbations = X_adv - X_clean
    distances = np.linalg.norm(perturbations, axis=1)
    
    ax2.hist(distances, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax2.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(distances):.3f}')
    ax2.set_xlabel('Perturbation Magnitude (L2 distance)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Perturbations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Decision boundary with samples
    ax3 = axes[1, 0]
    
    # Create mesh for decision boundary
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(grid_points).to(next(model.parameters()).device)
        outputs = model(grid_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy().reshape(xx.shape)
    
    # Plot decision boundary
    contour = ax3.contourf(xx, yy, probs, levels=20, cmap='RdBu_r', alpha=0.6)
    plt.colorbar(contour, ax=ax3, label='P(C2)')
    ax3.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)
    
    # Plot sample points (show first 50 to avoid clutter)
    sample_limit = min(50, len(indices))
    sample_indices = indices[:sample_limit]
    
    for i in sample_indices:
        ax3.plot([X_clean[i, 0], X_adv[i, 0]], [X_clean[i, 1], X_adv[i, 1]], 
                'k-', alpha=0.3, linewidth=1)
        ax3.scatter(X_clean[i, 0], X_clean[i, 1], 
                   c=['lightgreen' if y[i]==0 else 'lightblue'], s=30, marker='o', edgecolors='black')
        ax3.scatter(X_adv[i, 0], X_adv[i, 1], 
                   c=['red' if y[i]==0 else 'blue'], s=30, marker='s', edgecolors='black')
    
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Decision Boundary with Perturbations')
    
    # Plot 4: Success rate and metrics
    ax4 = axes[1, 1]
    
    # Compute metrics
    with torch.no_grad():
        X_adv_t = torch.FloatTensor(X_adv).to(next(model.parameters()).device)
        outputs = model(X_adv_t)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
        
        # Success rate
        success_rate = np.mean(preds != y)
        
        # L2 distance statistics
        l2_mean = np.mean(distances)
        l2_std = np.std(distances)
        
        # Misclassified counts
        misclassified_c1 = np.sum((y == 0) & (preds != y))
        misclassified_c2 = np.sum((y == 1) & (preds != y))
    
    # Create bar plot
    metrics = ['Success Rate', 'L2 Mean', 'L2 Std', 'C1 Miscls', 'C2 Miscls']
    values = [success_rate, l2_mean, l2_std, misclassified_c1, misclassified_c2]
    colors = ['red', 'blue', 'purple', 'orange', 'cyan']
    
    bars = ax4.bar(range(len(metrics)), values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(metrics)))
    ax4.set_xticklabels(metrics, rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if i == 0:  # Success rate as percentage
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        else:
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax4.set_ylabel('Value')
    ax4.set_title(f'Metrics for {attack_name.upper()}')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved to {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n--- Summary for {attack_name.upper()} ---")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average L2 Distance: {l2_mean:.4f} ± {l2_std:.4f}")
    print(f"C1 Misclassified: {misclassified_c1}/{np.sum(y==0)}")
    print(f"C2 Misclassified: {misclassified_c2}/{np.sum(y==1)}")

def create_summary_comparison(results, save_path=None):
    """
    Create a comprehensive comparison across all attacks.
    
    Args:
        results: Adversarial results dictionary
        save_path: Path to save the visualization
    """
    attacks = list(results['attacks'].keys())
    if not attacks:
        print("No attacks found in results")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Success rates
    ax1 = axes[0, 0]
    success_rates = [results['attacks'][att]['success_rate'] for att in attacks]
    bars = ax1.bar(range(len(attacks)), success_rates, 
                   color=['red', 'blue', 'green', 'orange'][:len(attacks)], 
                   alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(attacks)))
    ax1.set_xticklabels(attacks, rotation=45)
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Attack Success Rates')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # Plot 2: L2 distances
    ax2 = axes[0, 1]
    l2_means = []
    l2_stds = []
    
    for att in attacks:
        X_clean = results['clean']
        X_adv = results['attacks'][att]['examples']
        distances = np.linalg.norm(X_adv - X_clean, axis=1)
        l2_means.append(np.mean(distances))
        l2_stds.append(np.std(distances))
    
    x_pos = range(len(attacks))
    bars = ax2.bar(x_pos, l2_means, yerr=l2_stds, capsize=5,
                   color=['purple', 'cyan', 'magenta', 'yellow'][:len(attacks)], 
                   alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(attacks, rotation=45)
    ax2.set_ylabel('L2 Distance')
    ax2.set_title('Perturbation Magnitude (L2)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Attack effectiveness scatter
    ax3 = axes[1, 0]
    
    perturbation_sizes = []
    success_rates_plot = []
    labels = []
    
    for att in attacks:
        X_clean = results['clean']
        X_adv = results['attacks'][att]['examples']
        distances = np.linalg.norm(X_adv - X_clean, axis=1)
        
        perturbation_sizes.append(np.mean(distances))
        success_rates_plot.append(results['attacks'][att]['success_rate'])
        labels.append(att)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(attacks)))
    for i in range(len(attacks)):
        ax3.scatter(perturbation_sizes[i], success_rates_plot[i], 
                   s=200, c=[colors[i]], alpha=0.7, edgecolors='black', linewidth=2)
        ax3.annotate(labels[i], 
                    (perturbation_sizes[i], success_rates_plot[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Average L2 Perturbation')
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Effectiveness vs Stealthiness')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample comparison for each attack
    ax4 = axes[1, 1]
    
    # Show a single representative sample for each attack
    sample_idx = 0  # Use first sample
    if sample_idx < len(results['clean']):
        clean_pt = results['clean'][sample_idx]
        y_label = results['labels'][sample_idx]
        
        # Plot clean point
        ax4.scatter(clean_pt[0], clean_pt[1], s=200, c='green', 
                   marker='o', edgecolors='black', linewidth=2, label='Clean', zorder=5)
        
        # Plot adversarial points for each attack
        for i, att in enumerate(attacks):
            adv_pt = results['attacks'][att]['examples'][sample_idx]
            ax4.scatter(adv_pt[0], adv_pt[1], s=150, c=[colors[i]], 
                       marker='s', edgecolors='black', linewidth=1, alpha=0.8, zorder=4)
            ax4.annotate(att.upper(), (adv_pt[0], adv_pt[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Draw decision boundary
        x_range = np.linspace(-2, 2, 50)
        y_range = np.linspace(-2, 2, 50)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        from train_NN import BinaryNN
        model = BinaryNN(input_dim=2)
        model_path = "toy_example/models/toy_binary_nn.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            with torch.no_grad():
                grid_tensor = torch.FloatTensor(grid_points)
                outputs = model(grid_tensor)
                probs = torch.sigmoid(outputs).numpy().reshape(xx.shape)
            
            ax4.contour(xx, yy, probs, levels=[0.5], colors='black', linestyles='--', alpha=0.6)
        
        ax4.set_xlim(-2, 2)
        ax4.set_ylim(-2, 2)
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        ax4.set_title(f'Sample {sample_idx} (C{y_label}) Attack Comparison')
        ax4.legend()
    
    plt.suptitle('Adversarial Attack Analysis Summary', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary comparison saved to {save_path}")
    
    plt.show()

def main():
    """Main execution function."""
    print("=" * 70)
    print("Adversarial Attack Generation and Visualization for Toy Dataset")
    print("=" * 70)
    
    # Configuration
    data_path = "toy_example/data/circle_dataset.pkl"
    model_path = "toy_example/models/toy_binary_nn.pth"
    output_dir = "toy_example/results"
    plots_dir = "toy_example/plots"
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    X, y = load_dataset(data_path)
    # Convert labels from [-1, 1] to [0, 1] for binary classification
    y_binary = ((y + 1) / 2).astype(int)
    print(f"   Loaded {len(X)} samples")
    print(f"   Class distribution: C1={np.sum(y_binary==0)}, C2={np.sum(y_binary==1)}")
    
    # 2. Load model
    print("\n2. Loading model...")
    model = load_model(model_path, device)
    if model is None:
        print("   Failed to load model. Exiting.")
        return
    
    # 3. Generate adversarial examples
    print("\n3. Generating adversarial examples...")
    attacks = ['fgsm', 'bim-a', 'bim-b', 'jsma']
    
    # Check if results already exist
    results_file = os.path.join(output_dir, "adversarial_results.pkl")
    if os.path.exists(results_file):
        print("   Loading existing results...")
        results = load_adversarial_results(results_file)
    else:
        print("   Creating new results...")
        results = generate_adversarial_examples(
            model, X, y_binary, 
            attacks=attacks,
            eps=0.3,  # Epsilon for FGSM and BIM
            eps_iter=0.02  # Step size for BIM
        )
        save_adversarial_results(results, results_file)
    
    # 4. Visualize attack processes
    print("\n4. Visualizing attack processes...")
    
    # For iterative attacks (BIM, JSMA)
    for attack in ['bim-a', 'bim-b', 'jsma']:
        if attack in results['attacks']:
            print(f"   Creating visualization for {attack}...")
            viz_path = os.path.join(plots_dir, f"attack_process_{attack}.png")
            visualize_attack_process(results, attack, model, n_samples=5, save_path=viz_path)
    
    # 5. Compare clean vs adversarial
    print("\n5. Comparing clean vs adversarial examples...")
    for attack in attacks:
        if attack in results['attacks']:
            print(f"   Creating comparison for {attack}...")
            comp_path = os.path.join(plots_dir, f"comparison_{attack}.png")
            compare_clean_vs_adv(results, attack, model, n_samples=20, save_path=comp_path)
    
    # 6. Create summary comparison
    print("\n6. Creating summary comparison...")
    summary_path = os.path.join(plots_dir, "summary_comparison.png")
    create_summary_comparison(results, save_path=summary_path)
    
    print("\n" + "=" * 70)
    print("All visualizations created!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")
    
    # Print final summary
    print("\nFinal Summary:")
    for attack in attacks:
        if attack in results['attacks']:
            success = results['attacks'][attack]['success_rate']
            print(f"  {attack.upper()}: {success:.2%} success rate")

if __name__ == "__main__":
    main()