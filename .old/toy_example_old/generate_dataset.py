"""
Generate a 2D dataset with two Gaussian clusters:
- C1: label -1 (red)
- C2: label +1 (blue)

This script generates the data, saves it to disk, and creates visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.datasets import make_circles

# Set random seed for reproducibility
np.random.seed(42)

def generate_gaussian_dataset(n_samples_per_class=2000, 
                              c1_mean=(-1, 0), 
                              c2_mean=(1, 0),
                              c1_cov=[[0.09, 0], [0, 0.09]],
                              c2_cov=[[0.09, 0], [0, 0.09]]):
    """
    Generate a 2D Gaussian dataset with two classes.
    
    Parameters:
    - n_samples_per_class: Number of samples per class
    - c1_mean: Mean of class 1 (label -1)
    - c2_mean: Mean of class 2 (label +1)
    - c1_cov: Covariance matrix for class 1
    - c2_cov: Covariance matrix for class 2
    
    Returns:
    - X: Feature matrix (n_samples_per_class * 2, 2)
    - y: Label vector (n_samples_per_class * 2,)
    """
    
    # Generate C1 data (label -1)
    c1_data = np.random.multivariate_normal(
        mean=c1_mean, 
        cov=c1_cov, 
        size=n_samples_per_class
    )
    c1_labels = np.full(n_samples_per_class, -1)
    
    # Generate C2 data (label +1)
    c2_data = np.random.multivariate_normal(
        mean=c2_mean, 
        cov=c2_cov, 
        size=n_samples_per_class
    )
    c2_labels = np.full(n_samples_per_class, 1)
    
    # Combine datasets
    X = np.vstack([c1_data, c2_data])
    y = np.hstack([c1_labels, c2_labels])
    
    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

def generate_circle_dataset(n_samples=4000, noise=0.05, factor=0.3):
    """
    Generate a 2D nested circle dataset from sklearn, with labels converted to match our convention.
    
    Parameters:
    - n_samples: Total number of samples (will be split evenly between classes)
    - noise: Standard deviation of Gaussian noise added to the data
    - factor: Ratio between the inner and outer circle
    
    Returns:
    - X: Feature matrix (n_samples, 2)
    - y: Label vector (n_samples,) with values -1 (outer circle) and +1 (inner circle)
    """
    # Generate circles using sklearn
    # sklearn returns 0 for outer circle, 1 for inner circle
    X, y_sklearn = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    
    # Convert sklearn labels to our convention: outer circle = -1, inner circle = +1
    # sklearn: 0 = outer, 1 = inner
    # ours: -1 = outer, +1 = inner
    y = np.where(y_sklearn == 0, -1, 1)
    
    return X, y

def save_dataset(X, y, filepath, description=None):
    """Save dataset to disk using pickle."""
    if description is None:
        description = '2D Gaussian dataset with C1 (label=-1) and C2 (label=+1)'
    
    data = {
        'X': X,
        'y': y,
        'description': description,
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Dataset saved to {filepath}")
    print(f"  - Shape: X={X.shape}, y={y.shape}")
    print(f"  - Class distribution: C1(-1)={np.sum(y==-1)}, C2(+1)={np.sum(y==1)}")

def load_dataset(filepath):
    """Load dataset from disk."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']

def visualize_dataset(X, y, save_path=None, dataset_name="Dataset"):
    """Create comprehensive visualizations of the dataset."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name}: C1 (label=-1) vs C2 (label=+1)', fontsize=16)
    
    # Plot 1: Scatter plot
    ax1 = axes[0, 0]
    c1_mask = y == -1
    c2_mask = y == 1
    
    ax1.scatter(X[c1_mask, 0], X[c1_mask, 1], 
                c='red', alpha=0.6, s=20, label='C1 (label=-1)', edgecolors='black', linewidth=0.3)
    ax1.scatter(X[c2_mask, 0], X[c2_mask, 1], 
                c='blue', alpha=0.6, s=20, label='C2 (label=+1)', edgecolors='black', linewidth=0.3)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Scatter Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Histograms per feature
    ax2 = axes[0, 1]
    ax2.hist(X[c1_mask, 0], bins=30, alpha=0.6, color='red', label='C1 Feature 1', density=True)
    ax2.hist(X[c2_mask, 0], bins=30, alpha=0.6, color='blue', label='C2 Feature 1', density=True)
    ax2.set_xlabel('Feature 1 Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Feature 1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 2D Density estimation
    ax3 = axes[1, 0]
    # Create hexbin plot
    hb = ax3.hexbin(X[:, 0], X[:, 1], gridsize=20, cmap='coolwarm', mincnt=1, alpha=0.8)
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    ax3.set_title('Density Heatmap')
    plt.colorbar(hb, ax=ax3, label='Count')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Plot 4: Box plots
    ax4 = axes[1, 1]
    bp1 = ax4.boxplot([X[c1_mask, 0], X[c1_mask, 1]], 
                       positions=[1, 2], widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='red', alpha=0.6),
                       medianprops=dict(color='black'),
                       labels=['F1', 'F2'])
    bp2 = ax4.boxplot([X[c2_mask, 0], X[c2_mask, 1]], 
                       positions=[4, 5], widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='blue', alpha=0.6),
                       medianprops=dict(color='black'),
                       labels=['F1', 'F2'])
    ax4.set_ylabel('Feature Value')
    ax4.set_title('Feature Distributions by Class')
    ax4.set_xticks([1.5, 4.5])
    ax4.set_xticklabels(['C1 (-1)', 'C2 (+1)'])
    ax4.grid(True, alpha=0.3)
    
    # Add some statistics as text
    stats_text = f"""
    Dataset Statistics:
    Total samples: {len(X)}
    C1 samples: {np.sum(c1_mask)} 
    C2 samples: {np.sum(c2_mask)}
    
    Feature 1:
      C1 mean: {np.mean(X[c1_mask, 0]):.3f}, std: {np.std(X[c1_mask, 0]):.3f}
      C2 mean: {np.mean(X[c2_mask, 0]):.3f}, std: {np.std(X[c2_mask, 0]):.3f}
    
    Feature 2:
      C1 mean: {np.mean(X[c1_mask, 1]):.3f}, std: {np.std(X[c1_mask, 1]):.3f}
      C2 mean: {np.mean(X[c2_mask, 1]):.3f}, std: {np.std(X[c2_mask, 1]):.3f}
    """
    
    # Add text box to figure
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig

def visualize_boundary(X, y, save_path=None):
    """Create a decision boundary visualization using logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression
    clf = LogisticRegression(random_state=42)
    clf.fit(X_scaled, y)
    
    # Create mesh for decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict on mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu_r')
    
    c1_mask = y == -1
    c2_mask = y == 1
    
    plt.scatter(X_scaled[c1_mask, 0], X_scaled[c1_mask, 1], 
                c='red', alpha=0.7, s=20, label='C1 (label=-1)', edgecolors='black', linewidth=0.3)
    plt.scatter(X_scaled[c2_mask, 0], X_scaled[c2_mask, 1], 
                c='blue', alpha=0.7, s=20, label='C2 (label=+1)', edgecolors='black', linewidth=0.3)
    
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title('Dataset with Learned Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Boundary plot saved to {save_path}")
    
    plt.show()

def main():
    """Main execution function."""
    print("=" * 60)
    print("2D Dataset Generator (Gaussian and Circle)")
    print("=" * 60)
    
    # Parameters
    n_samples = 2000
    output_dir = "toy_example/data"
    
    # Generate Gaussian dataset
    print("\n1. Generating Gaussian dataset...")
    X_gauss, y_gauss = generate_gaussian_dataset(n_samples_per_class=n_samples)
    gauss_data_file = os.path.join(output_dir, "gaussian_dataset.pkl")
    save_dataset(X_gauss, y_gauss, gauss_data_file, 
                 description='2D Gaussian dataset with C1 (label=-1) and C2 (label=+1)')
    
    # Visualize Gaussian dataset
    print("\n2. Creating Gaussian dataset visualizations...")
    viz_file = os.path.join(output_dir, "gaussian_visualization.png")
    visualize_dataset(X_gauss, y_gauss, save_path=viz_file, dataset_name="Gaussian Dataset")
    
    # Decision boundary visualization
    print("\n3. Creating Gaussian decision boundary visualization...")
    boundary_file = os.path.join(output_dir, "gaussian_decision_boundary.png")
    visualize_boundary(X_gauss, y_gauss, save_path=boundary_file)
    
    # Generate Circle dataset
    print("\n4. Generating Circle dataset...")
    X_circle, y_circle = generate_circle_dataset(n_samples=n_samples*2, noise=0.05, factor=0.3)
    circle_data_file = os.path.join(output_dir, "circle_dataset.pkl")
    save_dataset(X_circle, y_circle, circle_data_file, 
                 description='2D Nested Circle dataset with C1 (outer, label=-1) and C2 (inner, label=+1)')
    
    # Visualize Circle dataset
    print("\n5. Creating Circle dataset visualizations...")
    circle_viz_file = os.path.join(output_dir, "circle_visualization.png")
    visualize_dataset(X_circle, y_circle, save_path=circle_viz_file, dataset_name="Circle Dataset")
    
    # Decision boundary visualization for circle
    print("\n6. Creating Circle decision boundary visualization...")
    circle_boundary_file = os.path.join(output_dir, "circle_decision_boundary.png")
    visualize_boundary(X_circle, y_circle, save_path=circle_boundary_file)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"Files created:")
    print(f"  - Gaussian data: {gauss_data_file}")
    print(f"  - Gaussian visualization: {viz_file}")
    print(f"  - Gaussian boundary: {boundary_file}")
    print(f"  - Circle data: {circle_data_file}")
    print(f"  - Circle visualization: {circle_viz_file}")
    print(f"  - Circle boundary: {circle_boundary_file}")
    print("\nYou can now use these datasets for training/testing models.")

if __name__ == "__main__":
    main()