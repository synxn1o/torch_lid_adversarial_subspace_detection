import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

# Import for 3D plotting
from mpl_toolkits.mplot3d import Axes3D

def load_characteristics(file_path):
    """Load extracted characteristics from .npy file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = np.load(file_path)
    X = data[:, :-1]
    y = data[:, -1]
    
    # Clean data: handle Inf and NaN
    if not np.isfinite(X).all():
        print(f"Warning: Found {np.sum(~np.isfinite(X))} non-finite values in {os.path.basename(file_path)}. Cleaning...")
        # Replace inf with max finite value (if any finite values exist)
        # Or replace with a large but finite number
        finite_max = np.finfo(X.dtype).max / 2 # Using half max to be safe
        X[np.isinf(X)] = np.nan # First convert inf to nan to use nan_to_num
        
        # Replace nan with 0.0 or a sensible default. For LID 0.0 means collapsed, which is ok.
        X = np.nan_to_num(X, nan=0.0, posinf=finite_max, neginf=-finite_max)
        
    return X, y

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train LR and evaluate."""
    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    # Predict
    y_probs = lr.predict_proba(X_test_scaled)[:, 1]
    y_pred = lr.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = auc(fpr, tpr)
    
    return {
        'model': lr,
        'scaler': scaler,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc_score,
        'fpr': fpr,
        'tpr': tpr,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_probs': y_probs,
        'X_test': X_test # Keep X_test for 3D plotting
    }

def plot_3d_features(X_features, y_labels, char_type, attack_name, output_dir):
    """
    Plots a 3D scatter plot of the first three feature dimensions.
    Assumes X_features has at least 3 dimensions.
    """
    if X_features.shape[1] < 3:
        print(f"Skipping 3D plot for {char_type} on {attack_name}: less than 3 feature dimensions.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Separate clean/noisy (y=0) and adversarial (y=1)
    clean_indices = y_labels == 0
    adv_indices = y_labels == 1

    ax.scatter(X_features[clean_indices, 0], X_features[clean_indices, 1], X_features[clean_indices, 2],
               c='blue', label='Clean/Noisy', alpha=0.6, edgecolors='w', s=50)
    ax.scatter(X_features[adv_indices, 0], X_features[adv_indices, 1], X_features[adv_indices, 2],
               c='red', label='Adversarial', alpha=0.6, edgecolors='w', s=50)

    ax.set_xlabel(f'{char_type.upper()} Feature 0')
    ax.set_ylabel(f'{char_type.upper()} Feature 1')
    ax.set_zlabel(f'{char_type.upper()} Feature 2')
    ax.set_title(f'3D {char_type.upper()} Features: {attack_name}')
    ax.legend()
    ax.grid(True)

    plot_path = os.path.join(output_dir, f'3d_plot_{char_type}_{attack_name}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  3D {char_type.upper()} plot saved to {plot_path}")


def plot_comparison_results(all_results, output_dir):
    """
    Plots ROC curves and AUC comparison for different characteristics across attacks.
    all_results structure: {attack_name: {char_type: {metrics_dict}}}
    """
    characteristics = sorted(list(next(iter(all_results.values())).keys())) # lid, kd, km
    attacks = sorted(list(all_results.keys()))

    # 1. ROC Curves per attack, comparing characteristics
    for attack in attacks:
        plt.figure(figsize=(10, 8))
        for char_type in characteristics:
            res = all_results[attack][char_type]
            plt.plot(res['fpr'], res['tpr'], lw=2, label=f'{char_type.upper()} (AUC = {res["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {attack} (Comparing Characteristics)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        roc_path = os.path.join(output_dir, f'roc_comparison_{attack}.png')
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"  ROC comparison plot for {attack} saved to {roc_path}")

    # 2. AUC Score Comparison Bar Chart
    metrics = ['auc', 'accuracy', 'precision', 'recall']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 6 * len(metrics)), sharex=True)
    
    if len(metrics) == 1: # Handle single subplot case
        axes = [axes]

    for i, metric_name in enumerate(metrics):
        ax = axes[i]
        
        bar_width = 0.8 / len(characteristics) # Adjust bar width based on number of characteristics
        index = np.arange(len(attacks))

        for j, char_type in enumerate(characteristics):
            values = [all_results[attack][char_type][metric_name] for attack in attacks]
            ax.bar(index + j * bar_width, values, bar_width, label=char_type.upper())

        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison Across Attacks and Characteristics')
        ax.set_xticks(index + bar_width * (len(characteristics) - 1) / 2)
        ax.set_xticklabels(attacks, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    auc_bar_path = os.path.join(output_dir, 'metrics_comparison_bar_chart.png')
    plt.savefig(auc_bar_path, dpi=300)
    plt.close()
    print(f"  Metrics comparison bar chart saved to {auc_bar_path}")

    # 3. Detector Output Distribution (for each attack and characteristic)
    # This can get very large, so let's make it conditional or plot fewer.
    # For now, keep it for each attack and characteristic.
    # We'll put them in a single figure with many subplots.
    
    num_plots = len(attacks) * len(characteristics)
    if num_plots > 0:
        fig_height = 4 * num_plots # Each plot is 4 inches high
        plt.figure(figsize=(12, fig_height))
        plot_idx = 1
        for attack in attacks:
            for char_type in characteristics:
                ax = plt.subplot(num_plots, 1, plot_idx)
                res = all_results[attack][char_type]
                
                probs_pos = res['y_probs'][res['y_true'] == 1]
                probs_neg = res['y_probs'][res['y_true'] == 0]
                
                sns.histplot(probs_neg, bins=30, alpha=0.5, label='Normal/Noisy', color='green', stat='density', ax=ax)
                sns.histplot(probs_pos, bins=30, alpha=0.5, label='Adversarial', color='red', stat='density', ax=ax)
                
                ax.set_title(f'Detector Output: {char_type.upper()} for {attack}')
                ax.set_xlabel('Probability of being Adversarial')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
                
        plt.tight_layout()
        dist_path = os.path.join(output_dir, "detection_prob_distributions_comparison.png")
        plt.savefig(dist_path, dpi=300)
        plt.close()
        print(f"  Detector probability distributions saved to {dist_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attack', default='all', type=str, help='Attack to process (fgsm, bim-a, etc. or all)')
    parser.add_argument('-r', '--characteristics', default='all', type=str, help='Characteristics to use (comma-separated: lid,kd,km, or all)')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data/characteristics")
    output_dir = os.path.join(base_dir, "results/detection_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which characteristics to process
    if args.characteristics == 'all':
        char_types_to_process = ['lid', 'kd', 'km']
    else:
        char_types_to_process = args.characteristics.split(',')

    all_attack_results = {} # Structure: {attack_name: {char_type: {metrics_dict}}}

    # Get list of attacks from existing LID files as a reference
    # Assuming LID files will always be present if any characteristic is processed.
    # Or find all unique attacks across all characteristic files.
    all_files_in_char_dir = os.listdir(data_dir)
    all_attacks_found = set()
    for f in all_files_in_char_dir:
        if f.endswith(".npy"):
            parts = f.split('_')
            if len(parts) >= 3 and parts[1] == 'toy': # e.g. lid_toy_fgsm.npy
                attack_name = parts[2].replace(".npy", "")
                all_attacks_found.add(attack_name)
    
    all_attacks_found = sorted(list(all_attacks_found))

    if args.attack == 'all':
        attacks_to_process = all_attacks_found
    else:
        attacks_to_process = [args.attack]
        if not all(att in all_attacks_found for att in attacks_to_process):
            print(f"Warning: Some specified attacks ({args.attack}) not found in existing characteristic files.")
            attacks_to_process = [att for att in attacks_to_process if att in all_attacks_found]
            if not attacks_to_process:
                print("No valid attacks to process based on existing files.")
                return

    print(f"Processing attacks: {attacks_to_process}")
    print(f"Processing characteristics: {char_types_to_process}")

    for attack_name in attacks_to_process:
        all_attack_results[attack_name] = {}
        for char_type in char_types_to_process:
            file_name_prefix = f"{char_type}_toy_{attack_name}.npy"
            file_path = os.path.join(data_dir, file_name_prefix)
            
            print(f"\nTraining detector for {char_type.upper()} with attack {attack_name}...")
            try:
                X, y = load_characteristics(file_path)
            except FileNotFoundError:
                print(f"  {char_type.upper()} file not found for {attack_name}. Skipping.")
                continue

            # Split into train/test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            print(f"  Train shape: {X_train.shape}")
            print(f"  Test shape: {X_test.shape}")
            
            res = train_and_evaluate(X_train, y_train, X_test, y_test)
            all_attack_results[attack_name][char_type] = res
            
            print(f"  AUC: {res['auc']:.4f}")
            print(f"  Accuracy: {res['accuracy']:.4f}")

            # Plot 3D features (only if at least 3 dimensions available)
            plot_3d_features(res['X_test'], res['y_true'], char_type, attack_name, output_dir)
            
    if not all_attack_results:
        print("No results to plot. Exiting.")
        return

    print("\nCreating comparison visualization plots...")
    plot_comparison_results(all_attack_results, output_dir)
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
