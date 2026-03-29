"""
Train a binary neural network on the toy Gaussian dataset.
Model: 2 hidden layers with 4 neurons each, ReLU activation, binary output.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class BinaryNN(nn.Module):
    """Binary neural network with 2 hidden layers of 4 neurons each."""
    
    def __init__(self, input_dim=2):
        super(BinaryNN, self).__init__()
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(input_dim, 4)
        # First hidden layer to second hidden layer
        self.fc2 = nn.Linear(4, 4)
        # Second hidden layer to output
        self.fc3 = nn.Linear(4, 1)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on final layer for BCEWithLogitsLoss
        return x
    
    def predict(self, x):
        """Get binary predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            return predictions, probs

def load_toy_data(data_path="toy_example/data/circle_dataset.pkl"):
    """Load the toy dataset generated earlier."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_dataset.py first.")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y = data['y']
    
    # Convert labels from [-1, 1] to [0, 1] for binary classification
    y_binary = (y + 1) / 2  # -1 -> 0, +1 -> 1
    
    return X, y, y_binary

def prepare_data(X, y_binary, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets."""
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders."""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_test_t, y_test_t

def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, device='cpu'):
    """Train the neural network."""
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"Training on device: {device}")
    print(f"Epochs: {epochs}, Learning rate: {learning_rate}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                
                # Get predictions
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    
    print("-" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_accuracies

def evaluate_model(model, X_test_t, y_test_t, device='cpu'):
    """Evaluate model on test set."""
    model.eval()
    
    with torch.no_grad():
        X_test_t = X_test_t.to(device)
        predictions, probabilities = model.predict(X_test_t)
        
        predictions = predictions.cpu().numpy().flatten()
        probabilities = probabilities.cpu().numpy().flatten()
        y_test = y_test_t.cpu().numpy().flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        print("\nTest Set Performance:")
        print("=" * 40)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"          Predicted")
        print(f"          0      1")
        print(f"True 0   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"      1   {cm[1,0]:4d}  {cm[1,1]:4d}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['C1 (0)', 'C2 (1)']))
        
        return predictions, probabilities, accuracy

def plot_training_history(train_losses, val_losses, val_accuracies, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(val_accuracies, label='Val Accuracy', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_decision_boundary(model, X, y, save_path=None, device='cpu'):
    """Plot decision boundary of the trained model."""
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.FloatTensor(mesh_points).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions, _ = model.predict(mesh_tensor)
        Z = predictions.cpu().numpy().reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu_r')
    
    # Plot data points
    c1_mask = y == 0  # C1 (original -1, converted to 0)
    c2_mask = y == 1  # C2 (original +1, converted to 1)
    
    plt.scatter(X[c1_mask, 0], X[c1_mask, 1], 
                c='red', alpha=0.7, s=20, label='C1 (label=0)', edgecolors='black', linewidth=0.3)
    plt.scatter(X[c2_mask, 0], X[c2_mask, 1], 
                c='blue', alpha=0.7, s=20, label='C2 (label=1)', edgecolors='black', linewidth=0.3)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Neural Network Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Decision boundary plot saved to {save_path}")
    
    plt.show()

def save_model(model, filepath):
    """Save the trained model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("Training Binary Neural Network on Toy Dataset")
    print("=" * 60)
    
    # Parameters
    data_path = "toy_example/data/circle_dataset.pkl"
    output_dir = "toy_example/models"
    plots_dir = "toy_example/plots"
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load data
    print("\n1. Loading toy dataset...")
    X, y_original, y_binary = load_toy_data(data_path)
    print(f"   Loaded {len(X)} samples")
    print(f"   Class distribution: C1(0)={np.sum(y_binary==0)}, C2(1)={np.sum(y_binary==1)}")
    
    # 2. Prepare data splits
    print("\n2. Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y_binary)
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # 3. Create dataloaders
    print("\n3. Creating data loaders...")
    train_loader, val_loader, test_loader, X_test_t, y_test_t = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
    )
    
    # 4. Initialize model
    print("\n4. Initializing model...")
    model = BinaryNN(input_dim=2).to(device)
    print(f"   Model architecture:")
    print(f"   - Input: 2 features")
    print(f"   - Hidden 1: 4 neurons (ReLU)")
    print(f"   - Hidden 2: 4 neurons (ReLU)")
    print(f"   - Output: 1 neuron (Binary)")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 5. Train model
    print("\n5. Training model...")
    trained_model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, epochs=200, learning_rate=0.001, device=device
    )
    
    # 6. Evaluate on test set
    print("\n6. Evaluating on test set...")
    predictions, probabilities, test_accuracy = evaluate_model(
        trained_model, X_test_t, y_test_t, device=device
    )
    
    # 7. Save model
    print("\n7. Saving model...")
    model_path = os.path.join(output_dir, "toy_binary_nn.pth")
    save_model(trained_model, model_path)
    
    # 8. Create visualizations
    print("\n8. Creating visualizations...")
    
    # Training history
    os.makedirs(plots_dir, exist_ok=True)
    history_plot = os.path.join(plots_dir, "training_history.png")
    plot_training_history(train_losses, val_losses, val_accuracies, save_path=history_plot)
    
    # Decision boundary
    boundary_plot = os.path.join(plots_dir, "decision_boundary.png")
    plot_decision_boundary(trained_model, X, y_binary, save_path=boundary_plot, device=device)
    
    # 9. Save training results
    print("\n9. Saving training results...")
    results = {
        'model_state': trained_model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy,
        'model_architecture': 'BinaryNN(2->4->4->1)'
    }
    
    results_path = os.path.join(output_dir, "training_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"   Results saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"\nFiles created:")
    print(f"  - Model: {model_path}")
    print(f"  - Results: {results_path}")
    print(f"  - History plot: {history_plot}")
    print(f"  - Boundary plot: {boundary_plot}")
    
    return trained_model, test_accuracy

if __name__ == "__main__":
    main()
