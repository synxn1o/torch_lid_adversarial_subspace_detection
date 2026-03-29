import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from util import get_data, get_model
import os
import time

def train(dataset, batch_size, epochs):
    print(f'Training on {dataset}...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Get data with augmentation
    train_loader, test_loader = get_data(dataset, batch_size=batch_size, augmentation=True)
    
    # Get model
    model = get_model(dataset).to(device)
    
    # Optimizer and Loss
    # Keras Adadelta default rho=0.95, PyTorch is 0.9. 
    # We'll use PyTorch default for now, but can adjust if needed.
    optimizer = optim.Adadelta(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        test_loss = test_loss / len(test_loader)
        
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Time: {end_time - start_time:.1f}s "
              f"Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
              f"Val Loss: {test_loss:.4f} Val Acc: {test_acc:.2f}%")
        
    # Save model
    if not os.path.exists('data'):
        os.makedirs('data')
    save_path = f'data/model_{dataset}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help="Dataset to use; either 'mnist', 'cifar', 'svhn' or 'all'", required=True, type=str)
    parser.add_argument('-e', '--epochs', help="The number of epochs to train for.", default=120, type=int)
    parser.add_argument('-b', '--batch_size', help="The batch size to use for training.", default=100, type=int)
    args = parser.parse_args()
    
    if args.dataset == 'all':
        for dataset in ['mnist', 'cifar', 'svhn']:
            train(dataset, args.batch_size, args.epochs)
    else:
        train(args.dataset, args.batch_size, args.epochs)

if __name__ == "__main__":
    main()
