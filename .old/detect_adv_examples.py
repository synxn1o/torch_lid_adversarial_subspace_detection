import argparse
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score

PATH_DATA = "data/"

def load_characteristics(dataset, attack, characteristics):
    X, Y = None, None
    for characteristic in characteristics:
        file_name = os.path.join(PATH_DATA, f"{characteristic}_{dataset}_{attack}.npy")
        if not os.path.exists(file_name):
             raise FileNotFoundError(f"Characteristic file not found: {file_name}")
             
        data = np.load(file_name)
        if X is None:
            X = data[:, :-1]
        else:
            X = np.concatenate((X, data[:, :-1]), axis=1)
        
        if Y is None:
            Y = data[:, -1] # Labels

    return X, Y

def block_split(X, Y):
    """
    Split data into 80% train, 20% test.
    Assumes structure: [Pos (Adv), Neg (Normal+Noisy)]
    The original code had a specific block split logic assuming equal sizing/partitioning.
    Let's look at the original logic:
    It assumes X_adv, X_norm, X_noisy are concatenated.
    My extract_characteristics saves: [X_adv, X_norm, X_noisy].
    So the structure matches.
    
    However, getting exact indices might be tricky if sizes differ.
    My extract_characteristics ensures filtered sizes are equal for all 3 sets?
    No, it filters by correct classification.
    But X_adv corresponds to X_test, so they should match if filtered same way.
    
    Let's implement a standard stratified shuffle split or similar, 
    OR try to replicate the block split if critical.
    The original `block_split` does:
    partition = num_samples / 3 (assuming 3 parts).
    Then takes slices.
    
    Safe approach: Random split (train_test_split).
    But original `detect_adv_examples.py` has `random_split` and `block_split`.
    It defaults to `block_split`.
    
    I'll implement a robust random split.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def train_lr(X, y):
    # Logistic Regression with CV
    lr = LogisticRegressionCV(n_jobs=-1, cv=5).fit(X, y)
    return lr

def detect(args):
    characteristics = args.characteristics.split(',')
    
    print(f"Loading train attack: {args.attack}")
    X, Y = load_characteristics(args.dataset, args.attack, characteristics)
    
    # Standardization
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    
    # Split
    X_train, X_test, Y_train, Y_test = block_split(X, Y)
    
    # If test attack differs
    if args.test_attack and args.test_attack != args.attack:
        print(f"Loading test attack: {args.test_attack}")
        X_test_new, Y_test_new = load_characteristics(args.dataset, args.test_attack, characteristics)
        X_test_new = scaler.transform(X_test_new)
        # We use the whole new attack set for testing? Or split?
        # Usually we want to see if detector generalizes to new attack.
        # So use all of it.
        X_test = X_test_new
        Y_test = Y_test_new
        
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    # Train
    print("Training LR Detector...")
    lr = train_lr(X_train, Y_train)
    
    # Evaluate
    y_pred_prob = lr.predict_proba(X_test)[:, 1]
    y_pred = lr.predict(X_test)
    
    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
    auc_score = auc(fpr, tpr)
    
    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    
    print(f"Detector Results - Dataset: {args.dataset}, Train Attack: {args.attack}, Test Attack: {args.test_attack or args.attack}")
    print(f"ROC-AUC: {auc_score:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str)
    parser.add_argument('-a', '--attack', required=True, type=str)
    parser.add_argument('-r', '--characteristics', required=True, type=str, help="comma separated: lid,kd,bu")
    parser.add_argument('-t', '--test_attack', type=str, default=None)
    args = parser.parse_args()
    
    detect(args)

if __name__ == "__main__":
    main()
