

#Task: Project - Visualize a Decision Tree.

#Train a DecisionTreeClassifier on the Iris or Mushroom dataset.

#Use sklearn.tree.plot_tree and graphviz to export a picture of the tree.

#Goal: Trace a single data point down the tree manually to verify the prediction.

"""
SOLUTIONS FOR DECISION TREE EXERCISES
=====================================
This file contains complete solutions for the Gini impurity 
and best split exercises.
"""

import numpy as np
from collections import Counter

# ============================================================================
# EXERCISE 1: Calculate Gini Impurity Manually
# ============================================================================

def calculate_gini(labels):
    """
    Calculate Gini impurity for a dataset.
    
    Formula: Gini = 1 - Σ(p_i²) where p_i is probability of class i
    
    Args:
        labels: List or array of class labels
        
    Returns:
        float: Gini impurity value (0 = pure, higher = more mixed)
    """
    if len(labels) == 0:
        return 0
    
    # Count occurrences of each class
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate probabilities
    gini = 1.0
    for count in class_counts.values():
        probability = count / total_samples
        gini -= probability ** 2
    
    return gini


print("=" * 80)
print("EXERCISE 1: Calculate Gini Impurity Manually")
print("=" * 80)
print("Task: Calculate Gini impurity for the following datasets:")
print("Formula: Gini = 1 - Σ(p_i²) where p_i is probability of class i\n")

# Test datasets
dataset1 = [0, 0, 0, 1, 1, 1]
dataset2 = [0, 0, 0, 0, 1, 1]
dataset3 = [0, 1, 2, 0, 1, 2]

# Calculate Gini for each dataset
gini1 = calculate_gini(dataset1)
gini2 = calculate_gini(dataset2)
gini3 = calculate_gini(dataset3)

print("Dataset 1:", dataset1)
print(f"  Classes: 3 zeros, 3 ones")
print(f"  P(0) = 3/6 = 0.5, P(1) = 3/6 = 0.5")
print(f"  Gini = 1 - (0.5² + 0.5²) = 1 - (0.25 + 0.25) = 0.5")
print(f"  Calculated Gini = {gini1:.4f}")
print()

print("Dataset 2:", dataset2)
print(f"  Classes: 4 zeros, 2 ones")
print(f"  P(0) = 4/6 = 0.667, P(1) = 2/6 = 0.333")
print(f"  Gini = 1 - (0.667² + 0.333²) = 1 - (0.444 + 0.111) = 0.444")
print(f"  Calculated Gini = {gini2:.4f}")
print()

print("Dataset 3:", dataset3)
print(f"  Classes: 2 zeros, 2 ones, 2 twos")
print(f"  P(0) = 2/6 = 0.333, P(1) = 2/6 = 0.333, P(2) = 2/6 = 0.333")
print(f"  Gini = 1 - (0.333² + 0.333² + 0.333²) = 1 - 3(0.111) = 0.667")
print(f"  Calculated Gini = {gini3:.4f}")
print()

print("INTERPRETATION:")
print(f"  • Gini = 0.0 means perfect purity (all same class)")
print(f"  • Gini = 0.5 means maximum impurity for binary classification")
print(f"  • Gini = 0.667 means maximum impurity for 3-class equal distribution")
print()

# ============================================================================
# EXERCISE 2: Find the Best Split
# ============================================================================

def calculate_information_gain(parent_labels, left_labels, right_labels):
    """
    Calculate information gain from a split.
    
    Information Gain = Gini(parent) - weighted_average(Gini(left), Gini(right))
    """
    n_parent = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    
    if n_left == 0 or n_right == 0:
        return 0
    
    # Calculate parent Gini
    gini_parent = calculate_gini(parent_labels)
    
    # Calculate weighted Gini for children
    gini_left = calculate_gini(left_labels)
    gini_right = calculate_gini(right_labels)
    
    weighted_gini = (n_left / n_parent) * gini_left + (n_right / n_parent) * gini_right
    
    # Information gain
    info_gain = gini_parent - weighted_gini
    
    return info_gain


def find_best_split_manual(X, y):
    """
    Find the best feature and threshold to split the data.
    
    Args:
        X: 2D array of features (n_samples, n_features)
        y: 1D array of labels
        
    Returns:
        tuple: (best_feature_idx, best_threshold, best_gain)
    """
    n_samples, n_features = X.shape
    best_gain = -1
    best_feature_idx = None
    best_threshold = None
    
    # Try each feature
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        
        # Get unique values to try as thresholds
        unique_values = np.unique(feature_values)
        
        # Try thresholds between consecutive values
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            # Split data
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            left_labels = y[left_mask]
            right_labels = y[right_mask]
            
            # Calculate information gain
            gain = calculate_information_gain(y, left_labels, right_labels)
            
            # Update best split
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_threshold = threshold
    
    return best_feature_idx, best_threshold, best_gain


print("=" * 80)
print("EXERCISE 2: Find the Best Split")
print("=" * 80)
print("Task: Find the best feature and threshold to maximize information gain\n")

# Test dataset
X_test = np.array([
    [1.0, 1.5],
    [2.0, 2.5],
    [3.0, 3.5],
    [4.0, 4.5],
    [5.0, 5.5],
    [6.0, 6.5]
])
y_test = np.array([0, 0, 0, 1, 1, 1])

print("Dataset:")
print("  Feature1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]")
print("  Feature2: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]")
print("  Labels:   [0, 0, 0, 1, 1, 1]")
print()

# Find best split
feature_idx, threshold, gain = find_best_split_manual(X_test, y_test)

print("RESULTS:")
print(f"  Best Feature: Feature{feature_idx + 1}")
print(f"  Best Threshold: {threshold:.2f}")
print(f"  Information Gain: {gain:.4f}")
print()

print("EXPLANATION:")
print(f"  Split rule: Feature{feature_idx + 1} <= {threshold:.2f}")
print(f"  • Left branch (<=): samples with Feature{feature_idx + 1} <= {threshold:.2f}")
print(f"  • Right branch (>): samples with Feature{feature_idx + 1} > {threshold:.2f}")
print()

# Show the split
left_mask = X_test[:, feature_idx] <= threshold
right_mask = ~left_mask

print("SPLIT DETAILS:")
print(f"  Left branch: {y_test[left_mask].tolist()} (Gini = {calculate_gini(y_test[left_mask]):.4f})")
print(f"  Right branch: {y_test[right_mask].tolist()} (Gini = {calculate_gini(y_test[right_mask]):.4f})")
print(f"  Original: {y_test.tolist()} (Gini = {calculate_gini(y_test):.4f})")
print()

# ============================================================================
# BONUS: Compare All Possible Splits
# ============================================================================

print("=" * 80)
print("BONUS: Comparison of All Possible Splits")
print("=" * 80)
print()

for feature_idx in range(X_test.shape[1]):
    print(f"Feature {feature_idx + 1}:")
    feature_values = X_test[:, feature_idx]
    unique_values = np.unique(feature_values)
    
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i + 1]) / 2
        
        left_mask = feature_values <= threshold
        right_mask = ~left_mask
        
        left_labels = y_test[left_mask]
        right_labels = y_test[right_mask]
        
        gain = calculate_information_gain(y_test, left_labels, right_labels)
        
        print(f"  Threshold {threshold:.2f}: Info Gain = {gain:.4f}")
        print(f"    Left:  {left_labels.tolist()}")
        print(f"    Right: {right_labels.tolist()}")
    print()
