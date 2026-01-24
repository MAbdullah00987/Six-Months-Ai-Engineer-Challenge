

#Part 3: Decision Trees - Practice Exercises

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# EXERCISE 1: Manual Gini Calculation (NumPy)



print("EXERCISE 1: Calculate Gini Impurity Manually")
print("""
Task: Calculate Gini impurity for the following datasets:
1. [0, 0, 0, 1, 1, 1]
2. [0, 0, 0, 0, 1, 1]
3. [0, 1, 2, 0, 1, 2]

Formula: Gini = 1 - Σ(p_i²) where p_i is probability of class i

Write your solution below:
""")

# YOUR CODE HERE
def calculate_gini(y):
    """
    Calculate Gini impurity
    
    Args:
        y: array of class labels
    
    Returns:
        float: Gini impurity value
    """
    # TODO: Implement this function
    pass

# Test your function
test_cases = [
    np.array([0, 0, 0, 1, 1, 1]),
    np.array([0, 0, 0, 0, 1, 1]),
    np.array([0, 1, 2, 0, 1, 2])
]

print("\nYour Results:")
for i, y in enumerate(test_cases, 1):
    gini = calculate_gini(y)
    print(f"Dataset {i}: Gini = {gini}")


# ============================================================================
# EXERCISE 2: Find Best Split (NumPy + Pandas)
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 2: Find the Best Split")
print("="*80)
print("""
Task: Given features and labels, find the best feature and threshold
to split the data that maximizes information gain.

Dataset:
    Feature1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    Feature2: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    Labels:   [0, 0, 0, 1, 1, 1]

Write a function that returns (best_feature, best_threshold, best_gain)
""")

# YOUR CODE HERE
def find_best_split_manual(X, y):
    """
    Find the best feature and threshold to split on
    
    Args:
        X: 2D numpy array of features (n_samples, n_features)
        y: 1D numpy array of labels
    
    Returns:
        tuple: (feature_index, threshold, information_gain)
    """
    # TODO: Implement this function
    # Hint: Try all features and all unique values as thresholds
    # Calculate information gain for each split
    # Return the split with maximum gain
    pass

# Test data
X_test = np.array([
    [1.0, 1.5],
    [2.0, 2.5],
    [3.0, 3.5],
    [4.0, 4.5],
    [5.0, 5.5],
    [6.0, 6.5]
])
y_test = np.array([0, 0, 0, 1, 1, 1])

print("\nYour Results:")
feature_idx, threshold, gain = find_best_split_manual(X_test, y_test)
print(f"Best Feature: {feature_idx}")
print(f"Best Threshold: {threshold}")
print(f"Information Gain: {gain}")


# ============================================================================
# EXERCISE 3: Statistical Validation (SciPy)
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 3: Statistical Validation of Splits")
print("="*80)
print("""
Task: Use ANOVA to determine which Iris feature is most important
for classification.

Steps:
1. Load Iris dataset
2. For each feature, perform one-way ANOVA across the 3 classes
3. Rank features by F-statistic
4. Compare with sklearn's feature_importances_
""")

# YOUR CODE HERE
from scipy.stats import f_oneway

def rank_features_by_anova(X, y, feature_names):
    """
    Rank features using ANOVA F-statistic
    
    Args:
        X: feature matrix
        y: labels
        feature_names: list of feature names
    
    Returns:
        DataFrame with features ranked by F-statistic
    """
    # TODO: Implement this function
    # Hint: For each feature, split data by class
    # Then run f_oneway on the groups
    pass

# Load data
iris = load_iris()

print("\nYour Results:")
results = rank_features_by_anova(iris.data, iris.target, iris.feature_names)
print(results)


# ============================================================================
# EXERCISE 4: Tree Visualization (Matplotlib)
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 4: Create Custom Tree Visualization")
print("="*80)
print("""
Task: Create a manual visualization of a decision tree showing:
1. Each node with its splitting rule
2. Gini impurity at each node
3. Number of samples at each node
4. Color-coded by majority class

Use matplotlib to draw rectangles and connecting lines.
""")

# YOUR CODE HERE
def visualize_tree_custom(clf, feature_names, class_names):
    """
    Create a custom tree visualization
    
    Args:
        clf: trained DecisionTreeClassifier
        feature_names: list of feature names
        class_names: list of class names
    """
    # TODO: Implement custom visualization
    # This is challenging! Use clf.tree_ to access tree structure
    # Draw nodes as rectangles, connect with lines
    pass

# Train a simple tree
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)

print("\nCreate your visualization here")
# visualize_tree_custom(clf, iris.feature_names, iris.target_names)


# ============================================================================
# EXERCISE 5: Pruning Analysis (sklearn + Matplotlib)
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 5: Analyze Effect of Pruning")
print("="*80)
print("""
Task: Train decision trees with different max_depth values (1 to 10)
and analyze:
1. Training accuracy vs depth
2. Testing accuracy vs depth
3. Number of leaves vs depth
4. Create plots showing overfitting

Identify the optimal depth for the Wine dataset.
""")

# YOUR CODE HERE
def analyze_pruning(X, y, max_depths=range(1, 11)):
    """
    Analyze effect of max_depth on tree performance
    
    Args:
        X: features
        y: labels
        max_depths: range of depths to try
    
    Returns:
        DataFrame with results
    """
    # TODO: Implement this function
    # Train trees with different depths
    # Record train/test accuracy and n_leaves for each
    pass

# Load Wine dataset
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3, random_state=42
)

print("\nYour Results:")
results = analyze_pruning(wine.data, wine.target)
print(results)


# ============================================================================
# EXERCISE 6: Feature Importance (pandas + seaborn)
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 6: Compare Feature Importance Methods")
print("="*80)
print("""
Task: For the Breast Cancer dataset, compare feature importance using:
1. sklearn's feature_importances_
2. Permutation importance
3. ANOVA F-scores

Create a side-by-side comparison plot.
""")

# YOUR CODE HERE
from sklearn.inspection import permutation_importance

def compare_feature_importance(X, y, feature_names):
    """
    Compare different feature importance methods
    
    Returns:
        DataFrame with all importance scores
    """
    # TODO: Implement comparison
    pass

# Load data
cancer = load_breast_cancer()

print("\nYour Results:")
importance_df = compare_feature_importance(
    cancer.data, cancer.target, cancer.feature_names
)
print(importance_df.head(10))


# ============================================================================
# EXERCISE 7: Real-world Application
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 7: Build a Complete Pipeline")
print("="*80)
print("""
Task: Create a complete decision tree pipeline:
1. Load and explore a dataset
2. Handle missing values and outliers
3. Split into train/test
4. Train multiple trees with different hyperparameters
5. Use cross-validation
6. Select best model
7. Visualize decision boundaries
8. Generate a classification report
9. Save the best model

Use the Wine dataset.
""")

# YOUR CODE HERE
class DecisionTreePipeline:
    """Complete pipeline for decision tree classification"""
    
    def __init__(self):
        self.best_model = None
        self.best_params = None
        
    def explore_data(self, X, y):
        """Explore and visualize dataset"""
        # TODO: Implement EDA
        pass
    
    def preprocess(self, X, y):
        """Handle missing values and outliers"""
        # TODO: Implement preprocessing
        pass
    
    def train_and_tune(self, X_train, y_train):
        """Train with different hyperparameters"""
        # TODO: Implement training with grid search
        pass
    
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation"""
        # TODO: Implement evaluation
        pass
    
    def visualize_results(self):
        """Create all necessary visualizations"""
        # TODO: Implement visualizations
        pass

# Run the pipeline
pipeline = DecisionTreePipeline()
print("\nImplement your pipeline here")


# ============================================================================
# SOLUTIONS
# ============================================================================

print("\n" + "="*80)
print("SOLUTIONS")
print("="*80)
print("Scroll down to see solutions...\n\n\n\n\n")


# SOLUTION 1: Gini Calculation
def calculate_gini_solution(y):
    """Calculate Gini impurity"""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

print("SOLUTION 1: Gini Calculation")
print("-" * 50)
for i, y in enumerate(test_cases, 1):
    gini = calculate_gini_solution(y)
    print(f"Dataset {i}: {y}")
    print(f"  Classes and counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Gini = {gini:.4f}")
    print()


# SOLUTION 2: Find Best Split
def find_best_split_solution(X, y):
    """Find best split using information gain"""
    def gini(y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def information_gain(y_parent, y_left, y_right):
        n = len(y_parent)
        n_l, n_r = len(y_left), len(y_right)
        parent_gini = gini(y_parent)
        child_gini = (n_l/n * gini(y_left)) + (n_r/n * gini(y_right))
        return parent_gini - child_gini
    
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            gain = information_gain(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain

print("\nSOLUTION 2: Find Best Split")
print("-" * 50)
feature_idx, threshold, gain = find_best_split_solution(X_test, y_test)
print(f"Best Feature: Feature {feature_idx}")
print(f"Best Threshold: {threshold:.2f}")
print(f"Information Gain: {gain:.4f}")


# SOLUTION 3: ANOVA Feature Ranking
def rank_features_by_anova_solution(X, y, feature_names):
    """Rank features using ANOVA"""
    results = []
    
    for i, feature_name in enumerate(feature_names):
        # Split feature by class
        groups = [X[y == c, i] for c in np.unique(y)]
        
        # Perform ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        results.append({
            'feature': feature_name,
            'f_statistic': f_stat,
            'p_value': p_value
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('f_statistic', ascending=False)
    return df

print("\nSOLUTION 3: ANOVA Feature Ranking")
print("-" * 50)
anova_results = rank_features_by_anova_solution(
    iris.data, iris.target, iris.feature_names
)
print(anova_results)

# Compare with sklearn
clf = DecisionTreeClassifier(random_state=42)
clf.fit(iris.data, iris.target)
print("\nsklearn feature_importances_:")
for name, importance in zip(iris.feature_names, clf.feature_importances_):
    print(f"{name:25s}: {importance:.4f}")


# SOLUTION 5: Pruning Analysis
def analyze_pruning_solution(X, y, max_depths=range(1, 11)):
    """Analyze effect of max_depth"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    results = []
    
    for depth in max_depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        n_leaves = clf.get_n_leaves()
        
        results.append({
            'max_depth': depth,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_leaves': n_leaves,
            'overfitting': train_acc - test_acc
        })
    
    df = pd.DataFrame(results)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Accuracy plot
    axes[0].plot(df['max_depth'], df['train_accuracy'], 'o-', label='Train')
    axes[0].plot(df['max_depth'], df['test_accuracy'], 's-', label='Test')
    axes[0].set_xlabel('Max Depth')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Depth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Leaves plot
    axes[1].plot(df['max_depth'], df['n_leaves'], 'o-', color='green')
    axes[1].set_xlabel('Max Depth')
    axes[1].set_ylabel('Number of Leaves')
    axes[1].set_title('Complexity vs Depth')
    axes[1].grid(True, alpha=0.3)
    
    # Overfitting plot
    axes[2].plot(df['max_depth'], df['overfitting'], 'o-', color='red')
    axes[2].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('Max Depth')
    axes[2].set_ylabel('Train - Test Accuracy')
    axes[2].set_title('Overfitting vs Depth')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pruning_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: pruning_analysis.png")
    
    return df

print("\nSOLUTION 5: Pruning Analysis")
print("-" * 50)
wine = load_wine()
pruning_results = analyze_pruning_solution(wine.data, wine.target)
print(pruning_results)
print(f"\nOptimal depth: {pruning_results.loc[pruning_results['test_accuracy'].idxmax(), 'max_depth']}")


