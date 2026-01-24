
#Project: Visualize a Decision Tree

#Train a Decision Tree on a simple dataset
#Use plot_tree() or graphviz to visualize
#Compare shallow vs deep trees
#Observe overfitting with different max_depth values


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 70)
print("DECISION TREE VISUALIZATION PROJECT")
print("=" * 70)

# ============================================================================
# PART 1: Generate a Simple Synthetic Dataset
# ============================================================================
print("\n📊 PART 1: Creating Synthetic Dataset")
print("-" * 70)

# Create a simple dataset for binary classification
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.1,
    random_state=42
)

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['Target'] = y

print(f"Dataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nClass Distribution:")
print(df['Target'].value_counts())

# ============================================================================
# PART 2: Visualize the Dataset
# ============================================================================
print("\n📈 PART 2: Visualizing Dataset")
print("-" * 70)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Feature_1'], df['Feature_2'], 
                     c=df['Target'], cmap='viridis', 
                     alpha=0.6, edgecolors='k', s=50)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Synthetic Dataset - Binary Classification', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Class')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dataset_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Dataset visualization saved as 'dataset_visualization.png'")

# ============================================================================
# PART 3: Split Data into Train and Test Sets
# ============================================================================
print("\n🔀 PART 3: Splitting Data")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================================================
# PART 4: Train Decision Trees with Different Depths
# ============================================================================
print("\n🌳 PART 4: Training Decision Trees with Different Depths")
print("-" * 70)

# Define different max_depth values
depths = [1, 2, 3, 5, 10, None]  # None means unlimited depth
models = {}
train_scores = []
test_scores = []

for depth in depths:
    # Train model
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Store model
    models[depth] = clf
    
    # Calculate scores
    train_score = accuracy_score(y_train, clf.predict(X_train))
    test_score = accuracy_score(y_test, clf.predict(X_test))
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    depth_str = str(depth) if depth is not None else "Unlimited"
    print(f"Depth {depth_str:>9} | Train Accuracy: {train_score:.4f} | Test Accuracy: {test_score:.4f}")

# ============================================================================
# PART 5: Visualize Trees (Shallow vs Deep)
# ============================================================================
print("\n🎨 PART 5: Visualizing Decision Trees")
print("-" * 70)

# Visualize trees with different depths
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for idx, depth in enumerate(depths):
    ax = axes[idx]
    depth_str = str(depth) if depth is not None else "Unlimited"
    
    plot_tree(models[depth], 
              feature_names=['Feature_1', 'Feature_2'],
              class_names=['Class_0', 'Class_1'],
              filled=True,
              rounded=True,
              ax=ax,
              fontsize=8)
    
    ax.set_title(f'Decision Tree (max_depth={depth_str})', 
                 fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('decision_trees_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Tree visualizations saved as 'decision_trees_comparison.png'")

# ============================================================================
# PART 6: Visualize Decision Boundaries
# ============================================================================
print("\n🎯 PART 6: Visualizing Decision Boundaries")
print("-" * 70)

def plot_decision_boundary(model, X, y, ax, title):
    """Plot decision boundary for a model"""
    h = 0.02  # step size in mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
               edgecolors='k', s=20, alpha=0.6)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title, fontweight='bold')

# Plot decision boundaries for selected depths
selected_depths = [1, 2, 5, None]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, depth in enumerate(selected_depths):
    depth_str = str(depth) if depth is not None else "Unlimited"
    title = f'Decision Boundary (max_depth={depth_str})'
    plot_decision_boundary(models[depth], X_test, y_test, axes[idx], title)

plt.tight_layout()
plt.savefig('decision_boundaries.png', dpi=300, bbox_inches='tight')
print("✓ Decision boundaries saved as 'decision_boundaries.png'")

# ============================================================================
# PART 7: Overfitting Analysis
# ============================================================================
print("\n📉 PART 7: Analyzing Overfitting")
print("-" * 70)

# Create overfitting plot
plt.figure(figsize=(12, 6))
depth_labels = [str(d) if d is not None else "Unlimited" for d in depths]

plt.plot(depth_labels, train_scores, 'o-', label='Training Accuracy', 
         linewidth=2, markersize=8, color='#2ecc71')
plt.plot(depth_labels, test_scores, 's-', label='Test Accuracy', 
         linewidth=2, markersize=8, color='#e74c3c')

# Add annotations for overfitting gap
for i, (train, test) in enumerate(zip(train_scores, test_scores)):
    gap = train - test
    if gap > 0.05:  # Significant overfitting
        plt.annotate(f'Gap: {gap:.3f}', 
                    xy=(i, (train + test) / 2),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=9, color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

plt.xlabel('Maximum Tree Depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Overfitting Analysis: Train vs Test Accuracy', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0.7, 1.05])
plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Overfitting analysis saved as 'overfitting_analysis.png'")

# ============================================================================
# PART 8: Detailed Analysis of Best Model
# ============================================================================
print("\n🏆 PART 8: Best Model Analysis")
print("-" * 70)

# Find best performing model on test set
best_idx = np.argmax(test_scores)
best_depth = depths[best_idx]
best_model = models[best_depth]

depth_str = str(best_depth) if best_depth is not None else "Unlimited"
print(f"Best Model: max_depth = {depth_str}")
print(f"Test Accuracy: {test_scores[best_idx]:.4f}")

# Get predictions
y_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title(f'Confusion Matrix (max_depth={depth_str})', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Class 0', 'Class 1']))

# ============================================================================
# PART 9: Feature Importance
# ============================================================================
print("\n⭐ PART 9: Feature Importance")
print("-" * 70)

feature_importance = best_model.feature_importances_
features = ['Feature_1', 'Feature_2']

plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance, color='skyblue', edgecolor='navy')
plt.xlabel('Importance', fontsize=12)
plt.title(f'Feature Importance (max_depth={depth_str})', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance saved as 'feature_importance.png'")

for feat, imp in zip(features, feature_importance):
    print(f"{feat}: {imp:.4f}")

# ============================================================================
# PART 10: Summary Statistics
# ============================================================================
print("\n📊 PART 10: Summary Statistics")
print("=" * 70)

summary_df = pd.DataFrame({
    'Max_Depth': [str(d) if d is not None else "Unlimited" for d in depths],
    'Train_Accuracy': train_scores,
    'Test_Accuracy': test_scores,
    'Overfitting_Gap': np.array(train_scores) - np.array(test_scores)
})

print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("✅ PROJECT COMPLETE!")
print("=" * 70)
print("\nGenerated Files:")
print("  1. dataset_visualization.png")
print("  2. decision_trees_comparison.png")
print("  3. decision_boundaries.png")
print("  4. overfitting_analysis.png")
print("  5. confusion_matrix.png")
print("  6. feature_importance.png")
print("\n🎓 Key Learnings:")
print("  • Shallow trees (depth 1-2) may underfit the data")
print("  • Deep trees (unlimited depth) often overfit the training data")
print("  • The best model balances training and test accuracy")
print("  • Overfitting is visible when train accuracy >> test accuracy")
print("=" * 70)