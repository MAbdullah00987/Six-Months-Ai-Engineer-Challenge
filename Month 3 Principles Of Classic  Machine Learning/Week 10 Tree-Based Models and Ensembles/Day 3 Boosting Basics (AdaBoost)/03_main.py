
#PART 3: CLASS WEIGHTS
#Mathematical Foundation of Class Weights

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import sympy as sp
from scipy.spatial.distance import euclidean

# PART 1: CREATE IMBALANCED DATASET
print("=" * 60)
print("CREATING IMBALANCED DATASET")
print("=" * 60)

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_clusters_per_class=2,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set class distribution:")
print(f"  Class 0: {np.sum(y_train == 0)} samples ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"  Class 1: {np.sum(y_train == 1)} samples ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")

# Baseline model (no handling of imbalance)
model_baseline = LogisticRegression(random_state=42, max_iter=1000)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)

print("\n" + "=" * 60)
print("BASELINE MODEL (No Imbalance Handling)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_baseline):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_baseline):.4f}")

# PART 2: SMOTE
print("\n" + "=" * 60)
print("SMOTE MATHEMATICAL FOUNDATION")
print("=" * 60)

# Define symbols
x1, x2, y1, y2, lambda_sym = sp.symbols('x1 x2 y1 y2 lambda')

# SMOTE formula
x_synthetic = x1 + lambda_sym * (x2 - x1)
y_synthetic = y1 + lambda_sym * (y2 - y1)

print("\nSMOTE Synthetic Sample Formula:")
print(f"x_new = x + λ(x_neighbor - x)")
print(f"x_new = {x_synthetic}")
print(f"y_new = {y_synthetic}")
print(f"\nWhere λ (lambda) is randomly selected from [0, 1]")

# Example calculation
print("\n" + "=" * 60)
print("NUMERICAL EXAMPLE")
print("=" * 60)

x_original = np.array([2, 3])
x_neighbor = np.array([5, 7])
lambda_val = 0.6

x_new = x_original + lambda_val * (x_neighbor - x_original)

print(f"Original point: {x_original}")
print(f"Neighbor point: {x_neighbor}")
print(f"Lambda value: {lambda_val}")
print(f"Synthetic point: {x_new}")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\n" + "=" * 60)
print("AFTER SMOTE RESAMPLING")
print("=" * 60)
print(f"Training set class distribution:")
print(f"  Class 0: {np.sum(y_train_smote == 0)} samples")
print(f"  Class 1: {np.sum(y_train_smote == 1)} samples")

# Train model with SMOTE
model_smote = LogisticRegression(random_state=42, max_iter=1000)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)

print("\n" + "=" * 60)
print("MODEL WITH SMOTE")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_smote):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_smote):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_smote):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_smote):.4f}")

# PART 3: CLASS WEIGHTS
print("\n" + "=" * 60)
print("CLASS WEIGHTS MATHEMATICAL FOUNDATION")
print("=" * 60)

# Calculate class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

print("\nFormula: weight_i = n_samples / (n_classes * n_samples_i)")
print(f"\nTotal samples: {len(y_train)}")
print(f"Number of classes: {len(classes)}")
print(f"Class 0 samples: {np.sum(y_train == 0)}")
print(f"Class 1 samples: {np.sum(y_train == 1)}")

print(f"\nCalculated Weights:")
for cls, weight in class_weight_dict.items():
    print(f"  Class {cls}: {weight:.4f}")

# Show how weights affect loss function
print("\n" + "=" * 60)
print("WEIGHTED LOSS FUNCTION")
print("=" * 60)
print("Standard Loss: L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]")
print("Weighted Loss: L = -[w₁*y*log(ŷ) + w₀*(1-y)*log(1-ŷ)]")
print(f"\nwhere w₀ = {class_weight_dict[0]:.4f}, w₁ = {class_weight_dict[1]:.4f}")

# Train model with class weights
model_weighted = LogisticRegression(
    class_weight='balanced', 
    random_state=42, 
    max_iter=1000
)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)

print("\n" + "=" * 60)
print("MODEL WITH CLASS WEIGHTS")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_weighted):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_weighted):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_weighted):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_weighted):.4f}")

# Compare all approaches
comparison_df = pd.DataFrame({
    'Method': ['Baseline', 'SMOTE', 'Class Weights'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_baseline),
        accuracy_score(y_test, y_pred_smote),
        accuracy_score(y_test, y_pred_weighted)
    ],
    'Precision': [
        precision_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_smote),
        precision_score(y_test, y_pred_weighted)
    ],
    'Recall': [
        recall_score(y_test, y_pred_baseline),
        recall_score(y_test, y_pred_smote),
        recall_score(y_test, y_pred_weighted)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_baseline),
        f1_score(y_test, y_pred_smote),
        f1_score(y_test, y_pred_weighted)
    ]
})

print("\n" + "=" * 60)
print("COMPARISON OF ALL METHODS")
print("=" * 60)
print(comparison_df.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(comparison_df['Method'], comparison_df[metric], 
                   color=['#95a5a6', '#3498db', '#2ecc71'], alpha=0.8)
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('methods_comparison.png', dpi=300, bbox_inches='tight')
plt.show()