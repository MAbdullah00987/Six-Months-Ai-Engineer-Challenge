
#Day 3: Boosting Basics (AdaBoost)
#Objective: Understand the shift from "parallel" (Bagging) to "sequential" (Boosting).
#Random Forests & Bagging
#Goal: Master ensemble methods - Bagging and Random Forests

#Concept: Weak Learners, Weighted Errors. How AdaBoost focuses on the "hard-to-classify" examples.

#Task: Project - Compare Bagging and Boosting.

#Use sklearn.ensemble.AdaBoostClassifier.

#Run a comparison script: Random Forest (Bagging) vs. AdaBoost on a complex dataset (e.g., Moons dataset in sklearn).

#Project: Heart Disease Prediction

#Compare single Decision Tree vs Random Forest
#Evaluate both models
#Analyze performance differences

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

#Problems With Imbalanced Data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#PART 1: IMBALANCED DATA 
#Understanding the Problem

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.95, 0.05],  # 95% class 0, 5% class 1
    random_state=42
)

# Visualize class imbalance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution
unique, counts = np.unique(y, return_counts=True)
axes[0].bar(['Class 0 (Negative)', 'Class 1 (Positive)'], counts, 
            color=['#3498db', '#e74c3c'], alpha=0.7)
axes[0].set_ylabel('Count')
axes[0].set_title('Class Distribution (Imbalanced)', fontweight='bold')
axes[0].text(0, counts[0]+20, f'{counts[0]} samples', ha='center')
axes[0].text(1, counts[1]+20, f'{counts[1]} samples', ha='center')

# Pie chart
axes[1].pie(counts, labels=['Class 0 (95%)', 'Class 1 (5%)'], 
            autopct='%1.1f%%', colors=['#3498db', '#e74c3c'],
            startangle=90)
axes[1].set_title('Class Proportion', fontweight='bold')

plt.tight_layout()
plt.savefig('imbalanced_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("Dataset Statistics:")
print(f"Total samples: {len(y)}")
print(f"Class 0: {counts[0]} ({counts[0]/len(y)*100:.1f}%)")
print(f"Class 1: {counts[1]} ({counts[1]/len(y)*100:.1f}%)")
print(f"Imbalance Ratio: {counts[0]/counts[1]:.2f}:1")


#Problems with Imbalanced Data


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train baseline model (no handling of imbalance)
baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

# Metrics
print("=" * 60)
print("BASELINE MODEL (No Imbalance Handling)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_baseline):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_baseline):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_baseline))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline))

# Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred_baseline)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix - Baseline Model', fontweight='bold', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('baseline_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()



#PART 2: SMOTE (Synthetic Minority Over-sampling Technique)

import numpy as np
from scipy.spatial.distance import euclidean
import sympy as sp

#Implementing Smote
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Mathematical Foundation of SMOTE
print("=" * 60)
print("SMOTE MATHEMATICAL FOUNDATION")
print("=" * 60)

# Define symbols
x1, x2, y1, y2, lambda_sym = sp.symbols('x1 x2 y1 y2 lambda')

# SMOTE formula: x_new = x + λ(x_neighbor - x)
# where λ ∈ [0, 1] is random
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
print(f"Original point:  {x_original}")
print(f"Neighbor point:  {x_neighbor}")
print(f"Lambda value:    {lambda_val}")
print(f"Synthetic point: {x_new}")

#Implementing Smote

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("=" * 60)
print("AFTER SMOTE")
print("=" * 60)
unique_smote, counts_smote = np.unique(y_train_smote, return_counts=True)
print(f"Class 0: {counts_smote[0]} samples")
print(f"Class 1: {counts_smote[1]} samples")
print(f"New Ratio: {counts_smote[0]/counts_smote[1]:.2f}:1")

# Visualize SMOTE effect using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_train_smote_pca = pca.transform(X_train_smote)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before SMOTE
minority_before = X_train_pca[y_train == 1]
majority_before = X_train_pca[y_train == 0]

axes[0].scatter(majority_before[:, 0], majority_before[:, 1], 
                alpha=0.5, label='Class 0 (Majority)', c='blue', s=30)
axes[0].scatter(minority_before[:, 0], minority_before[:, 1], 
                alpha=0.8, label='Class 1 (Minority)', c='red', s=50, marker='^')
axes[0].set_title('Before SMOTE', fontweight='bold', fontsize=14)
axes[0].set_xlabel('First Principal Component')
axes[0].set_ylabel('Second Principal Component')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# After SMOTE
minority_after = X_train_smote_pca[y_train_smote == 1]
majority_after = X_train_smote_pca[y_train_smote == 0]

axes[1].scatter(majority_after[:, 0], majority_after[:, 1], 
                alpha=0.5, label='Class 0 (Majority)', c='blue', s=30)
axes[1].scatter(minority_after[:, 0], minority_after[:, 1], 
                alpha=0.8, label='Class 1 (Minority - with synthetic)', 
                c='red', s=50, marker='^')
axes[1].set_title('After SMOTE', fontweight='bold', fontsize=14)
axes[1].set_xlabel('First Principal Component')
axes[1].set_ylabel('Second Principal Component')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('smote_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Train model with SMOTE data
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


#PART 4: THRESHOLD ADJUSTMENT
#Understanding Decision Thresholds

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# Get probability predictions
y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba_baseline)
roc_auc = roc_auc_score(y_test, y_proba_baseline)

# Calculate Precision-Recall curve
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_proba_baseline)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC Curve
axes[0].plot(fpr, tpr, color='#e74c3c', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random Classifier')
axes[0].set_xlabel('False Positive Rate', fontweight='bold')
axes[0].set_ylabel('True Positive Rate', fontweight='bold')
axes[0].set_title('ROC Curve', fontweight='bold', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Precision-Recall Curve
axes[1].plot(recall_vals, precision_vals, color='#3498db', lw=2)
axes[1].set_xlabel('Recall', fontweight='bold')
axes[1].set_ylabel('Precision', fontweight='bold')
axes[1].set_title('Precision-Recall Curve', fontweight='bold', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=np.sum(y_test)/len(y_test), color='gray', 
                linestyle='--', label='Baseline')
axes[1].legend()

plt.tight_layout()
plt.savefig('threshold_curves.png', dpi=300, bbox_inches='tight')
plt.show()

#Finding Optimal Threshold

# Find optimal threshold using F1-score
f1_scores = []
thresholds_to_test = np.linspace(0.1, 0.9, 50)

for threshold in thresholds_to_test:
    y_pred_threshold = (y_proba_baseline >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold)
    f1_scores.append(f1)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds_to_test[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print("=" * 60)
print("OPTIMAL THRESHOLD ANALYSIS")
print("=" * 60)
print(f"Default threshold: 0.50")
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1-Score at default: {f1_score(y_test, y_pred_baseline):.4f}")
print(f"F1-Score at optimal: {optimal_f1:.4f}")

# Apply optimal threshold
y_pred_optimal = (y_proba_baseline >= optimal_threshold).astype(int)

print("\n" + "=" * 60)
print("RESULTS WITH OPTIMAL THRESHOLD")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_optimal):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_optimal):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_optimal):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_optimal):.4f}")

# Visualize threshold impact
plt.figure(figsize=(12, 6))
plt.plot(thresholds_to_test, f1_scores, color='#2ecc71', lw=2, label='F1-Score')
plt.axvline(x=optimal_threshold, color='red', linestyle='--', lw=2, 
            label=f'Optimal Threshold = {optimal_threshold:.3f}')
plt.axvline(x=0.5, color='gray', linestyle='--', lw=2, label='Default Threshold = 0.50')
plt.xlabel('Classification Threshold', fontweight='bold', fontsize=12)
plt.ylabel('F1-Score', fontweight='bold', fontsize=12)
plt.title('F1-Score vs Classification Threshold', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('optimal_threshold.png', dpi=300, bbox_inches='tight')
plt.show()


#PART 5: INTRODUCTION TO BOOSTING
#Boosting Philosophy

print("=" * 70)
print("BOOSTING vs BAGGING: FUNDAMENTAL DIFFERENCE")
print("=" * 70)

print("\n📦 BAGGING (Bootstrap Aggregating):")
print("   • Parallel training: all models trained independently")
print("   • Equal weight to all training samples")
print("   • Example: Random Forest")
print("   • Goal: Reduce variance")
print("   • Formula: ŷ = (1/M) Σ f_m(x)")

print("\n🚀 BOOSTING:")
print("   • Sequential training: each model learns from previous mistakes")
print("   • Adaptive weights: hard examples get more attention")
print("   • Example: AdaBoost, Gradient Boosting")
print("   • Goal: Reduce bias")
print("   • Formula: ŷ = Σ α_m * f_m(x)")

# Visualization concept
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bagging
axes[0].text(0.5, 0.9, 'BAGGING', ha='center', fontsize=16, 
             fontweight='bold', transform=axes[0].transAxes)
for i in range(3):
    y_pos = 0.7 - i*0.25
    axes[0].add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.1, 
                                     fill=True, color='lightblue', 
                                     edgecolor='black', linewidth=2))
    axes[0].text(0.5, y_pos, f'Model {i+1}', ha='center', va='center',
                fontsize=12, transform=axes[0].transAxes)
    
axes[0].annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.2),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'),
                transform=axes[0].transAxes)
axes[0].text(0.5, 0.08, 'Average Predictions', ha='center', fontsize=12,
            fontweight='bold', transform=axes[0].transAxes)
axes[0].axis('off')
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# Boosting
axes[1].text(0.5, 0.9, 'BOOSTING', ha='center', fontsize=16, 
             fontweight='bold', transform=axes[1].transAxes)
for i in range(3):
    y_pos = 0.7 - i*0.25
    color_intensity = 0.3 + i*0.3
    axes[1].add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.1, 
                                     fill=True, color=(1, color_intensity, color_intensity), 
                                     edgecolor='black', linewidth=2))
    axes[1].text(0.5, y_pos, f'Model {i+1} (α_{i+1})', ha='center', va='center',
                fontsize=12, transform=axes[1].transAxes)
    if i < 2:
        axes[1].annotate('', xy=(0.5, y_pos-0.15), xytext=(0.5, y_pos-0.05),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                        transform=axes[1].transAxes)
        axes[1].text(0.7, y_pos-0.1, 'Learns from\nerrors', ha='left', 
                    fontsize=9, transform=axes[1].transAxes)

axes[1].annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.2),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'),
                transform=axes[1].transAxes)
axes[1].text(0.5, 0.08, 'Weighted Sum', ha='center', fontsize=12,
            fontweight='bold', transform=axes[1].transAxes)
axes[1].axis('off')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('bagging_vs_boosting.png', dpi=300, bbox_inches='tight')
plt.show()
