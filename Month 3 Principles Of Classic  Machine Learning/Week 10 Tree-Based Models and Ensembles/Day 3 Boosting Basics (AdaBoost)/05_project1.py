

#Task: Project - Compare Bagging and Boosting.

#Use sklearn.ensemble.AdaBoostClassifier.

#Run a comparison script: Random Forest (Bagging) vs. AdaBoost on a complex dataset
#(e.g., Moons dataset in sklearn).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 80)
print("BAGGING vs BOOSTING COMPARISON PROJECT")
print("=" * 80)
print("\n")

# ============================================================================
# 1. CREATE MULTIPLE DATASETS FOR COMPREHENSIVE COMPARISON
# ============================================================================
print("1. CREATING DATASETS")
print("-" * 80)

datasets = {}

# Dataset 1: Moons (Non-linear, complex)
X_moons, y_moons = make_moons(n_samples=1000, noise=0.3, random_state=42)
datasets['Moons'] = (X_moons, y_moons)
print("✓ Moons dataset created (1000 samples, 2 features)")

# Dataset 2: Circles (Highly non-linear)
X_circles, y_circles = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)
datasets['Circles'] = (X_circles, y_circles)
print("✓ Circles dataset created (1000 samples, 2 features)")

# Dataset 3: Complex Classification (Multi-feature)
X_complex, y_complex = make_classification(n_samples=1000, n_features=20, 
                                           n_informative=15, n_redundant=5,
                                           n_classes=2, random_state=42)
datasets['Complex'] = (X_complex, y_complex)
print("✓ Complex dataset created (1000 samples, 20 features)")

print("\n")

# ============================================================================
# 2. VISUALIZE DATASETS
# ============================================================================
print("2. VISUALIZING DATASETS")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, (X, y)) in enumerate(datasets.items()):
    if X.shape[1] == 2:  # Only plot 2D datasets
        axes[idx].scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', alpha=0.6)
        axes[idx].scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', alpha=0.6)
        axes[idx].set_title(f'{name} Dataset', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

# For complex dataset, show feature importance later
axes[2].text(0.5, 0.5, f'Complex Dataset\n20 Features\nVisualization: See Feature Importance', 
             ha='center', va='center', fontsize=12, transform=axes[2].transAxes)
axes[2].set_xlim(0, 1)
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('01_datasets_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Dataset visualization saved as '01_datasets_visualization.png'")
plt.show()

print("\n")

# ============================================================================
# 3. MODEL TRAINING AND EVALUATION FUNCTION
# ============================================================================
print("3. DEFINING MODELS")
print("-" * 80)

def evaluate_models(X, y, dataset_name):
    """
    Train and evaluate both Random Forest and AdaBoost models
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features (helpful for some datasets)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest (Bagging)': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'AdaBoost (Boosting)': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} on {dataset_name} dataset...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Calculate metrics
        results[name] = {
            'model': model,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1_score': f1_score(y_test, y_pred_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_test': X_test_scaled
        }
        
        print(f"  Test Accuracy: {results[name]['test_accuracy']:.4f}")
        print(f"  ROC-AUC: {results[name]['roc_auc']:.4f}")
        print(f"  CV Score: {results[name]['cv_mean']:.4f} (±{results[name]['cv_std']:.4f})")
    
    return results

print("✓ Model evaluation function defined")
print("\n")

# ============================================================================
# 4. TRAIN AND EVALUATE ON ALL DATASETS
# ============================================================================
print("4. TRAINING AND EVALUATING MODELS")
print("=" * 80)

all_results = {}

for dataset_name, (X, y) in datasets.items():
    print(f"\n{'=' * 80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 80}")
    all_results[dataset_name] = evaluate_models(X, y, dataset_name)

print("\n")

# ============================================================================
# 5. CREATE COMPARISON DATAFRAME
# ============================================================================
print("5. CREATING RESULTS COMPARISON TABLE")
print("-" * 80)

comparison_data = []

for dataset_name, results in all_results.items():
    for model_name, metrics in results.items():
        comparison_data.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Train Accuracy': metrics['train_accuracy'],
            'Test Accuracy': metrics['test_accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'CV Mean': metrics['cv_mean'],
            'CV Std': metrics['cv_std']
        })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))
print("\n✓ Comparison table created")

# Save to CSV
comparison_df.to_csv('comparison_results.csv', index=False)
print("✓ Results saved to 'comparison_results.csv'")
print("\n")

# ============================================================================
# 6. STATISTICAL COMPARISON
# ============================================================================
print("6. STATISTICAL COMPARISON")
print("-" * 80)

for dataset_name in datasets.keys():
    print(f"\n{dataset_name} Dataset:")
    rf_acc = comparison_df[(comparison_df['Dataset']==dataset_name) & 
                           (comparison_df['Model']=='Random Forest (Bagging)')]['Test Accuracy'].values[0]
    ab_acc = comparison_df[(comparison_df['Dataset']==dataset_name) & 
                           (comparison_df['Model']=='AdaBoost (Boosting)')]['Test Accuracy'].values[0]
    
    diff = rf_acc - ab_acc
    if abs(diff) < 0.01:
        print(f"  Performance is similar (difference: {diff:.4f})")
    elif diff > 0:
        print(f"  Random Forest performs better by {diff:.4f}")
    else:
        print(f"  AdaBoost performs better by {abs(diff):.4f}")

print("\n")

# ============================================================================
# 7. VISUALIZATION: PERFORMANCE METRICS COMPARISON
# ============================================================================
print("7. CREATING PERFORMANCE VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_to_plot = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    
    pivot_data = comparison_df.pivot(index='Dataset', columns='Model', values=metric)
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pivot_data['Random Forest (Bagging)'], 
                   width, label='Random Forest', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, pivot_data['AdaBoost (Boosting)'], 
                   width, label='AdaBoost', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets.keys())
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('02_performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Performance metrics visualization saved")
plt.show()

# ============================================================================
# 8. CONFUSION MATRICES
# ============================================================================
print("8. CREATING CONFUSION MATRICES")
print("-" * 80)

fig, axes = plt.subplots(3, 2, figsize=(12, 16))

for idx, (dataset_name, results) in enumerate(all_results.items()):
    for j, (model_name, metrics) in enumerate(results.items()):
        ax = axes[idx, j]
        
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar_kws={'label': 'Count'})
        ax.set_title(f'{model_name}\n{dataset_name} Dataset', 
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('03_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrices saved")
plt.show()

# ============================================================================
# 9. ROC CURVES
# ============================================================================
print("9. CREATING ROC CURVES")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (dataset_name, results) in enumerate(all_results.items()):
    ax = axes[idx]
    
    for model_name, metrics in results.items():
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
        roc_auc = metrics['roc_auc']
        
        color = '#2ecc71' if 'Random Forest' in model_name else '#e74c3c'
        linestyle = '-' if 'Random Forest' in model_name else '--'
        
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2,
               label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title(f'ROC Curve - {dataset_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ ROC curves saved")
plt.show()

# ============================================================================
# 10. LEARNING CURVES
# ============================================================================
print("10. CREATING LEARNING CURVES")
print("-" * 80)

fig, axes = plt.subplots(3, 2, figsize=(14, 16))

for idx, (dataset_name, (X, y)) in enumerate(datasets.items()):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'Random Forest (Bagging)': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost (Boosting)': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3), 
            n_estimators=100, 
            random_state=42
        )
    }
    
    for j, (model_name, model) in enumerate(models.items()):
        ax = axes[idx, j]
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_scaled, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', color='#3498db', 
               label='Training score', linewidth=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='#3498db')
        
        ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c',
               label='Validation score', linewidth=2)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='#e74c3c')
        
        ax.set_xlabel('Training Set Size', fontsize=10, fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}\n{dataset_name} Dataset', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_learning_curves.png', dpi=300, bbox_inches='tight')
print("✓ Learning curves saved")
plt.show()

# ============================================================================
# 11. DECISION BOUNDARY VISUALIZATION (2D datasets only)
# ============================================================================
print("11. CREATING DECISION BOUNDARIES")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for idx, dataset_name in ['Moons', 'Circles']:
    X, y = datasets[dataset_name]
    results = all_results[dataset_name]
    
    for j, (model_name, metrics) in enumerate(results.items()):
        ax = axes[idx, j]
        
        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict on mesh
        scaler = StandardScaler()
        scaler.fit(X)
        Z = metrics['model'].predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                           edgecolors='black', s=50, alpha=0.7)
        
        ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}\n{dataset_name} Dataset\nAccuracy: {metrics["test_accuracy"]:.3f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_decision_boundaries.png', dpi=300, bbox_inches='tight')
print("✓ Decision boundaries saved")
plt.show()

# ============================================================================
# 12. FEATURE IMPORTANCE (Complex dataset)
# ============================================================================
print("12. ANALYZING FEATURE IMPORTANCE")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

complex_results = all_results['Complex']

for idx, (model_name, metrics) in enumerate(complex_results.items()):
    model = metrics['model']
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        ax = axes[idx]
        ax.barh(range(len(indices)), importances[indices], color='#3498db', alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([f'Feature {i}' for i in indices])
        ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax.set_title(f'Top 15 Feature Importances\n{model_name}', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

plt.tight_layout()
plt.savefig('07_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance analysis saved")
plt.show()

# ============================================================================
# 13. FINAL SUMMARY
# ============================================================================
print("\n")
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\nKEY FINDINGS:")
print("-" * 80)

for dataset_name in datasets.keys():
    print(f"\n{dataset_name} Dataset:")
    dataset_results = comparison_df[comparison_df['Dataset'] == dataset_name]
    
    rf_row = dataset_results[dataset_results['Model'] == 'Random Forest (Bagging)'].iloc[0]
    ab_row = dataset_results[dataset_results['Model'] == 'AdaBoost (Boosting)'].iloc[0]
    
    print(f"  Random Forest - Accuracy: {rf_row['Test Accuracy']:.4f}, F1: {rf_row['F1-Score']:.4f}")
    print(f"  AdaBoost      - Accuracy: {ab_row['Test Accuracy']:.4f}, F1: {ab_row['F1-Score']:.4f}")
    
    if rf_row['Test Accuracy'] > ab_row['Test Accuracy']:
        print(f"  Winner: Random Forest (+{rf_row['Test Accuracy'] - ab_row['Test Accuracy']:.4f})")
    else:
        print(f"  Winner: AdaBoost (+{ab_row['Test Accuracy'] - rf_row['Test Accuracy']:.4f})")

print("\n" + "=" * 80)
print("COMPARISON INSIGHTS:")
print("=" * 80)
print("""
1. BAGGING (Random Forest):
   - Reduces variance by averaging multiple models
   - Builds independent trees in parallel
   - Less prone to overfitting
   - Better with high-variance, low-bias base learners
   - More robust to outliers

2. BOOSTING (AdaBoost):
   - Reduces bias by focusing on hard examples
   - Builds trees sequentially, each correcting previous errors
   - Can be more accurate but prone to overfitting
   - Better with low-variance, high-bias base learners
   - More sensitive to noisy data and outliers

3. When to use which:
   - Use Random Forest when you have noisy data or outliers
   - Use AdaBoost when you need to squeeze out maximum accuracy
   - Random Forest is generally more stable and easier to tune
   - AdaBoost can perform better on clean, well-structured data
""")


