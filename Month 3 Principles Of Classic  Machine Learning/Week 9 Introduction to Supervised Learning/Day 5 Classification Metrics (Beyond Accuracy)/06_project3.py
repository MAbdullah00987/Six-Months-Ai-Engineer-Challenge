
#Project 3: Feature Scaling Impact**
# Show how StandardScaler affects logistic regression
# Compare scaled vs unscaled performance
# Visualize the difference


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


print("PROJECT 3: FEATURE SCALING IMPACT ON LOGISTIC REGRESSION")


# STEP 1: Generate Synthetic Dataset with Different Scales
print("\n[STEP 1] Generating synthetic dataset with features at different scales...")

np.random.seed(42)

# Generate base classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

# Create features with drastically different scales
# Feature 1: Small scale (0-1)
X[:, 0] = X[:, 0] * 0.5

# Feature 2: Medium scale (0-100)
X[:, 1] = X[:, 1] * 50 + 50

# Feature 3: Large scale (0-10000)
X[:, 2] = X[:, 2] * 5000 + 5000

# Feature 4: Very large scale (0-1000000)
X[:, 3] = X[:, 3] * 500000 + 500000

# Create DataFrame
feature_names = ['Feature_1 (0-1)', 'Feature_2 (0-100)', 
                 'Feature_3 (0-10K)', 'Feature_4 (0-1M)']
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\n" + "="*70)
print("Feature Statistics (Unscaled)")
print("="*70)
print(df[feature_names].describe())


# STEP 2: Split Data

print("\n[STEP 2] Splitting data into training and test sets (70-30)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


# STEP 3: Apply StandardScaler
print("\n[STEP 3] Applying StandardScaler to features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaler parameters:")
print("Mean:", scaler.mean_)
print("Standard Deviation:", np.sqrt(scaler.var_))

print("\n" + "="*70)
print("Feature Statistics (Scaled)")
print("="*70)
df_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
print(df_scaled.describe())


# STEP 4: Train Logistic Regression Models
print("\n[STEP 4] Training Logistic Regression models...")

# Model 1: Without scaling
print("\nTraining model WITHOUT scaling...")
model_unscaled = LogisticRegression(random_state=42, max_iter=1000)
model_unscaled.fit(X_train, y_train)

# Model 2: With scaling
print("Training model WITH scaling...")
model_scaled = LogisticRegression(random_state=42, max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)


# STEP 5: Make Predictions

print("\n[STEP 5] Making predictions...")

y_pred_unscaled = model_unscaled.predict(X_test)
y_pred_scaled = model_scaled.predict(X_test_scaled)

y_pred_proba_unscaled = model_unscaled.predict_proba(X_test)[:, 1]
y_pred_proba_scaled = model_scaled.predict_proba(X_test_scaled)[:, 1]


# STEP 6: Evaluate Performance
print("MODEL PERFORMANCE COMPARISON")


def evaluate_model(y_true, y_pred, model_name):
    """Calculate and display model metrics"""
    print(f"\n{model_name}")
    print("-" * 50)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }

metrics_unscaled = evaluate_model(y_test, y_pred_unscaled, "UNSCALED MODEL")
metrics_scaled = evaluate_model(y_test, y_pred_scaled, "SCALED MODEL")

# Classification Reports
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORTS")
print("="*70)
print("\nUnscaled Model:")
print(classification_report(y_test, y_pred_unscaled))

print("\nScaled Model:")
print(classification_report(y_test, y_pred_scaled))


# STEP 7: Analyze Coefficients
print("MODEL COEFFICIENTS ANALYSIS")


print("\nUnscaled Model Coefficients:")
coef_unscaled = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model_unscaled.coef_[0],
    'Abs_Coefficient': np.abs(model_unscaled.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)
print(coef_unscaled)
print(f"Intercept: {model_unscaled.intercept_[0]:.6f}")

print("\nScaled Model Coefficients:")
coef_scaled = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model_scaled.coef_[0],
    'Abs_Coefficient': np.abs(model_scaled.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)
print(coef_scaled)
print(f"Intercept: {model_scaled.intercept_[0]:.6f}")


# STEP 8: Statistical Analysis with Statsmodels

print("STATSMODELS LOGISTIC REGRESSION ANALYSIS")


# Add constant for statsmodels
X_train_sm = sm.add_constant(X_train_scaled)
X_test_sm = sm.add_constant(X_test_scaled)

# Fit statsmodels logistic regression
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit(disp=0)

print("\nStatsmodels Summary (Scaled Data):")
print(result.summary())


# STEP 9: Visualizations
print("\n[STEP 9] Creating visualizations...")

fig = plt.figure(figsize=(16, 12))

# Subplot 1: Feature Distributions (Unscaled)
ax1 = plt.subplot(3, 3, 1)
for i, col in enumerate(feature_names):
    plt.hist(df[col], alpha=0.5, label=col, bins=30)
plt.title('Feature Distributions (Unscaled)', fontsize=12, fontweight='bold')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend(fontsize=8)
plt.yscale('log')

# Subplot 2: Feature Distributions (Scaled)
ax2 = plt.subplot(3, 3, 2)
for i, col in enumerate(feature_names):
    plt.hist(df_scaled[col], alpha=0.5, label=col, bins=30)
plt.title('Feature Distributions (Scaled)', fontsize=12, fontweight='bold')
plt.xlabel('Standardized Value')
plt.ylabel('Frequency')
plt.legend(fontsize=8)

# Subplot 3: Coefficient Comparison
ax3 = plt.subplot(3, 3, 3)
x_pos = np.arange(len(feature_names))
width = 0.35
plt.bar(x_pos - width/2, model_unscaled.coef_[0], width, 
        label='Unscaled', alpha=0.8, color='coral')
plt.bar(x_pos + width/2, model_scaled.coef_[0], width, 
        label='Scaled', alpha=0.8, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Model Coefficients Comparison', fontsize=12, fontweight='bold')
plt.xticks(x_pos, [f'F{i+1}' for i in range(len(feature_names))], rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Subplot 4: Confusion Matrix - Unscaled
ax4 = plt.subplot(3, 3, 4)
cm_unscaled = confusion_matrix(y_test, y_pred_unscaled)
sns.heatmap(cm_unscaled, annot=True, fmt='d', cmap='Reds', 
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix (Unscaled)', fontsize=12, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Subplot 5: Confusion Matrix - Scaled
ax5 = plt.subplot(3, 3, 5)
cm_scaled = confusion_matrix(y_test, y_pred_scaled)
sns.heatmap(cm_scaled, annot=True, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix (Scaled)', fontsize=12, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Subplot 6: ROC Curves
ax6 = plt.subplot(3, 3, 6)
fpr_unscaled, tpr_unscaled, _ = roc_curve(y_test, y_pred_proba_unscaled)
fpr_scaled, tpr_scaled, _ = roc_curve(y_test, y_pred_proba_scaled)
auc_unscaled = auc(fpr_unscaled, tpr_unscaled)
auc_scaled = auc(fpr_scaled, tpr_scaled)

plt.plot(fpr_unscaled, tpr_unscaled, label=f'Unscaled (AUC={auc_unscaled:.3f})', 
         linewidth=2, color='coral')
plt.plot(fpr_scaled, tpr_scaled, label=f'Scaled (AUC={auc_scaled:.3f})', 
         linewidth=2, color='skyblue')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# Subplot 7: Performance Metrics Comparison
ax7 = plt.subplot(3, 3, 7)
metrics_df = pd.DataFrame({
    'Unscaled': metrics_unscaled,
    'Scaled': metrics_scaled
})
metrics_df.plot(kind='bar', ax=ax7, color=['coral', 'skyblue'], alpha=0.8)
plt.title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model')
plt.ylim([0, 1.1])
plt.grid(axis='y', alpha=0.3)

# Subplot 8: Feature Scale Comparison
ax8 = plt.subplot(3, 3, 8)
scales = [df[col].max() - df[col].min() for col in feature_names]
plt.barh(range(len(feature_names)), scales, color='lightcoral', alpha=0.7)
plt.yticks(range(len(feature_names)), [f'F{i+1}' for i in range(len(feature_names))])
plt.xlabel('Scale Range (Max - Min)')
plt.title('Feature Scale Differences (Unscaled)', fontsize=12, fontweight='bold')
plt.xscale('log')
plt.grid(axis='x', alpha=0.3)

# Subplot 9: Coefficient Magnitude Comparison
ax9 = plt.subplot(3, 3, 9)
coef_comparison = pd.DataFrame({
    'Unscaled': np.abs(model_unscaled.coef_[0]),
    'Scaled': np.abs(model_scaled.coef_[0])
}, index=[f'F{i+1}' for i in range(len(feature_names))])
coef_comparison.plot(kind='bar', ax=ax9, color=['coral', 'skyblue'], alpha=0.8)
plt.title('Absolute Coefficient Magnitudes', fontsize=12, fontweight='bold')
plt.ylabel('|Coefficient|')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_scaling_impact.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'feature_scaling_impact.png'")
plt.show()


# STEP 10: Key Insights and Conclusions

print("KEY INSIGHTS AND CONCLUSIONS")

print("\n1. FEATURE SCALING IMPACT:")
print(f"   - Features ranged from 0-1 to 0-1M before scaling")
print(f"   - After scaling, all features have mean=0 and std=1")

print("\n2. MODEL PERFORMANCE:")
accuracy_diff = metrics_scaled['Accuracy'] - metrics_unscaled['Accuracy']
print(f"   - Unscaled Model Accuracy: {metrics_unscaled['Accuracy']:.4f}")
print(f"   - Scaled Model Accuracy:   {metrics_scaled['Accuracy']:.4f}")
print(f"   - Improvement: {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")

print("\n3. COEFFICIENT INTERPRETATION:")
print(f"   - Unscaled coefficients vary by {np.abs(model_unscaled.coef_[0]).max():.6f}")
print(f"   - Scaled coefficients are more interpretable and comparable")
print(f"   - Scaled coefficients show true feature importance")

print("\n4. CONVERGENCE:")
print(f"   - Unscaled model iterations: {model_unscaled.n_iter_[0]}")
print(f"   - Scaled model iterations:   {model_scaled.n_iter_[0]}")

print("\n5. RECOMMENDATIONS:")
print("    Always use StandardScaler for logistic regression")
print("    Scaling improves convergence and interpretability")
print("    Scaled coefficients reflect true feature importance")
print("    Essential when features have different units/scales")


