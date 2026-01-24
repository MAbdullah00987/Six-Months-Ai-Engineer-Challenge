

#Task: Project - Heart Disease Prediction.

#Compare a single DecisionTree vs. a RandomForestClassifier.

#Observe how the Random Forest reduces Variance (overfitting).


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("HEART DISEASE PREDICTION PROJECT")
print("Decision Tree vs Random Forest Comparison")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n[STEP 1] Loading Heart Disease Dataset...")

# Using UCI Heart Disease dataset structure
# Creating a synthetic dataset (you can replace with real data)
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(29, 80, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples),  # chest pain type
    'trestbps': np.random.randint(90, 200, n_samples),  # resting blood pressure
    'chol': np.random.randint(120, 400, n_samples),  # cholesterol
    'fbs': np.random.randint(0, 2, n_samples),  # fasting blood sugar
    'restecg': np.random.randint(0, 3, n_samples),  # resting ECG
    'thalach': np.random.randint(70, 200, n_samples),  # max heart rate
    'exang': np.random.randint(0, 2, n_samples),  # exercise induced angina
    'oldpeak': np.random.uniform(0, 6, n_samples),  # ST depression
    'slope': np.random.randint(0, 3, n_samples),
    'ca': np.random.randint(0, 4, n_samples),  # number of major vessels
    'thal': np.random.randint(0, 4, n_samples)
}

# Create target with some correlation to features
target = ((data['age'] > 55).astype(int) + 
          (data['cp'] > 1).astype(int) + 
          (data['thalach'] < 130).astype(int) + 
          (data['chol'] > 250).astype(int) + 
          np.random.randint(0, 2, n_samples)) > 2

data['target'] = target.astype(int)

df = pd.DataFrame(data)

print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset Info:")
print(df.info())

print(f"\nStatistical Summary:")
print(df.describe())

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n[STEP 2] Exploratory Data Analysis...")

# Check for missing values
print(f"\nMissing Values:")
print(df.isnull().sum())

# Target distribution
print(f"\nTarget Distribution:")
print(df['target'].value_counts())
print(f"Percentage with Heart Disease: {df['target'].mean()*100:.2f}%")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Target Distribution
axes[0, 0].pie(df['target'].value_counts(), labels=['No Disease', 'Disease'], 
               autopct='%1.1f%%', startangle=90, colors=['#90EE90', '#FF6B6B'])
axes[0, 0].set_title('Heart Disease Distribution', fontsize=14, fontweight='bold')

# 2. Age Distribution by Target
df[df['target']==0]['age'].hist(ax=axes[0, 1], bins=20, alpha=0.6, label='No Disease', color='green')
df[df['target']==1]['age'].hist(ax=axes[0, 1], bins=20, alpha=0.6, label='Disease', color='red')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Age Distribution by Heart Disease', fontsize=14, fontweight='bold')
axes[0, 1].legend()

# 3. Correlation Heatmap
corr = df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=axes[1, 0], cbar_kws={'shrink': 0.8})
axes[1, 0].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

# 4. Max Heart Rate vs Age
scatter = axes[1, 1].scatter(df['age'], df['thalach'], c=df['target'], 
                             cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Max Heart Rate')
axes[1, 1].set_title('Max Heart Rate vs Age', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=axes[1, 1], label='Heart Disease')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("✓ EDA visualizations saved as 'eda_analysis.png'")
plt.show()

# ============================================================================
# STEP 3: STATISTICAL ANALYSIS WITH STATSMODELS
# ============================================================================
print("\n[STEP 3] Statistical Analysis...")

# Logistic Regression for feature importance
X_stats = df.drop('target', axis=1)
y_stats = df['target']
X_stats_const = sm.add_constant(X_stats)

logit_model = sm.Logit(y_stats, X_stats_const)
result = logit_model.fit(disp=0)

print("\nLogistic Regression Summary (Top Features):")
print(result.summary().tables[1])

# ============================================================================
# STEP 4: DATA PREPARATION
# ============================================================================
print("\n[STEP 4] Preparing Data for Machine Learning...")

X = df.drop('target', axis=1)
y = df['target']

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature Scaling (optional but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 5: DECISION TREE MODEL
# ============================================================================
print("\n[STEP 5] Training Decision Tree Classifier...")

dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

dt_model.fit(X_train, y_train)

# Predictions
y_train_pred_dt = dt_model.predict(X_train)
y_test_pred_dt = dt_model.predict(X_test)

# Metrics
train_acc_dt = accuracy_score(y_train, y_train_pred_dt)
test_acc_dt = accuracy_score(y_test, y_test_pred_dt)

print(f"\nDecision Tree Results:")
print(f"Training Accuracy: {train_acc_dt:.4f}")
print(f"Test Accuracy: {test_acc_dt:.4f}")
print(f"Variance (Overfitting): {train_acc_dt - test_acc_dt:.4f}")

print(f"\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_test_pred_dt))

# ============================================================================
# STEP 6: RANDOM FOREST MODEL
# ============================================================================
print("\n[STEP 6] Training Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Metrics
train_acc_rf = accuracy_score(y_train, y_train_pred_rf)
test_acc_rf = accuracy_score(y_test, y_test_pred_rf)

print(f"\nRandom Forest Results:")
print(f"Training Accuracy: {train_acc_rf:.4f}")
print(f"Test Accuracy: {test_acc_rf:.4f}")
print(f"Variance (Overfitting): {train_acc_rf - test_acc_rf:.4f}")

print(f"\nClassification Report (Random Forest):")
print(classification_report(y_test, y_test_pred_rf))

# ============================================================================
# STEP 7: COMPARISON AND VARIANCE REDUCTION
# ============================================================================
print("\n[STEP 7] Comparing Models - Variance Reduction Analysis...")

comparison = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Train Accuracy': [train_acc_dt, train_acc_rf],
    'Test Accuracy': [test_acc_dt, test_acc_rf],
    'Variance (Overfitting)': [train_acc_dt - test_acc_dt, train_acc_rf - test_acc_rf]
})

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(comparison.to_string(index=False))
print("="*70)

variance_reduction = ((train_acc_dt - test_acc_dt) - (train_acc_rf - test_acc_rf))
print(f"\n✓ Variance Reduction by Random Forest: {variance_reduction:.4f}")
print(f"✓ Reduction Percentage: {(variance_reduction/(train_acc_dt - test_acc_dt))*100:.2f}%")

# ============================================================================
# STEP 8: CROSS-VALIDATION
# ============================================================================
print("\n[STEP 8] Cross-Validation Analysis...")

cv_scores_dt = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

print(f"\nDecision Tree CV Scores: {cv_scores_dt}")
print(f"Mean: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std() * 2:.4f})")

print(f"\nRandom Forest CV Scores: {cv_scores_rf}")
print(f"Mean: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================
print("\n[STEP 9] Creating Comparison Visualizations...")

fig = plt.figure(figsize=(16, 12))

# 1. Accuracy Comparison
ax1 = plt.subplot(2, 3, 1)
models = ['Decision Tree', 'Random Forest']
train_scores = [train_acc_dt, train_acc_rf]
test_scores = [test_acc_dt, test_acc_rf]

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, train_scores, width, label='Train', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, test_scores, width, label='Test', color='#e74c3c', alpha=0.8)

ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Variance (Overfitting) Comparison
ax2 = plt.subplot(2, 3, 2)
variance_scores = [train_acc_dt - test_acc_dt, train_acc_rf - test_acc_rf]
colors = ['#e74c3c', '#2ecc71']
bars = ax2.bar(models, variance_scores, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Variance (Train - Test)')
ax2.set_title('Overfitting Comparison (Lower is Better)', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Confusion Matrix - Decision Tree
ax3 = plt.subplot(2, 3, 3)
cm_dt = confusion_matrix(y_test, y_test_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar_kws={'shrink': 0.8})
ax3.set_title('Confusion Matrix - Decision Tree', fontweight='bold')
ax3.set_ylabel('Actual')
ax3.set_xlabel('Predicted')

# 4. Confusion Matrix - Random Forest
ax4 = plt.subplot(2, 3, 4)
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax4, cbar_kws={'shrink': 0.8})
ax4.set_title('Confusion Matrix - Random Forest', fontweight='bold')
ax4.set_ylabel('Actual')
ax4.set_xlabel('Predicted')

# 5. Feature Importance - Random Forest
ax5 = plt.subplot(2, 3, 5)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

ax5.barh(feature_importance['feature'], feature_importance['importance'], color='#9b59b6', alpha=0.7)
ax5.set_xlabel('Importance')
ax5.set_title('Top 10 Feature Importance (Random Forest)', fontweight='bold')
ax5.invert_yaxis()

# 6. ROC Curve
ax6 = plt.subplot(2, 3, 6)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

auc_dt = auc(fpr_dt, tpr_dt)
auc_rf = auc(fpr_rf, tpr_rf)

ax6.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.3f})', linewidth=2, color='#e74c3c')
ax6.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', linewidth=2, color='#2ecc71')
ax6.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax6.set_xlabel('False Positive Rate')
ax6.set_ylabel('True Positive Rate')
ax6.set_title('ROC Curve Comparison', fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Model comparison visualizations saved as 'model_comparison.png'")
plt.show()

# ============================================================================
# STEP 10: LEARNING CURVES (VARIANCE VISUALIZATION)
# ============================================================================
print("\n[STEP 10] Generating Learning Curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Decision Tree Learning Curve
train_sizes, train_scores_dt, val_scores_dt = learning_curve(
    dt_model, X_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_mean_dt = train_scores_dt.mean(axis=1)
train_std_dt = train_scores_dt.std(axis=1)
val_mean_dt = val_scores_dt.mean(axis=1)
val_std_dt = val_scores_dt.std(axis=1)

ax1.plot(train_sizes, train_mean_dt, label='Training Score', color='#3498db', linewidth=2, marker='o')
ax1.fill_between(train_sizes, train_mean_dt - train_std_dt, train_mean_dt + train_std_dt, alpha=0.2, color='#3498db')
ax1.plot(train_sizes, val_mean_dt, label='Validation Score', color='#e74c3c', linewidth=2, marker='s')
ax1.fill_between(train_sizes, val_mean_dt - val_std_dt, val_mean_dt + val_std_dt, alpha=0.2, color='#e74c3c')
ax1.set_xlabel('Training Set Size')
ax1.set_ylabel('Accuracy Score')
ax1.set_title('Learning Curve - Decision Tree (Shows Overfitting)', fontweight='bold', fontsize=14)
ax1.legend(loc='best')
ax1.grid(alpha=0.3)

# Random Forest Learning Curve
train_sizes, train_scores_rf, val_scores_rf = learning_curve(
    rf_model, X_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_mean_rf = train_scores_rf.mean(axis=1)
train_std_rf = train_scores_rf.std(axis=1)
val_mean_rf = val_scores_rf.mean(axis=1)
val_std_rf = val_scores_rf.std(axis=1)

ax2.plot(train_sizes, train_mean_rf, label='Training Score', color='#3498db', linewidth=2, marker='o')
ax2.fill_between(train_sizes, train_mean_rf - train_std_rf, train_mean_rf + train_std_rf, alpha=0.2, color='#3498db')
ax2.plot(train_sizes, val_mean_rf, label='Validation Score', color='#2ecc71', linewidth=2, marker='s')
ax2.fill_between(train_sizes, val_mean_rf - val_std_rf, val_mean_rf + val_std_rf, alpha=0.2, color='#2ecc71')
ax2.set_xlabel('Training Set Size')
ax2.set_ylabel('Accuracy Score')
ax2.set_title('Learning Curve - Random Forest (Reduces Overfitting)', fontweight='bold', fontsize=14)
ax2.legend(loc='best')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
print("✓ Learning curves saved as 'learning_curves.png'")
plt.show()


# FINAL SUMMARY


print("="*70)
print(f"\n📊 Dataset: {len(df)} samples, {len(df.columns)-1} features")
print(f"📈 Target Distribution: {df['target'].value_counts().to_dict()}")
print(f"\n🌳 Decision Tree:")
print(f"   - Train Accuracy: {train_acc_dt:.4f}")
print(f"   - Test Accuracy: {test_acc_dt:.4f}")
print(f"   - Overfitting (Variance): {train_acc_dt - test_acc_dt:.4f}")
print(f"   - AUC Score: {auc_dt:.4f}")
print(f"\n🌲 Random Forest:")
print(f"   - Train Accuracy: {train_acc_rf:.4f}")
print(f"   - Test Accuracy: {test_acc_rf:.4f}")
print(f"   - Overfitting (Variance): {train_acc_rf - test_acc_rf:.4f}")
print(f"   - AUC Score: {auc_rf:.4f}")
print(f"\n✨ Key Insight:")
print(f"   Random Forest reduced overfitting by {variance_reduction:.4f}")
print(f"   ({(variance_reduction/(train_acc_dt - test_acc_dt))*100:.2f}% improvement)")
print(f"\n💡 Why Random Forest Reduces Variance:")
print(f"   1. Uses multiple trees (ensemble) instead of single tree")
print(f"   2. Each tree trained on random subset (bagging)")
print(f"   3. Random feature selection at each split")
print(f"   4. Averages predictions → smoother decision boundary")

print("\n Project Complete! All visualizations saved.")
print("📁 Files created:")
print("   - eda_analysis.png")
print("   - model_comparison.png")
print("   - learning_curves.png")