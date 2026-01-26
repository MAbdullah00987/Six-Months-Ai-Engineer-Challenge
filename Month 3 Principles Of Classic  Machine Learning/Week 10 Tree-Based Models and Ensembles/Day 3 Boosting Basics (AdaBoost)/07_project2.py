#Project: Heart Disease Prediction

#Compare single Decision Tree vs Random Forest
#Evaluate both models
#Analyze performance differences

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("HEART DISEASE PREDICTION: DECISION TREE vs RANDOM FOREST")
print("="*70)

# ============================================================================
# 1. CREATE SYNTHETIC HEART DISEASE DATASET
# ============================================================================
print("\n[STEP 1] Creating Synthetic Heart Disease Dataset...")

np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'age': np.random.randint(29, 80, n_samples),
    'sex': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
    'cp': np.random.choice([0, 1, 2, 3], n_samples),  # Chest pain type
    'trestbps': np.random.randint(90, 200, n_samples),  # Resting blood pressure
    'chol': np.random.randint(126, 400, n_samples),  # Cholesterol
    'fbs': np.random.choice([0, 1], n_samples),  # Fasting blood sugar > 120
    'restecg': np.random.choice([0, 1, 2], n_samples),  # Resting ECG
    'thalach': np.random.randint(71, 202, n_samples),  # Max heart rate
    'exang': np.random.choice([0, 1], n_samples),  # Exercise induced angina
    'oldpeak': np.random.uniform(0, 6.2, n_samples),  # ST depression
    'slope': np.random.choice([0, 1, 2], n_samples),  # Slope of peak exercise
    'ca': np.random.choice([0, 1, 2, 3, 4], n_samples),  # Number of major vessels
    'thal': np.random.choice([0, 1, 2, 3], n_samples)  # Thalassemia
}

df = pd.DataFrame(data)

# Create target variable with realistic logic
# Higher risk with: older age, male, high cholesterol, low max HR, exercise angina
risk_score = (
    (df['age'] > 55) * 0.3 +
    (df['sex'] == 1) * 0.2 +
    (df['chol'] > 240) * 0.2 +
    (df['thalach'] < 120) * 0.2 +
    (df['exang'] == 1) * 0.3 +
    (df['cp'] > 1) * 0.25 +
    (df['ca'] > 0) * 0.2 +
    np.random.uniform(0, 0.3, n_samples)
)

df['target'] = (risk_score > 0.6).astype(int)  # 1: Disease, 0: No disease

print(f"Dataset created with {n_samples} samples and {len(df.columns)} features")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n[STEP 2] Exploratory Data Analysis...")

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Target Distribution ---")
print(df['target'].value_counts())
print(f"\nDisease Prevalence: {df['target'].mean()*100:.2f}%")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Target distribution
axes[0, 0].pie(df['target'].value_counts(), labels=['No Disease', 'Disease'], 
               autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Heart Disease Distribution', fontsize=14, fontweight='bold')

# Age distribution by target
df[df['target']==0]['age'].hist(bins=20, alpha=0.5, label='No Disease', 
                                 color='green', ax=axes[0, 1])
df[df['target']==1]['age'].hist(bins=20, alpha=0.5, label='Disease', 
                                 color='red', ax=axes[0, 1])
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Age Distribution by Disease Status', fontsize=14, fontweight='bold')
axes[0, 1].legend()

# Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=axes[1, 0])
axes[1, 0].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

# Box plot: Cholesterol by target
df.boxplot(column='chol', by='target', ax=axes[1, 1])
axes[1, 1].set_xlabel('Target (0=No Disease, 1=Disease)')
axes[1, 1].set_ylabel('Cholesterol')
axes[1, 1].set_title('Cholesterol Distribution by Disease Status', fontsize=14, fontweight='bold')
plt.suptitle('')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ EDA visualizations saved as 'eda_analysis.png'")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n[STEP 3] Data Preprocessing...")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Data preprocessing completed")

# ============================================================================
# 4. MODEL TRAINING - DECISION TREE
# ============================================================================
print("\n[STEP 4] Training Decision Tree Classifier...")

# Initialize Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Train model
dt_model.fit(X_train_scaled, y_train)

# Predictions
dt_train_pred = dt_model.predict(X_train_scaled)
dt_test_pred = dt_model.predict(X_test_scaled)
dt_test_proba = dt_model.predict_proba(X_test_scaled)[:, 1]

print("✓ Decision Tree trained successfully")

# ============================================================================
# 5. MODEL TRAINING - RANDOM FOREST
# ============================================================================
print("\n[STEP 5] Training Random Forest Classifier...")

# Initialize Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# Train model
rf_model.fit(X_train_scaled, y_train)

# Predictions
rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)
rf_test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("✓ Random Forest trained successfully")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n[STEP 6] Model Evaluation...")

def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Comprehensive model evaluation"""
    print(f"\n--- {model_name} Performance ---")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Disease', 'Disease']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'cm': cm
    }

# Evaluate both models
dt_metrics = evaluate_model(y_test, dt_test_pred, dt_test_proba, "Decision Tree")
rf_metrics = evaluate_model(y_test, rf_test_pred, rf_test_proba, "Random Forest")

# ============================================================================
# 7. CROSS-VALIDATION
# ============================================================================
print("\n[STEP 7] Cross-Validation Analysis...")

dt_cv_scores = cross_val_score(dt_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"\nDecision Tree CV Scores: {dt_cv_scores}")
print(f"Decision Tree Mean CV Accuracy: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std():.4f})")

print(f"\nRandom Forest CV Scores: {rf_cv_scores}")
print(f"Random Forest Mean CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

# ============================================================================
# 8. STATISTICAL COMPARISON
# ============================================================================
print("\n[STEP 8] Statistical Comparison of Models...")

# Paired t-test on CV scores
t_stat, p_value = stats.ttest_rel(rf_cv_scores, dt_cv_scores)
print(f"\nPaired t-test results:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Significant difference between models (p < 0.05)")
else:
    print("✗ No significant difference between models (p >= 0.05)")

# McNemar's test
from statsmodels.stats.contingency_tables import mcnemar

# Create contingency table
dt_correct = (dt_test_pred == y_test).astype(int)
rf_correct = (rf_test_pred == y_test).astype(int)

contingency_table = pd.crosstab(dt_correct, rf_correct)
print(f"\nContingency Table (Correct Predictions):")
print(contingency_table)

result = mcnemar(contingency_table, exact=False, correction=True)
print(f"\nMcNemar's Test:")
print(f"Statistic: {result.statistic:.4f}")
print(f"p-value: {result.pvalue:.4f}")

# ============================================================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[STEP 9] Feature Importance Analysis...")

# Decision Tree feature importance
dt_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Decision Tree Feature Importance ---")
print(dt_importance)

# Random Forest feature importance
rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Random Forest Feature Importance ---")
print(rf_importance)

# ============================================================================
# 10. VISUALIZATION
# ============================================================================
print("\n[STEP 10] Creating Comprehensive Visualizations...")

fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison - Metrics
ax1 = plt.subplot(2, 3, 1)
metrics_comparison = pd.DataFrame({
    'Decision Tree': [dt_metrics['accuracy'], dt_metrics['precision'], 
                      dt_metrics['recall'], dt_metrics['f1'], dt_metrics['auc']],
    'Random Forest': [rf_metrics['accuracy'], rf_metrics['precision'], 
                      rf_metrics['recall'], rf_metrics['f1'], rf_metrics['auc']]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])

metrics_comparison.plot(kind='bar', ax=ax1, color=['#3498db', '#2ecc71'])
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_ylim([0, 1])
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# 2. Confusion Matrix - Decision Tree
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(dt_metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')

# 3. Confusion Matrix - Random Forest
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(rf_metrics['cm'], annot=True, fmt='d', cmap='Greens', ax=ax3)
ax3.set_title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

# 4. ROC Curve
ax4 = plt.subplot(2, 3, 4)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_test_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_test_proba)

ax4.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_metrics["auc"]:.3f})', 
         color='#3498db', linewidth=2)
ax4.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_metrics["auc"]:.3f})', 
         color='#2ecc71', linewidth=2)
ax4.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Feature Importance - Random Forest
ax5 = plt.subplot(2, 3, 5)
top_features = rf_importance.head(10)
ax5.barh(top_features['feature'], top_features['importance'], color='#2ecc71')
ax5.set_xlabel('Importance')
ax5.set_title('Top 10 Features - Random Forest', fontsize=14, fontweight='bold')
ax5.invert_yaxis()

# 6. Cross-Validation Scores
ax6 = plt.subplot(2, 3, 6)
cv_comparison = pd.DataFrame({
    'Decision Tree': dt_cv_scores,
    'Random Forest': rf_cv_scores
})
cv_comparison.plot(kind='box', ax=ax6, color={'boxes': '#3498db', 'whiskers': '#3498db'})
ax6.set_ylabel('Accuracy Score')
ax6.set_title('Cross-Validation Scores Distribution', fontsize=14, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Comprehensive visualizations saved as 'model_comparison_analysis.png'")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY & RECOMMENDATIONS")
print("="*70)

print(f"\n📊 Dataset Overview:")
print(f"   - Total Samples: {len(df)}")
print(f"   - Features: {len(X.columns)}")
print(f"   - Disease Prevalence: {df['target'].mean()*100:.2f}%")

print(f"\n🎯 Model Performance (Test Set):")
print(f"\n   Decision Tree:")
print(f"   - Accuracy: {dt_metrics['accuracy']:.4f}")
print(f"   - F1-Score: {dt_metrics['f1']:.4f}")
print(f"   - AUC-ROC: {dt_metrics['auc']:.4f}")

print(f"\n   Random Forest:")
print(f"   - Accuracy: {rf_metrics['accuracy']:.4f}")
print(f"   - F1-Score: {rf_metrics['f1']:.4f}")
print(f"   - AUC-ROC: {rf_metrics['auc']:.4f}")

improvement = ((rf_metrics['accuracy'] - dt_metrics['accuracy']) / dt_metrics['accuracy']) * 100
print(f"\n📈 Random Forest Improvement: {improvement:+.2f}%")

print(f"\n🔬 Statistical Significance:")
print(f"   - p-value (t-test): {p_value:.4f}")
print(f"   - Significant: {'Yes' if p_value < 0.05 else 'No'}")

print(f"\n🏆 Winner: {'Random Forest' if rf_metrics['accuracy'] > dt_metrics['accuracy'] else 'Decision Tree'}")

print("\n💡 Key Insights:")
print("   1. Random Forest generally outperforms single Decision Tree")
print("   2. Ensemble methods reduce overfitting and improve generalization")
print("   3. Feature importance helps identify key risk factors")
print("   4. Cross-validation ensures robust model evaluation")

print("\n" + "="*70)
print("Project completed successfully! ✓")
print("="*70)