
#roject 2: Diabetes Prediction**
# Use Pima Indians Diabetes dataset
# Complete preprocessing pipeline
# Try both logistic regression and regularized versions
# Create ROC curve and AUC score

"""
Diabetes Prediction using Pima Indians Diabetes Dataset
Complete preprocessing pipeline with Logistic Regression and Regularized versions
Includes ROC curve and AUC score analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, roc_auc_score, accuracy_score)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("DIABETES PREDICTION PROJECT - PIMA INDIANS DATASET")
print("="*80)

# ============================================================================
# 1. LOAD THE DATASET
# ============================================================================
print("\n1. LOADING DATASET...")
print("-"*80)

# Load Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url, names=columns)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n\n2. EXPLORATORY DATA ANALYSIS")
print("-"*80)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df['Outcome'].value_counts())
print(f"\nClass Balance: {df['Outcome'].value_counts(normalize=True)}")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n\n3. DATA PREPROCESSING")
print("-"*80)

# Identify zero values that should be missing
# (Glucose, BloodPressure, SkinThickness, Insulin, BMI cannot be zero)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\nZero values (potentially missing) in each column:")
for col in cols_with_zeros:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} ({zero_count/len(df)*100:.2f}%)")

# Replace zeros with NaN
df_clean = df.copy()
df_clean[cols_with_zeros] = df_clean[cols_with_zeros].replace(0, np.nan)

print("\nMissing values after replacing zeros with NaN:")
print(df_clean.isnull().sum())

# Impute missing values with median (robust to outliers)
for col in cols_with_zeros:
    median_val = df_clean[col].median()
    df_clean[col].fillna(median_val, inplace=True)
    print(f"Imputed {col} with median: {median_val:.2f}")

# Check for outliers using IQR method
print("\n\nOutlier Detection (IQR method):")
for col in df_clean.columns[:-1]:  # Exclude 'Outcome'
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    print(f"{col}: {outliers} outliers")

# ============================================================================
# 4. FEATURE ENGINEERING & ANALYSIS
# ============================================================================
print("\n\n4. FEATURE ANALYSIS")
print("-"*80)

# Correlation analysis
print("\nCorrelation with Outcome:")
correlations = df_clean.corr()['Outcome'].sort_values(ascending=False)
print(correlations)

# Multicollinearity check using VIF
X_vif = df_clean.drop('Outcome', axis=1)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
print("\nVariance Inflation Factor (VIF) - Multicollinearity Check:")
print(vif_data)
print("(VIF > 10 indicates high multicollinearity)")

# ============================================================================
# 5. PREPARE DATA FOR MODELING
# ============================================================================
print("\n\n5. PREPARING DATA FOR MODELING")
print("-"*80)

# Split features and target
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))

# Feature scaling (important for regularized models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled using StandardScaler")

# ============================================================================
# 6. MODEL TRAINING
# ============================================================================
print("\n\n6. MODEL TRAINING")
print("-"*80)

# Dictionary to store models and results
models = {}
results = {}

# 6.1 Standard Logistic Regression
print("\n6.1 Standard Logistic Regression")
print("-"*40)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
models['Standard LR'] = lr

# Predictions
y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Evaluation
acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
results['Standard LR'] = {'accuracy': acc_lr, 'auc': auc_lr}

print(f"Accuracy: {acc_lr:.4f}")
print(f"AUC Score: {auc_lr:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# 6.2 L1 Regularized (Lasso) Logistic Regression
print("\n6.2 L1 Regularized (Lasso) Logistic Regression")
print("-"*40)
lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)

# Grid search for best C parameter
param_grid_l1 = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_l1 = GridSearchCV(lr_l1, param_grid_l1, cv=5, scoring='roc_auc')
grid_l1.fit(X_train_scaled, y_train)
models['L1 (Lasso) LR'] = grid_l1.best_estimator_

print(f"Best C parameter: {grid_l1.best_params_['C']}")

# Predictions
y_pred_l1 = grid_l1.predict(X_test_scaled)
y_pred_proba_l1 = grid_l1.predict_proba(X_test_scaled)[:, 1]

# Evaluation
acc_l1 = accuracy_score(y_test, y_pred_l1)
auc_l1 = roc_auc_score(y_test, y_pred_proba_l1)
results['L1 (Lasso) LR'] = {'accuracy': acc_l1, 'auc': auc_l1}

print(f"Accuracy: {acc_l1:.4f}")
print(f"AUC Score: {auc_l1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_l1))

# 6.3 L2 Regularized (Ridge) Logistic Regression
print("\n6.3 L2 Regularized (Ridge) Logistic Regression")
print("-"*40)
lr_l2 = LogisticRegression(penalty='l2', random_state=42, max_iter=1000)

# Grid search for best C parameter
param_grid_l2 = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_l2 = GridSearchCV(lr_l2, param_grid_l2, cv=5, scoring='roc_auc')
grid_l2.fit(X_train_scaled, y_train)
models['L2 (Ridge) LR'] = grid_l2.best_estimator_

print(f"Best C parameter: {grid_l2.best_params_['C']}")

# Predictions
y_pred_l2 = grid_l2.predict(X_test_scaled)
y_pred_proba_l2 = grid_l2.predict_proba(X_test_scaled)[:, 1]

# Evaluation
acc_l2 = accuracy_score(y_test, y_pred_l2)
auc_l2 = roc_auc_score(y_test, y_pred_proba_l2)
results['L2 (Ridge) LR'] = {'accuracy': acc_l2, 'auc': auc_l2}

print(f"Accuracy: {acc_l2:.4f}")
print(f"AUC Score: {auc_l2:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_l2))

# 6.4 ElasticNet Logistic Regression
print("\n6.4 ElasticNet Logistic Regression (L1 + L2)")
print("-"*40)
lr_elastic = LogisticRegression(penalty='elasticnet', solver='saga', 
                                l1_ratio=0.5, random_state=42, max_iter=1000)

# Grid search for best C parameter
param_grid_elastic = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_elastic = GridSearchCV(lr_elastic, param_grid_elastic, cv=5, scoring='roc_auc')
grid_elastic.fit(X_train_scaled, y_train)
models['ElasticNet LR'] = grid_elastic.best_estimator_

print(f"Best C parameter: {grid_elastic.best_params_['C']}")

# Predictions
y_pred_elastic = grid_elastic.predict(X_test_scaled)
y_pred_proba_elastic = grid_elastic.predict_proba(X_test_scaled)[:, 1]

# Evaluation
acc_elastic = accuracy_score(y_test, y_pred_elastic)
auc_elastic = roc_auc_score(y_test, y_pred_proba_elastic)
results['ElasticNet LR'] = {'accuracy': acc_elastic, 'auc': auc_elastic}

print(f"Accuracy: {acc_elastic:.4f}")
print(f"AUC Score: {auc_elastic:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_elastic))

# ============================================================================
# 7. MODEL COMPARISON
# ============================================================================
print("\n\n7. MODEL COMPARISON")
print("-"*80)

results_df = pd.DataFrame(results).T
print("\nModel Performance Summary:")
print(results_df)

print(f"\nBest Model (by AUC): {results_df['auc'].idxmax()}")
print(f"Best AUC Score: {results_df['auc'].max():.4f}")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n\n8. CREATING VISUALIZATIONS...")
print("-"*80)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 8.1 Correlation Heatmap
plt.subplot(3, 3, 1)
sns.heatmap(df_clean.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# 8.2 Class Distribution
plt.subplot(3, 3, 2)
df_clean['Outcome'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Outcome (0=No Diabetes, 1=Diabetes)')
plt.ylabel('Count')
plt.xticks(rotation=0)

# 8.3 Feature Distributions
plt.subplot(3, 3, 3)
df_clean[df_clean.columns[:-1]].hist(bins=20, figsize=(16, 12), layout=(3, 3))
plt.suptitle('Feature Distributions', y=1.02, fontsize=14, fontweight='bold')

# 8.4 ROC Curves for all models
plt.subplot(3, 3, 4)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_l1, tpr_l1, _ = roc_curve(y_test, y_pred_proba_l1)
fpr_l2, tpr_l2, _ = roc_curve(y_test, y_pred_proba_l2)
fpr_elastic, tpr_elastic, _ = roc_curve(y_test, y_pred_proba_elastic)

plt.plot(fpr_lr, tpr_lr, label=f'Standard LR (AUC={auc_lr:.3f})', linewidth=2)
plt.plot(fpr_l1, tpr_l1, label=f'L1 Lasso (AUC={auc_l1:.3f})', linewidth=2)
plt.plot(fpr_l2, tpr_l2, label=f'L2 Ridge (AUC={auc_l2:.3f})', linewidth=2)
plt.plot(fpr_elastic, tpr_elastic, label=f'ElasticNet (AUC={auc_elastic:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison', fontsize=12, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# 8.5 Confusion Matrix - Best Model
plt.subplot(3, 3, 5)
best_model_name = results_df['auc'].idxmax()
if best_model_name == 'Standard LR':
    y_pred_best = y_pred_lr
elif best_model_name == 'L1 (Lasso) LR':
    y_pred_best = y_pred_l1
elif best_model_name == 'L2 (Ridge) LR':
    y_pred_best = y_pred_l2
else:
    y_pred_best = y_pred_elastic

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 8.6 Feature Importance (Coefficients)
plt.subplot(3, 3, 6)
best_model = models[best_model_name]
coef = best_model.coef_[0]
features = X.columns
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coef})
coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title(f'Feature Importance - {best_model_name}', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 8.7 Model Comparison - Accuracy
plt.subplot(3, 3, 7)
results_df['accuracy'].plot(kind='bar', color='steelblue')
plt.title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim([results_df['accuracy'].min() - 0.05, results_df['accuracy'].max() + 0.05])
plt.grid(True, alpha=0.3)

# 8.8 Model Comparison - AUC
plt.subplot(3, 3, 8)
results_df['auc'].plot(kind='bar', color='coral')
plt.title('Model AUC Score Comparison', fontsize=12, fontweight='bold')
plt.ylabel('AUC Score')
plt.xticks(rotation=45, ha='right')
plt.ylim([results_df['auc'].min() - 0.05, results_df['auc'].max() + 0.05])
plt.grid(True, alpha=0.3)

# 8.9 Box plots for features by outcome
plt.subplot(3, 3, 9)
df_melt = df_clean.melt(id_vars='Outcome', var_name='Feature', value_name='Value')
glucose_data = df_clean[['Glucose', 'Outcome']]
sns.boxplot(x='Outcome', y='Glucose', data=glucose_data, palette='Set2')
plt.title('Glucose Levels by Outcome', fontsize=12, fontweight='bold')
plt.xlabel('Outcome (0=No Diabetes, 1=Diabetes)')

plt.tight_layout()
plt.savefig('diabetes_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'diabetes_analysis.png'")
plt.show()

# ============================================================================
# 9. STATSMODELS ANALYSIS
# ============================================================================
print("\n\n9. STATSMODELS LOGISTIC REGRESSION ANALYSIS")
print("-"*80)

# Add constant for statsmodels
X_train_sm = sm.add_constant(X_train_scaled)
X_test_sm = sm.add_constant(X_test_scaled)

# Fit logistic regression using statsmodels
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

print(result.summary())

# Statistical significance of features
print("\n\nFeature Significance (p-values):")
p_values = pd.DataFrame({
    'Feature': ['Intercept'] + list(X.columns),
    'P-value': result.pvalues,
    'Significant': result.pvalues < 0.05
})
print(p_values)


print("\n\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\nDataset: Pima Indians Diabetes")
print(f"Total samples: {len(df_clean)}")
print(f"Features: {len(X.columns)}")
print(f"Train/Test split: 80/20")
print(f"\nBest Model: {best_model_name}")
print(f"Best AUC Score: {results_df['auc'].max():.4f}")
print(f"Best Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")
print("\nAll models trained successfully!")
print("="*80)
