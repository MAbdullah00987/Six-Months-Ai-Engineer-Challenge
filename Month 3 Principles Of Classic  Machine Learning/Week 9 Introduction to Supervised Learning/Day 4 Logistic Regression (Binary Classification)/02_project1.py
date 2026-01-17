
#Project 1: Diabetes Prediction.
# Dataset: Pima Indians Diabetes Dataset (available on Kaggle/Scikit-learn).
# Predict if a patient has diabetes based on health metrics.
# Analyze the model.coef_ to see which feature (e.g., Glucose, BMI) is the strongest predictor.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("DIABETES PREDICTION PROJECT - PIMA INDIANS DATASET")
print("=" * 70)


# STEP 1: LOAD AND EXPLORE THE DATA

print("\n[STEP 1] Loading Dataset...")

# Load the Pima Indians Diabetes Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url, names=column_names)

print(f"\nDataset Shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1] - 1}")

print("\n" + "=" * 70)
print("First 5 rows of the dataset:")
print("=" * 70)
print(df.head())

print("\n" + "=" * 70)
print("Dataset Information:")
print("=" * 70)
print(df.info())

print("\n" + "=" * 70)
print("Statistical Summary:")
print("=" * 70)
print(df.describe())

print("\n" + "=" * 70)
print("Missing Values:")
print("=" * 70)
print(df.isnull().sum())

print("\n" + "=" * 70)
print("Class Distribution:")
print("=" * 70)
print(df['Outcome'].value_counts())
print(f"\nPercentage with Diabetes: {df['Outcome'].mean() * 100:.2f}%")
print(f"Percentage without Diabetes: {(1 - df['Outcome'].mean()) * 100:.2f}%")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 2] Data Preprocessing...")
print("=" * 70)

# Handle zero values (zeros are impossible for some features)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\nZero values in key columns (medically impossible):")
for col in zero_cols:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} zeros ({zero_count/len(df)*100:.2f}%)")

# Replace zeros with NaN and then with median
df_clean = df.copy()
for col in zero_cols:
    df_clean[col] = df_clean[col].replace(0, np.nan)
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

print("\nZeros replaced with median values.")

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 3] Exploratory Data Analysis...")
print("=" * 70)

# Correlation analysis
print("\nCorrelation with Outcome (Diabetes):")
correlations = df_clean.corr()['Outcome'].sort_values(ascending=False)
print(correlations)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Correlation heatmap
sns.heatmap(df_clean.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[0, 0])
axes[0, 0].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

# 2. Distribution of Outcome
outcome_counts = df_clean['Outcome'].value_counts()
axes[0, 1].bar(['No Diabetes', 'Diabetes'], outcome_counts.values, 
               color=['green', 'red'], alpha=0.7)
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Distribution of Diabetes Outcome', fontsize=14, fontweight='bold')
for i, v in enumerate(outcome_counts.values):
    axes[0, 1].text(i, v + 10, str(v), ha='center', fontweight='bold')

# 3. Feature importance (correlation bar plot)
feature_corr = correlations.drop('Outcome').abs().sort_values(ascending=True)
axes[1, 0].barh(range(len(feature_corr)), feature_corr.values, color='skyblue')
axes[1, 0].set_yticks(range(len(feature_corr)))
axes[1, 0].set_yticklabels(feature_corr.index)
axes[1, 0].set_xlabel('Absolute Correlation with Outcome')
axes[1, 0].set_title('Feature Correlation with Diabetes', fontsize=14, fontweight='bold')

# 4. Box plot for top features
top_features = correlations.drop('Outcome').abs().nlargest(4).index
df_melted = df_clean.melt(id_vars='Outcome', value_vars=top_features)
sns.boxplot(data=df_melted, x='variable', y='value', hue='Outcome', ax=axes[1, 1])
axes[1, 1].set_title('Top 4 Features Distribution by Outcome', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Feature')
axes[1, 1].set_ylabel('Value')
axes[1, 1].legend(title='Diabetes', labels=['No', 'Yes'])

plt.tight_layout()
plt.savefig('diabetes_eda.png', dpi=300, bbox_inches='tight')
print("\n✓ EDA visualizations saved as 'diabetes_eda.png'")

# Distribution plots for all features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(df_clean.columns[:-1]):
    df_clean[df_clean['Outcome'] == 0][col].hist(bins=30, alpha=0.5, 
                                                   label='No Diabetes', 
                                                   color='green', ax=axes[idx])
    df_clean[df_clean['Outcome'] == 1][col].hist(bins=30, alpha=0.5, 
                                                   label='Diabetes', 
                                                   color='red', ax=axes[idx])
    axes[idx].set_title(col, fontweight='bold')
    axes[idx].legend()
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Feature distributions saved as 'feature_distributions.png'")

# ============================================================================
# STEP 4: MULTICOLLINEARITY CHECK (VIF)
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 4] Checking for Multicollinearity (VIF)...")
print("=" * 70)

X_vif = df_clean.drop('Outcome', axis=1)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False)

print("\nVariance Inflation Factor (VIF):")
print(vif_data)
print("\nNote: VIF > 10 indicates high multicollinearity")

# ============================================================================
# STEP 5: PREPARE DATA FOR MODELING
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 5] Preparing Data for Modeling...")
print("=" * 70)

# Split features and target
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Features standardized (mean=0, std=1)")

# ============================================================================
# STEP 6: BUILD LOGISTIC REGRESSION MODEL
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 6] Building Logistic Regression Model...")
print("=" * 70)

# Train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("\n✓ Model trained successfully!")

# ============================================================================
# STEP 7: ANALYZE MODEL COEFFICIENTS
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 7] ANALYZING MODEL COEFFICIENTS (model.coef_)")
print("=" * 70)

# Get coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
})
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

print("\nLogistic Regression Coefficients (Sorted by Importance):")
print(coefficients)

print(f"\n{'='*70}")
print("STRONGEST PREDICTOR:")
print(f"{'='*70}")
strongest_feature = coefficients.iloc[0]['Feature']
strongest_coef = coefficients.iloc[0]['Coefficient']
print(f"Feature: {strongest_feature}")
print(f"Coefficient: {strongest_coef:.4f}")
print(f"\nInterpretation: A 1 standard deviation increase in {strongest_feature}")
print(f"is associated with a {'positive' if strongest_coef > 0 else 'negative'} change")
print(f"in the log-odds of having diabetes.")

# Visualize coefficients
plt.figure(figsize=(12, 6))
colors = ['red' if x > 0 else 'blue' for x in coefficients['Coefficient']]
plt.barh(range(len(coefficients)), coefficients['Coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(coefficients)), coefficients['Feature'])
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Logistic Regression Coefficients (Feature Importance)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('model_coefficients.png', dpi=300, bbox_inches='tight')
print("\n✓ Coefficient plot saved as 'model_coefficients.png'")

# ============================================================================
# STEP 8: STATSMODELS ANALYSIS (for p-values and statistical significance)
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 8] Statistical Analysis with Statsmodels...")
print("=" * 70)

# Add constant for statsmodels
X_train_sm = sm.add_constant(X_train_scaled)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

print("\nStatsmodels Logistic Regression Summary:")
print(result.summary())

# Create a detailed coefficient summary
coef_summary = pd.DataFrame({
    'Feature': ['Intercept'] + list(X.columns),
    'Coefficient': result.params.values,
    'Std Error': result.bse.values,
    'z-score': result.tvalues.values,
    'P-value': result.pvalues.values,
    'Significant': ['Yes' if p < 0.05 else 'No' for p in result.pvalues.values]
})

print("\n" + "=" * 70)
print("DETAILED COEFFICIENT ANALYSIS:")
print("=" * 70)
print(coef_summary.to_string(index=False))

significant_features = coef_summary[coef_summary['Significant'] == 'Yes']['Feature'].tolist()
print(f"\nStatistically Significant Features (p < 0.05): {', '.join(significant_features[1:])}")

# ============================================================================
# STEP 9: MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 9] Model Evaluation...")
print("=" * 70)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Accuracy
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"\nTraining Accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
print(f"Testing Accuracy: {test_score:.4f} ({test_score*100:.2f}%)")

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Classification Report
print("\n" + "=" * 70)
print("Classification Report:")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ============================================================================
# STEP 10: VISUALIZATION OF RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 10] Creating Evaluation Visualizations...")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Evaluation plots saved as 'model_evaluation.png'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"""
✓ Dataset: Pima Indians Diabetes Dataset
✓ Total Samples: {df.shape[0]}
✓ Features: {df.shape[1] - 1}
✓ Target: Diabetes (Yes/No)

MODEL PERFORMANCE:
- Training Accuracy: {train_score*100:.2f}%
- Testing Accuracy: {test_score*100:.2f}%
- ROC-AUC Score: {roc_auc:.2f}
- Mean CV Score: {cv_scores.mean()*100:.2f}%

STRONGEST PREDICTOR: {strongest_feature}
- Coefficient: {strongest_coef:.4f}
- Interpretation: Higher {strongest_feature} values are associated with 
  {'increased' if strongest_coef > 0 else 'decreased'} probability of diabetes

TOP 3 PREDICTORS:
""")

for i in range(min(3, len(coefficients))):
    feat = coefficients.iloc[i]['Feature']
    coef = coefficients.iloc[i]['Coefficient']
    print(f"  {i+1}. {feat}: {coef:.4f}")

print("All visualizations have been saved successfully!")
print("Files created:")
print("  - diabetes_eda.png")
print("  - feature_distributions.png")
print("  - model_coefficients.png")
print("  - model_evaluation.png")
