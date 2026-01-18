

#Project 2: Titanic Survival Prediction**
# Handle missing data
# Encode categorical features
# Scale numerical features
# Evaluate with multiple metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("TITANIC SURVIVAL PREDICTION PROJECT")
print("=" * 80)


# 1. LOAD DATA

print("\n1. LOADING DATA...")

# For demonstration, creating sample Titanic dataset
# In practice, you would load from: pd.read_csv('titanic.csv')
np.random.seed(42)
n_samples = 891

data = pd.DataFrame({
    'PassengerId': range(1, n_samples + 1),
    'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
    'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
    'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
    'Age': np.random.normal(29.7, 14.5, n_samples),
    'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.68, 0.23, 0.05, 0.02, 0.01, 0.01]),
    'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.015, 0.003, 0.001, 0.001]),
    'Fare': np.random.exponential(32, n_samples),
    'Embarked': np.random.choice(['C', 'Q', 'S', np.nan], n_samples, p=[0.19, 0.09, 0.72, 0.002])
})

# Introduce missing values to simulate real data
data.loc[np.random.choice(data.index, 177, replace=False), 'Age'] = np.nan
data.loc[np.random.choice(data.index, 1, replace=False), 'Fare'] = np.nan

print(f"Dataset loaded successfully!")
print(f"Shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())

# 2. EXPLORATORY DATA ANALYSIS

print("\n" + "=" * 80)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

print("\nMissing Values:")
missing = data.isnull().sum()
missing_pct = 100 * missing / len(data)
missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Count'] > 0])

print("\nSurvival Rate:")
survival_rate = data['Survived'].value_counts(normalize=True)
print(survival_rate)

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Survival Distribution
axes[0, 0].pie(data['Survived'].value_counts(), labels=['Died', 'Survived'], 
               autopct='%1.1f%%', startangle=90, colors=['#ff6b6b', '#4ecdc4'])
axes[0, 0].set_title('Survival Distribution')

# Survival by Gender
pd.crosstab(data['Sex'], data['Survived'], normalize='index').plot(
    kind='bar', ax=axes[0, 1], color=['#ff6b6b', '#4ecdc4'])
axes[0, 1].set_title('Survival Rate by Gender')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Proportion')
axes[0, 1].legend(['Died', 'Survived'])

# Survival by Class
pd.crosstab(data['Pclass'], data['Survived'], normalize='index').plot(
    kind='bar', ax=axes[0, 2], color=['#ff6b6b', '#4ecdc4'])
axes[0, 2].set_title('Survival Rate by Class')
axes[0, 2].set_xlabel('Passenger Class')
axes[0, 2].set_ylabel('Proportion')
axes[0, 2].legend(['Died', 'Survived'])

# Age Distribution
data['Age'].hist(bins=30, ax=axes[1, 0], color='skyblue', edgecolor='black')
axes[1, 0].set_title('Age Distribution')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Frequency')

# Fare Distribution
data['Fare'].hist(bins=30, ax=axes[1, 1], color='lightcoral', edgecolor='black')
axes[1, 1].set_title('Fare Distribution')
axes[1, 1].set_xlabel('Fare')
axes[1, 1].set_ylabel('Frequency')

# Correlation Heatmap
numeric_cols = data.select_dtypes(include=[np.number]).columns
correlation = data[numeric_cols].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 2])
axes[1, 2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nEDA visualizations saved as 'eda_visualizations.png'")


# 3. DATA PREPROCESSING

print("3. DATA PREPROCESSING")


# Create a copy for preprocessing
df = data.copy()

# 3.1 Handle Missing Data
print("\n3.1 Handling Missing Data...")

# Age: Fill with median grouped by Pclass and Sex
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)

# Fare: Fill with median
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Embarked: Fill with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("Missing values after imputation:")
print(df.isnull().sum().sum(), "total missing values")

# 3.2 Feature Engineering
print("\n3.2 Feature Engineering...")

# Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Is Alone
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Age Groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# Fare Groups
df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])

print("New features created: FamilySize, IsAlone, AgeGroup, FareGroup")

# 3.3 Encode Categorical Features
print("\n3.3 Encoding Categorical Features...")

# Binary encoding for Sex
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding for Embarked
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)

# One-hot encoding for AgeGroup
age_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
df = pd.concat([df, age_dummies], axis=1)

# One-hot encoding for FareGroup
fare_dummies = pd.get_dummies(df['FareGroup'], prefix='FareGroup')
df = pd.concat([df, fare_dummies], axis=1)

# Drop original categorical columns
df.drop(['Embarked', 'AgeGroup', 'FareGroup', 'PassengerId'], axis=1, inplace=True)

print("Categorical features encoded successfully")
print(f"Dataset shape after encoding: {df.shape}")

# 3.4 Prepare Features and Target
print("\n3.4 Preparing Features and Target...")

X = df.drop('Survived', axis=1)
y = df['Survived']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")

# 3.5 Train-Test Split
print("\n3.5 Splitting Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 3.6 Feature Scaling
print("\n3.6 Scaling Features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")


# 4. MODEL TRAINING AND EVALUATION


print("4. MODEL TRAINING AND EVALUATION")


# Dictionary to store results
results = {}

# 4.1 Logistic Regression
print("\n4.1 Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
}

# 4.2 Decision Tree
print("4.2 Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_pred_proba_dt = dt.predict_proba(X_test)[:, 1]

results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'precision': precision_score(y_test, y_pred_dt),
    'recall': recall_score(y_test, y_pred_dt),
    'f1': f1_score(y_test, y_pred_dt),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_dt)
}

# 4.3 Random Forest
print("4.3 Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
}

# 4.4 Gradient Boosting
print("4.4 Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]

results['Gradient Boosting'] = {
    'accuracy': accuracy_score(y_test, y_pred_gb),
    'precision': precision_score(y_test, y_pred_gb),
    'recall': recall_score(y_test, y_pred_gb),
    'f1': f1_score(y_test, y_pred_gb),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_gb)
}

# 4.5 Support Vector Machine
print("4.5 Support Vector Machine...")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
y_pred_proba_svm = svm.predict_proba(X_test_scaled)[:, 1]

results['SVM'] = {
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'precision': precision_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm),
    'f1': f1_score(y_test, y_pred_svm),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_svm)
}


# 5. RESULTS COMPARISON

print("5. MODEL COMPARISON")


results_df = pd.DataFrame(results).T
print("\nModel Performance Metrics:")
print(results_df.round(4))

# Find best model
best_model = results_df['accuracy'].idxmax()
print(f"\nBest Model (by Accuracy): {best_model}")
print(f"Accuracy: {results_df.loc[best_model, 'accuracy']:.4f}")


# 6. VISUALIZE RESULTS


print("6. VISUALIZING RESULTS")


fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 6.1 Metrics Comparison
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
for i, metric in enumerate(metrics):
    row = i // 3
    col = i % 3
    results_df[metric].plot(kind='bar', ax=axes[row, col], color='skyblue')
    axes[row, col].set_title(f'{metric.upper()} Comparison')
    axes[row, col].set_ylabel('Score')
    axes[row, col].set_ylim([0, 1])
    axes[row, col].grid(axis='y', alpha=0.3)
    for tick in axes[row, col].get_xticklabels():
        tick.set_rotation(45)

# 6.2 ROC Curves
axes[1, 2].plot([0, 1], [0, 1], 'k--', label='Random')
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
axes[1, 2].plot(fpr_lr, tpr_lr, label=f'LR (AUC={results["Logistic Regression"]["roc_auc"]:.3f})')
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
axes[1, 2].plot(fpr_rf, tpr_rf, label=f'RF (AUC={results["Random Forest"]["roc_auc"]:.3f})')
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)
axes[1, 2].plot(fpr_gb, tpr_gb, label=f'GB (AUC={results["Gradient Boosting"]["roc_auc"]:.3f})')
axes[1, 2].set_xlabel('False Positive Rate')
axes[1, 2].set_ylabel('True Positive Rate')
axes[1, 2].set_title('ROC Curves')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Model comparison visualization saved as 'model_comparison.png'")


# 7. FEATURE IMPORTANCE (Random Forest)

print("7. FEATURE IMPORTANCE ANALYSIS")


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'].head(10), feature_importance['importance'].head(10))
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("Feature importance visualization saved as 'feature_importance.png'")


# 8. CONFUSION MATRIX FOR BEST MODEL


print("8. CONFUSION MATRIX")

# Using Random Forest as an example
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=['Died', 'Survived']))


print("PROJECT COMPLETED SUCCESSFULLY!")
print("\nGenerated Files:")
print("1. eda_visualizations.png")
print("2. model_comparison.png")
print("3. feature_importance.png")
print("4. confusion_matrix.png")