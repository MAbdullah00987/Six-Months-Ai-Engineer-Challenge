

#Project 1: Titanic Survival Prediction (Part 1).
#Build a Logistic Regression model to predict survival.
#Generate a Confusion Matrix heatmap using Seaborn.
#Calculate Precision (of all predicted survivors, how many actually survived?) vs. Recall (of all actual survivors, how many did we find?).

#TITANIC SURVIVAL PREDICTION USING LOGISTIC REGRESSION
#Learning Objectives:
# Build an end-to-end ML pipeline
# Understand logistic regression for binary classification
#Master confusion matrix and evaluation metrics
# Learn proper data preprocessing techniques

# ============================================================================
# STEP 1: IMPORT LIBRARIES AND LOAD DATA
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Set visual style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the Titanic dataset from seaborn
print("=" * 70)
print("LOADING TITANIC DATASET")
print("=" * 70)
df = sns.load_dataset('titanic')

# Display first few rows to understand the data
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)
print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumn Information:")
print(df.info())

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Check for missing values
print("\nMissing Values Count:")
print(df.isnull().sum())

print("\nMissing Values Percentage:")
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
print(missing_pct[missing_pct > 0])

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Column Explanations
print("\n" + "=" * 70)
print("WHAT EACH COLUMN MEANS:")
print("=" * 70)
print("""
- survived: 0 = Died, 1 = Survived (TARGET VARIABLE)
- pclass: Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)
- sex: Gender of passenger
- age: Age in years
- sibsp: Number of siblings/spouses aboard
- parch: Number of parents/children aboard
- fare: Passenger fare paid
- embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- class: Same as pclass but categorical
- deck: Deck location (extracted from cabin)
""")

# Survival rate overview
print("\n" + "=" * 70)
print("SURVIVAL STATISTICS")
print("=" * 70)
survival_rate = df['survived'].value_counts()
print(f"\nTotal Passengers: {len(df)}")
print(f"Survived: {survival_rate[1]} ({survival_rate[1]/len(df)*100:.2f}%)")
print(f"Died: {survival_rate[0]} ({survival_rate[0]/len(df)*100:.2f}%)")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Create a copy for preprocessing
data = df.copy()

# Step 3.1: Handle Missing Values
print("\nHandling Missing Values...")

# Age: Fill with median (robust to outliers)
age_median = data['age'].median()
data['age'].fillna(age_median, inplace=True)
print(f"✓ Filled {df['age'].isnull().sum()} missing Age values with median: {age_median:.1f}")

# Embarked: Fill with mode (most common port)
embarked_mode = data['embarked'].mode()[0]
data['embarked'].fillna(embarked_mode, inplace=True)
print(f"✓ Filled {df['embarked'].isnull().sum()} missing Embarked values with mode: {embarked_mode}")

# Deck: Too many missing values (77%), we'll drop this column
print(f"✓ Dropped 'deck' column (77% missing values)")

# Step 3.2: Feature Selection
# Select relevant features for prediction
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
data = data[features + ['survived']]

print(f"\nSelected Features: {features}")
print(f"Target Variable: survived")

# Step 3.3: Encode Categorical Variables
print("\nEncoding Categorical Variables...")

# Sex: Male=1, Female=0
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
print("✓ Encoded 'sex': male=1, female=0")

# Embarked: Create dummy variables (one-hot encoding)
data = pd.get_dummies(data, columns=['embarked'], prefix='embarked', drop_first=True)
print("✓ One-hot encoded 'embarked' (dropped first category to avoid multicollinearity)")

print("\nPreprocessed Dataset Shape:", data.shape)
print("\nFinal Features:")
print(data.columns.tolist())

# Verify no missing values remain
print(f"\nMissing values remaining: {data.isnull().sum().sum()}")

# ============================================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================

print("\n" + "=" * 70)
print("SPLITTING DATA")
print("=" * 70)

# Separate features (X) and target variable (y)
X = data.drop('survived', axis=1)  # All columns except 'survived'
y = data['survived']                # Only the 'survived' column

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split into training (80%) and testing (20%) sets
# random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

print("""
WHY TRAIN-TEST SPLIT?
- Training set: Used to teach the model patterns
- Testing set: Used to evaluate how well model generalizes to unseen data
- This prevents overfitting (memorizing training data)
""")

# ============================================================================
# STEP 5: BUILD LOGISTIC REGRESSION MODEL
# ============================================================================

print("\n" + "=" * 70)
print("BUILDING LOGISTIC REGRESSION MODEL")
print("=" * 70)

# Create the model
# max_iter=1000 ensures convergence
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model on training data
print("\nTraining the model...")
model.fit(X_train, y_train)
print("✓ Model training complete!")

# Make predictions on test data
y_pred = model.predict(X_test)

print(f"\nPredictions made for {len(y_pred)} test samples")
print(f"Sample predictions (first 10): {y_pred[:10]}")
print(f"Actual values (first 10):      {y_test[:10].values}")

# ============================================================================
# STEP 6: GENERATE CONFUSION MATRIX HEATMAP
# ============================================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("""
UNDERSTANDING THE CONFUSION MATRIX:
┌─────────────────────────────────────────┐
│              Predicted                  │
│           Died (0)  Survived (1)        │
├─────────────────────────────────────────┤
│ Actual                                  │
│ Died (0)      TN         FP             │
│ Survived (1)  FN         TP             │
└─────────────────────────────────────────┘

- True Negative (TN): Correctly predicted Died
- False Positive (FP): Predicted Survived but actually Died (Type I Error)
- False Negative (FN): Predicted Died but actually Survived (Type II Error)
- True Positive (TP): Correctly predicted Survived
""")

# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# Create heatmap visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Died (0)', 'Survived (1)'],
            yticklabels=['Died (0)', 'Survived (1)'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Titanic Survival Prediction', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix heatmap saved as 'confusion_matrix.png'")
plt.show()

# ============================================================================
# STEP 7: CALCULATE AND EXPLAIN EVALUATION METRICS
# ============================================================================

print("\n" + "=" * 70)
print("EVALUATION METRICS")
print("=" * 70)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Manual calculation to show the formulas
print("\nMETRIC CALCULATIONS (Manual):")
print("─" * 70)

# Accuracy
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
print(f"\n1. ACCURACY = (TP + TN) / (TP + TN + FP + FN)")
print(f"   Accuracy = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn})")
print(f"   Accuracy = {accuracy_manual:.4f} or {accuracy_manual*100:.2f}%")
print(f"   → Out of all predictions, {accuracy_manual*100:.2f}% were correct")

# Precision
precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
print(f"\n2. PRECISION = TP / (TP + FP)")
print(f"   Precision = {tp} / ({tp} + {fp})")
print(f"   Precision = {precision_manual:.4f} or {precision_manual*100:.2f}%")
print(f"   → Of all passengers we predicted would SURVIVE, {precision_manual*100:.2f}% actually survived")
print(f"   → This tells us how RELIABLE our survival predictions are")

# Recall (Sensitivity)
recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\n3. RECALL (Sensitivity) = TP / (TP + FN)")
print(f"   Recall = {tp} / ({tp} + {fn})")
print(f"   Recall = {recall_manual:.4f} or {recall_manual*100:.2f}%")
print(f"   → Of all passengers who actually SURVIVED, we correctly identified {recall_manual*100:.2f}%")
print(f"   → This tells us how many survivors we FOUND")

# F1-Score
f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0
print(f"\n4. F1-SCORE = 2 × (Precision × Recall) / (Precision + Recall)")
print(f"   F1-Score = 2 × ({precision_manual:.4f} × {recall_manual:.4f}) / ({precision_manual:.4f} + {recall_manual:.4f})")
print(f"   F1-Score = {f1_manual:.4f}")
print(f"   → Harmonic mean of Precision and Recall (balanced measure)")

# Verify with sklearn
print("\n" + "=" * 70)
print("SKLEARN VERIFICATION:")
print("─" * 70)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Detailed classification report
print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# ============================================================================
# STEP 8: INTERPRETATION AND SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)

# Create summary table
summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score': [accuracy, precision, recall, f1],
    'Percentage': [f"{accuracy*100:.2f}%", f"{precision*100:.2f}%", 
                   f"{recall*100:.2f}%", f"{f1*100:.2f}%"],
    'Interpretation': [
        'Overall correctness of predictions',
        'Reliability of survival predictions',
        'Coverage of actual survivors',
        'Balance between Precision & Recall'
    ]
})

print("\n", summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("WHEN TO PRIORITIZE WHICH METRIC?")
print("=" * 70)
print("""
PRECISION Priority (Minimize False Positives):
- Use Case: Life insurance policies, medical treatment approval
- Goal: Be very sure when predicting survival
- Example: "Only give life vests to passengers we're CERTAIN will survive"

RECALL Priority (Minimize False Negatives):
- Use Case: Disease screening, emergency evacuations
- Goal: Don't miss any survivors
- Example: "Better to save some who might not need it than miss actual survivors"

BALANCED (F1-Score):
- Use Case: General classification, when both errors matter equally
- Goal: Good balance between precision and recall
""")

print("\n" + "=" * 70)
print("SUGGESTIONS FOR MODEL IMPROVEMENT")
print("=" * 70)
print("""
1. FEATURE ENGINEERING:
   - Create family_size = sibsp + parch + 1
   - Extract title from name (Mr., Mrs., Miss., Master.)
   - Create age groups (child, young adult, adult, senior)
   - Combine pclass and fare for wealth indicator

2. HANDLE CLASS IMBALANCE:
   - Use class_weight='balanced' in LogisticRegression
   - Try SMOTE (Synthetic Minority Over-sampling)
   - Adjust decision threshold based on business needs

3. FEATURE SCALING:
   - Standardize numerical features (age, fare)
   - This can improve convergence and performance

4. TRY OTHER MODELS:
   - Random Forest Classifier
   - Gradient Boosting (XGBoost, LightGBM)
   - Support Vector Machines (SVM)
   - Neural Networks

5. HYPERPARAMETER TUNING:
   - Use GridSearchCV or RandomizedSearchCV
   - Tune C (regularization parameter)
   - Try different solvers and penalties

6. CROSS-VALIDATION:
   - Use k-fold cross-validation (k=5 or 10)
   - Get more reliable performance estimates
   - Detect overfitting
""")



