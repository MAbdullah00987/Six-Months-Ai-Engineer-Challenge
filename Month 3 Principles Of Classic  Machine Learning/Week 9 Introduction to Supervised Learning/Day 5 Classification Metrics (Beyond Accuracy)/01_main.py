

#Day 5: Classification Metrics (Beyond Accuracy)

#Objective: Understand why "Accuracy" is a bad metric for imbalanced data (e.g., detecting rare diseases).
#Concept: Confusion Matrix (TP, TN, FP, FN), Precision, Recall, F1-Score, ROC Curve, AUC.
 
#Learning Objectives:
# Learn evaluation metrics (accuracy, precision, recall, F1)
# Understand confusion matrix
# Master train-test split
# Learn feature scaling importance

#Study Materials:
# Géron Chapter 3: "Classification" (pages 85-112)
# Andrew Ng Course 1, Week 3: "Evaluating a Learning Algorithm"
# Scikit-Learn docs: Model Evaluation


#Project 1: Titanic Survival Prediction (Part 1).
#Build a Logistic Regression model to predict survival.
#Generate a Confusion Matrix heatmap using Seaborn.
#Calculate Precision (of all predicted survivors, how many actually survived?) vs. Recall (of all actual survivors, how many did we find?).

#Project 2: Titanic Survival Prediction**
# Handle missing data
# Encode categorical features
# Scale numerical features
# Evaluate with multiple metrics

#Project 3: Feature Scaling Impact**
# Show how StandardScaler affects logistic regression
# Compare scaled vs unscaled performance
# Visualize the difference

#Topic 1: Why Accuracy Fails - Complete Implementation


"""
TOPIC 1: Why Accuracy Fails - The Imbalanced Data Problem
Complete implementation with NumPy, Pandas, Matplotlib, Seaborn, SymPy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import symbols, simplify, latex
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("TOPIC 1: WHY ACCURACY IS MISLEADING FOR IMBALANCED DATA")
# PART 1: Create Real-World Imbalanced Dataset (Rare Disease Detection)

print("\n PART 1: Creating Imbalanced Medical Dataset")
print("-" * 80)

# Set random seed for reproducibility
np.random.seed(42)

# Create imbalanced dataset: 99% healthy, 1% disease
n_samples = 10000
disease_rate = 0.01  # 1% have disease
n_disease = int(n_samples * disease_rate)
n_healthy = n_samples - n_disease

print(f"Total Samples: {n_samples}")
print(f"Disease Cases: {n_disease} ({disease_rate*100}%)")
print(f"Healthy Cases: {n_healthy} ({(1-disease_rate)*100}%)")

# Generate features (e.g., blood test results)
# Disease patients have slightly different feature distributions
X_disease = np.random.randn(n_disease, 5) + np.array([1.5, 2.0, 1.0, 1.8, 1.2])
X_healthy = np.random.randn(n_healthy, 5)

# Combine features and labels
X = np.vstack([X_disease, X_healthy])
y = np.concatenate([np.ones(n_disease), np.zeros(n_healthy)])

# Create DataFrame for better visualization
feature_names = ['Blood_Pressure', 'Glucose', 'Cholesterol', 'Heart_Rate', 'BMI']
df = pd.DataFrame(X, columns=feature_names)
df['Disease'] = y

print("\n Dataset Info:")
print(df.head(10))
print("\n Class Distribution:")
print(df['Disease'].value_counts())


# PART 2: Visualize Data Imbalance
print("\n PART 2: Visualizing Data Imbalance")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Pie Chart
colors = ['#FF6B6B', '#4ECDC4']
disease_counts = df['Disease'].value_counts()
axes[0, 0].pie(disease_counts, labels=['Healthy (0)', 'Disease (1)'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
axes[0, 0].set_title('Class Distribution - Pie Chart', fontsize=14, fontweight='bold')

# 2. Bar Plot
axes[0, 1].bar(['Healthy', 'Disease'], disease_counts.values, color=colors)
axes[0, 1].set_ylabel('Count', fontsize=12)
axes[0, 1].set_title('Class Distribution - Bar Chart', fontsize=14, fontweight='bold')
for i, v in enumerate(disease_counts.values):
    axes[0, 1].text(i, v + 100, str(v), ha='center', fontweight='bold')

# 3. Imbalance Ratio Visualization
imbalance_ratio = n_healthy / n_disease
axes[1, 0].barh(['Imbalance\nRatio'], [imbalance_ratio], color='#95E1D3')
axes[1, 0].set_xlabel('Ratio (Healthy : Disease)', fontsize=12)
axes[1, 0].set_title(f'Imbalance Ratio: {imbalance_ratio:.0f}:1', 
                     fontsize=14, fontweight='bold')
axes[1, 0].text(imbalance_ratio/2, 0, f'{imbalance_ratio:.0f}:1', 
                ha='center', va='center', fontsize=16, fontweight='bold')

# 4. Feature Distribution Comparison
feature_to_plot = 'Glucose'
axes[1, 1].hist(df[df['Disease']==0][feature_to_plot], bins=50, 
                alpha=0.7, label='Healthy', color=colors[0])
axes[1, 1].hist(df[df['Disease']==1][feature_to_plot], bins=20, 
                alpha=0.7, label='Disease', color=colors[1])
axes[1, 1].set_xlabel(feature_to_plot, fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title(f'{feature_to_plot} Distribution by Class', 
                     fontsize=14, fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('01_data_imbalance.png', dpi=300, bbox_inches='tight')
print("Saved: 01_data_imbalance.png")
plt.show()


# PART 3: The "Dumb Classifier" Problem

print("\n PART 3: The Dumb Classifier That Achieves 99% Accuracy")
print("-" * 80)

# Create a "dumb" classifier that always predicts "Healthy" (0)
y_pred_dumb = np.zeros(len(y))

# Calculate accuracy
accuracy_dumb = accuracy_score(y, y_pred_dumb)
print(f"\n Dumb Classifier Strategy: Always predict 'Healthy'")
print(f"Accuracy: {accuracy_dumb:.4f} ({accuracy_dumb*100:.2f}%)")
print("\n THIS LOOKS AMAZING... BUT IS COMPLETELY USELESS!")

# Let's see what it actually does
correct_predictions = np.sum(y_pred_dumb == y)
incorrect_predictions = np.sum(y_pred_dumb != y)

print(f"\n Breakdown:")
print(f"   Correct predictions: {correct_predictions} ({correct_predictions/len(y)*100:.2f}%)")
print(f"  Incorrect predictions: {incorrect_predictions} ({incorrect_predictions/len(y)*100:.2f}%)")
print(f"\n  CRITICAL PROBLEM:")
print(f"  • Missed ALL {n_disease} disease cases!")
print(f"  • 100% of disease patients told they're healthy!")
print(f"  • In healthcare, this is CATASTROPHIC!")


# PART 4: Mathematical Proof with SymPy

print("\n PART 4: Mathematical Proof - Why Accuracy is Misleading")
print("-" * 80)

# Define symbolic variables
TP, TN, FP, FN = symbols('TP TN FP FN', positive=True, real=True)

# Accuracy formula
accuracy_formula = (TP + TN) / (TP + TN + FP + FN)

print("\n📐 Accuracy Formula:")
print(f"Accuracy = (TP + TN) / (TP + TN + FP + FN)")

# For our dumb classifier
print("\n Dumb Classifier Values:")
print(f"  TP (True Positives): 0 (predicted no one has disease)")
print(f"  TN (True Negatives): {n_healthy} (correctly said healthy people are healthy)")
print(f"  FP (False Positives): 0 (didn't predict anyone has disease)")
print(f"  FN (False Negatives): {n_disease} (missed all disease cases!)")

# Substitute values
accuracy_calculated = accuracy_formula.subs({
    TP: 0,
    TN: n_healthy,
    FP: 0,
    FN: n_disease
})

print(f"\nAccuracy = (0 + {n_healthy}) / (0 + {n_healthy} + 0 + {n_disease})")
print(f"Accuracy = {n_healthy} / {n_samples}")
print(f"Accuracy = {float(accuracy_calculated):.4f} or {float(accuracy_calculated)*100:.2f}%")

print("\n Why This Happens:")
print(f"  • The dataset is {(1-disease_rate)*100}% healthy")
print(f"  • Predicting 'healthy' for everyone gives {(1-disease_rate)*100}% accuracy")
print(f"  • Accuracy is dominated by the majority class!")


# PART 5: Compare Multiple Classifiers
print("\n PART 5: Comparing Different Classifiers")
print("-" * 80)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Initialize classifiers
classifiers = {
    'Always Healthy': DummyClassifier(strategy='most_frequent'),
    'Random Guess': DummyClassifier(strategy='uniform'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = []

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Count predictions
    n_pred_disease = np.sum(y_pred == 1)
    n_pred_healthy = np.sum(y_pred == 0)
    
    results.append({
        'Classifier': name,
        'Accuracy': acc,
        'Predicted Disease': n_pred_disease,
        'Predicted Healthy': n_pred_healthy
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Predicted {n_pred_disease} disease cases")
    print(f"  Predicted {n_pred_healthy} healthy cases")

# Create comparison DataFrame
results_df = pd.DataFrame(results)
print("\n Summary Table:")
print(results_df.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
axes[0].bar(results_df['Classifier'], results_df['Accuracy'], color=['#FF6B6B', '#FFA07A', '#4ECDC4'])
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Accuracy Comparison - MISLEADING!', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 1])
for i, v in enumerate(results_df['Accuracy']):
    axes[0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')

# Predictions breakdown
x = np.arange(len(results_df))
width = 0.35
axes[1].bar(x - width/2, results_df['Predicted Disease'], width, 
            label='Predicted Disease', color='#FF6B6B')
axes[1].bar(x + width/2, results_df['Predicted Healthy'], width, 
            label='Predicted Healthy', color='#4ECDC4')
axes[1].set_ylabel('Number of Predictions', fontsize=12)
axes[1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(results_df['Classifier'])
axes[1].legend()

plt.tight_layout()
plt.savefig('02_accuracy_misleading.png', dpi=300, bbox_inches='tight')
print(" Saved: 02_accuracy_misleading.png")
plt.show()


print("""
1️  ACCURACY IS MISLEADING for imbalanced datasets
    • A classifier can have 99% accuracy but be completely useless
    • It's dominated by the majority class

2⃣  THE PROBLEM:
    • In rare disease detection, we CARE MOST about finding the 1%
    • Missing disease cases (False Negatives) is catastrophic
    • Accuracy doesn't tell us about this!

3️  WHAT WE NEED:
    • Metrics that focus on the minority class
    • Confusion Matrix to see TP, TN, FP, FN
    • Precision: Of those we said have disease, how many actually do?
    • Recall: Of those who have disease, how many did we catch?
    • F1-Score: Balance between Precision and Recall

4️  REAL-WORLD IMPACT:
    • Healthcare: Missing cancer diagnosis
    • Fraud Detection: Missing fraudulent transactions
    • Spam Detection: Missing actual spam

  NEXT: We'll learn Confusion Matrix and better metrics!
""")

