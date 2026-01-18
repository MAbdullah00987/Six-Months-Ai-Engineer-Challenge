

#Topic 2: Confusion Matrix - Complete Implementation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

print("="*80)
print("TOPIC 2: CONFUSION MATRIX - THE FOUNDATION")
print("="*80)

# ============================================================================
# PART 1: Understanding Confusion Matrix Components
# ============================================================================
print("\n PART 1: What is a Confusion Matrix?")
print("-" * 80)

print("""
A Confusion Matrix shows the performance of a classification model:

                    Predicted
                 Negative | Positive
              -------------------------
Actual Neg  |    TN      |    FP     |  ← False Positive (Type I Error)
Actual Pos  |    FN      |    TP     |  ← False Negative (Type II Error)
              -------------------------
                 ↑             ↑
          False Negative  True Positive

Key Components:
• TP (True Positive): Correctly predicted positive (GOOD!)
• TN (True Negative): Correctly predicted negative (GOOD!)
• FP (False Positive): Incorrectly predicted positive (BAD - Type I Error)
• FN (False Negative): Incorrectly predicted negative (BAD - Type II Error)

Real-World Examples:
├─ TP: Patient has disease, we detected it 
├─ TN: Patient is healthy, we said healthy 
├─ FP: Patient is healthy, we said disease  (False alarm)
└─ FN: Patient has disease, we missed it  (CRITICAL!)
""")

# ============================================================================
# PART 2: Build Confusion Matrix from Scratch
# ============================================================================
print("\n🔨 PART 2: Building Confusion Matrix from Scratch with NumPy")
print("-" * 80)

def confusion_matrix_from_scratch(y_true, y_pred):
    """
    Calculate confusion matrix using pure NumPy
    
    Parameters:
    -----------
    y_true : array-like, true labels
    y_pred : array-like, predicted labels
    
    Returns:
    --------
    cm : 2D array [[TN, FP], [FN, TP]]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate each component
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # Create matrix in standard format
    cm = np.array([[TN, FP],
                   [FN, TP]])
    
    return cm, {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

# Create sample data
np.random.seed(42)
n_samples = 1000
disease_rate = 0.1

# Generate imbalanced dataset
n_disease = int(n_samples * disease_rate)
n_healthy = n_samples - n_disease

X_disease = np.random.randn(n_disease, 3) + np.array([2, 1.5, 1])
X_healthy = np.random.randn(n_healthy, 3)

X = np.vstack([X_disease, X_healthy])
y = np.concatenate([np.ones(n_disease), np.zeros(n_healthy)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train a simple model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate confusion matrix from scratch
cm_scratch, components = confusion_matrix_from_scratch(y_test, y_pred)

print(" Confusion Matrix (from scratch):")
print(cm_scratch)
print(f"\n Components:")
print(f"  True Positives (TP):  {components['TP']:4d}  ← Disease correctly detected")
print(f"  True Negatives (TN):  {components['TN']:4d}  ← Healthy correctly identified")
print(f"  False Positives (FP): {components['FP']:4d}  ← False alarm (Type I Error)")
print(f"  False Negatives (FN): {components['FN']:4d}  ← Missed disease (Type II Error)")

# Verify with sklearn
cm_sklearn = confusion_matrix(y_test, y_pred)
print(f"\n Verification with sklearn:")
print(cm_sklearn)
print(f"Match: {np.array_equal(cm_scratch, cm_sklearn)}")

# ============================================================================
# PART 3: Visualize Confusion Matrix
# ============================================================================
print("\n PART 3: Visualizing Confusion Matrix")
print("-" * 80)

def plot_confusion_matrix(cm, components, title="Confusion Matrix"):
    """Create beautiful confusion matrix visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='black',
                ax=ax, annot_kws={'size': 16, 'weight': 'bold'})
    
    # Labels
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Tick labels
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=12)
    ax.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=12)
    
    # Add component labels
    TN, FP = cm[0]
    FN, TP = cm[1]
    
    ax.text(0.5, 0.2, 'TN', ha='center', va='center', 
            fontsize=20, color='darkblue', weight='bold')
    ax.text(1.5, 0.2, 'FP\n(Type I)', ha='center', va='center', 
            fontsize=20, color='darkred', weight='bold')
    ax.text(0.5, 1.2, 'FN\n(Type II)', ha='center', va='center', 
            fontsize=20, color='darkred', weight='bold')
    ax.text(1.5, 1.2, 'TP', ha='center', va='center', 
            fontsize=20, color='darkgreen', weight='bold')
    
    # Add percentage annotations
    total = cm.sum()
    ax.text(0.5, 0.8, f'{TN/total*100:.1f}%', ha='center', va='center', 
            fontsize=12, color='gray')
    ax.text(1.5, 0.8, f'{FP/total*100:.1f}%', ha='center', va='center', 
            fontsize=12, color='gray')
    ax.text(0.5, 1.8, f'{FN/total*100:.1f}%', ha='center', va='center', 
            fontsize=12, color='gray')
    ax.text(1.5, 1.8, f'{TP/total*100:.1f}%', ha='center', va='center', 
            fontsize=12, color='gray')
    
    plt.tight_layout()
    return fig

fig = plot_confusion_matrix(cm_scratch, components, 
                            "Confusion Matrix - Logistic Regression")
plt.savefig('03_confusion_matrix_single.png', dpi=300, bbox_inches='tight')
print(" Saved: 03_confusion_matrix_single.png")
plt.show()

# ============================================================================
# PART 4: Interpret Each Quadrant with Real Examples
# ============================================================================
print("\n💡 PART 4: Real-World Interpretation")
print("-" * 80)

total_predictions = cm_scratch.sum()
TN, FP = cm_scratch[0]
FN, TP = cm_scratch[1]

print(f"\n Medical Diagnosis Interpretation (Total: {total_predictions} patients):\n")

print(f" TRUE NEGATIVES (TN = {TN}):")
print(f"   • {TN} patients were healthy AND we correctly said healthy")
print(f"   • Good! No unnecessary treatment/stress")
print(f"   • {TN/total_predictions*100:.1f}% of all predictions\n")

print(f" FALSE POSITIVES (FP = {FP}) - Type I Error:")
print(f"   • {FP} patients were healthy BUT we said they have disease")
print(f"   • Problem: Unnecessary treatment, anxiety, medical costs")
print(f"   • {FP/total_predictions*100:.1f}% of all predictions\n")

print(f" FALSE NEGATIVES (FN = {FN}) - Type II Error:")
print(f"   • {FN} patients had disease BUT we missed it")
print(f"   • CRITICAL: Disease progresses untreated!")
print(f"   • {FN/total_predictions*100:.1f}% of all predictions\n")

print(f" TRUE POSITIVES (TP = {TP}):")
print(f"   • {TP} patients had disease AND we detected it")
print(f"   • Excellent! Early treatment possible")
print(f"   • {TP/total_predictions*100:.1f}% of all predictions\n")

# Calculate some basic insights
total_actual_disease = TP + FN
total_actual_healthy = TN + FP
total_predicted_disease = TP + FP
total_predicted_healthy = TN + FN

print(f" Summary:")
print(f"  • Actual disease cases: {total_actual_disease}")
print(f"  • Actual healthy cases: {total_actual_healthy}")
print(f"  • Predicted disease: {total_predicted_disease}")
print(f"  • Predicted healthy: {total_predicted_healthy}")

# ============================================================================
# PART 5: Compare Multiple Classifiers
# ============================================================================
print("\n PART 5: Comparing Multiple Classifiers")
print("-" * 80)

classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, clf) in enumerate(classifiers.items()):
    # Train and predict
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                ax=axes[idx], cbar=False,
                linewidths=2, linecolor='black',
                annot_kws={'size': 14, 'weight': 'bold'})
    
    axes[idx].set_title(name, fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=11)
    axes[idx].set_ylabel('Actual' if idx == 0 else '', fontsize=11)
    axes[idx].set_xticklabels(['Neg', 'Pos'], fontsize=10)
    axes[idx].set_yticklabels(['Neg', 'Pos'], fontsize=10)
    
    # Add labels
    TN, FP = cm[0]
    FN, TP = cm[1]
    
    axes[idx].text(0.5, 0.25, 'TN', ha='center', va='center', 
                  fontsize=12, color='blue', weight='bold')
    axes[idx].text(1.5, 0.25, 'FP', ha='center', va='center', 
                  fontsize=12, color='red', weight='bold')
    axes[idx].text(0.5, 1.25, 'FN', ha='center', va='center', 
                  fontsize=12, color='red', weight='bold')
    axes[idx].text(1.5, 1.25, 'TP', ha='center', va='center', 
                  fontsize=12, color='green', weight='bold')
    
    print(f"\n{name}:")
    print(f"  TP: {TP:3d}  |  TN: {TN:3d}")
    print(f"  FP: {FP:3d}  |  FN: {FN:3d}")

plt.tight_layout()
plt.savefig('04_confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
print("\n Saved: 04_confusion_matrix_comparison.png")
plt.show()

# ============================================================================
# PART 6: Interactive Analysis
# ============================================================================
print("\n PART 6: Detailed Component Analysis")
print("-" * 80)

# Create comprehensive comparison table
comparison_data = []

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    TN, FP = cm[0]
    FN, TP = cm[1]
    
    comparison_data.append({
        'Classifier': name,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Total Correct': TP + TN,
        'Total Wrong': FP + FN,
        'Accuracy': (TP + TN) / (TP + TN + FP + FN)
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n📊 Confusion Matrix Components Comparison:")
print(comparison_df.to_string(index=False))

# Visualize component comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

components_to_plot = ['TP', 'TN', 'FP', 'FN']
colors_map = {'TP': '#2ecc71', 'TN': '#3498db', 'FP': '#e74c3c', 'FN': '#e67e22'}
titles = {
    'TP': 'True Positives (Good!)', 
    'TN': 'True Negatives (Good!)',
    'FP': 'False Positives (Type I Error)', 
    'FN': 'False Negatives (Type II Error)'
}

for idx, component in enumerate(components_to_plot):
    row = idx // 2
    col = idx % 2
    
    axes[row, col].bar(comparison_df['Classifier'], 
                       comparison_df[component], 
                       color=colors_map[component])
    axes[row, col].set_title(titles[component], fontsize=13, fontweight='bold')
    axes[row, col].set_ylabel('Count', fontsize=11)
    axes[row, col].tick_params(axis='x', rotation=15)
    
    # Add value labels
    for i, v in enumerate(comparison_df[component]):
        axes[row, col].text(i, v + 1, str(v), ha='center', 
                           fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('05_components_comparison.png', dpi=300, bbox_inches='tight')
print("\n Saved: 05_components_comparison.png")
plt.show()

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "="*80)
print(" KEY TAKEAWAYS - CONFUSION MATRIX")
print("="*80)
print("""
1️  CONFUSION MATRIX STRUCTURE:
                 Predicted
             Negative | Positive
         -----------------------
    Neg  |    TN     |    FP    |  ← Type I Error (False Alarm)
    Pos  |    FN     |    TP    |  ← Type II Error (Miss)
         -----------------------

2️  WHAT EACH COMPONENT MEANS:
    • TP (True Positive):  Correctly detected disease
    • TN (True Negative):  Correctly identified healthy
    • FP (False Positive):  False alarm - said disease but healthy
    • FN (False Negative):  Missed disease - said healthy but disease

3️  WHY CONFUSION MATRIX > ACCURACY:
    • Shows WHERE the model makes mistakes
    • Reveals if model is biased towards one class
    • Helps us understand Type I vs Type II errors
    • Foundation for better metrics (Precision, Recall, F1)

4️  CRITICAL QUESTION:
    "Which error is worse in your application?"
    • Medical: FN worse (missing disease is critical)
    • Spam: FP worse (blocking important email)
    • Fraud: Balance (catch fraud, don't block customers)

  NEXT: Precision & Recall - Focusing on what matters!
""")

print("\n Topic 2 Complete! Ready for Topic 3: Precision")
