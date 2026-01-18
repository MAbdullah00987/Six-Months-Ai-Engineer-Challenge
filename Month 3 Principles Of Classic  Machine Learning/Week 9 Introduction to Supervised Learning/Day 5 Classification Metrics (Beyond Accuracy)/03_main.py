
#Topic 3: Precision - Complete Implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import symbols, simplify, latex, Rational
from sklearn.metrics import precision_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

print("="*80)
print("TOPIC 3: PRECISION - WHEN FALSE POSITIVES MATTER")
print("="*80)


# PART 1: Mathematical Foundation with SymPy

print("\n PART 1: Mathematical Foundation with SymPy")
print("-" * 80)

# Define symbolic variables
TP, TN, FP, FN = symbols('TP TN FP FN', positive=True, real=True)

# Precision formula
precision_formula = TP / (TP + FP)

print("\n Precision Formula (Symbolic):")
print(f"Precision = TP / (TP + FP)")
print(f"LaTeX: {latex(precision_formula)}")

print("\n What Precision Answers:")
print("   'Of all the cases I predicted as POSITIVE, how many were actually positive?'")
print("\n   Precision = Correct Positive Predictions / All Positive Predictions")
print("   Precision = TP / (TP + FP)")

# Example calculation with SymPy
print("\n Example Calculation:")
tp_val, fp_val = 80, 20
precision_calc = precision_formula.subs({TP: tp_val, FP: fp_val})
print(f"   If TP = {tp_val}, FP = {fp_val}")
print(f"   Precision = {tp_val} / ({tp_val} + {fp_val}) = {precision_calc} = {float(precision_calc):.3f}")

# Why precision matters
print("\n Why Precision is Important:")
print("""
HIGH Precision is critical when False Positives are COSTLY:

1️ Spam Email Detection:
   • FP = Important email marked as spam (TERRIBLE!)
   • High precision means: "If I mark it as spam, I'm very confident"
   • Better to let some spam through (FN) than block important email (FP)

2️  Cancer Screening (Follow-up tests):
   • FP = Healthy person told they might have cancer (causes anxiety, expensive tests)
   • High precision = "If I say cancer, I'm very confident"
   • Trade-off: Might miss some cases (FN), but avoid false alarms

3️  Criminal Justice:
   • FP = Innocent person convicted (CATASTROPHIC!)
   • High precision = "Only convict if very certain"
   • "Better 10 guilty go free than 1 innocent suffer"

4️  Credit Card Fraud:
   • FP = Legitimate transaction blocked (customer annoyed)
   • High precision = "Only block if really suspicious"
   • Balance: Don't annoy customers unnecessarily
""")


# PART 2: Implementation from Scratch with NumPy

print("\n PART 2: Precision Implementation from Scratch")
print("-" * 80)

def precision_from_scratch(y_true, y_pred):
    """
    Calculate precision using pure NumPy
    
    Precision = TP / (TP + FP)
    
    Parameters:
    -----------
    y_true : array-like, true labels (0 or 1)
    y_pred : array-like, predicted labels (0 or 1)
    
    Returns:
    --------
    precision : float, precision score
    components : dict, TP and FP counts
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate True Positives and False Positives
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    
    # Calculate precision
    # Handle division by zero
    if (TP + FP) == 0:
        precision = 0.0  # or np.nan, depending on use case
    else:
        precision = TP / (TP + FP)
    
    return precision, {'TP': TP, 'FP': FP, 'Total_Predicted_Positive': TP + FP}

# Create sample predictions
np.random.seed(42)
n_samples = 1000

# Generate imbalanced dataset
y_true = np.concatenate([np.ones(100), np.zeros(900)])
# Simulated predictions (not perfect)
y_pred = y_true.copy()
# Add some errors
error_indices = np.random.choice(n_samples, size=150, replace=False)
y_pred[error_indices] = 1 - y_pred[error_indices]

# Calculate precision from scratch
prec_scratch, components = precision_from_scratch(y_true, y_pred)

print(f" From Scratch Calculation:")
print(f"   True Positives (TP):  {components['TP']}")
print(f"   False Positives (FP): {components['FP']}")
print(f"   Total Predicted Positive: {components['Total_Predicted_Positive']}")
print(f"   Precision = {components['TP']} / {components['Total_Predicted_Positive']} = {prec_scratch:.4f}")

# Verify with sklearn
prec_sklearn = precision_score(y_true, y_pred)
print(f"\n sklearn precision_score: {prec_sklearn:.4f}")
print(f"   Match: {np.isclose(prec_scratch, prec_sklearn)}")

# Get full confusion matrix for context
cm = confusion_matrix(y_true, y_pred)
TN, FP_cm = cm[0]
FN, TP_cm = cm[1]

print(f"\n Full Confusion Matrix Context:")
print(f"   TP = {TP_cm}, TN = {TN}, FP = {FP_cm}, FN = {FN}")
print(f"   Interpretation: Of {TP_cm + FP_cm} positive predictions, {TP_cm} were correct")


# PART 3: Understanding Precision with Different Scenarios

print("\n PART 3: Precision in Different Scenarios")
print("-" * 80)

scenarios = [
    ("Conservative Model", [90, 10]),  # [TP, FP] - very careful
    ("Balanced Model", [70, 30]),       # [TP, FP] - balanced
    ("Aggressive Model", [95, 80]),     # [TP, FP] - predicts positive often
]

print("\nComparing Three Models:\n")
scenario_results = []

for name, (tp, fp) in scenarios:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    scenario_results.append({
        'Model': name,
        'TP': tp,
        'FP': fp,
        'Predicted Positive': tp + fp,
        'Precision': precision
    })
    
    print(f"{name}:")
    print(f"   TP = {tp}, FP = {fp}")
    print(f"   Predicted {tp + fp} as positive, {tp} were correct")
    print(f"   Precision = {tp}/{tp + fp} = {precision:.3f} ({precision*100:.1f}%)")
    print()

scenario_df = pd.DataFrame(scenario_results)
print(" Summary Table:")
print(scenario_df.to_string(index=False))

# Visualize scenarios
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart comparison
x = np.arange(len(scenario_df))
width = 0.35

axes[0].bar(x - width/2, scenario_df['TP'], width, label='True Positives', 
           color='#2ecc71', alpha=0.8)
axes[0].bar(x + width/2, scenario_df['FP'], width, label='False Positives', 
           color='#e74c3c', alpha=0.8)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('TP vs FP Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(scenario_df['Model'], rotation=15, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Precision comparison
bars = axes[1].bar(scenario_df['Model'], scenario_df['Precision'], 
                   color=['#3498db', '#9b59b6', '#e67e22'], alpha=0.8)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target: 0.8')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=15)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, scenario_df['Precision'])):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('06_precision_scenarios.png', dpi=300, bbox_inches='tight')
print("\n Saved: 06_precision_scenarios.png")
plt.show()


# PART 4: Precision-Threshold Analysis

print("\n PART 4: How Precision Changes with Decision Threshold")
print("-" * 80)

# Create realistic dataset
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n) * 0.5) > 0
y = y.astype(int)

# Make imbalanced
y[np.random.choice(n, size=int(n*0.85), replace=False)] = 0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Get probability predictions
y_proba = model.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = np.linspace(0, 1, 100)
precisions = []
n_positives = []
tps = []
fps = []

for threshold in thresholds:
    y_pred_threshold = (y_proba >= threshold).astype(int)
    
    # Calculate precision
    tp = np.sum((y_test == 1) & (y_pred_threshold == 1))
    fp = np.sum((y_test == 0) & (y_pred_threshold == 1))
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0  # No predictions, technically perfect precision
    
    precisions.append(precision)
    n_positives.append(tp + fp)
    tps.append(tp)
    fps.append(fp)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Precision vs Threshold
axes[0].plot(thresholds, precisions, linewidth=2.5, color='#3498db')
axes[0].fill_between(thresholds, precisions, alpha=0.3, color='#3498db')
axes[0].set_xlabel('Decision Threshold', fontsize=12)
axes[0].set_ylabel('Precision', fontsize=12)
axes[0].set_title('Precision vs Threshold', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='High Precision (0.9)')
axes[0].axvline(x=0.5, color='g', linestyle='--', alpha=0.5, label='Default (0.5)')
axes[0].legend()

# Number of Positive Predictions vs Threshold
axes[1].plot(thresholds, n_positives, linewidth=2.5, color='#e74c3c')
axes[1].fill_between(thresholds, n_positives, alpha=0.3, color='#e74c3c')
axes[1].set_xlabel('Decision Threshold', fontsize=12)
axes[1].set_ylabel('# Positive Predictions', fontsize=12)
axes[1].set_title('Predictions vs Threshold', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

# TP and FP vs Threshold
axes[2].plot(thresholds, tps, linewidth=2.5, color='#2ecc71', label='True Positives')
axes[2].plot(thresholds, fps, linewidth=2.5, color='#e67e22', label='False Positives')
axes[2].fill_between(thresholds, tps, alpha=0.2, color='#2ecc71')
axes[2].fill_between(thresholds, fps, alpha=0.2, color='#e67e22')
axes[2].set_xlabel('Decision Threshold', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_title('TP and FP vs Threshold', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('07_precision_threshold.png', dpi=300, bbox_inches='tight')
print(" Saved: 07_precision_threshold.png")
plt.show()

print("\n Key Observations:")
print("   • Higher threshold → Higher precision (fewer FP)")
print("   • Higher threshold → Fewer positive predictions")
print("   • Higher threshold → Might miss true positives (lower recall)")
print("   • Trade-off: Precision ↑ as we become more conservative")


# PART 5: Real-World Application Example

print("\n PART 5: Real-World Example - Email Spam Detection")
print("-" * 80)

print("""
Scenario: Email Spam Classifier

Two approaches:
1. HIGH PRECISION: Only mark as spam if VERY confident
2. LOW PRECISION: Mark as spam more liberally

Let's compare:
""")

# Simulate email classification
np.random.seed(42)
n_emails = 1000
actual_spam = np.random.rand(n_emails) < 0.3  # 30% spam

# High precision classifier (conservative)
high_prec_pred = actual_spam.copy()
# Only mark as spam if very certain (miss some spam, rarely wrong)
high_prec_pred[np.random.choice(np.where(actual_spam)[0], size=40, replace=False)] = False
# Very few false positives
high_prec_pred[np.random.choice(np.where(~actual_spam)[0], size=5, replace=False)] = True

# Low precision classifier (aggressive)
low_prec_pred = actual_spam.copy()
# Mark more emails as spam (catch more spam, but more mistakes)
low_prec_pred[np.random.choice(np.where(~actual_spam)[0], size=100, replace=False)] = True

# Calculate metrics
hp_precision, hp_comp = precision_from_scratch(actual_spam, high_prec_pred)
lp_precision, lp_comp = precision_from_scratch(actual_spam, low_prec_pred)

print(f"\n High Precision Classifier (Conservative):")
print(f"   TP = {hp_comp['TP']}, FP = {hp_comp['FP']}")
print(f"   Precision = {hp_precision:.3f}")
print(f"   → Out of {hp_comp['Total_Predicted_Positive']} emails marked spam,")
print(f"     {hp_comp['TP']} actually were spam ({hp_precision*100:.1f}% correct)")
print(f"    Rarely blocks important emails!")
print(f"    Some spam gets through")

print(f"\n Low Precision Classifier (Aggressive):")
print(f"   TP = {lp_comp['TP']}, FP = {lp_comp['FP']}")
print(f"   Precision = {lp_precision:.3f}")
print(f"   → Out of {lp_comp['Total_Predicted_Positive']} emails marked spam,")
print(f"     {lp_comp['TP']} actually were spam ({lp_precision*100:.1f}% correct)")
print(f"    Catches more spam")
print(f"    Blocks many important emails!")


# CONCLUSION

print("\n" + "="*80)
print(" KEY TAKEAWAYS - PRECISION")
print("="*80)
print("""
1️  PRECISION FORMULA:
    Precision = TP / (TP + FP)
    "Of my positive predictions, how many were correct?"

2️  WHEN TO PRIORITIZE PRECISION:
     False Positives are COSTLY
     Better to miss some positives than have false alarms
    Examples: Spam detection, fraud alerts, job candidate screening

3️  HIGH PRECISION MEANS:
     When you predict positive, you're usually right
     Very few false alarms
     But might miss some true positives (low recall trade-off)

4️  PRECISION TRADE-OFF:
    • Higher threshold → Higher precision, Lower recall
    • Lower threshold → Lower precision, Higher recall
    • Must balance based on application!

5️  PRECISION ALONE IS NOT ENOUGH:
    • A model that predicts positive only once can have perfect precision
    • Need to consider Recall too (how many positives we actually catch)
    • This leads us to F1-Score (harmonic mean of Precision & Recall)

  NEXT: Recall (Sensitivity) - When missing positives is critical!
""")

print("\n Topic 3 Complete! Ready for Topic 4: Recall")
print("="*80)