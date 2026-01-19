
#Project 3: Customer Purchase Prediction**
# Binary classification problem
# Feature engineering from age and salary
# Implement decision boundary visualization
# Calculate business metrics (profit/loss from predictions)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("CUSTOMER PURCHASE PREDICTION PROJECT")
print("="*60)

# ============================================================================
# STEP 1: GENERATE SYNTHETIC DATASET
# ============================================================================
print("\n1. GENERATING SYNTHETIC CUSTOMER DATA...")
np.random.seed(42)

n_samples = 400

# Generate features
age = np.random.randint(18, 65, n_samples)
salary = np.random.randint(15000, 150000, n_samples)

# Create purchase probability based on age and salary
# Higher salary and certain age ranges are more likely to purchase
age_factor = (age - 40)**2 / 1000  # Peak around 40
salary_factor = (salary - 50000) / 100000
purchase_prob = 1 / (1 + np.exp(-(salary_factor - age_factor + np.random.normal(0, 0.3, n_samples))))

# Generate binary purchase outcome
purchased = (purchase_prob > 0.5).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'EstimatedSalary': salary,
    'Purchased': purchased
})

print(f"Dataset created: {df.shape[0]} samples, {df.shape[1]} features")
print(f"\nPurchase distribution:\n{df['Purchased'].value_counts()}")
print(f"\nDataset Preview:")
print(df.head(10))

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("2. EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\nStatistical Summary:")
print(df.describe())

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution of Age
axes[0, 0].hist(df['Age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].grid(True, alpha=0.3)

# Distribution of Salary
axes[0, 1].hist(df['EstimatedSalary'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Estimated Salary')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Salary Distribution')
axes[0, 1].grid(True, alpha=0.3)

# Purchase by Age
for purchase in [0, 1]:
    subset = df[df['Purchased'] == purchase]
    axes[1, 0].scatter(subset['Age'], subset['EstimatedSalary'], 
                      label=f'Purchased: {purchase}', alpha=0.6, s=50)
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Estimated Salary')
axes[1, 0].set_title('Purchase Pattern: Age vs Salary')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Purchase Distribution
purchase_counts = df['Purchased'].value_counts()
axes[1, 1].bar(['No Purchase', 'Purchase'], purchase_counts.values, 
               color=['lightcoral', 'lightgreen'], edgecolor='black')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Purchase Distribution')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ EDA visualizations saved as 'eda_analysis.png'")
plt.show()

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*60)
print("3. FEATURE ENGINEERING")
print("="*60)

# Create new features
df['Age_Salary_Ratio'] = df['Age'] / (df['EstimatedSalary'] / 1000)
df['Age_Squared'] = df['Age'] ** 2
df['Salary_Log'] = np.log1p(df['EstimatedSalary'])
df['Age_Salary_Interaction'] = df['Age'] * df['EstimatedSalary'] / 1000

# Create age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 65], labels=['Young', 'Middle', 'Senior'])
df['Salary_Group'] = pd.cut(df['EstimatedSalary'], bins=[0, 50000, 100000, 200000], 
                             labels=['Low', 'Medium', 'High'])

print("\nEngineered Features:")
print(df[['Age', 'EstimatedSalary', 'Age_Salary_Ratio', 'Age_Squared', 
          'Salary_Log', 'Age_Salary_Interaction']].head())

print(f"\nAge Groups Distribution:\n{df['Age_Group'].value_counts()}")
print(f"\nSalary Groups Distribution:\n{df['Salary_Group'].value_counts()}")

# ============================================================================
# STEP 4: DATA PREPARATION
# ============================================================================
print("\n" + "="*60)
print("4. DATA PREPARATION")
print("="*60)

# Select features for modeling
feature_cols = ['Age', 'EstimatedSalary', 'Age_Salary_Ratio', 'Age_Squared', 
                'Salary_Log', 'Age_Salary_Interaction']

X = df[feature_cols].values
y = df['Purchased'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                      random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed ✓")

# ============================================================================
# STEP 5: MODEL TRAINING
# ============================================================================
print("\n" + "="*60)
print("5. MODEL TRAINING")
print("="*60)

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    results[name] = {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'predictions': y_pred_test
    }
    
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Testing Accuracy: {test_acc:.4f}")

# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================
print("\n" + "="*60)
print("6. DETAILED MODEL EVALUATION")
print("="*60)

# Use Logistic Regression for detailed analysis
best_model = results['Logistic Regression']['model']
y_pred = results['Logistic Regression']['predictions']

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Purchase', 'Purchase']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'])
plt.title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# ROC Curve
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ ROC curve saved as 'roc_curve.png'")
plt.show()

# ============================================================================
# STEP 7: DECISION BOUNDARY VISUALIZATION
# ============================================================================
print("\n" + "="*60)
print("7. DECISION BOUNDARY VISUALIZATION")
print("="*60)

# Train a simple 2D model for visualization (Age vs Salary only)
X_simple = df[['Age', 'EstimatedSalary']].values
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y, test_size=0.25, random_state=42, stratify=y)

scaler_simple = StandardScaler()
X_train_simple_scaled = scaler_simple.fit_transform(X_train_simple)
X_test_simple_scaled = scaler_simple.transform(X_test_simple)

model_simple = LogisticRegression(random_state=42)
model_simple.fit(X_train_simple_scaled, y_train_simple)

# Create mesh grid
h = 0.02
x_min, x_max = X_train_simple_scaled[:, 0].min() - 1, X_train_simple_scaled[:, 0].max() + 1
y_min, y_max = X_train_simple_scaled[:, 1].min() - 1, X_train_simple_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on mesh
Z = model_simple.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlGn')
plt.colorbar(label='Prediction')

# Plot training points
scatter = plt.scatter(X_train_simple_scaled[:, 0], X_train_simple_scaled[:, 1], 
                     c=y_train_simple, cmap='RdYlGn', edgecolors='black', 
                     s=50, alpha=0.8, label='Training Data')

plt.xlabel('Age (Scaled)', fontsize=12)
plt.ylabel('Estimated Salary (Scaled)', fontsize=12)
plt.title('Decision Boundary: Customer Purchase Prediction', 
          fontsize=14, fontweight='bold')
plt.legend(*scatter.legend_elements(), title="Purchased", loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
print("\n✓ Decision boundary saved as 'decision_boundary.png'")
plt.show()

# ============================================================================
# STEP 8: BUSINESS METRICS CALCULATION
# ============================================================================
print("\n" + "="*60)
print("8. BUSINESS METRICS & PROFIT/LOSS ANALYSIS")
print("="*60)

# Define business costs and revenues
MARKETING_COST = 50  # Cost to market to one customer
PRODUCT_PROFIT = 300  # Profit if customer purchases
OPPORTUNITY_COST = 0  # Cost of not marketing to a potential customer

# Calculate business metrics
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix Breakdown:")
print(f"True Negatives (TN):  {tn} - Correctly predicted no purchase")
print(f"False Positives (FP): {fp} - Incorrectly predicted purchase")
print(f"False Negatives (FN): {fn} - Missed potential customers")
print(f"True Positives (TP):  {tp} - Correctly predicted purchase")

# Calculate costs
cost_marketing = (tp + fp) * MARKETING_COST
revenue_from_purchases = tp * PRODUCT_PROFIT
cost_of_missed_opportunities = fn * (PRODUCT_PROFIT - MARKETING_COST)

net_profit = revenue_from_purchases - cost_marketing

print("\n" + "-"*60)
print("FINANCIAL ANALYSIS")
print("-"*60)
print(f"Marketing Cost (TP + FP):        ${cost_marketing:,.2f}")
print(f"Revenue from Purchases (TP):     ${revenue_from_purchases:,.2f}")
print(f"Missed Opportunity Cost (FN):    ${cost_of_missed_opportunities:,.2f}")
print(f"\nNET PROFIT:                      ${net_profit:,.2f}")

# Compare with naive strategy (market to everyone)
total_customers = len(y_test)
actual_purchases = y_test.sum()
naive_cost = total_customers * MARKETING_COST
naive_revenue = actual_purchases * PRODUCT_PROFIT
naive_profit = naive_revenue - naive_cost

print("\n" + "-"*60)
print("COMPARISON: NAIVE STRATEGY (Market to Everyone)")
print("-"*60)
print(f"Marketing Cost (All):            ${naive_cost:,.2f}")
print(f"Revenue from Purchases:          ${naive_revenue:,.2f}")
print(f"Net Profit:                      ${naive_profit:,.2f}")

profit_improvement = net_profit - naive_profit
improvement_pct = (profit_improvement / abs(naive_profit)) * 100 if naive_profit != 0 else 0

print("\n" + "-"*60)
print("MODEL ADVANTAGE")
print("-"*60)
print(f"Profit Improvement:              ${profit_improvement:,.2f}")
print(f"Improvement Percentage:          {improvement_pct:.2f}%")

# Visualize business metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Cost-Revenue breakdown
categories = ['Marketing\nCost', 'Revenue from\nPurchases', 'Net\nProfit']
values = [cost_marketing, revenue_from_purchases, net_profit]
colors = ['red', 'green', 'blue']

axes[0].bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Amount ($)', fontsize=12)
axes[0].set_title('Model-Based Strategy: Financial Breakdown', 
                  fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Comparison: Model vs Naive
strategies = ['ML Model\nStrategy', 'Naive Strategy\n(Market to All)']
profits = [net_profit, naive_profit]
colors_comparison = ['green' if p > 0 else 'red' for p in profits]

axes[1].bar(strategies, profits, color=colors_comparison, edgecolor='black', alpha=0.7)
axes[1].set_ylabel('Net Profit ($)', fontsize=12)
axes[1].set_title('Strategy Comparison: Net Profit', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig('business_metrics.png', dpi=300, bbox_inches='tight')
print("\n✓ Business metrics visualization saved as 'business_metrics.png'")
plt.show()

# ============================================================================
# STEP 9: STATISTICAL ANALYSIS WITH STATSMODELS
# ============================================================================
print("\n" + "="*60)
print("9. STATISTICAL ANALYSIS (Statsmodels)")
print("="*60)

# Prepare data for statsmodels
X_train_sm = sm.add_constant(X_train_scaled[:, :2])  # Use first 2 features for simplicity
logit_model = sm.Logit(y_train, X_train_sm)
logit_result = logit_model.fit(disp=0)

print("\nLogistic Regression Summary (Statsmodels):")
print(logit_result.summary())

# ============================================================================
# STEP 10: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*60)
print("10. FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature importance from Random Forest
rf_model = results['Random Forest']['model']
feature_importance = rf_model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(importance_df)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], 
         color='teal', edgecolor='black', alpha=0.7)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance for Purchase Prediction', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance visualization saved as 'feature_importance.png'")
plt.show()


print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print(f"\nDataset: {df.shape[0]} customers analyzed")
print(f"Features: {len(feature_cols)} engineered features")
print(f"Models trained: {len(models)}")
print(f"Best model: Logistic Regression (Test Accuracy: {results['Logistic Regression']['test_accuracy']:.4f})")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Net Profit with ML Model: ${net_profit:,.2f}")
print(f"Profit Improvement vs Naive: ${profit_improvement:,.2f} ({improvement_pct:.2f}%)")

