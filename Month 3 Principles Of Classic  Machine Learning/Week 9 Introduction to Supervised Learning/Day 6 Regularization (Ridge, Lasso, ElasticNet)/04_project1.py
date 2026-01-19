
#Task: Project - Regularization Exploration.
#Use the Housing dataset again. Add meaningful noise or extra useless features.
#Train Lasso regression and watch it force coefficients of useless features to zero. This is automatic feature selection!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*70)
print("REGULARIZATION EXPLORATION: Lasso Feature Selection")
print("="*70)

# 1. Load the California Housing Dataset
print("\n[1] Loading California Housing Dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseValue')

print(f"Original Dataset Shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"\nFirst few rows:")
print(X.head())

# 2. Add Meaningful Noise Features
print("\n[2] Adding Noise and Useless Features...")
np.random.seed(42)

# Add random noise features (completely useless)
X['random_noise_1'] = np.random.randn(len(X))
X['random_noise_2'] = np.random.uniform(0, 100, len(X))
X['random_noise_3'] = np.random.randn(len(X)) * 50

# Add features that are just scaled versions (redundant)
X['noise_uniform'] = np.random.uniform(-10, 10, len(X))
X['noise_normal'] = np.random.normal(0, 5, len(X))

# Add some correlated noise (slightly related but not useful)
X['weak_noise'] = X['MedInc'] * 0.01 + np.random.randn(len(X)) * 10

print(f"Dataset Shape After Adding Noise: {X.shape}")
print(f"New Features Added: 6 useless features")
print(f"\nAll Features: {list(X.columns)}")

# 3. Split the data
print("\n[3] Splitting Data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardize features
print("\n[4] Standardizing Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier interpretation
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# 5. Train Multiple Models for Comparison
print("\n[5] Training Models...")

# Linear Regression (No Regularization)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Ridge Regression (L2 Regularization)
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# Lasso Regression (L1 Regularization)
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

# 6. Evaluate Models
print("\n[6] Model Performance Comparison:")
print("-" * 70)

models = {
    'Linear Regression': (lr, y_pred_lr),
    'Ridge Regression': (ridge, y_pred_ridge),
    'Lasso Regression': (lasso, y_pred_lasso)
}

for name, (model, y_pred) in models.items():
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")

# 7. Analyze Coefficients
print("\n[7] Coefficient Analysis:")
print("=" * 70)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Linear_Reg': lr.coef_,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_
})

coef_df = coef_df.sort_values('Lasso', key=abs, ascending=False)
print("\nCoefficients (sorted by absolute Lasso value):")
print(coef_df.to_string(index=False))

# Count zero coefficients in Lasso
zero_coefs = (coef_df['Lasso'] == 0).sum()
print(f"\n✓ Lasso forced {zero_coefs} coefficients to EXACTLY ZERO!")
print(f"✓ This is automatic feature selection in action!")

# 8. Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Coefficient Comparison
ax1 = plt.subplot(2, 2, 1)
x_pos = np.arange(len(coef_df))
width = 0.25

ax1.bar(x_pos - width, coef_df['Linear_Reg'], width, label='Linear Reg', alpha=0.8)
ax1.bar(x_pos, coef_df['Ridge'], width, label='Ridge', alpha=0.8)
ax1.bar(x_pos + width, coef_df['Lasso'], width, label='Lasso', alpha=0.8)

ax1.set_xlabel('Features', fontsize=11)
ax1.set_ylabel('Coefficient Value', fontsize=11)
ax1.set_title('Coefficient Comparison: Linear vs Ridge vs Lasso', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(coef_df['Feature'], rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.grid(True, alpha=0.3)

# Plot 2: Lasso Coefficients Only (Highlight Zero Coefficients)
ax2 = plt.subplot(2, 2, 2)
colors = ['red' if c == 0 else 'green' for c in coef_df['Lasso']]
bars = ax2.barh(coef_df['Feature'], coef_df['Lasso'], color=colors, alpha=0.7)

ax2.set_xlabel('Coefficient Value', fontsize=11)
ax2.set_ylabel('Features', fontsize=11)
ax2.set_title(f'Lasso Coefficients (Red = Zero, {zero_coefs} features eliminated)', 
              fontsize=13, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Absolute Coefficient Values
ax3 = plt.subplot(2, 2, 3)
coef_abs = pd.DataFrame({
    'Feature': X.columns,
    'Linear_Reg': np.abs(lr.coef_),
    'Ridge': np.abs(ridge.coef_),
    'Lasso': np.abs(lasso.coef_)
})
coef_abs = coef_abs.sort_values('Lasso', ascending=False)

x_pos = np.arange(len(coef_abs))
ax3.bar(x_pos - width, coef_abs['Linear_Reg'], width, label='Linear Reg', alpha=0.8)
ax3.bar(x_pos, coef_abs['Ridge'], width, label='Ridge', alpha=0.8)
ax3.bar(x_pos + width, coef_abs['Lasso'], width, label='Lasso', alpha=0.8)

ax3.set_xlabel('Features', fontsize=11)
ax3.set_ylabel('Absolute Coefficient Value', fontsize=11)
ax3.set_title('Absolute Coefficient Values (Feature Importance)', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(coef_abs['Feature'], rotation=45, ha='right', fontsize=9)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Prediction Comparison
ax4 = plt.subplot(2, 2, 4)
ax4.scatter(y_test, y_pred_lr, alpha=0.3, label='Linear Reg', s=20)
ax4.scatter(y_test, y_pred_ridge, alpha=0.3, label='Ridge', s=20)
ax4.scatter(y_test, y_pred_lasso, alpha=0.5, label='Lasso', s=20)

min_val = min(y_test.min(), y_pred_lasso.min())
max_val = max(y_test.max(), y_pred_lasso.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

ax4.set_xlabel('Actual Values', fontsize=11)
ax4.set_ylabel('Predicted Values', fontsize=11)
ax4.set_title('Predictions vs Actual Values', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. Lasso Path Analysis (Different Alpha Values)
print("\n[8] Lasso Path: Testing Different Alpha Values...")
print("=" * 70)

alphas = np.logspace(-4, 1, 50)
coefs = []

for alpha in alphas:
    lasso_temp = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso_temp.fit(X_train_scaled, y_train)
    coefs.append(lasso_temp.coef_)

coefs = np.array(coefs)

# Plot Lasso Path
plt.figure(figsize=(14, 8))
for i in range(coefs.shape[1]):
    plt.plot(alphas, coefs[:, i], label=X.columns[i])

plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Lasso Path: How Coefficients Shrink with Regularization', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print("KEY INSIGHTS:")
print(f" Original features: 8")
print(f" Useless features added: 6")
print(f" Total features: {X.shape[1]}")
print(f"\n Lasso eliminated {zero_coefs} features automatically!")
print(f" Features with zero coefficients are:")

zero_features = coef_df[coef_df['Lasso'] == 0]['Feature'].tolist()
for feat in zero_features:
    print(f"  - {feat}")

print(f"\n✓ Important features retained by Lasso:")
important_features = coef_df[coef_df['Lasso'] != 0].sort_values('Lasso', key=abs, ascending=False)
for _, row in important_features.iterrows():
    print(f"  - {row['Feature']}: {row['Lasso']:.4f}")

print("CONCLUSION:")
print("Lasso regression successfully identified and eliminated useless features")
print("by forcing their coefficients to exactly zero. This is automatic feature")
print("selection through L1 regularization!")
