
#Project 2: Regularization Exploration**
# Use a dataset with many features (e.g., California Housing)
# Compare Linear, Ridge, and Lasso regression
# Plot regularization paths
# Analyze coefficient shrinkage

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 80)
print("REGULARIZATION EXPLORATION: Ridge vs Lasso Regression")
print("=" * 80)

# 1. Load and explore the dataset
print("\n1. Loading California Housing Dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseValue')

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"\nTarget statistics:")
print(y.describe())

# 2. Split and scale the data
print("\n2. Splitting and Scaling Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# 3. Train models with different regularization
print("\n3. Training Models...")

# Linear Regression (no regularization)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# Lasso Regression (L1 regularization)
lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

# 4. Compare model performance
print("\n4. Model Performance Comparison:")
print("-" * 80)

models = {
    'Linear Regression': (lr, y_pred_lr),
    'Ridge (α=1.0)': (ridge, y_pred_ridge),
    'Lasso (α=0.01)': (lasso, y_pred_lasso)
}

results = []
for name, (model, y_pred) in models.items():
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2
    })
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

results_df = pd.DataFrame(results)

# 5. Coefficient Analysis
print("\n5. Coefficient Comparison:")
print("-" * 80)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Linear': lr.coef_,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_
})

print(coef_df.to_string(index=False))

# Count zero coefficients
print(f"\nZero coefficients in Lasso: {np.sum(np.abs(lasso.coef_) < 1e-10)}")

# 6. Regularization Path for Ridge
print("\n6. Computing Regularization Paths...")

alphas_ridge = np.logspace(-2, 4, 100)
coefs_ridge = []

for alpha in alphas_ridge:
    ridge_temp = Ridge(alpha=alpha, random_state=42)
    ridge_temp.fit(X_train_scaled, y_train)
    coefs_ridge.append(ridge_temp.coef_)

coefs_ridge = np.array(coefs_ridge)

# 7. Regularization Path for Lasso
alphas_lasso = np.logspace(-3, 1, 100)
coefs_lasso = []

for alpha in alphas_lasso:
    lasso_temp = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso_temp.fit(X_train_scaled, y_train)
    coefs_lasso.append(lasso_temp.coef_)

coefs_lasso = np.array(coefs_lasso)

# 8. Cross-validation to find optimal alpha
print("\n8. Finding Optimal Alpha via Cross-Validation...")

ridge_cv = RidgeCV(alphas=alphas_ridge, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Optimal Ridge alpha: {ridge_cv.alpha_:.4f}")

lasso_cv = LassoCV(alphas=alphas_lasso, cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Optimal Lasso alpha: {lasso_cv.alpha_:.4f}")

# 9. Visualization
print("\n9. Creating Visualizations...")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Model Performance Comparison
ax1 = plt.subplot(3, 2, 1)
metrics = ['RMSE', 'MAE', 'R² Score']
x_pos = np.arange(len(results_df))
width = 0.25

for i, metric in enumerate(metrics):
    values = results_df[metric].values
    ax1.bar(x_pos + i*width, values, width, label=metric, alpha=0.8)

ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Coefficient Magnitudes
ax2 = plt.subplot(3, 2, 2)
x_pos = np.arange(len(X.columns))
width = 0.25

ax2.bar(x_pos - width, np.abs(lr.coef_), width, label='Linear', alpha=0.8)
ax2.bar(x_pos, np.abs(ridge.coef_), width, label='Ridge', alpha=0.8)
ax2.bar(x_pos + width, np.abs(lasso.coef_), width, label='Lasso', alpha=0.8)

ax2.set_xlabel('Features', fontsize=11, fontweight='bold')
ax2.set_ylabel('|Coefficient|', fontsize=11, fontweight='bold')
ax2.set_title('Coefficient Magnitude Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(X.columns, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Ridge Regularization Path
ax3 = plt.subplot(3, 2, 3)
for i in range(coefs_ridge.shape[1]):
    ax3.plot(alphas_ridge, coefs_ridge[:, i], label=X.columns[i], linewidth=2)

ax3.axvline(ridge_cv.alpha_, color='red', linestyle='--', linewidth=2, 
            label=f'Optimal α={ridge_cv.alpha_:.2f}')
ax3.set_xscale('log')
ax3.set_xlabel('Alpha (λ)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Coefficient Value', fontsize=11, fontweight='bold')
ax3.set_title('Ridge Regularization Path (L2)', fontsize=12, fontweight='bold')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Lasso Regularization Path
ax4 = plt.subplot(3, 2, 4)
for i in range(coefs_lasso.shape[1]):
    ax4.plot(alphas_lasso, coefs_lasso[:, i], label=X.columns[i], linewidth=2)

ax4.axvline(lasso_cv.alpha_, color='red', linestyle='--', linewidth=2,
            label=f'Optimal α={lasso_cv.alpha_:.4f}')
ax4.set_xscale('log')
ax4.set_xlabel('Alpha (λ)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Coefficient Value', fontsize=11, fontweight='bold')
ax4.set_title('Lasso Regularization Path (L1)', fontsize=12, fontweight='bold')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax4.grid(alpha=0.3)

# Plot 5: Coefficient Shrinkage Heatmap
ax5 = plt.subplot(3, 2, 5)
coef_comparison = pd.DataFrame({
    'Linear': lr.coef_,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_
}, index=X.columns)

sns.heatmap(coef_comparison.T, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, ax=ax5, cbar_kws={'label': 'Coefficient Value'})
ax5.set_title('Coefficient Shrinkage Heatmap', fontsize=12, fontweight='bold')
ax5.set_xlabel('Features', fontsize=11, fontweight='bold')
ax5.set_ylabel('Model', fontsize=11, fontweight='bold')

# Plot 6: Prediction Scatter Plot
ax6 = plt.subplot(3, 2, 6)
sample_size = 500
idx = np.random.choice(len(y_test), sample_size, replace=False)

ax6.scatter(y_test.iloc[idx], y_pred_lr[idx], alpha=0.5, label='Linear', s=30)
ax6.scatter(y_test.iloc[idx], y_pred_ridge[idx], alpha=0.5, label='Ridge', s=30)
ax6.scatter(y_test.iloc[idx], y_pred_lasso[idx], alpha=0.5, label='Lasso', s=30)

min_val = min(y_test.min(), y_pred_lr.min())
max_val = max(y_test.max(), y_pred_lr.max())
ax6.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

ax6.set_xlabel('True Values', fontsize=11, fontweight='bold')
ax6.set_ylabel('Predictions', fontsize=11, fontweight='bold')
ax6.set_title('Predictions vs True Values (Sample)', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'regularization_analysis.png'")
plt.show()

# 10. Summary and Insights
print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)

print("\n1. REGULARIZATION EFFECTS:")
print(f"   - Ridge shrinks coefficients but keeps all features")
print(f"   - Lasso performs feature selection (zeros out {np.sum(np.abs(lasso.coef_) < 1e-10)} features)")

print("\n2. COEFFICIENT SHRINKAGE:")
max_shrink_ridge = X.columns[np.argmax(np.abs(lr.coef_ - ridge.coef_))]
max_shrink_lasso = X.columns[np.argmax(np.abs(lr.coef_ - lasso.coef_))]
print(f"   - Ridge shrinks '{max_shrink_ridge}' the most")
print(f"   - Lasso shrinks '{max_shrink_lasso}' the most")

print("\n3. MODEL SELECTION:")
best_model = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
print(f"   - Best performing model: {best_model}")
print(f"   - Optimal Ridge alpha: {ridge_cv.alpha_:.4f}")
print(f"   - Optimal Lasso alpha: {lasso_cv.alpha_:.4f}")

print("\n4. WHEN TO USE:")
print("   - Linear: Simple baseline, interpretable")
print("   - Ridge: Multicollinearity, keep all features")
print("   - Lasso: Feature selection, sparse models")

print("Analysis complete! Check 'regularization_analysis.png' for visualizations.")
