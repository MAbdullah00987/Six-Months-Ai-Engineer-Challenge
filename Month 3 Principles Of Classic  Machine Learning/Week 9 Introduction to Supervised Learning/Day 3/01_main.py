

#Day 3: Polynomial Regression & The Bias-Variance Tradeoff
#Objective: What if a straight line doesn't fit the data? Learn to model curves.
#Concept: Underfitting (High Bias) vs. Overfitting (High Variance).
#Technique: PolynomialFeatures to generate powers of features ($x^2, x^3$).

#Advanced Regression & Regularization**

#Learning Objectives:
# Understand overfitting and underfitting
# Learn Ridge and Lasso regression
# Explore polynomial features

#Study Materials:**
# Géron Chapter 4: "Regularized Linear Models" (pages 132-145)
# Andrew Ng Course 1, Week 3: "Regularization"
# Article: "Ridge vs Lasso Regression"


#Task: Project - Feature Scaling Impact.
#Create a dataset with a curve. 
#Fit a Linear model (Underfit).Fit a Polynomial model degree 2 (Good fit).Fit a Polynomial model degree 20 (Overfit).
#Crucial: Apply StandardScaler before training to see why scaling matters for convergence.

#Project 2: Regularization Exploration**
# Use a dataset with many features (e.g., California Housing)
# Compare Linear, Ridge, and Lasso regression
# Plot regularization paths
# Analyze coefficient shrinkage

#Project 3: Fish Weight Prediction**
# Apply polynomial features
# Compare different degrees of polynomial
# Use Ridge regression to prevent overfitting

#Part 1: Polynomial Regression 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from scipy import optimize
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
import sympy as sp
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*70)
print("PART 1: UNDERSTANDING THE PROBLEM - LINEAR VS POLYNOMIAL")
print("="*70)

# Generate non-linear data
np.random.seed(42)
n_samples = 100
X = np.linspace(-3, 3, n_samples)
y_true = 0.5 * X**3 - 2 * X**2 + X + 1  # True cubic relationship
y = y_true + np.random.normal(0, 2, n_samples)  # Add noise

# Create DataFrame
df = pd.DataFrame({'X': X, 'y': y, 'y_true': y_true})
print("\nDataset Preview:")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")

# Visualize the data
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.scatter(X, y, alpha=0.6, label='Noisy Data', color='blue')
ax.plot(X, y_true, 'r--', linewidth=2, label='True Function: 0.5x³ - 2x² + x + 1')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Non-Linear Data: Why Linear Regression Fails', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("PART 2: POLYNOMIAL FEATURES WITH SKLEARN")
print("="*70)

# Reshape X for sklearn
X_reshaped = X.reshape(-1, 1)

# Create polynomial features of different degrees
degrees = [1, 2, 3, 5, 10]
models = {}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, degree in enumerate(degrees):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_reshaped)
    
    print(f"\nDegree {degree}:")
    print(f"  Original features: {X_reshaped.shape[1]}")
    print(f"  Polynomial features: {X_poly.shape[1]}")
    print(f"  Feature names: {poly.get_feature_names_out(['X'])}")
    
    # Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    models[degree] = {
        'poly': poly,
        'model': model,
        'mse': mse,
        'r2': r2,
        'coefficients': model.coef_
    }
    
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Coefficients: {model.coef_[:min(5, len(model.coef_))]}")
    
    # Plot
    axes[idx].scatter(X, y, alpha=0.4, s=20, label='Data')
    axes[idx].plot(X, y_pred, 'r-', linewidth=2, label=f'Degree {degree} fit')
    axes[idx].plot(X, y_true, 'g--', linewidth=1, alpha=0.7, label='True function')
    axes[idx].set_title(f'Degree {degree}\nMSE: {mse:.2f}, R²: {r2:.3f}', fontweight='bold')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

# Hide last subplot
axes[-1].axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("PART 3: BIAS-VARIANCE TRADEOFF VISUALIZATION")
print("="*70)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42
)

train_errors = []
test_errors = []
degree_range = range(1, 16)

for degree in degree_range:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    
    print(f"Degree {degree:2d}: Train MSE = {train_mse:8.4f}, Test MSE = {test_mse:8.4f}")

# Bias-Variance Tradeoff Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Left plot: Training vs Test Error
ax1.plot(degree_range, train_errors, 'o-', label='Training Error', linewidth=2, markersize=8)
ax1.plot(degree_range, test_errors, 's-', label='Test Error', linewidth=2, markersize=8)
ax1.axvline(x=3, color='green', linestyle='--', alpha=0.7, label='Optimal (degree=3)')
ax1.set_xlabel('Polynomial Degree', fontsize=12)
ax1.set_ylabel('Mean Squared Error', fontsize=12)
ax1.set_title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add regions
ax1.axvspan(1, 2, alpha=0.2, color='red', label='Underfitting (High Bias)')
ax1.axvspan(8, 15, alpha=0.2, color='orange', label='Overfitting (High Variance)')
ax1.axvspan(2, 8, alpha=0.1, color='green')

# Right plot: R² score
r2_train = [1 - (te / np.var(y_train)) for te in train_errors]
r2_test = [1 - (te / np.var(y_test)) for te in test_errors]

ax2.plot(degree_range, r2_train, 'o-', label='Training R²', linewidth=2, markersize=8)
ax2.plot(degree_range, r2_test, 's-', label='Test R²', linewidth=2, markersize=8)
ax2.axvline(x=3, color='green', linestyle='--', alpha=0.7, label='Optimal')
ax2.set_xlabel('Polynomial Degree', fontsize=12)
ax2.set_ylabel('R² Score', fontsize=12)
ax2.set_title('Model Performance: R² Score', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("PART 4: UNDERFITTING VS OVERFITTING - DETAILED ANALYSIS")
print("="*70)

# Create three models: underfitting, optimal, overfitting
scenarios = {
    'Underfitting (degree=1)': 1,
    'Optimal (degree=3)': 3,
    'Overfitting (degree=15)': 15
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (label, degree) in enumerate(scenarios.items()):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_reshaped)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    # Generate smooth curve for visualization
    X_smooth = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_smooth_poly = poly.transform(X_smooth)
    y_smooth_pred = model.predict(X_smooth_poly)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    axes[idx].scatter(X, y, alpha=0.5, s=30, label='Data', color='blue')
    axes[idx].plot(X_smooth, y_smooth_pred, 'r-', linewidth=2.5, label=f'Degree {degree} fit')
    axes[idx].plot(X, y_true, 'g--', linewidth=2, alpha=0.7, label='True function')
    axes[idx].set_title(f'{label}\nMSE: {mse:.2f}, R²: {r2:.3f}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('y')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    
    print(f"\n{label}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Number of parameters: {degree + 1}")

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("PART 5: REGULARIZATION - RIDGE REGRESSION (L2)")
print("="*70)

# Use high-degree polynomial to demonstrate regularization
degree = 10
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X_reshaped)

# Split data
X_train_poly, X_test_poly, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# Try different alpha values (regularization strength)
alphas = [0, 0.001, 0.01, 0.1, 1, 10, 100]
ridge_results = []

print(f"\nRidge Regression with Polynomial Degree {degree}:")
print("-" * 60)

for alpha in alphas:
    if alpha == 0:
        model = LinearRegression()
    else:
        model = Ridge(alpha=alpha)
    
    model.fit(X_train_poly, y_train)
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    ridge_results.append({
        'alpha': alpha,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'model': model
    })
    
    print(f"Alpha = {alpha:7.3f}: Train MSE = {train_mse:8.4f}, Test MSE = {test_mse:8.4f}")

# Visualize Ridge Regression
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: MSE vs Alpha
ax1 = axes[0, 0]
ax1.semilogx([r['alpha'] if r['alpha'] > 0 else 0.0001 for r in ridge_results], 
             [r['train_mse'] for r in ridge_results], 'o-', label='Training MSE', markersize=8)
ax1.semilogx([r['alpha'] if r['alpha'] > 0 else 0.0001 for r in ridge_results], 
             [r['test_mse'] for r in ridge_results], 's-', label='Test MSE', markersize=8)
ax1.set_xlabel('Regularization Parameter (α)', fontsize=11)
ax1.set_ylabel('Mean Squared Error', fontsize=11)
ax1.set_title('Ridge Regression: MSE vs Regularization Strength', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Coefficient Magnitude vs Alpha
ax2 = axes[0, 1]
for i in range(min(5, X_poly.shape[1])):
    coefs = [r['model'].coef_[i] if hasattr(r['model'], 'coef_') else 0 for r in ridge_results]
    ax2.semilogx([r['alpha'] if r['alpha'] > 0 else 0.0001 for r in ridge_results], 
                 coefs, 'o-', label=f'Coef {i+1}', markersize=6)
ax2.set_xlabel('Regularization Parameter (α)', fontsize=11)
ax2.set_ylabel('Coefficient Value', fontsize=11)
ax2.set_title('Ridge: Coefficient Shrinkage', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Predictions with no regularization
ax3 = axes[1, 0]
model_no_reg = ridge_results[0]['model']
X_smooth = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_smooth_poly = poly.transform(X_smooth)
y_pred_no_reg = model_no_reg.predict(X_smooth_poly)
ax3.scatter(X, y, alpha=0.5, s=20, label='Data')
ax3.plot(X_smooth, y_pred_no_reg, 'r-', linewidth=2, label='No Regularization (α=0)')
ax3.plot(X, y_true, 'g--', linewidth=2, alpha=0.7, label='True function')
ax3.set_xlabel('X')
ax3.set_ylabel('y')
ax3.set_title('Without Regularization (Overfitting)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Predictions with optimal regularization
ax4 = axes[1, 1]
best_ridge = min(ridge_results[1:], key=lambda x: x['test_mse'])
model_reg = best_ridge['model']
y_pred_reg = model_reg.predict(X_smooth_poly)
ax4.scatter(X, y, alpha=0.5, s=20, label='Data')
ax4.plot(X_smooth, y_pred_reg, 'r-', linewidth=2, label=f'Ridge (α={best_ridge["alpha"]})')
ax4.plot(X, y_true, 'g--', linewidth=2, alpha=0.7, label='True function')
ax4.set_xlabel('X')
ax4.set_ylabel('y')
ax4.set_title('With Regularization (Better Generalization)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("PART 6: LASSO REGRESSION (L1) - FEATURE SELECTION")
print("="*70)

lasso_results = []

print(f"\nLasso Regression with Polynomial Degree {degree}:")
print("-" * 60)

for alpha in alphas:
    if alpha == 0:
        continue
    
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Count non-zero coefficients
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
    
    lasso_results.append({
        'alpha': alpha,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'n_nonzero': n_nonzero,
        'model': model
    })
    
    print(f"Alpha = {alpha:7.3f}: Train MSE = {train_mse:8.4f}, Test MSE = {test_mse:8.4f}, "
          f"Non-zero coefs = {n_nonzero}/{degree}")

# Visualize Lasso Regression
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: MSE vs Alpha
ax1 = axes[0, 0]
ax1.semilogx([r['alpha'] for r in lasso_results], 
             [r['train_mse'] for r in lasso_results], 'o-', label='Training MSE', markersize=8)
ax1.semilogx([r['alpha'] for r in lasso_results], 
             [r['test_mse'] for r in lasso_results], 's-', label='Test MSE', markersize=8)
ax1.set_xlabel('Regularization Parameter (α)', fontsize=11)
ax1.set_ylabel('Mean Squared Error', fontsize=11)
ax1.set_title('Lasso Regression: MSE vs Regularization Strength', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Number of non-zero coefficients
ax2 = axes[0, 1]
ax2.semilogx([r['alpha'] for r in lasso_results], 
             [r['n_nonzero'] for r in lasso_results], 'o-', color='purple', markersize=8, linewidth=2)
ax2.set_xlabel('Regularization Parameter (α)', fontsize=11)
ax2.set_ylabel('Number of Non-Zero Coefficients', fontsize=11)
ax2.set_title('Lasso: Feature Selection (Sparsity)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Coefficient paths
ax3 = axes[1, 0]
for i in range(min(5, X_poly.shape[1])):
    coefs = [r['model'].coef_[i] for r in lasso_results]
    ax3.semilogx([r['alpha'] for r in lasso_results], 
                 coefs, 'o-', label=f'Coef {i+1}', markersize=6)
ax3.set_xlabel('Regularization Parameter (α)', fontsize=11)
ax3.set_ylabel('Coefficient Value', fontsize=11)
ax3.set_title('Lasso: Coefficient Paths (Feature Selection)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Best Lasso fit
ax4 = axes[1, 1]
best_lasso = min(lasso_results, key=lambda x: x['test_mse'])
model_lasso = best_lasso['model']
y_pred_lasso = model_lasso.predict(X_smooth_poly)
ax4.scatter(X, y, alpha=0.5, s=20, label='Data')
ax4.plot(X_smooth, y_pred_lasso, 'r-', linewidth=2, 
         label=f'Lasso (α={best_lasso["alpha"]}, {best_lasso["n_nonzero"]} features)')
ax4.plot(X, y_true, 'g--', linewidth=2, alpha=0.7, label='True function')
ax4.set_xlabel('X')
ax4.set_ylabel('y')
ax4.set_title('Lasso Regression: Sparse Solution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("PART 7: RIDGE VS LASSO - DIRECT COMPARISON")
print("="*70)

# Compare coefficient shrinkage
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Coefficients comparison
alpha_compare = 1.0
ridge_model = Ridge(alpha=alpha_compare).fit(X_train_poly, y_train)
lasso_model = Lasso(alpha=alpha_compare, max_iter=10000).fit(X_train_poly, y_train)
linear_model = LinearRegression().fit(X_train_poly, y_train)

coef_indices = np.arange(len(linear_model.coef_))

ax1 = axes[0]
ax1.bar(coef_indices - 0.2, linear_model.coef_, 0.2, label='No Regularization', alpha=0.8)
ax1.bar(coef_indices, ridge_model.coef_, 0.2, label=f'Ridge (α={alpha_compare})', alpha=0.8)
ax1.bar(coef_indices + 0.2, lasso_model.coef_, 0.2, label=f'Lasso (α={alpha_compare})', alpha=0.8)
ax1.set_xlabel('Coefficient Index', fontsize=11)
ax1.set_ylabel('Coefficient Value', fontsize=11)
ax1.set_title('Coefficient Comparison', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Predictions comparison
ax2 = axes[1]
y_ridge = ridge_model.predict(X_smooth_poly)
y_lasso = lasso_model.predict(X_smooth_poly)
y_linear = linear_model.predict(X_smooth_poly)

ax2.scatter(X, y, alpha=0.4, s=15, label='Data', color='gray')
ax2.plot(X_smooth, y_linear, '--', linewidth=2, label='No Regularization', alpha=0.7)
ax2.plot(X_smooth, y_ridge, '-', linewidth=2, label='Ridge')
ax2.plot(X_smooth, y_lasso, '-', linewidth=2, label='Lasso')
ax2.plot(X, y_true, 'k--', linewidth=1.5, alpha=0.5, label='True function')
ax2.set_xlabel('X')
ax2.set_ylabel('y')
ax2.set_title('Prediction Comparison', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Performance metrics
ax3 = axes[2]
models_comp = {
    'No Reg': linear_model,
    'Ridge': ridge_model,
    'Lasso': lasso_model
}

train_scores = []
test_scores = []
model_names = []

for name, model in models_comp.items():
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    train_scores.append(train_r2)
    test_scores.append(test_r2)
    model_names.append(name)

x_pos = np.arange(len(model_names))
width = 0.35

ax3.bar(x_pos - width/2, train_scores, width, label='Train R²', alpha=0.8)
ax3.bar(x_pos + width/2, test_scores, width, label='Test R²', alpha=0.8)
ax3.set_ylabel('R² Score', fontsize=11)
ax3.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_names)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print summary
print("\nPerformance Summary:")
print("-" * 60)
for i, name in enumerate(model_names):
    print(f"{name:15s}: Train R² = {train_scores[i]:.4f}, Test R² = {test_scores[i]:.4f}")

print("\n" + "="*70)
print("PART 8: CROSS-VALIDATION FOR MODEL SELECTION")
print("="*70)

from sklearn.model_selection import cross_val_score, KFold

# Test different polynomial degrees with cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
degree_range = range(1, 11)
cv_scores = []

print("\nCross-Validation Results (5-Fold):")
print("-" * 60)

for degree in degree_range:
    # Create pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    
    # Cross-validation
    scores = cross_val_score(pipeline, X_reshaped, y, cv=cv, 
                            scoring='neg_mean_squared_error')
    mse_scores = -scores
    
    cv_scores.append({
        'degree': degree,
        'mean_mse': mse_scores.mean(),
        'std_mse': mse_scores.std()
    })
    
    print(f"Degree {degree:2d}: MSE = {mse_scores.mean():8.4f} ± {mse_scores.std():6.4f}")

# Plot cross-validation results
fig, ax = plt.subplots(figsize=(10, 6))

degrees = [s['degree'] for s in cv_scores]
means = [s['mean_mse'] for s in cv_scores]
stds = [s['std_mse'] for s in cv_scores]

ax.errorbar(degrees, means, yerr=stds, marker='o', markersize=8, 
            capsize=5, capthick=2, linewidth=2, label='CV MSE ± std')
ax.set_xlabel('Polynomial Degree', fontsize=12)
ax.set_ylabel('Cross-Validation MSE', fontsize=12)
ax.set_title('Cross-Validation: Model Selection', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Mark optimal degree
optimal_idx = np.argmin(means)
optimal_degree = degrees[optimal_idx]
ax.axvline(x=optimal_degree, color='red', linestyle='--', linewidth=2, 
           label=f'Optimal Degree = {optimal_degree}')
ax.legend()

plt.tight_layout()
plt.show()

print(f"\nOptimal Polynomial Degree: {optimal_degree}")

print("\n" + "="*70)
print("PART 9: LEARNING CURVES - DIAGNOSING BIAS-VARIANCE")
print("="*70)

from sklearn.model_selection import learning_curve

# Create learning curves for different scenarios
scenarios = [
    ('Underfitting (degree=1)', 1, 1.0),
    ('Optimal (degree=3)', 3, 1.0),
    ('Overfitting (degree=10)', 10, 0.0)
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (title, degree, alpha) in enumerate(scenarios):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('ridge', Ridge(alpha=alpha))
    ])
    
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_reshaped, y, 
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error',
        random_state=42
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    val_scores_std = val_scores.std(axis=1)
    
    ax = axes[idx]
    ax.plot(train_sizes, train_scores_mean, 'o-', linewidth=2, label='Training Error', markersize=8)
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2)
    
    ax.plot(train_sizes, val_scores_mean, 's-', linewidth=2, label='Validation Error', markersize=8)
    ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.2)
    
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('Mean Squared Error', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nKey Observations:")
print("  - Underfitting: Both errors are high and converge at a high value")
print("  - Optimal: Errors converge at a low value with small gap")
print("  - Overfitting: Large gap between training and validation errors")

print("\n" + "="*70)
print("PART 10: SYMBOLIC MATHEMATICS WITH SYMPY")
print("="*70)

# Use SymPy to understand polynomial regression symbolically
x = sp.Symbol('x')
coeffs = sp.symbols('a0 a1 a2 a3')

# Define polynomial
poly_expr = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3

print("\nSymbolic Polynomial Expression:")
print(f"  y = {poly_expr}")

# Derivatives (for understanding gradient descent)
first_deriv = sp.diff(poly_expr, x)
second_deriv = sp.diff(first_deriv, x)

print("\nFirst Derivative (slope):")
print(f"  dy/dx = {first_deriv}")

print("\nSecond Derivative (curvature):")
print(f"  d²y/dx² = {second_deriv}")

# L2 Regularization term symbolically
l2_penalty = sum(c**2 for c in coeffs[1:])
print("\nL2 Regularization Term (Ridge):")
print(f"  Penalty = λ * ({l2_penalty})")

# L1 Regularization term symbolically
l1_penalty = sum(sp.Abs(c) for c in coeffs[1:])
print("\nL1 Regularization Term (Lasso):")
print(f"  Penalty = λ * ({l1_penalty})")

# Fit actual polynomial and display symbolically
degree_fit = 3
poly_features = PolynomialFeatures(degree=degree_fit, include_bias=True)
X_poly_fit = poly_features.fit_transform(X_reshaped)
model_fit = LinearRegression()
model_fit.fit(X_poly_fit, y)

fitted_expr = sum(model_fit.coef_[i] * x**i if i > 0 else model_fit.intercept_ 
                  for i in range(len(model_fit.coef_)))

print(f"\nFitted Polynomial (degree={degree_fit}):")
print(f"  y = {sp.N(fitted_expr, 4)}")

# Evaluate at specific points
x_vals = [-2, 0, 2]
print("\nEvaluating fitted polynomial:")
for x_val in x_vals:
    y_val = float(fitted_expr.subs(x, x_val))
    print(f"  f({x_val:2.0f}) = {y_val:8.4f}")

print("\n" + "="*70)
print("PART 11: STATISTICAL ANALYSIS WITH SCIPY")
print("="*70)

# Fit polynomial and analyze residuals
degree_stat = 3
poly_stat = PolynomialFeatures(degree=degree_stat, include_bias=False)
X_poly_stat = poly_stat.fit_transform(X_reshaped)
model_stat = LinearRegression()
model_stat.fit(X_poly_stat, y)
y_pred_stat = model_stat.predict(X_poly_stat)
residuals = y - y_pred_stat

print("\nResidual Analysis:")
print("-" * 60)
print(f"Mean of residuals: {np.mean(residuals):.6f} (should be ≈ 0)")
print(f"Std of residuals: {np.std(residuals):.4f}")
print(f"Min residual: {np.min(residuals):.4f}")
print(f"Max residual: {np.max(residuals):.4f}")

# Normality test (Shapiro-Wilk)
stat_shapiro, p_shapiro = stats.shapiro(residuals)
print(f"\nShapiro-Wilk Test (Normality of Residuals):")
print(f"  Statistic: {stat_shapiro:.6f}")
print(f"  P-value: {p_shapiro:.6f}")
print(f"  Result: {'Residuals appear normal' if p_shapiro > 0.05 else 'Residuals not normal'} (α=0.05)")

# Durbin-Watson test (autocorrelation)
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(residuals)
print(f"\nDurbin-Watson Test (Autocorrelation):")
print(f"  Statistic: {dw_stat:.4f}")
print(f"  Result: {'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Possible autocorrelation'}")

# Visualize residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs Fitted
ax1 = axes[0, 0]
ax1.scatter(y_pred_stat, residuals, alpha=0.6, s=30)
ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax1.set_xlabel('Fitted Values', fontsize=11)
ax1.set_ylabel('Residuals', fontsize=11)
ax1.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Q-Q plot
ax2 = axes[0, 1]
stats.probplot(residuals, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Histogram of residuals
ax3 = axes[1, 0]
ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7, density=True)
mu, sigma = residuals.mean(), residuals.std()
x_hist = np.linspace(residuals.min(), residuals.max(), 100)
ax3.plot(x_hist, stats.norm.pdf(x_hist, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
ax3.set_xlabel('Residuals', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Scale-Location plot
ax4 = axes[1, 1]
residuals_standardized = residuals / np.std(residuals)
ax4.scatter(y_pred_stat, np.sqrt(np.abs(residuals_standardized)), alpha=0.6, s=30)
ax4.set_xlabel('Fitted Values', fontsize=11)
ax4.set_ylabel('√|Standardized Residuals|', fontsize=11)
ax4.set_title('Scale-Location Plot (Homoscedasticity)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("PART 12: STATSMODELS - DETAILED REGRESSION ANALYSIS")
print("="*70)

# Use statsmodels for detailed statistical analysis
X_sm = sm.add_constant(X_poly_stat)
model_sm = sm.OLS(y, X_sm)
results_sm = model_sm.fit()

print("\nOLS Regression Results:")
print("=" * 78)
print(results_sm.summary())

# Additional diagnostics
print("\n\nAdditional Diagnostics:")
print("-" * 60)

# Condition number (multicollinearity check)
print(f"Condition Number: {np.linalg.cond(X_sm):.2f}")
print(f"  (>30 suggests multicollinearity issues)")

# VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["Feature"] = [f"X^{i}" for i in range(X_poly_stat.shape[1])]
vif_data["VIF"] = [variance_inflation_factor(X_poly_stat, i) for i in range(X_poly_stat.shape[1])]
print("\nVariance Inflation Factors (VIF):")
print(vif_data.to_string(index=False))
print("  (VIF > 10 indicates problematic multicollinearity)")

# Heteroscedasticity test (Breusch-Pagan)
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(results_sm.resid, X_sm)
print(f"\nBreusch-Pagan Test (Heteroscedasticity):")
print(f"  LM Statistic: {bp_test[0]:.4f}")
print(f"  P-value: {bp_test[1]:.6f}")
print(f"  Result: {'Homoscedastic' if bp_test[1] > 0.05 else 'Heteroscedastic'} (α=0.05)")

print("\n" + "="*70)
print("PART 13: COMPREHENSIVE COMPARISON - ALL METHODS")
print("="*70)

# Compare all methods on test set
methods_comparison = {}

# 1. Linear Regression
poly_linear = PolynomialFeatures(degree=1, include_bias=False)
X_train_lin, X_test_lin, y_train_comp, y_test_comp = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42
)
X_train_poly_lin = poly_linear.fit_transform(X_train_lin)
X_test_poly_lin = poly_linear.transform(X_test_lin)
model_linear = LinearRegression().fit(X_train_poly_lin, y_train_comp)
methods_comparison['Linear'] = {
    'model': model_linear,
    'X_test': X_test_poly_lin,
    'degree': 1
}

# 2. Polynomial Regression (degree 3)
poly_3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly_3 = poly_3.fit_transform(X_train_lin)
X_test_poly_3 = poly_3.transform(X_test_lin)
model_poly3 = LinearRegression().fit(X_train_poly_3, y_train_comp)
methods_comparison['Polynomial (d=3)'] = {
    'model': model_poly3,
    'X_test': X_test_poly_3,
    'degree': 3
}

# 3. Ridge Regression
poly_ridge = PolynomialFeatures(degree=10, include_bias=False)
X_train_poly_ridge = poly_ridge.fit_transform(X_train_lin)
X_test_poly_ridge = poly_ridge.transform(X_test_lin)
model_ridge_final = Ridge(alpha=1.0).fit(X_train_poly_ridge, y_train_comp)
methods_comparison['Ridge (d=10)'] = {
    'model': model_ridge_final,
    'X_test': X_test_poly_ridge,
    'degree': 10
}

# 4. Lasso Regression
model_lasso_final = Lasso(alpha=0.1, max_iter=10000).fit(X_train_poly_ridge, y_train_comp)
methods_comparison['Lasso (d=10)'] = {
    'model': model_lasso_final,
    'X_test': X_test_poly_ridge,
    'degree': 10
}

# 5. ElasticNet
model_elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000).fit(X_train_poly_ridge, y_train_comp)
methods_comparison['ElasticNet (d=10)'] = {
    'model': model_elastic,
    'X_test': X_test_poly_ridge,
    'degree': 10
}

# Evaluate all methods
results_df = []

print("\nComparison of All Methods:")
print("=" * 90)
print(f"{'Method':<20} {'Train MSE':>12} {'Test MSE':>12} {'Train R²':>10} {'Test R²':>10} {'Degree':>8}")
print("-" * 90)

for method_name, method_data in methods_comparison.items():
    model = method_data['model']
    X_test_method = method_data['X_test']
    degree = method_data['degree']
    
    # Get corresponding training data
    if degree == 1:
        X_train_method = X_train_poly_lin
    elif degree == 3:
        X_train_method = X_train_poly_3
    else:
        X_train_method = X_train_poly_ridge
    
    y_train_pred = model.predict(X_train_method)
    y_test_pred = model.predict(X_test_method)
    
    train_mse = mean_squared_error(y_train_comp, y_train_pred)
    test_mse = mean_squared_error(y_test_comp, y_test_pred)
    train_r2 = r2_score(y_train_comp, y_train_pred)
    test_r2 = r2_score(y_test_comp, y_test_pred)
    
    print(f"{method_name:<20} {train_mse:12.4f} {test_mse:12.4f} {train_r2:10.4f} {test_r2:10.4f} {degree:8d}")
    
    results_df.append({
        'Method': method_name,
        'Train_MSE': train_mse,
        'Test_MSE': test_mse,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Degree': degree
    })

results_comparison = pd.DataFrame(results_df)

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: MSE Comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(results_comparison))
width = 0.35
ax1.bar(x_pos - width/2, results_comparison['Train_MSE'], width, label='Train MSE', alpha=0.8)
ax1.bar(x_pos + width/2, results_comparison['Test_MSE'], width, label='Test MSE', alpha=0.8)
ax1.set_ylabel('Mean Squared Error', fontsize=11)
ax1.set_title('MSE Comparison Across Methods', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_comparison['Method'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: R² Comparison
ax2 = axes[0, 1]
ax2.bar(x_pos - width/2, results_comparison['Train_R2'], width, label='Train R²', alpha=0.8)
ax2.bar(x_pos + width/2, results_comparison['Test_R2'], width, label='Test R²', alpha=0.8)
ax2.set_ylabel('R² Score', fontsize=11)
ax2.set_title('R² Score Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_comparison['Method'], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Overfitting Gap
ax3 = axes[1, 0]
gap = results_comparison['Train_MSE'] - results_comparison['Test_MSE']
colors = ['green' if g < 0 else 'red' for g in gap]
ax3.bar(x_pos, np.abs(gap), color=colors, alpha=0.7)
ax3.set_ylabel('|Train MSE - Test MSE|', fontsize=11)
ax3.set_title('Overfitting Gap (Red = Test worse)', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(results_comparison['Method'], rotation=45, ha='right')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Predictions visualization
ax4 = axes[1, 1]
X_smooth = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

colors_plot = plt.cm.tab10(np.linspace(0, 1, len(methods_comparison)))
for idx, (method_name, method_data) in enumerate(methods_comparison.items()):
    model = method_data['model']
    degree = method_data['degree']
    
    if degree == 1:
        poly_smooth = poly_linear
    elif degree == 3:
        poly_smooth = poly_3
    else:
        poly_smooth = poly_ridge
    
    X_smooth_poly = poly_smooth.transform(X_smooth)
    y_smooth_pred = model.predict(X_smooth_poly)
    
    ax4.plot(X_smooth, y_smooth_pred, linewidth=2, label=method_name, color=colors_plot[idx])

ax4.scatter(X, y, alpha=0.3, s=15, color='gray', label='Data')
ax4.plot(X, y_true, 'k--', linewidth=2, alpha=0.7, label='True function')
ax4.set_xlabel('X')
ax4.set_ylabel('y')
ax4.set_title('All Methods - Predictions', fontsize=12, fontweight='bold')
ax4.legend(loc='best', fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("PART 14: FINAL SUMMARY AND KEY TAKEAWAYS")
print("="*70)

'''
KEY CONCEPTS LEARNED:
====================

1. POLYNOMIAL FEATURES:
   - Transform x into [x, x², x³, ...] to model non-linear relationships
   - sklearn.preprocessing.PolynomialFeatures makes this easy
   - Higher degree = more flexibility but risk of overfitting

2. BIAS-VARIANCE TRADEOFF:
   - Underfitting (High Bias): Model too simple, poor on both train & test
   - Optimal: Model complexity matches data complexity
   - Overfitting (High Variance): Model too complex, memorizes noise

3. REGULARIZATION:
   - Ridge (L2): Shrinks all coefficients, keeps all features
     - Penalty: λ * Σ(coefficient²)
     - Good when all features are relevant
   
   - Lasso (L1): Can zero out coefficients, performs feature selection
     - Penalty: λ * Σ|coefficient|
     - Good for sparse models
   
   - ElasticNet: Combines Ridge and Lasso
     - Penalty: λ * [α * L1 + (1-α) * L2]

4. MODEL SELECTION:
   - Use cross-validation to choose hyperparameters
   - Learning curves diagnose bias-variance issues
   - Always evaluate on unseen test data

5. STATISTICAL VALIDATION:
   - Check residuals for normality and homoscedasticity
   - Look for patterns in residual plots
   - VIF detects multicollinearity
   - Use statsmodels for comprehensive analysis

PRACTICAL TIPS:
==============
✓ Start simple, add complexity if needed
✓ Always split data: train/validation/test
✓ Visualize your data before modeling
✓ Check residual plots to validate assumptions
✓ Use regularization with high-degree polynomials
✓ Cross-validation prevents overfitting
✓ Consider domain knowledge when choosing model complexity

LIBRARIES USED:
===============
• NumPy: Array operations, linear algebra
• Pandas: Data manipulation and analysis
• Matplotlib: Basic plotting
• Seaborn: Statistical visualizations
• Scikit-learn: Machine learning models and preprocessing
• SciPy: Statistical tests
• Statsmodels: Detailed statistical analysis
• SymPy: Symbolic mathematics

NEXT STEPS:
===========
1. Try this on real datasets
2. Experiment with different regularization strengths
3. Combine with feature engineering
4. Explore other regularization methods (ElasticNet, etc.)
5. Learn about model ensembles (combine multiple models)
'''

"""
POLYNOMIAL REGRESSION & BIAS-VARIANCE TRADEOFF
Complete Guide with Python Libraries
================================================

Topics Covered:
1. Understanding Linear vs Polynomial Regression
2. PolynomialFeatures in sklearn
3. Underfitting (High Bias)
4. Overfitting (High Variance)
5. Ridge Regression (L2 Regularization)
6. Lasso Regression (L1 Regularization)
7. Cross-Validation & Model Selection
8. Symbolic Math with SymPy
9. Statistical Analysis with SciPy & Statsmodels
10. Visualization with Matplotlib & Seaborn
"""