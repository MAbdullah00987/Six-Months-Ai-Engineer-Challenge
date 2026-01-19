
#JSon Prompts Used here 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
import sympy as sp
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HOUR 1: MATHEMATICAL FOUNDATION")
print("="*80)

# Exercise 1.1: Understand the loss functions
print("\n📚 Exercise 1.1: Derive and visualize loss functions")
print("-" * 60)

w = sp.Symbol('w')
y, x, alpha = sp.symbols('y x alpha')

# Basic loss
mse = (y - w*x)**2
print(f"MSE Loss: {mse}")

# Ridge loss
ridge_loss = mse + alpha * w**2
print(f"\nRidge Loss: {ridge_loss}")
ridge_derivative = sp.diff(ridge_loss, w)
print(f"Ridge derivative: {sp.simplify(ridge_derivative)}")

# Lasso loss
lasso_loss = mse + alpha * sp.Abs(w)
print(f"\nLasso Loss: {lasso_loss}")
print("Note: Lasso derivative has discontinuity at w=0")

# TODO: YOUR TASK - Set w=2, x=3, y=7, alpha=0.5 and compute actual loss values
w_val, x_val, y_val, alpha_val = 2, 3, 7, 0.5
mse_val = float(mse.subs([(w, w_val), (x, x_val), (y, y_val)]))
ridge_val = float(ridge_loss.subs([(w, w_val), (x, x_val), (y, y_val), (alpha, alpha_val)]))
lasso_val = float(lasso_loss.subs([(w, w_val), (x, x_val), (y, y_val), (alpha, alpha_val)]))

print(f"\n✅ With w={w_val}, x={x_val}, y={y_val}, alpha={alpha_val}:")
print(f"   MSE: {mse_val:.2f}")
print(f"   Ridge: {ridge_val:.2f}")
print(f"   Lasso: {lasso_val:.2f}")

# Exercise 1.2: Visualize constraint regions
print("\n📚 Exercise 1.2: Geometric intuition - Why Lasso creates sparsity")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Ridge constraint (circle)
theta = np.linspace(0, 2*np.pi, 100)
t = 1.0  # constraint budget
w1_ridge = t * np.cos(theta)
w2_ridge = t * np.sin(theta)

# Lasso constraint (diamond)
w1_lasso = np.concatenate([np.linspace(0, t, 25), np.linspace(t, 0, 25),
                           np.linspace(0, -t, 25), np.linspace(-t, 0, 25)])
w2_lasso = np.concatenate([np.linspace(t, 0, 25), np.linspace(0, -t, 25),
                           np.linspace(-t, 0, 25), np.linspace(0, t, 25)])

# Contour lines (loss function)
w1_grid = np.linspace(-2, 2, 100)
w2_grid = np.linspace(-2, 2, 100)
W1, W2 = np.meshgrid(w1_grid, w2_grid)
# Assume optimal is at (1.5, 1.5)
Z = (W1 - 1.5)**2 + (W2 - 1.5)**2

# Plot Ridge
ax = axes[0]
ax.contour(W1, W2, Z, levels=15, cmap='viridis', alpha=0.6)
ax.plot(w1_ridge, w2_ridge, 'r-', linewidth=3, label='Ridge constraint: w₁² + w₂² ≤ t')
ax.plot(1.5, 1.5, 'k*', markersize=15, label='Unconstrained optimum')
ax.set_xlabel('w₁', fontsize=12)
ax.set_ylabel('w₂', fontsize=12)
ax.set_title('Ridge (L2): Circular constraint\nSolution rarely at axes', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# Plot Lasso
ax = axes[1]
ax.contour(W1, W2, Z, levels=15, cmap='viridis', alpha=0.6)
ax.plot(w1_lasso, w2_lasso, 'r-', linewidth=3, label='Lasso constraint: |w₁| + |w₂| ≤ t')
ax.plot(1.5, 1.5, 'k*', markersize=15, label='Unconstrained optimum')
ax.plot(1, 0, 'ro', markersize=10, label='Lasso solution (sparse!)')
ax.set_xlabel('w₁', fontsize=12)
ax.set_ylabel('w₂', fontsize=12)
ax.set_title('Lasso (L1): Diamond constraint\nSolution often at corners (w₂=0)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('hour1_geometric_intuition.png', dpi=150)
print("✅ Saved: hour1_geometric_intuition.png")
plt.show()

print("\n" + "="*80)
print("HOUR 2: BUILD FROM SCRATCH WITH NUMPY")
print("="*80)

# Exercise 2.1: Ridge from scratch
print("\n📚 Exercise 2.1: Implement Ridge Regression from scratch")
print("-" * 60)

# Generate data
np.random.seed(42)
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)
true_weights = np.array([3, -2, 0.5, 0, -1])
y = X @ true_weights + np.random.randn(n_samples) * 0.5

print(f"Data: {n_samples} samples, {n_features} features")
print(f"True weights: {true_weights}")

# Method 1: Closed form solution
def ridge_closed_form(X, y, alpha):
    """Ridge using analytical solution: w = (X'X + αI)^(-1) X'y"""
    n_features = X.shape[1]
    I = np.eye(n_features)
    w = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return w

# Method 2: Gradient descent
def ridge_gradient_descent(X, y, alpha, lr=0.01, iterations=1000):
    """Ridge using gradient descent"""
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    for i in range(iterations):
        predictions = X @ w
        gradient = -2/n_samples * X.T @ (y - predictions) + 2 * alpha * w
        w -= lr * gradient
    
    return w

# Test both methods
alpha_test = 1.0
w_closed = ridge_closed_form(X, y, alpha_test)
w_gradient = ridge_gradient_descent(X, y, alpha_test, lr=0.01, iterations=2000)

print(f"\n✅ Ridge with α={alpha_test}:")
print(f"   True weights:    {true_weights}")
print(f"   Closed form:     {w_closed}")
print(f"   Gradient desc:   {w_gradient}")
print(f"   Difference:      {np.linalg.norm(w_closed - w_gradient):.6f}")

# TODO: YOUR TASK - Try different alpha values and see how weights change
print("\n📝 YOUR TASK: Test α = 0.01, 0.1, 1.0, 10.0, 100.0")
alphas_test = [0.01, 0.1, 1.0, 10.0, 100.0]
for a in alphas_test:
    w = ridge_closed_form(X, y, a)
    print(f"α={a:6.2f}: weights={w}, L2 norm={np.linalg.norm(w):.3f}")

print("\n" + "="*80)
print("HOUR 3: SKLEARN PROFESSIONAL IMPLEMENTATION")
print("="*80)

# Exercise 3.1: Create dataset with irrelevant features
print("\n📚 Exercise 3.1: Feature selection with Lasso")
print("-" * 60)

np.random.seed(42)
n_samples = 200
n_features = 20
n_informative = 5

X = np.random.randn(n_samples, n_features)
# Only first 5 features are informative
true_coef = np.zeros(n_features)
true_coef[:n_informative] = [5, -3, 2, -4, 1]
y = X @ true_coef + np.random.randn(n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Train models
models = {
    'Linear': LinearRegression(),
    'Ridge (α=1)': Ridge(alpha=1.0),
    'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=5000),
    'Lasso (α=0.5)': Lasso(alpha=0.5, max_iter=5000),
    'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000)
}

print("\nModel Performance:")
print(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10} {'Non-zero':>10}")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    train_r2 = model.score(X_train_sc, y_train)
    test_r2 = model.score(X_test_sc, y_test)
    
    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(np.abs(model.coef_) > 0.01)
        print(f"{name:<20} {train_r2:>10.4f} {test_r2:>10.4f} {n_nonzero:>10}/{n_features}")

# TODO: YOUR TASK - Which model best identifies the 5 true features?
print("\n📝 YOUR TASK: Compare coefficient patterns")
lasso = Lasso(alpha=0.5, max_iter=5000).fit(X_train_sc, y_train)
print(f"\nTrue coefficients:  {true_coef}")
print(f"Lasso found:        {lasso.coef_}")
print(f"Lasso correctly identified features: {np.where(np.abs(lasso.coef_) > 0.1)[0]}")

print("\n" + "="*80)
print("HOUR 4: HYPERPARAMETER TUNING")
print("="*80)

# Exercise 4.1: Grid search for optimal alpha
print("\n📚 Exercise 4.1: Find optimal alpha using cross-validation")
print("-" * 60)

alphas = np.logspace(-3, 2, 50)

# Ridge
ridge_scores = []
for alpha in alphas:
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='r2')
    ridge_scores.append(scores.mean())

# Lasso
lasso_scores = []
for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=5000)
    scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='r2')
    lasso_scores.append(scores.mean())

# Find optimal
ridge_best_idx = np.argmax(ridge_scores)
lasso_best_idx = np.argmax(lasso_scores)

print(f"✅ Optimal Ridge α: {alphas[ridge_best_idx]:.4f} (CV R²: {ridge_scores[ridge_best_idx]:.4f})")
print(f"✅ Optimal Lasso α: {alphas[lasso_best_idx]:.4f} (CV R²: {lasso_scores[lasso_best_idx]:.4f})")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(alphas, ridge_scores, 'b-', linewidth=2, label='Ridge CV score')
plt.axvline(alphas[ridge_best_idx], color='r', linestyle='--', label=f'Optimal α={alphas[ridge_best_idx]:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (log scale)', fontsize=11)
plt.ylabel('CV R² Score', fontsize=11)
plt.title('Ridge: Finding Optimal Regularization Strength', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(alphas, lasso_scores, 'g-', linewidth=2, label='Lasso CV score')
plt.axvline(alphas[lasso_best_idx], color='r', linestyle='--', label=f'Optimal α={alphas[lasso_best_idx]:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (log scale)', fontsize=11)
plt.ylabel('CV R² Score', fontsize=11)
plt.title('Lasso: Finding Optimal Regularization Strength', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hour4_hyperparameter_tuning.png', dpi=150)
print("✅ Saved: hour4_hyperparameter_tuning.png")
plt.show()

print("\n" + "="*80)
print("HOUR 5: STATISTICAL ANALYSIS WITH STATSMODELS")
print("="*80)

# Exercise 5.1: Statistical inference
print("\n📚 Exercise 5.1: Statistical analysis of regression")
print("-" * 60)

# OLS with statsmodels
X_train_sm = sm.add_constant(X_train_sc)
ols_model = sm.OLS(y_train, X_train_sm).fit()

print("\nOLS Summary (first 5 features):")
print(ols_model.summary().tables[1])

# Residual analysis
ridge_optimal = Ridge(alpha=alphas[ridge_best_idx]).fit(X_train_sc, y_train)
residuals = y_test - ridge_optimal.predict(X_test_sc)

# Normality test
stat, p_val = stats.shapiro(residuals)
print(f"\n✅ Shapiro-Wilk normality test: p-value = {p_val:.4f}")
if p_val > 0.05:
    print("   → Residuals appear normally distributed ✓")
else:
    print("   → Residuals may not be normally distributed ✗")

# Visualize residuals
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Residuals vs predicted
axes[0].scatter(ridge_optimal.predict(X_test_sc), residuals, alpha=0.6)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residual Plot\n(Should be random around 0)')
axes[0].grid(True, alpha=0.3)

# QQ plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot\n(Should be linear for normality)')
axes[1].grid(True, alpha=0.3)

# Histogram
axes[2].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Residuals')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Residual Distribution\n(Should be bell-shaped)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hour5_residual_analysis.png', dpi=150)
print("✅ Saved: hour5_residual_analysis.png")
plt.show()

print("\n" + "="*80)
print("HOUR 6: COMPREHENSIVE VISUALIZATION")
print("="*80)

# Exercise 6.1: Regularization paths
print("\n📚 Exercise 6.1: Visualize regularization paths")
print("-" * 60)

alphas_path = np.logspace(-2, 2, 100)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas_path:
    ridge = Ridge(alpha=alpha).fit(X_train_sc, y_train)
    lasso = Lasso(alpha=alpha, max_iter=5000).fit(X_train_sc, y_train)
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# Create mega visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Ridge path
ax = axes[0, 0]
for i in range(n_features):
    color = 'red' if i < n_informative else 'gray'
    alpha_val = 1.0 if i < n_informative else 0.3
    ax.plot(alphas_path, ridge_coefs[:, i], color=color, alpha=alpha_val, linewidth=2)
ax.set_xscale('log')
ax.set_xlabel('Alpha', fontsize=11)
ax.set_ylabel('Coefficient value', fontsize=11)
ax.set_title('Ridge Path\n(Red = true features, Gray = noise)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

# 2. Lasso path
ax = axes[0, 1]
for i in range(n_features):
    color = 'red' if i < n_informative else 'gray'
    alpha_val = 1.0 if i < n_informative else 0.3
    ax.plot(alphas_path, lasso_coefs[:, i], color=color, alpha=alpha_val, linewidth=2)
ax.set_xscale('log')
ax.set_xlabel('Alpha', fontsize=11)
ax.set_ylabel('Coefficient value', fontsize=11)
ax.set_title('Lasso Path\n(Gray features → 0, Red features survive)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

# 3. Feature selection
ax = axes[0, 2]
n_features_selected = [np.sum(np.abs(coef) > 0.01) for coef in lasso_coefs]
ax.plot(alphas_path, n_features_selected, linewidth=3, color='blue')
ax.axhline(y=n_informative, color='r', linestyle='--', linewidth=2, 
           label=f'True # features = {n_informative}')
ax.set_xscale('log')
ax.set_xlabel('Alpha', fontsize=11)
ax.set_ylabel('# Selected features', fontsize=11)
ax.set_title('Lasso Feature Selection\n(Higher α → fewer features)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Coefficient magnitude comparison
ax = axes[1, 0]
x_pos = np.arange(n_features)
width = 0.25
linear_m = LinearRegression().fit(X_train_sc, y_train)
ridge_m = Ridge(alpha=1.0).fit(X_train_sc, y_train)
lasso_m = Lasso(alpha=0.5, max_iter=5000).fit(X_train_sc, y_train)

colors_true = ['red' if i < n_informative else 'gray' for i in range(n_features)]
ax.bar(x_pos - width, linear_m.coef_, width, label='Linear', alpha=0.7)
ax.bar(x_pos, ridge_m.coef_, width, label='Ridge', alpha=0.7)
ax.bar(x_pos + width, lasso_m.coef_, width, label='Lasso', alpha=0.7)
ax.set_xlabel('Feature index', fontsize=11)
ax.set_ylabel('Coefficient', fontsize=11)
ax.set_title('Coefficient Comparison\n(First 5 are true features)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 5. Train vs Test performance
ax = axes[1, 1]
models_compare = ['Linear', 'Ridge\n(α=1)', 'Lasso\n(α=0.5)', 'ElasticNet']
train_scores = []
test_scores = []

for name, model in [('Linear', LinearRegression()),
                     ('Ridge', Ridge(alpha=1.0)),
                     ('Lasso', Lasso(alpha=0.5, max_iter=5000)),
                     ('ElasticNet', ElasticNet(alpha=0.5, l1_ratio=0.5))]:
    model.fit(X_train_sc, y_train)
    train_scores.append(model.score(X_train_sc, y_train))
    test_scores.append(model.score(X_test_sc, y_test))

x_pos = np.arange(len(models_compare))
width = 0.35
ax.bar(x_pos - width/2, train_scores, width, label='Train R²', alpha=0.8)
ax.bar(x_pos + width/2, test_scores, width, label='Test R²', alpha=0.8)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(models_compare)
ax.set_title('Train vs Test Performance\n(Smaller gap = less overfit)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

# 6. Prediction scatter
ax = axes[1, 2]
best_model = Lasso(alpha=alphas[lasso_best_idx], max_iter=5000).fit(X_train_sc, y_train)
y_pred = best_model.predict(X_test_sc)
ax.scatter(y_test, y_pred, alpha=0.6, s=50)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual values', fontsize=11)
ax.set_ylabel('Predicted values', fontsize=11)
ax.set_title(f'Best Model Predictions\nTest R² = {best_model.score(X_test_sc, y_test):.4f}', 
             fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hour6_comprehensive_analysis.png', dpi=150)
print("✅ Saved: hour6_comprehensive_analysis.png")
plt.show()

print("\n" + "="*80)
print("FINAL SUMMARY & NEXT STEPS")
print("="*80)

print("""
🎉 CONGRATULATIONS! You've completed the regularization workshop!

📊 What you learned today:
   ✓ Mathematical foundation of L1 and L2 regularization
   ✓ Why Lasso creates sparsity (geometric interpretation)
   ✓ Implementation from scratch using numpy
   ✓ Professional implementation with sklearn
   ✓ Hyperparameter tuning with cross-validation
   ✓ Statistical analysis with statsmodels
   ✓ Comprehensive visualization with matplotlib and seaborn

🎯 Key insights:
   • Ridge (L2): Use when ALL features matter, prevents overfitting
   • Lasso (L1): Use for FEATURE SELECTION, creates sparse models
   • ElasticNet: Best of both worlds, handles correlated features
   • Always use cross-validation to choose α
   • Always standardize features before regularization

📝 Homework for tomorrow:
   1. Apply this to a real dataset (house prices, stocks, etc.)
   2. Implement coordinate descent for Lasso from scratch
   3. Try Group Lasso for categorical variables
   4. Explore Bayesian interpretation of Ridge regression

🔗 Next topics to learn:
   → Polynomial regression + regularization
   → Cross-validation strategies (k-fold, stratified, time-series)
   → Feature engineering techniques
   → Ensemble methods (Random Forest, Gradient Boosting)

💪 Practice exercises:
   1. Create dataset where Ridge > Lasso (explain why)
   2. Create dataset where Lasso > Ridge (explain why)
   3. Build end-to-end pipeline with feature engineering
   4. Compare regularization vs. dropout in neural networks
""")

print("\n✅ All files saved! Review the visualizations and notes.")
print("="*80)