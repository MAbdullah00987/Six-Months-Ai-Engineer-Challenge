
#Day 6: Regularization (Ridge, Lasso, ElasticNet)
#Objective: Stop your model from memorizing the noise (Overfitting).

#Concept: L1 Regularization (Lasso) vs. L2 Regularization (Ridge).

#Learning Objectives:
# Apply everything learned to real datasets
# Practice end-to-end ML workflow
# Build portfolio-ready projects

#Project 1: Regularization Exploration.

#Use the Housing dataset again. Add meaningful noise or extra useless features.

#Train Lasso regression and watch it force coefficients of useless features to zero. This is automatic feature selection!

#Project 2: Diabetes Prediction**
# Use Pima Indians Diabetes dataset
# Complete preprocessing pipeline
# Try both logistic regression and regularized versions
# Create ROC curve and AUC score

#Project 3: Customer Purchase Prediction**
# Binary classification problem
# Feature engineering from age and salary
# Implement decision boundary visualization
# Calculate business metrics (profit/loss from predictions)

#Part 1: Complete Regularization Learning System
#Master Ridge, Lasso, and ElasticNet with hands-on examples


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import sympy as sp
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("PART 1: MATHEMATICAL FOUNDATION WITH SYMPY")
print("="*80)

# Define symbolic variables
w, x, y, lambda_l1, lambda_l2, alpha = sp.symbols('w x y lambda_L1 lambda_L2 alpha')
n = sp.Symbol('n', positive=True)

# 1. Basic Linear Regression Loss
print("\n1. Basic Linear Regression (No Regularization)")
print("-" * 60)
loss_basic = (y - w*x)**2
print(f"Loss Function: L = {loss_basic}")
derivative_basic = sp.diff(loss_basic, w)
print(f"Derivative ∂L/∂w = {derivative_basic}")
print(f"Simplified: {sp.simplify(derivative_basic)}")

# 2. L2 Regularization (Ridge)
print("\n2. Ridge Regression (L2 Regularization)")
print("-" * 60)
loss_ridge = (y - w*x)**2 + lambda_l2 * w**2
print(f"Loss Function: L = {loss_ridge}")
derivative_ridge = sp.diff(loss_ridge, w)
print(f"Derivative ∂L/∂w = {derivative_ridge}")
print(f"Simplified: {sp.simplify(derivative_ridge)}")
optimal_w_ridge = sp.solve(derivative_ridge, w)
print(f"Optimal w (Ridge): {optimal_w_ridge}")

# 3. L1 Regularization (Lasso)
print("\n3. Lasso Regression (L1 Regularization)")
print("-" * 60)
loss_lasso = (y - w*x)**2 + lambda_l1 * sp.Abs(w)
print(f"Loss Function: L = {loss_lasso}")
print("Note: L1 penalty |w| creates sparsity (forces some weights to exactly 0)")

# 4. ElasticNet (L1 + L2)
print("\n4. ElasticNet (L1 + L2 Combined)")
print("-" * 60)
loss_elastic = (y - w*x)**2 + alpha * lambda_l1 * sp.Abs(w) + (1-alpha) * lambda_l2 * w**2
print(f"Loss Function: L = {sp.simplify(loss_elastic)}")
print("ElasticNet balances feature selection (L1) and weight shrinkage (L2)")

print("\n" + "="*80)
print("PART 2: NUMPY - BUILD REGULARIZATION FROM SCRATCH")
print("="*80)

# Generate synthetic data with noise
np.random.seed(42)
n_samples = 100
X_true = np.linspace(0, 10, n_samples)
y_true = 2.5 * X_true + 1.0
noise = np.random.normal(0, 3, n_samples)
y_noisy = y_true + noise

# Add polynomial features to create overfitting scenario
X_poly = np.column_stack([X_true**i for i in range(1, 8)])  # Up to x^7

class RegularizationFromScratch:
    def __init__(self, reg_type='ridge', alpha=1.0, l1_ratio=0.5):
        self.reg_type = reg_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.weights = None
        
    def fit(self, X, y, lr=0.01, iterations=1000):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        
        for i in range(iterations):
            # Predictions
            y_pred = X @ self.weights
            
            # Gradient of MSE
            gradient = -2/n_samples * X.T @ (y - y_pred)
            
            # Add regularization gradient
            if self.reg_type == 'ridge':
                gradient += 2 * self.alpha * self.weights
            elif self.reg_type == 'lasso':
                gradient += self.alpha * np.sign(self.weights)
            elif self.reg_type == 'elasticnet':
                gradient += self.alpha * (
                    self.l1_ratio * np.sign(self.weights) + 
                    (1 - self.l1_ratio) * 2 * self.weights
                )
            
            # Update weights
            self.weights -= lr * gradient
            
        return self
    
    def predict(self, X):
        return X @ self.weights

# Train models
ridge_scratch = RegularizationFromScratch('ridge', alpha=10.0)
lasso_scratch = RegularizationFromScratch('lasso', alpha=0.5)
elastic_scratch = RegularizationFromScratch('elasticnet', alpha=1.0)

ridge_scratch.fit(X_poly, y_noisy, lr=0.001, iterations=2000)
lasso_scratch.fit(X_poly, y_noisy, lr=0.001, iterations=2000)
elastic_scratch.fit(X_poly, y_noisy, lr=0.001, iterations=2000)

print("\nWeights comparison (from scratch):")
print(f"Ridge weights: {ridge_scratch.weights}")
print(f"Lasso weights: {lasso_scratch.weights}")
print(f"ElasticNet weights: {elastic_scratch.weights}")
print(f"\nNumber of near-zero weights in Lasso: {np.sum(np.abs(lasso_scratch.weights) < 0.1)}")

print("\n" + "="*80)
print("PART 3: SKLEARN - PROFESSIONAL IMPLEMENTATION")
print("="*80)

# Create dataset with more features for better demonstration
np.random.seed(42)
n_samples = 200
n_features = 15

# True model has only 5 relevant features
X = np.random.randn(n_samples, n_features)
true_weights = np.zeros(n_features)
true_weights[[0, 3, 5, 8, 12]] = [5, -3, 2, -4, 3]  # Only 5 non-zero
y = X @ true_weights + np.random.randn(n_samples) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train different models
models = {
    'Linear': LinearRegression(),
    'Ridge (α=0.1)': Ridge(alpha=0.1),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Ridge (α=10.0)': Ridge(alpha=10.0),
    'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=5000),
    'Lasso (α=0.5)': Lasso(alpha=0.5, max_iter=5000),
    'Lasso (α=1.0)': Lasso(alpha=1.0, max_iter=5000),
    'ElasticNet (α=0.5, l1=0.5)': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(np.abs(model.coef_) > 0.01)
    else:
        n_nonzero = n_features
    
    results.append({
        'Model': name,
        'Train R²': train_score,
        'Test R²': test_score,
        'MSE': mse,
        'MAE': mae,
        'Non-zero Coefs': n_nonzero
    })

results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("PART 4: PANDAS - DATA ANALYSIS & COMPARISON")
print("="*80)

# Analyze overfitting
results_df['Overfit Gap'] = results_df['Train R²'] - results_df['Test R²']
print("\nOverfitting Analysis (Train R² - Test R²):")
print(results_df[['Model', 'Train R²', 'Test R²', 'Overfit Gap']].to_string(index=False))

# Feature importance analysis
print("\nFeature Selection Analysis:")
for name, model in models.items():
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        print(f"\n{name}:")
        print(f"  Max |coef|: {np.max(np.abs(coefs)):.4f}")
        print(f"  Min |coef|: {np.min(np.abs(coefs)):.4f}")
        print(f"  Non-zero: {np.sum(np.abs(coefs) > 0.01)}/{n_features}")

print("\n" + "="*80)
print("PART 5: SCIPY - OPTIMIZATION & STATISTICS")
print("="*80)

# Custom optimization with scipy
def ridge_loss(w, X, y, alpha):
    predictions = X @ w
    mse = np.mean((y - predictions)**2)
    penalty = alpha * np.sum(w**2)
    return mse + penalty

# Optimize using scipy
w_init = np.random.randn(n_features)
result = minimize(ridge_loss, w_init, args=(X_train_scaled, y_train, 1.0), method='BFGS')

print("\nScipy Optimization Results:")
print(f"Success: {result.success}")
print(f"Iterations: {result.nit}")
print(f"Final loss: {result.fun:.6f}")

# Statistical tests on residuals
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
residuals = y_test - ridge_model.predict(X_test_scaled)

# Normality test
_, p_value_normality = stats.shapiro(residuals)
print(f"\nShapiro-Wilk test (residuals normality): p={p_value_normality:.4f}")

# Homoscedasticity
print(f"Residual std: {np.std(residuals):.4f}")

print("\n" + "="*80)
print("PART 6: STATSMODELS - STATISTICAL REGRESSION ANALYSIS")
print("="*80)

# OLS with statsmodels
X_train_sm = sm.add_constant(X_train_scaled)
X_test_sm = sm.add_constant(X_test_scaled)

ols_model = OLS(y_train, X_train_sm).fit()
print("\nOLS Regression Summary (First features):")
print(ols_model.summary().tables[1])

# Regularized regression path
alphas = np.logspace(-2, 2, 50)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X_train_scaled, y_train)
    lasso = Lasso(alpha=alpha, max_iter=5000).fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

print("\nRegularization path computed for visualization")

print("\n" + "="*80)
print("PART 7: MATPLOTLIB - COMPREHENSIVE VISUALIZATION")
print("="*80)

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 12))

# 1. Regularization Paths - Ridge
ax1 = plt.subplot(3, 3, 1)
for i in range(n_features):
    ax1.plot(alphas, ridge_coefs[:, i], alpha=0.7)
ax1.set_xscale('log')
ax1.set_xlabel('Alpha (log scale)', fontsize=10)
ax1.set_ylabel('Coefficient value', fontsize=10)
ax1.set_title('Ridge Regularization Path\n(Coefficients shrink smoothly)', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)

# 2. Regularization Paths - Lasso
ax2 = plt.subplot(3, 3, 2)
for i in range(n_features):
    ax2.plot(alphas, lasso_coefs[:, i], alpha=0.7)
ax2.set_xscale('log')
ax2.set_xlabel('Alpha (log scale)', fontsize=10)
ax2.set_ylabel('Coefficient value', fontsize=10)
ax2.set_title('Lasso Regularization Path\n(Coefficients become exactly 0)', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)

# 3. Coefficient comparison
ax3 = plt.subplot(3, 3, 3)
x_pos = np.arange(n_features)
width = 0.25
linear_model = LinearRegression().fit(X_train_scaled, y_train)
ridge_model = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
lasso_model = Lasso(alpha=0.5, max_iter=5000).fit(X_train_scaled, y_train)

ax3.bar(x_pos - width, linear_model.coef_, width, label='Linear', alpha=0.8)
ax3.bar(x_pos, ridge_model.coef_, width, label='Ridge', alpha=0.8)
ax3.bar(x_pos + width, lasso_model.coef_, width, label='Lasso', alpha=0.8)
ax3.set_xlabel('Feature index', fontsize=10)
ax3.set_ylabel('Coefficient value', fontsize=10)
ax3.set_title('Coefficient Comparison\n(Ridge shrinks, Lasso zeros out)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# 4. Performance vs Alpha - Ridge
ax4 = plt.subplot(3, 3, 4)
test_scores_ridge = []
train_scores_ridge = []
for alpha in alphas:
    model = Ridge(alpha=alpha).fit(X_train_scaled, y_train)
    train_scores_ridge.append(model.score(X_train_scaled, y_train))
    test_scores_ridge.append(model.score(X_test_scaled, y_test))

ax4.plot(alphas, train_scores_ridge, label='Train R²', linewidth=2)
ax4.plot(alphas, test_scores_ridge, label='Test R²', linewidth=2)
ax4.set_xscale('log')
ax4.set_xlabel('Alpha (log scale)', fontsize=10)
ax4.set_ylabel('R² Score', fontsize=10)
ax4.set_title('Ridge: Performance vs Regularization\n(Find sweet spot)', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Performance vs Alpha - Lasso
ax5 = plt.subplot(3, 3, 5)
test_scores_lasso = []
train_scores_lasso = []
for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=5000).fit(X_train_scaled, y_train)
    train_scores_lasso.append(model.score(X_train_scaled, y_train))
    test_scores_lasso.append(model.score(X_test_scaled, y_test))

ax5.plot(alphas, train_scores_lasso, label='Train R²', linewidth=2)
ax5.plot(alphas, test_scores_lasso, label='Test R²', linewidth=2)
ax5.set_xscale('log')
ax5.set_xlabel('Alpha (log scale)', fontsize=10)
ax5.set_ylabel('R² Score', fontsize=10)
ax5.set_title('Lasso: Performance vs Regularization\n(Automatic feature selection)', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Number of non-zero coefficients vs Alpha
ax6 = plt.subplot(3, 3, 6)
n_nonzero = [np.sum(np.abs(coef) > 0.01) for coef in lasso_coefs]
ax6.plot(alphas, n_nonzero, linewidth=2, color='red')
ax6.set_xscale('log')
ax6.set_xlabel('Alpha (log scale)', fontsize=10)
ax6.set_ylabel('Number of non-zero coefficients', fontsize=10)
ax6.set_title('Lasso Feature Selection\n(Higher α = fewer features)', fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.fill_between(alphas, 0, n_nonzero, alpha=0.3, color='red')

# 7. Residuals plot
ax7 = plt.subplot(3, 3, 7)
ridge_pred = ridge_model.predict(X_test_scaled)
ax7.scatter(ridge_pred, y_test - ridge_pred, alpha=0.6, s=50)
ax7.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax7.set_xlabel('Predicted values', fontsize=10)
ax7.set_ylabel('Residuals', fontsize=10)
ax7.set_title('Residual Plot (Ridge)\n(Should be random around 0)', fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. Predicted vs Actual
ax8 = plt.subplot(3, 3, 8)
ax8.scatter(y_test, ridge_pred, alpha=0.6, s=50)
ax8.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect prediction')
ax8.set_xlabel('Actual values', fontsize=10)
ax8.set_ylabel('Predicted values', fontsize=10)
ax8.set_title(f'Predictions vs Actual (Ridge)\nR² = {ridge_model.score(X_test_scaled, y_test):.3f}', 
              fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Model comparison heatmap
ax9 = plt.subplot(3, 3, 9)
comparison_data = results_df[['Model', 'Test R²', 'MSE', 'Non-zero Coefs']].set_index('Model')
sns.heatmap(comparison_data.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax9, 
            cbar_kws={'label': 'Value'})
ax9.set_title('Model Performance Heatmap\n(Green = Better)', fontweight='bold')

plt.tight_layout()
plt.savefig('regularization_complete_analysis.png', dpi=150, bbox_inches='tight')
print("\nComprehensive visualization saved as 'regularization_complete_analysis.png'")
plt.show()

print("\n" + "="*80)
print("PART 8: SEABORN - STATISTICAL VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Distribution of coefficients
ax = axes[0, 0]
coef_data = pd.DataFrame({
    'Linear': linear_model.coef_,
    'Ridge': ridge_model.coef_,
    'Lasso': lasso_model.coef_
})
coef_data_melted = coef_data.melt(var_name='Model', value_name='Coefficient')
sns.violinplot(data=coef_data_melted, x='Model', y='Coefficient', ax=ax)
ax.set_title('Coefficient Distribution by Model\n(Ridge pulls toward 0, Lasso zeros out)', 
             fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)

# 2. Pairplot of metrics
ax = axes[0, 1]
metrics_df = results_df[['Train R²', 'Test R²', 'MSE']].copy()
metrics_df['Model Type'] = results_df['Model'].apply(lambda x: x.split('(')[0].strip())
scatter_data = metrics_df[metrics_df['Model Type'].isin(['Ridge', 'Lasso', 'ElasticNet'])]
sns.scatterplot(data=scatter_data, x='Train R²', y='Test R²', 
                hue='Model Type', s=200, ax=ax, alpha=0.7)
ax.plot([0, 1], [0, 1], 'k--', label='No overfitting')
ax.set_title('Train vs Test R²\n(Closer to line = less overfit)', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Regularization strength impact
ax = axes[1, 0]
alpha_analysis = pd.DataFrame({
    'Alpha': alphas,
    'Ridge Test R²': test_scores_ridge,
    'Lasso Test R²': test_scores_lasso
})
ax.plot(alpha_analysis['Alpha'], alpha_analysis['Ridge Test R²'], 
        label='Ridge', linewidth=2, marker='o', markersize=4)
ax.plot(alpha_analysis['Alpha'], alpha_analysis['Lasso Test R²'], 
        label='Lasso', linewidth=2, marker='s', markersize=4)
ax.set_xscale('log')
ax.set_xlabel('Alpha (Regularization Strength)', fontsize=11)
ax.set_ylabel('Test R² Score', fontsize=11)
ax.set_title('Optimal Regularization Strength\n(Peak shows best α value)', 
             fontweight='bold', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Feature importance heatmap
ax = axes[1, 1]
feature_importance = pd.DataFrame({
    'Feature': [f'F{i}' for i in range(n_features)],
    'True': true_weights,
    'Linear': linear_model.coef_,
    'Ridge': ridge_model.coef_,
    'Lasso': lasso_model.coef_
}).set_index('Feature')

sns.heatmap(feature_importance.T, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=ax, cbar_kws={'label': 'Coefficient Value'})
ax.set_title('Feature Importance Comparison\n(Lasso selects important features)', 
             fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('regularization_seaborn_analysis.png', dpi=150, bbox_inches='tight')
print("\nSeaborn visualizations saved as 'regularization_seaborn_analysis.png'")
plt.show()

print("\n" + "="*80)
print("SUMMARY & KEY INSIGHTS")
print("="*80)

print("""
 KEY TAKEAWAYS:

1. RIDGE (L2) REGULARIZATION:
   ✓ Shrinks all coefficients smoothly toward zero
   ✓ Never exactly zeros out features
   ✓ Best when ALL features are potentially relevant
   ✓ Penalty: λ × Σ(w²)

2. LASSO (L1) REGULARIZATION:
   ✓ Forces some coefficients to EXACTLY zero
   ✓ Performs automatic feature selection
   ✓ Best when only SOME features are relevant
   ✓ Penalty: λ × Σ|w|

3. ELASTICNET (L1 + L2):
   ✓ Combines benefits of both
   ✓ Balance controlled by l1_ratio parameter
   ✓ Best for high-dimensional data with correlated features

4. WHEN TO USE WHAT:
   → Many features, all relevant? → Ridge
   → Many features, few relevant? → Lasso
   → High correlation between features? → ElasticNet
   → Not sure? → Try all and compare!

5. HYPERPARAMETER TUNING:
   → Higher α = More regularization = Simpler model
   → Lower α = Less regularization = More complex model
   → Use cross-validation to find optimal α

From this analysis:
""")

best_model = results_df.loc[results_df['Test R²'].idxmax()]
print(f"\nBest performing model: {best_model['Model']}")
print(f"  Test R²: {best_model['Test R²']:.4f}")
print(f"  Features used: {best_model['Non-zero Coefs']}/{n_features}")
print(f"  Overfitting gap: {best_model['Overfit Gap']:.4f}")

print("\nAll visualizations saved! Check the PNG files.")
