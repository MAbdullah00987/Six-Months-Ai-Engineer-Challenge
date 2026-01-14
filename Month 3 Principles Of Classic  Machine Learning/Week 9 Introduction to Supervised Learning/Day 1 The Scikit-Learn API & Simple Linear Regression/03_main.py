
#Part 3: SymPy - Mathematical Foundations of ML

"""
DAY 1 - PART 3: SymPy for ML Mathematics
Understanding the mathematical foundations of linear regression
"""

import sympy as sp
from sympy import symbols, Matrix, diff, simplify, solve, latex
from sympy.stats import Normal, density, E, variance
import numpy as np

print("="*70)
print("SECTION 1: Linear Regression Mathematics")
print("="*70)

# 1.1 Define symbolic variables
x, y, w0, w1, m = symbols('x y w_0 w_1 m')
xi, yi = symbols('x_i y_i')

# Simple linear regression model: y = w0 + w1*x
print("\n1.1 LINEAR MODEL:")
model = w0 + w1 * x
print(f"ŷ = {model}")
print(f"LaTeX: {latex(model)}")

# 1.2 Cost function (Mean Squared Error)
print("\n1.2 COST FUNCTION (MSE):")
prediction = w0 + w1 * xi
error = yi - prediction
squared_error = error**2
mse = squared_error / m  # simplified version for one sample

print(f"Error: e = y - ŷ = {error}")
print(f"Squared Error: {squared_error}")
print(f"MSE (single sample): J = {mse}")

# For multiple samples
n = symbols('n', positive=True, integer=True)
J = sp.Sum((yi - (w0 + w1*xi))**2, (xi, 1, n)) / n
print(f"\nMSE (n samples): J = {J}")

# 1.3 Gradient Descent - Partial Derivatives
print("\n" + "="*70)
print("SECTION 2: Gradient Descent Derivations")
print("="*70)

# Define loss function explicitly
loss = (yi - (w0 + w1*xi))**2

# Compute gradients
grad_w0 = diff(loss, w0)
grad_w1 = diff(loss, w1)

print("\n2.1 PARTIAL DERIVATIVES:")
print(f"∂L/∂w₀ = {grad_w0}")
print(f"∂L/∂w₁ = {grad_w1}")

print("\n2.2 SIMPLIFIED:")
print(f"∂L/∂w₀ = {simplify(grad_w0)}")
print(f"∂L/∂w₁ = {simplify(grad_w1)}")

# 2.3 Gradient descent update rules
alpha = symbols('alpha', positive=True)  # learning rate
print("\n2.3 GRADIENT DESCENT UPDATE RULES:")
print(f"w₀ := w₀ - α · ∂L/∂w₀")
print(f"w₁ := w₁ - α · ∂L/∂w₁")

# Substitute to get explicit update
w0_new = w0 - alpha * grad_w0
w1_new = w1 - alpha * grad_w1
print(f"\nw₀_new = {simplify(w0_new)}")
print(f"w₁_new = {simplify(w1_new)}")

print("\n" + "="*70)
print("SECTION 3: Normal Equation (Closed-Form Solution)")
print("="*70)

# For y = w0 + w1*x, we want to minimize MSE
# Setting derivatives to zero gives normal equation
print("\n3.1 OPTIMAL WEIGHTS (set gradients = 0):")

# Define sum notation
x_mean, y_mean = symbols('bar{x} bar{y}')
xi, yi, n = symbols('x_i y_i n')

# These are the derived formulas
w1_optimal = sp.Sum((xi - x_mean)*(yi - y_mean), (xi, 1, n)) / sp.Sum((xi - x_mean)**2, (xi, 1, n))
w0_optimal = y_mean - w1_optimal * x_mean

print(f"\nw₁* = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²")
print(f"w₀* = ȳ - w₁* · x̄")

# 3.2 Matrix form (multiple features)
print("\n3.2 MATRIX FORM (Multiple Linear Regression):")
print("X = design matrix (n × (d+1)) with bias column")
print("y = target vector (n × 1)")
print("w = weight vector ((d+1) × 1)")
print("\nModel: ŷ = Xw")
print("Normal Equation: w* = (XᵀX)⁻¹Xᵀy")

# Demonstrate with symbolic matrices
n, d = 3, 2  # 3 samples, 2 features
X_sym = Matrix([
    [1, symbols('x_{11}'), symbols('x_{12}')],
    [1, symbols('x_{21}'), symbols('x_{22}')],
    [1, symbols('x_{31}'), symbols('x_{32}')]
])
y_sym = Matrix([symbols('y_1'), symbols('y_2'), symbols('y_3')])

print("\n3.3 EXAMPLE WITH SYMBOLIC MATRICES:")
print(f"X = \n{X_sym}")
print(f"\ny = {y_sym.T}")

# Normal equation components
XtX = X_sym.T @ X_sym
Xty = X_sym.T @ y_sym

print(f"\nXᵀX = \n{XtX}")
print(f"\nXᵀy = {Xty.T}")

print("\n" + "="*70)
print("SECTION 4: Regularization Mathematics")
print("="*70)

# 4.1 Ridge Regression (L2 regularization)
print("\n4.1 RIDGE REGRESSION (L2):")
lambda_param = symbols('lambda', positive=True)
w_vec = symbols('w_1:4')  # w1, w2, w3

# Cost function with L2 penalty
J_ridge = sp.Sum((yi - (w0 + w1*xi))**2, (xi, 1, n)) / n + lambda_param * w1**2
print(f"J_ridge = MSE + λ·w₁²")
print(f"        = {J_ridge}")

# Ridge normal equation
print("\nRidge Normal Equation:")
print("w* = (XᵀX + λI)⁻¹Xᵀy")
print("where I is identity matrix")

# 4.2 Lasso Regression (L1 regularization)
print("\n4.2 LASSO REGRESSION (L1):")
J_lasso = sp.Sum((yi - (w0 + w1*xi))**2, (xi, 1, n)) / n + lambda_param * sp.Abs(w1)
print(f"J_lasso = MSE + λ·|w₁|")
print("Note: L1 promotes sparsity (feature selection)")

print("\n" + "="*70)
print("SECTION 5: Statistical Foundations")
print("="*70)

# 5.1 Assumptions of linear regression
print("\n5.1 KEY ASSUMPTIONS:")
print("1. Linearity: y = Xw + ε")
print("2. Independence: errors εᵢ are independent")
print("3. Homoscedasticity: Var(ε) = σ² (constant)")
print("4. Normality: ε ~ N(0, σ²)")

# 5.2 Confidence intervals
print("\n5.2 CONFIDENCE INTERVALS FOR WEIGHTS:")
print("For weight wⱼ:")
print("wⱼ ± t_{α/2,n-d-1} · SE(wⱼ)")
print("where SE(wⱼ) = σ̂ · sqrt([(XᵀX)⁻¹]ⱼⱼ)")

# 5.3 R-squared derivation
print("\n5.3 R² (Coefficient of Determination):")
y_i, y_pred_i, y_bar = symbols('y_i hat{y}_i bar{y}')
ss_res = sp.Sum((y_i - y_pred_i)**2, (xi, 1, n))
ss_tot = sp.Sum((y_i - y_bar)**2, (xi, 1, n))
r_squared = 1 - ss_res/ss_tot

print(f"SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)")
print(f"SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)")
print(f"R² = 1 - SS_res/SS_tot = {r_squared}")
print("Interpretation: proportion of variance explained")

print("\n" + "="*70)
print("SECTION 6: Numerical Example with Actual Values")
print("="*70)

# Numerical example
X_vals = np.array([1, 2, 3, 4, 5])
y_vals = np.array([2, 4, 5, 4, 5])

# Compute optimal weights using formulas
x_mean_val = np.mean(X_vals)
y_mean_val = np.mean(y_vals)

numerator = np.sum((X_vals - x_mean_val) * (y_vals - y_mean_val))
denominator = np.sum((X_vals - x_mean_val)**2)
w1_val = numerator / denominator
w0_val = y_mean_val - w1_val * x_mean_val

print("\n6.1 DATA:")
print(f"X = {X_vals}")
print(f"y = {y_vals}")

print("\n6.2 COMPUTED VALUES:")
print(f"x̄ = {x_mean_val:.3f}")
print(f"ȳ = {y_mean_val:.3f}")
print(f"Σ(xᵢ - x̄)(yᵢ - ȳ) = {numerator:.3f}")
print(f"Σ(xᵢ - x̄)² = {denominator:.3f}")

print("\n6.3 OPTIMAL WEIGHTS:")
print(f"w₁* = {w1_val:.3f}")
print(f"w₀* = {w0_val:.3f}")
print(f"\nModel: ŷ = {w0_val:.3f} + {w1_val:.3f}x")

# Predictions and R²
y_pred = w0_val + w1_val * X_vals
ss_res_val = np.sum((y_vals - y_pred)**2)
ss_tot_val = np.sum((y_vals - y_mean_val)**2)
r2_val = 1 - ss_res_val/ss_tot_val

print("\n6.4 MODEL QUALITY:")
print(f"SS_res = {ss_res_val:.3f}")
print(f"SS_tot = {ss_tot_val:.3f}")
print(f"R² = {r2_val:.3f}")
print(f"Interpretation: Model explains {r2_val*100:.1f}% of variance")

print("\n" + "="*70)
print("KEY MATHEMATICAL CONCEPTS")
print("="*70)
print("""
✓ CORE FORMULAS DERIVED:
1. Loss function: L = (y - ŷ)²
2. Gradient: ∂L/∂w = -2(y - ŷ)x
3. Update rule: w := w - α·∂L/∂w
4. Normal equation: w* = (XᵀX)⁻¹Xᵀy
5. Ridge penalty: λ·||w||₂²
6. Lasso penalty: λ·||w||₁
7. R² = 1 - SS_res/SS_tot

INSIGHTS:
- Gradient descent iteratively minimizes loss
- Normal equation gives closed-form solution
- Regularization adds penalty to prevent overfitting
- L2 (Ridge) shrinks weights; L1 (Lasso) can zero them
- R² measures goodness of fit (0 to 1)
- These formulas are implemented in sklearn!
""")