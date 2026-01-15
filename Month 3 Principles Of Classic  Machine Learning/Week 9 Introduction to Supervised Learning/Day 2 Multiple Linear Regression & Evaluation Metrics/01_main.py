
#Day 2: Multiple Linear Regression & Evaluation Metrics
#Objective: Predict a continuous value based on multiple input features.
#Concept: The Normal Equation vs. Gradient Descent, Residuals.
#Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), 
#Root Mean Squared Error (RMSE), $R^2$ Score.
 
#Project 1: California Housing Prediction.
#Note: The "Boston Housing" dataset is deprecated in Scikit-Learn due to ethical issues. 
#Use sklearn.datasets.fetch_california_housing instead.
#Predict median house values based on 8 different features (income, age, rooms, etc.).Calculate RMSE to see how far off your predictions are in dollars.

#P#roject 2: Salary Prediction
# Dataset: Create or find a salary vs experience dataset
#Build your first linear regression model
#Visualize the regression line
#Calculate R² score and MSE

#Part 1. Mathematical Foundation with SymPy

import sympy as sp
import numpy as np
from sympy import symbols, Matrix, transpose, latex, simplify

print("="*60)
print("MULTIPLE LINEAR REGRESSION - MATHEMATICAL FOUNDATIONS")
print("="*60)

# 1. SYMBOLIC REPRESENTATION OF LINEAR REGRESSION
print("\n1. LINEAR REGRESSION MODEL")
print("-" * 60)

# Define symbols
x1, x2, theta0, theta1, theta2, y = symbols('x_1 x_2 theta_0 theta_1 theta_2 y')

# Hypothesis function
h = theta0 + theta1*x1 + theta2*x2
print(f"Hypothesis: h(x) = {h}")

# Matrix form
print("\nMatrix Form: y = Xθ")
X = Matrix([[1, symbols('x_1^{(1)}'), symbols('x_2^{(1)}')],
            [1, symbols('x_1^{(2)}'), symbols('x_2^{(2)}')],
            [1, symbols('x_1^{(3)}'), symbols('x_2^{(3)}')]])
theta = Matrix([theta0, theta1, theta2])
print(f"\nX shape: (m, n+1) where m=samples, n=features")
print(f"θ shape: (n+1, 1)")

# 2. COST FUNCTION (MSE)
print("\n\n2. COST FUNCTION - Mean Squared Error")
print("-" * 60)

m = symbols('m', positive=True, integer=True)
y_pred, y_true = symbols('y_pred y_true')
residual = y_pred - y_true

print(f"Residual: ε = {residual}")
print(f"Squared Error: ε² = {residual**2}")

# MSE formula
print("\nMean Squared Error:")
print("J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²")

# 3. NORMAL EQUATION DERIVATION
print("\n\n3. NORMAL EQUATION (Closed-Form Solution)")
print("-" * 60)

print("To minimize J(θ), we set ∂J/∂θ = 0")
print("\nDerivation:")
print("J(θ) = (1/2m)(Xθ - y)ᵀ(Xθ - y)")
print("∂J/∂θ = (1/m)Xᵀ(Xθ - y) = 0")
print("\nSolving for θ:")
print("XᵀXθ = Xᵀy")
print("θ = (XᵀX)⁻¹Xᵀy  ← NORMAL EQUATION")

# Numerical example with actual matrices
print("\n\n4. CONCRETE EXAMPLE")
print("-" * 60)

# Sample data: 3 samples, 2 features
X_concrete = np.array([
    [1, 1, 2],  # [1, x1, x2] for sample 1
    [1, 2, 3],  # [1, x1, x2] for sample 2
    [1, 3, 4]   # [1, x1, x2] for sample 3
])
y_concrete = np.array([[5], [7], [9]])

print("Given Data:")
print(f"X = \n{X_concrete}")
print(f"\ny = \n{y_concrete}")

# Calculate Normal Equation
XtX = X_concrete.T @ X_concrete
Xty = X_concrete.T @ y_concrete
XtX_inv = np.linalg.inv(XtX)
theta_solution = XtX_inv @ Xty

print("\nCalculations:")
print(f"XᵀX = \n{XtX}")
print(f"\n(XᵀX)⁻¹ = \n{XtX_inv}")
print(f"\nXᵀy = \n{Xty}")
print(f"\nθ = (XᵀX)⁻¹Xᵀy = \n{theta_solution}")

print(f"\nSolution: θ₀={theta_solution[0,0]:.4f}, θ₁={theta_solution[1,0]:.4f}, θ₂={theta_solution[2,0]:.4f}")

# 5. GRADIENT DESCENT - SYMBOLIC DERIVATIVES
print("\n\n5. GRADIENT DESCENT - ANALYTICAL GRADIENTS")
print("-" * 60)

# Define cost function symbolically
h_sym = theta0 + theta1*x1 + theta2*x2
J = (h_sym - y)**2 / 2

print("Cost for one sample: J = (h(x) - y)²/2")
print(f"J = {J}")

# Compute partial derivatives
dJ_dtheta0 = sp.diff(J, theta0)
dJ_dtheta1 = sp.diff(J, theta1)
dJ_dtheta2 = sp.diff(J, theta2)

print("\nPartial Derivatives:")
print(f"∂J/∂θ₀ = {dJ_dtheta0}")
print(f"∂J/∂θ₁ = {dJ_dtheta1}")
print(f"∂J/∂θ₂ = {dJ_dtheta2}")

print("\nGradient Descent Update Rule:")
alpha = symbols('alpha', positive=True)
print(f"θⱼ := θⱼ - α(∂J/∂θⱼ)")
print(f"\nFor our example:")
print(f"θ₀ := θ₀ - α·{dJ_dtheta0}")
print(f"θ₁ := θ₁ - α·{dJ_dtheta1}")
print(f"θ₂ := θ₂ - α·{dJ_dtheta2}")

# 6. EVALUATION METRICS - SYMBOLIC
print("\n\n6. EVALUATION METRICS")
print("-" * 60)

n = symbols('n', positive=True, integer=True)
y_i, y_pred_i, y_mean = symbols('y_i y_pred_i bar{y}')

print("Mean Absolute Error (MAE):")
print("MAE = (1/n)Σ|yᵢ - ŷᵢ|")

print("\nMean Squared Error (MSE):")
print("MSE = (1/n)Σ(yᵢ - ŷᵢ)²")

print("\nRoot Mean Squared Error (RMSE):")
print("RMSE = √MSE = √[(1/n)Σ(yᵢ - ŷᵢ)²]")

print("\nR² Score (Coefficient of Determination):")
print("R² = 1 - (SS_res / SS_tot)")
print("where:")
print("  SS_res = Σ(yᵢ - ŷᵢ)²  (Residual Sum of Squares)")
print("  SS_tot = Σ(yᵢ - ȳ)²   (Total Sum of Squares)")

SS_res = (y_i - y_pred_i)**2
SS_tot = (y_i - y_mean)**2
R2 = 1 - (SS_res / SS_tot)

print(f"\nR² ∈ [0, 1] for typical cases")
print(f"R² = 1: Perfect fit")
print(f"R² = 0: Model as good as mean baseline")

print("Mathematical foundation complete!")
