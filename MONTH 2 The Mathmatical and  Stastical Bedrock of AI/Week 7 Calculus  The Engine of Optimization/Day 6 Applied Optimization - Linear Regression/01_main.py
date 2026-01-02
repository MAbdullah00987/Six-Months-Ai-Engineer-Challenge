#Day 6: Applied Optimization - Linear Regression
#Focus: Using calculus for real-world machine learning problems
#Learning Activities (2 hours)
#Review: Linear regression theory and loss functions
#Study: Deriving the gradient of mean squared error
#Read: Connection between calculus and machine learning

#Project 1: Linear Regression with Gradient Descent (2.5 hours)
#Generate synthetic 2D data (y = 3x + 2 + noise)
#Implement gradient descent to find best-fit line
#Use MSE loss function: L = (1/n)Σ(y_pred - y_true)²
#Derive and implement gradients for slope and intercept
#Visualize:

#Data points and evolving fit line
#Loss over iterations
#Parameter convergence



#Project 2: Stochastic Gradient Descent (SGD) (1.5 hours)
#Implement basic SGD (update with single random sample per iteration)
#Compare with batch gradient descent on same dataset
#Visualize convergence paths side by side
#Analyze trade-offs (speed vs. stability)

#Deliverable: Two implementations with comparative analysis and visualizations


#Applied Optimization - Linear Regression: Complete Guide.
#Learning Structure
#Mathematical Foundation (SymPy for calculus)
#Data Manipulation (NumPy, Pandas)
#Visualization (Matplotlib, Seaborn)
#Animation (Manim for understanding concepts)
#Implementation (Building from scratch)

#Part 1: Mathematical Foundation - Derivatives with SymPy

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables
w, b, x, y = sp.symbols('w b x y')

print("="*60)
print("TOPIC 1: DERIVATIVES - THE OPTIMIZATION FOUNDATION")
print("="*60)

# 1. Simple function and its derivative
print("\n1. Understanding Derivatives Geometrically")
print("-" * 50)

# Define a simple quadratic function
f = w**2 - 4*w + 5
print(f"Function: f(w) = {f}")

# Calculate derivative
df = sp.diff(f, w)
print(f"Derivative: f'(w) = {df}")

# Find minimum (where derivative = 0)
critical_points = sp.solve(df, w)
print(f"Critical point (minimum): w = {critical_points}")

# Evaluate at minimum
min_value = f.subs(w, critical_points[0])
print(f"Minimum value: f({critical_points[0]}) = {min_value}")

# 2. Linear Regression Loss Function
print("\n2. Mean Squared Error Loss Function")
print("-" * 50)

# MSE for single data point: L = (y - (wx + b))²
prediction = w*x + b
error = y - prediction
loss = error**2

print(f"Prediction: ŷ = {prediction}")
print(f"Error: e = y - ŷ = {error}")
print(f"Loss (squared error): L = {loss}")

# 3. Partial Derivatives - The Gradient
print("\n3. Computing Gradients (Partial Derivatives)")
print("-" * 50)

# Expand the loss
loss_expanded = sp.expand(loss)
print(f"Expanded Loss: L = {loss_expanded}")

# Compute partial derivatives
dL_dw = sp.diff(loss, w)
dL_db = sp.diff(loss, b)

print(f"\n∂L/∂w = {dL_dw}")
print(f"∂L/∂b = {dL_db}")

# Simplify
dL_dw_simplified = sp.simplify(dL_dw)
dL_db_simplified = sp.simplify(dL_db)

print(f"\nSimplified:")
print(f"∂L/∂w = {dL_dw_simplified}")
print(f"∂L/∂b = {dL_db_simplified}")

# 4. Average over multiple data points (MSE)
print("\n4. Mean Squared Error for Multiple Points")
print("-" * 50)

# For n data points: MSE = (1/n) * Σ(yᵢ - (w*xᵢ + b))²
n = sp.Symbol('n', positive=True)
i = sp.Symbol('i', integer=True)

print("MSE = (1/n) * Σ(yᵢ - (w*xᵢ + b))²")
print("\nFor a single point, gradient is:")
print(f"∂L/∂w = {dL_dw_simplified}")
print(f"∂L/∂b = {dL_db_simplified}")

print("\nFor n points, we sum and average:")
print("∂MSE/∂w = (2/n) * Σ(xᵢ * (w*xᵢ + b - yᵢ))")
print("∂MSE/∂b = (2/n) * Σ(w*xᵢ + b - yᵢ)")

# 5. Numerical Example
print("\n5. Numerical Example: Computing Gradient")
print("-" * 50)

# Sample data point
x_val, y_val = 2.0, 5.0
w_val, b_val = 1.0, 0.5

# Compute gradient numerically
grad_w = dL_dw.subs([(x, x_val), (y, y_val), (w, w_val), (b, b_val)])
grad_b = dL_db.subs([(x, x_val), (y, y_val), (w, w_val), (b, b_val)])

prediction_val = w_val * x_val + b_val
error_val = y_val - prediction_val

print(f"Data point: (x={x_val}, y={y_val})")
print(f"Parameters: w={w_val}, b={b_val}")
print(f"Prediction: ŷ = {prediction_val}")
print(f"Error: {error_val}")
print(f"Gradient: ∂L/∂w = {grad_w}, ∂L/∂b = {grad_b}")

# 6. Visualization
print("\n6. Visualizing Loss Function")
print("-" * 50)

# Create a range of w values
w_values = np.linspace(-2, 6, 100)

# Fix x=2, y=5, b=0 and plot loss as function of w
x_fixed, y_fixed, b_fixed = 2.0, 5.0, 0.0
loss_func = sp.lambdify(w, loss.subs([(x, x_fixed), (y, y_fixed), (b, b_fixed)]))
loss_values = [loss_func(w_val) for w_val in w_values]

# Find optimal w analytically
w_optimal = y_fixed / x_fixed  # Since b=0, optimal is w = y/x

plt.figure(figsize=(12, 4))

# Plot 1: Loss function
plt.subplot(1, 2, 1)
plt.plot(w_values, loss_values, 'b-', linewidth=2, label='Loss L(w)')
plt.axvline(w_optimal, color='r', linestyle='--', label=f'Optimal w = {w_optimal:.2f}')
plt.scatter([w_optimal], [loss_func(w_optimal)], color='r', s=100, zorder=5)
plt.xlabel('w (weight)', fontsize=12)
plt.ylabel('Loss L(w)', fontsize=12)
plt.title('Loss Function vs Weight', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Derivative (gradient)
deriv_func = sp.lambdify(w, dL_dw.subs([(x, x_fixed), (y, y_fixed), (b, b_fixed)]))
deriv_values = [float(deriv_func(w_val)) for w_val in w_values]

plt.subplot(1, 2, 2)
plt.plot(w_values, deriv_values, 'g-', linewidth=2, label="∂L/∂w")
plt.axhline(0, color='k', linestyle='-', alpha=0.3)
plt.axvline(w_optimal, color='r', linestyle='--', label=f'Optimal w = {w_optimal:.2f}')
plt.scatter([w_optimal], [0], color='r', s=100, zorder=5)
plt.xlabel('w (weight)', fontsize=12)
plt.ylabel('∂L/∂w (gradient)', fontsize=12)
plt.title('Gradient (Derivative) vs Weight', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('derivatives_foundation.png', dpi=150, bbox_inches='tight')
plt.show()

