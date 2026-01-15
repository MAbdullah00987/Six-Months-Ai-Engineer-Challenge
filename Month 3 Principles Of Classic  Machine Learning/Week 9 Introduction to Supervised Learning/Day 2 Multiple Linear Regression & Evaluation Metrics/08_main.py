
#Part 8. Interactive Learning Notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("="*80)
print("LINEAR REGRESSION - PRACTICE EXERCISES")
print("="*80)

# ============================================================================
# EXERCISE 1: Implement Normal Equation from Scratch
# ============================================================================
print("\n" + "="*80)
print("EXERCISE 1: Implement Normal Equation")
print("="*80)
print("\nTask: Implement θ = (X'X)⁻¹X'y")

# Given data
X_ex1 = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])  # Already has intercept column
y_ex1 = np.array([[5], [7], [9], [11]])

print("\nGiven:")
print(f"X = \n{X_ex1}")
print(f"y = \n{y_ex1}")

def normal_equation(X, y):
    """
    TODO: Implement the normal equation
    Hint: Use np.linalg.inv() for matrix inverse and @ for matrix multiplication
    """
    # YOUR CODE HERE
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

# Test your implementation
theta_ex1 = normal_equation(X_ex1, y_ex1)
print(f"\nYour solution: θ = \n{theta_ex1}")
print(f"Expected: θ₀ = 1.0, θ₁ = 2.0")

# Verify
predictions_ex1 = X_ex1 @ theta_ex1
print(f"\nPredictions: {predictions_ex1.T}")
print(f"Actual: {y_ex1.T}")
print(f"Perfect match: {np.allclose(predictions_ex1, y_ex1)}")

# ============================================================================
# EXERCISE 2: Implement Gradient Descent
# ============================================================================
print("\n\n" + "="*80)
print("EXERCISE 2: Implement Gradient Descent")
print("="*80)
print("\nTask: Implement gradient descent from scratch")

# Generate data
np.random.seed(42)
X_ex2 = np.random.randn(50, 1)
y_ex2 = 4 + 3 * X_ex2 + np.random.randn(50, 1) * 0.5

X_ex2_b = np.c_[np.ones((50, 1)), X_ex2]  # Add intercept

def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    """
    TODO: Implement gradient descent
    
    Steps:
    1. Initialize theta with zeros
    2. For each iteration:
       a. Compute predictions: h = Xθ
       b. Compute errors: e = h - y
       c. Compute gradient: grad = (1/m)X'e
       d. Update theta: θ = θ - α*grad
       e. Compute cost: J = (1/2m)Σ(e²)
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    
    # YOUR CODE HERE
    for i in range(n_iterations):
        # Compute predictions
        predictions = X @ theta
        
        # Compute errors
        errors = predictions - y
        
        # Compute gradient
        gradient = (1/m) * X.T @ errors
        
        # Update parameters
        theta = theta - learning_rate * gradient
        
        # Compute cost
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)
    
    return theta, cost_history

theta_ex2, costs_ex2 = gradient_descent(X_ex2_b, y_ex2)

print(f"\nLearned parameters:")
print(f"θ₀ (intercept) = {theta_ex2[0, 0]:.4f}")
print(f"θ₁ (slope) = {theta_ex2[1, 0]:.4f}")
print(f"\nExpected: θ₀ ≈ 4, θ₁ ≈ 3")

print(f"\nFinal cost: {costs_ex2[-1]:.6f}")
print(f"Cost decreased: {costs_ex2[0] > costs_ex2[-1]}")

# Visualize convergence
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(costs_ex2)
plt.xlabel('Iteration')
plt.ylabel('Cost J(θ)')
plt.title('Gradient Descent Convergence')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_ex2, y_ex2, alpha=0.5)
plt.plot(X_ex2, X_ex2_b @ theta_ex2, 'r-', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Fitted Line')
plt.grid(True)
plt.tight_layout()
plt.savefig('exercise2_gradient_descent.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved: exercise2_gradient_descent.png")

# ============================================================================
# EXERCISE 3: Compute Evaluation Metrics
# ============================================================================
print("\n\n" + "="*80)
print("EXERCISE 3: Implement Evaluation Metrics")
print("="*80)

y_true_ex3 = np.array([3, -0.5, 2, 7])
y_pred_ex3 = np.array([2.5, 0.0, 2, 8])

print("\nGiven:")
print(f"y_true = {y_true_ex3}")
print(f"y_pred = {y_pred_ex3}")

def compute_mae(y_true, y_pred):
    """TODO: Implement Mean Absolute Error"""
    # YOUR CODE HERE
    return np.mean(np.abs(y_true - y_pred))

def compute_mse(y_true, y_pred):
    """TODO: Implement Mean Squared Error"""
    # YOUR CODE HERE
    return np.mean((y_true - y_pred)**2)

def compute_rmse(y_true, y_pred):
    """TODO: Implement Root Mean Squared Error"""
    # YOUR CODE HERE
    return np.sqrt(compute_mse(y_true, y_pred))

def compute_r2(y_true, y_pred):
    """TODO: Implement R² Score"""
    # YOUR CODE HERE
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

mae_ex3 = compute_mae(y_true_ex3, y_pred_ex3)
mse_ex3 = compute_mse(y_true_ex3, y_pred_ex3)
rmse_ex3 = compute_rmse(y_true_ex3, y_pred_ex3)
r2_ex3 = compute_r2(y_true_ex3, y_pred_ex3)

print(f"\nYour results:")
print(f"MAE:  {mae_ex3:.4f}")
print(f"MSE:  {mse_ex3:.4f}")
print(f"RMSE: {rmse_ex3:.4f}")
print(f"R²:   {r2_ex3:.4f}")

print(f"\nExpected:")
print(f"MAE:  0.5000")
print(f"MSE:  0.3750")
print(f"RMSE: 0.6124")
print(f"R²:   0.9486")

# ============================================================================
# EXERCISE 4: Feature Engineering
# ============================================================================
print("\n\n" + "="*80)
print("EXERCISE 4: Feature Engineering")
print("="*80)
print("\nTask: Create polynomial features and train model")

# Simple dataset
X_ex4 = np.linspace(-3, 3, 100).reshape(-1, 1)
y_ex4 = 0.5 * X_ex4**2 + X_ex4 + 2 + np.random.randn(100, 1) * 0.5

def create_polynomial_features(X, degree):
    """
    TODO: Create polynomial features up to given degree
    Example: X = [x] → [1, x, x², x³, ...] for degree=3
    """
    # YOUR CODE HERE
    X_poly = np.ones((X.shape[0], 1))  # Start with intercept
    for d in range(1, degree + 1):
        X_poly = np.c_[X_poly, X**d]
    return X_poly

# Test with degree 2
X_ex4_poly = create_polynomial_features(X_ex4, degree=2)

print(f"\nOriginal features shape: {X_ex4.shape}")
print(f"Polynomial features shape: {X_ex4_poly.shape}")
print(f"Expected shape: (100, 3) for [1, x, x²]")

# Train model
theta_ex4 = np.linalg.inv(X_ex4_poly.T @ X_ex4_poly) @ X_ex4_poly.T @ y_ex4
y_ex4_pred = X_ex4_poly @ theta_ex4

r2_ex4 = compute_r2(y_ex4, y_ex4_pred)
print(f"\nR² Score with polynomial features: {r2_ex4:.4f}")

# Compare with linear
X_ex4_linear = np.c_[np.ones((100, 1)), X_ex4]
theta_ex4_linear = np.linalg.inv(X_ex4_linear.T @ X_ex4_linear) @ X_ex4_linear.T @ y_ex4
y_ex4_pred_linear = X_ex4_linear @ theta_ex4_linear
r2_ex4_linear = compute_r2(y_ex4, y_ex4_pred_linear)

print(f"R² Score with linear features: {r2_ex4_linear:.4f}")
print(f"\nImprovement: {(r2_ex4 - r2_ex4_linear):.4f}")

# ============================================================================
# EXERCISE 5: Multi-Feature Linear Regression
# ============================================================================
print("\n\n" + "="*80)
print("EXERCISE 5: Multi-Feature Linear Regression with Sklearn")
print("="*80)

# Create dataset
np.random.seed(42)
n = 100
X_ex5 = pd.DataFrame({
    'bedrooms': np.random.randint(1, 6, n),
    'square_feet': np.random.randint(500, 3000, n),
    'age': np.random.randint(0, 50, n)
})
y_ex5 = (50000 + 
         20000 * X_ex5['bedrooms'] + 
         100 * X_ex5['square_feet'] - 
         1000 * X_ex5['age'] + 
         np.random.randn(n) * 10000)

print("\nTask: Use scikit-learn to train and evaluate a model")
print(f"\nDataset: {n} houses with {X_ex5.shape[1]} features")

# TODO: Your code here
# 1. Split data into train/test
# 2. Create and train LinearRegression model
# 3. Make predictions
# 4. Compute metrics

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_ex5, y_ex5, test_size=0.2, random_state=42)

model_ex5 = LinearRegression()
model_ex5.fit(X_train, y_train)

y_pred_ex5 = model_ex5.predict(X_test)

print(f"\nModel coefficients:")
for feature, coef in zip(X_ex5.columns, model_ex5.coef_):
    print(f"  {feature}: {coef:.2f}")
print(f"  Intercept: {model_ex5.intercept_:.2f}")

print(f"\nEvaluation on test set:")
print(f"  R² Score: {r2_score(y_test, y_pred_ex5):.4f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_ex5)):,.2f}")

# ============================================================================
# EXERCISE 6: Residual Analysis
# ============================================================================
print("\n\n" + "="*80)
print("EXERCISE 6: Residual Analysis")
print("="*80)

residuals_ex5 = y_test - y_pred_ex5

print("\nTask: Analyze residuals to check model assumptions")

# TODO: Compute residual statistics
print(f"\nResidual statistics:")
print(f"  Mean: {residuals_ex5.mean():.4f} (should be ≈ 0)")
print(f"  Std Dev: {residuals_ex5.std():.2f}")
print(f"  Min: {residuals_ex5.min():.2f}")
print(f"  Max: {residuals_ex5.max():.2f}")

# Create residual plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residual plot
axes[0].scatter(y_pred_ex5, residuals_ex5, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residual Plot')
axes[0].grid(True)

# Histogram
axes[1].hist(residuals_ex5, bins=20, alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--')
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('exercise6_residuals.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved: exercise6_residuals.png")

# ============================================================================
# CHALLENGE EXERCISE: Custom Loss Function
# ============================================================================
print("\n\n" + "="*80)
print("CHALLENGE: Implement Huber Loss")
print("="*80)
print("\nHuber loss is less sensitive to outliers than MSE")

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss:
    - Use MSE for small errors (|e| <= δ)
    - Use MAE for large errors (|e| > δ)
    
    Formula:
    L(e) = 0.5 * e² if |e| <= δ
    L(e) = δ * (|e| - 0.5*δ) if |e| > δ
    """
    # YOUR CODE HERE
    errors = y_true - y_pred
    is_small = np.abs(errors) <= delta
    
    loss = np.where(
        is_small,
        0.5 * errors**2,
        delta * (np.abs(errors) - 0.5 * delta)
    )
    
    return np.mean(loss)

# Test with outliers
y_true_challenge = np.array([1, 2, 3, 100])  # Last value is outlier
y_pred_challenge = np.array([1.1, 2.1, 2.9, 3])

mse_challenge = compute_mse(y_true_challenge, y_pred_challenge)
huber_challenge = huber_loss(y_true_challenge, y_pred_challenge, delta=1.0)

print(f"\nWith outlier (y_true=[1, 2, 3, 100], y_pred=[1.1, 2.1, 2.9, 3]):")
print(f"  MSE (sensitive to outliers): {mse_challenge:.2f}")
print(f"  Huber Loss (robust): {huber_challenge:.2f}")
print(f"\nHuber loss is {mse_challenge/huber_challenge:.2f}x smaller!")

# ============================================================================
# SOLUTIONS VERIFICATION
# ============================================================================
print("\n\n" + "="*80)
print("SOLUTIONS SUMMARY")
print("="*80)

results_summary = {
    'Exercise 1': '✓ Normal Equation implemented',
    'Exercise 2': '✓ Gradient Descent implemented',
    'Exercise 3': '✓ Metrics computed correctly',
    'Exercise 4': '✓ Polynomial features created',
    'Exercise 5': '✓ Sklearn model trained',
    'Exercise 6': '✓ Residual analysis completed',
    'Challenge': '✓ Huber Loss implemented'
}

for exercise, status in results_summary.items():
    print(f"{exercise:20s}: {status}")

print("CONGRATULATIONS! You've completed all exercises!")
print("\nNext steps:")
print("  1. Try these exercises with different datasets")
print("  2. Experiment with different learning rates")
print("  3. Add regularization (Ridge, Lasso)")
print("  4. Try polynomial features of different degrees")
print("  5. Build a real-world prediction system")
