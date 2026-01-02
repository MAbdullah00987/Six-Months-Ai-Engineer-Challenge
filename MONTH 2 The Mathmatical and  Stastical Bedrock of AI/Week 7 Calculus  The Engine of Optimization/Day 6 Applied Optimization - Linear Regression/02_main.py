
#Part 2: NumPy Operations - Vectorized Gradient Computation

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*60)
print("TOPIC 2: NUMPY - VECTORIZED GRADIENT COMPUTATION")
print("="*60)

# 1. Generate synthetic linear data
print("\n1. Data Generation")
print("-" * 50)

# True parameters
true_w = 2.5
true_b = 1.0

# Generate data
n_samples = 100
X = np.linspace(0, 10, n_samples)
noise = np.random.randn(n_samples) * 1.5
y = true_w * X + true_b + noise

print(f"Generated {n_samples} data points")
print(f"True parameters: w = {true_w}, b = {true_b}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nFirst 5 data points:")
print(f"X: {X[:5]}")
print(f"y: {y[:5]}")

# 2. Prediction function (vectorized)
print("\n2. Vectorized Prediction")
print("-" * 50)

def predict(X, w, b):
    """Compute predictions for all data points at once"""
    return w * X + b

# Test with initial parameters
w_init, b_init = 1.0, 0.0
y_pred = predict(X, w_init, b_init)

print(f"Initial parameters: w = {w_init}, b = {b_init}")
print(f"Predictions shape: {y_pred.shape}")
print(f"First 5 predictions: {y_pred[:5]}")

# 3. Loss computation (vectorized)
print("\n3. Mean Squared Error Loss")
print("-" * 50)

def mse_loss(y_true, y_pred):
    """Compute MSE loss"""
    errors = y_true - y_pred
    squared_errors = errors ** 2
    mse = np.mean(squared_errors)
    return mse

initial_loss = mse_loss(y, y_pred)
print(f"Initial loss (MSE): {initial_loss:.4f}")

# Step by step breakdown
errors = y - y_pred
squared_errors = errors ** 2
print(f"\nBreakdown:")
print(f"Errors (first 5): {errors[:5]}")
print(f"Squared errors (first 5): {squared_errors[:5]}")
print(f"Mean of squared errors: {np.mean(squared_errors):.4f}")

# 4. Gradient computation (vectorized)
print("\n4. Computing Gradients")
print("-" * 50)

def compute_gradients(X, y, w, b):
    """
    Compute gradients of MSE loss w.r.t. w and b
    
    ∂MSE/∂w = (2/n) * Σ(xᵢ * (ŷᵢ - yᵢ))
    ∂MSE/∂b = (2/n) * Σ(ŷᵢ - yᵢ)
    
    Where ŷᵢ = w*xᵢ + b
    """
    n = len(X)
    
    # Predictions
    y_pred = predict(X, w, b)
    
    # Errors (note: we use y_pred - y, which gives us the right direction)
    errors = y_pred - y
    
    # Gradients
    dw = (2/n) * np.sum(X * errors)
    db = (2/n) * np.sum(errors)
    
    return dw, db

dw, db = compute_gradients(X, y, w_init, b_init)
print(f"Gradients at w={w_init}, b={b_init}:")
print(f"∂MSE/∂w = {dw:.4f}")
print(f"∂MSE/∂b = {db:.4f}")

# 5. Manual gradient verification (loop vs vectorized)
print("\n5. Verification: Loop vs Vectorized")
print("-" * 50)

# Loop version
def compute_gradients_loop(X, y, w, b):
    n = len(X)
    dw = 0
    db = 0
    
    for i in range(n):
        y_pred_i = w * X[i] + b
        error_i = y_pred_i - y[i]
        dw += (2/n) * X[i] * error_i
        db += (2/n) * error_i
    
    return dw, db

import time

# Time loop version
start = time.time()
dw_loop, db_loop = compute_gradients_loop(X, y, w_init, b_init)
time_loop = time.time() - start

# Time vectorized version
start = time.time()
dw_vec, db_vec = compute_gradients(X, y, w_init, b_init)
time_vec = time.time() - start

print(f"Loop version:      dw={dw_loop:.6f}, db={db_loop:.6f}, time={time_loop*1000:.4f}ms")
print(f"Vectorized version: dw={dw_vec:.6f}, db={db_vec:.6f}, time={time_vec*1000:.4f}ms")
print(f"Speedup: {time_loop/time_vec:.1f}x faster")
print(f"Results match: {np.allclose(dw_loop, dw_vec) and np.allclose(db_loop, db_vec)}")

# 6. Gradient descent step
print("\n6. Gradient Descent Update")
print("-" * 50)

def gradient_descent_step(X, y, w, b, learning_rate):
    """Perform one step of gradient descent"""
    dw, db = compute_gradients(X, y, w, b)
    
    # Update parameters in opposite direction of gradient
    w_new = w - learning_rate * dw
    b_new = b - learning_rate * db
    
    return w_new, b_new

learning_rate = 0.01
w_new, b_new = gradient_descent_step(X, y, w_init, b_init, learning_rate)

print(f"Learning rate: {learning_rate}")
print(f"Before: w={w_init:.4f}, b={b_init:.4f}, loss={initial_loss:.4f}")
print(f"After:  w={w_new:.4f}, b={b_new:.4f}, loss={mse_loss(y, predict(X, w_new, b_new)):.4f}")
print(f"Change: Δw={w_new-w_init:.4f}, Δb={b_new-b_init:.4f}")

# 7. Visualization
print("\n7. Visualizing Data and Initial Fit")
print("-" * 50)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Data and predictions
ax = axes[0]
ax.scatter(X, y, alpha=0.5, s=20, label='Data', color='blue')
ax.plot(X, predict(X, w_init, b_init), 'r-', linewidth=2, label=f'Initial (w={w_init}, b={b_init})')
ax.plot(X, predict(X, true_w, true_b), 'g--', linewidth=2, label=f'True (w={true_w}, b={true_b})')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Data and Model', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Residuals
ax = axes[1]
residuals = y - predict(X, w_init, b_init)
ax.scatter(X, residuals, alpha=0.5, s=20, color='red')
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Residuals (y - ŷ)', fontsize=12)
ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Error distribution
ax = axes[2]
ax.hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Residual Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('numpy_vectorization.png', dpi=150, bbox_inches='tight')
plt.show()
