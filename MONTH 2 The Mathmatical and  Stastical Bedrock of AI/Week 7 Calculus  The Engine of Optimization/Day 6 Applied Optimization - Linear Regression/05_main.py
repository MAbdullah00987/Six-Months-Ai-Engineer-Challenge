
#Part 4: Mathematical Foundations - Complete Derivations

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, diff, simplify, expand, Matrix, latex

# Enable pretty printing
sp.init_printing(use_unicode=True)

print("="*70)
print("MATHEMATICAL FOUNDATIONS - COMPLETE DERIVATIONS")
print("="*70)

# ===== PART 1: SINGLE VARIABLE LINEAR REGRESSION =====
print("\n" + "="*70)
print("PART 1: SINGLE VARIABLE LINEAR REGRESSION")
print("="*70)

print("\n1.1 Model Definition")
print("-" * 70)

# Define symbols
w, b, x, y = symbols('w b x y', real=True)
n = symbols('n', positive=True, integer=True)

# Model
y_pred = w * x + b

print("Linear Model:")
print(f"  ŷ = {y_pred}")
print(f"\nWhere:")
print(f"  w = weight (slope)")
print(f"  b = bias (intercept)")
print(f"  x = input feature")
print(f"  ŷ = predicted output")

print("\n1.2 Loss Function (Single Data Point)")
print("-" * 70)

# Single point loss
error = y - y_pred
loss_single = error**2

print(f"Error: e = y - ŷ = {error}")
print(f"Squared Error: L = e² = {expand(loss_single)}")

print("\n1.3 Computing Gradients")
print("-" * 70)

# Partial derivatives
dL_dw = diff(loss_single, w)
dL_db = diff(loss_single, b)

print(f"∂L/∂w = {dL_dw}")
print(f"∂L/∂b = {dL_db}")

# Simplified
dL_dw_simplified = simplify(dL_dw)
dL_db_simplified = simplify(dL_db)

print(f"\nSimplified:")
print(f"∂L/∂w = {dL_dw_simplified}")
print(f"∂L/∂b = {dL_db_simplified}")

print("\n1.4 Mean Squared Error (Multiple Points)")
print("-" * 70)

# For multiple points, we sum and average
print("For n data points: (x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)")
print("\nMSE = (1/n) Σᵢ (yᵢ - (w·xᵢ + b))²")
print("\nGradients:")
print("∂MSE/∂w = (1/n) Σᵢ ∂Lᵢ/∂w")
print("         = (1/n) Σᵢ 2(w·xᵢ + b - yᵢ)·xᵢ")
print("         = (2/n) Σᵢ xᵢ·(w·xᵢ + b - yᵢ)")
print("\n∂MSE/∂b = (2/n) Σᵢ (w·xᵢ + b - yᵢ)")

# ===== PART 2: MULTIPLE LINEAR REGRESSION =====
print("\n" + "="*70)
print("PART 2: MULTIPLE LINEAR REGRESSION (VECTOR FORM)")
print("="*70)

print("\n2.1 Vector Notation")
print("-" * 70)

print("For m features:")
print("  X = [x₁, x₂, ..., xₘ]ᵀ  (feature vector)")
print("  w = [w₁, w₂, ..., wₘ]ᵀ  (weight vector)")
print("  ŷ = wᵀX + b = Σⱼ wⱼxⱼ + b")

print("\n2.2 Matrix Form (n samples, m features)")
print("-" * 70)

print("Design Matrix:")
print("     ┌                    ┐")
print("     │ x₁₁  x₁₂  ...  x₁ₘ │")
print("  X =│ x₂₁  x₂₂  ...  x₂ₘ │  shape: (n, m)")
print("     │  ⋮    ⋮    ⋱    ⋮  │")
print("     │ xₙ₁  xₙ₂  ...  xₙₘ │")
print("     └                    ┘")
print("\nPredictions: ŷ = Xw + b·1  (vectorized)")

print("\n2.3 MSE Loss in Matrix Form")
print("-" * 70)

print("MSE = (1/n) ||y - ŷ||² = (1/n) ||y - Xw - b·1||²")
print("    = (1/n) (y - Xw - b·1)ᵀ(y - Xw - b·1)")

print("\n2.4 Gradient Derivation (Matrix Calculus)")
print("-" * 70)

print("Using matrix calculus rules:")
print("\n∂MSE/∂w = (2/n) Xᵀ(Xw + b·1 - y)")
print("∂MSE/∂b = (2/n) 1ᵀ(Xw + b·1 - y) = (2/n) Σᵢ(ŷᵢ - yᵢ)")

# Demonstrate with symbolic 2x2 example
print("\n2.5 Concrete Example (2 features, 3 samples)")
print("-" * 70)

w1, w2 = symbols('w1 w2', real=True)
x11, x12, x21, x22, x31, x32 = symbols('x11 x12 x21 x22 x31 x32', real=True)
y1_sym, y2_sym, y3_sym = symbols('y1 y2 y3', real=True)

# Single prediction
pred1 = w1*x11 + w2*x12 + b
pred2 = w1*x21 + w2*x22 + b
pred3 = w1*x31 + w2*x32 + b

# MSE
mse = ((y1_sym - pred1)**2 + (y2_sym - pred2)**2 + (y3_sym - pred3)**2) / 3

print(f"Predictions:")
print(f"  ŷ₁ = {pred1}")
print(f"  ŷ₂ = {pred2}")
print(f"  ŷ₃ = {pred3}")

print(f"\nMSE = {mse}")

# Compute gradient w.r.t. w1
grad_w1 = diff(mse, w1)
grad_w1_simplified = simplify(grad_w1)

print(f"\n∂MSE/∂w₁ = {grad_w1_simplified}")

# ===== PART 3: GRADIENT DESCENT UPDATE RULES =====
print("\n" + "="*70)
print("PART 3: GRADIENT DESCENT UPDATE RULES")
print("="*70)

print("\n3.1 Basic Gradient Descent")
print("-" * 70)

alpha = symbols('alpha', positive=True)
t = symbols('t', integer=True)

print("Update rule:")
print(f"  w⁽ᵗ⁺¹⁾ = w⁽ᵗ⁾ - α · ∂MSE/∂w")
print(f"  b⁽ᵗ⁺¹⁾ = b⁽ᵗ⁾ - α · ∂MSE/∂b")
print(f"\nWhere α is the learning rate")

print("\n3.2 Gradient Descent with Momentum")
print("-" * 70)

beta = symbols('beta', real=True)
v_w, v_b = symbols('v_w v_b', real=True)

print("Velocity update:")
print(f"  v_w⁽ᵗ⁺¹⁾ = β·v_w⁽ᵗ⁾ + (1-β)·∂MSE/∂w")
print(f"  v_b⁽ᵗ⁺¹⁾ = β·v_b⁽ᵗ⁾ + (1-β)·∂MSE/∂b")
print(f"\nParameter update:")
print(f"  w⁽ᵗ⁺¹⁾ = w⁽ᵗ⁾ - α·v_w⁽ᵗ⁺¹⁾")
print(f"  b⁽ᵗ⁺¹⁾ = b⁽ᵗ⁾ - α·v_b⁽ᵗ⁺¹⁾")

print("\n3.3 Adam Optimizer")
print("-" * 70)

beta1, beta2, epsilon = symbols('beta1 beta2 epsilon', positive=True)
m_w, v_w_adam = symbols('m_w v_w', real=True)

print("First moment (mean):")
print(f"  m_w⁽ᵗ⁺¹⁾ = β₁·m_w⁽ᵗ⁾ + (1-β₁)·∂MSE/∂w")
print(f"\nSecond moment (variance):")
print(f"  v_w⁽ᵗ⁺¹⁾ = β₂·v_w⁽ᵗ⁾ + (1-β₂)·(∂MSE/∂w)²")
print(f"\nBias correction:")
print(f"  m̂_w = m_w⁽ᵗ⁺¹⁾ / (1 - β₁ᵗ)")
print(f"  v̂_w = v_w⁽ᵗ⁺¹⁾ / (1 - β₂ᵗ)")
print(f"\nParameter update:")
print(f"  w⁽ᵗ⁺¹⁾ = w⁽ᵗ⁾ - α · m̂_w / (√v̂_w + ε)")

# ===== PART 4: ANALYTICAL SOLUTION (NORMAL EQUATIONS) =====
print("\n" + "="*70)
print("PART 4: ANALYTICAL SOLUTION (NORMAL EQUATIONS)")
print("="*70)

print("\n4.1 Closed-Form Solution")
print("-" * 70)

print("To minimize MSE, set gradients to zero:")
print("  ∇MSE = 0")
print("\nFor linear regression:")
print("  ∂MSE/∂w = 0  →  Xᵀ(Xw - y) = 0")
print("  →  XᵀXw = Xᵀy")
print("  →  w = (XᵀX)⁻¹Xᵀy")
print("\nThis is the Normal Equation (analytical solution)")

print("\n4.2 When to Use Gradient Descent vs Normal Equation")
print("-" * 70)
print("Normal Equation:")
print("  Pros: Exact solution, no hyperparameters")
print("  Cons: O(n³) complexity, needs matrix inversion")
print("  Use when: n < 10,000 features")
print("\nGradient Descent:")
print("  Pros: O(kn²) complexity, works for large n")
print("  Cons: Needs tuning, iterative")
print("  Use when: n > 10,000 features or online learning")

# ===== PART 5: NUMERICAL EXAMPLE =====
print("\n" + "="*70)
print("PART 5: NUMERICAL VERIFICATION")
print("="*70)

print("\n5.1 Small Dataset Example")
print("-" * 70)

# Create small dataset
X_data = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features
y_data = np.array([7, 11, 15])  # targets
w_true = np.array([1, 2])  # True weights
b_true = 2  # True bias

print("Data:")
print(f"X = \n{X_data}")
print(f"y = {y_data}")

# Predictions with true parameters
y_pred_true = X_data @ w_true + b_true
print(f"\nTrue predictions: ŷ = {y_pred_true}")
print(f"Targets:          y = {y_data}")

# Compute loss
mse_true = np.mean((y_data - y_pred_true)**2)
print(f"MSE with true parameters: {mse_true:.6f}")

print("\n5.2 Gradient Computation")
print("-" * 70)

# Start with wrong parameters
w_init = np.array([0.0, 0.0])
b_init = 0.0

y_pred_init = X_data @ w_init + b_init
errors = y_pred_init - y_data

# Compute gradients
n_samples = len(X_data)
grad_w = (2/n_samples) * (X_data.T @ errors)
grad_b = (2/n_samples) * np.sum(errors)

print(f"Initial parameters: w={w_init}, b={b_init}")
print(f"Initial predictions: {y_pred_init}")
print(f"Errors: {errors}")
print(f"Gradients: ∂MSE/∂w = {grad_w}, ∂MSE/∂b = {grad_b:.4f}")

# One step of gradient descent
lr = 0.1
w_new = w_init - lr * grad_w
b_new = b_init - lr * grad_b

y_pred_new = X_data @ w_new + b_new
mse_new = np.mean((y_data - y_pred_new)**2)

print(f"\nAfter one GD step (lr={lr}):")
print(f"New parameters: w={w_new}, b={b_new:.4f}")
print(f"New MSE: {mse_new:.4f}")

print("\n5.3 Normal Equation Solution")
print("-" * 70)

# Add bias column
X_with_bias = np.c_[X_data, np.ones(n_samples)]

# Normal equation: w = (XᵀX)⁻¹Xᵀy
XTX = X_with_bias.T @ X_with_bias
XTy = X_with_bias.T @ y_data
params = np.linalg.solve(XTX, XTy)

w_optimal = params[:2]
b_optimal = params[2]

print(f"Optimal parameters (Normal Equation):")
print(f"w = {w_optimal}")
print(f"b = {b_optimal:.4f}")

y_pred_optimal = X_data @ w_optimal + b_optimal
mse_optimal = np.mean((y_data - y_pred_optimal)**2)
print(f"MSE with optimal parameters: {mse_optimal:.10f}")

# ===== PART 6: VISUALIZATION =====
print("\n" + "="*70)
print("PART 6: VISUALIZING MATHEMATICAL CONCEPTS")
print("="*70)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss surface (2D slice)
ax = axes[0, 0]
w_range = np.linspace(-2, 4, 100)
b_range = np.linspace(-2, 4, 100)
W, B = np.meshgrid(w_range, b_range)
Z = np.zeros_like(W)

# Compute loss for single feature example
x_val, y_val = 2.0, 5.0
for i in range(len(w_range)):
    for j in range(len(b_range)):
        y_pred_ij = W[i,j] * x_val + B[i,j]
        Z[i,j] = (y_val - y_pred_ij)**2

contour = ax.contour(W, B, Z, levels=20, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)
ax.scatter([2.5], [0], color='red', s=200, marker='*', 
          edgecolors='white', linewidths=2, label='Optimal', zorder=5)
ax.set_xlabel('w', fontsize=12)
ax.set_ylabel('b', fontsize=12)
ax.set_title('Loss Surface L(w,b)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Gradient vectors
ax = axes[0, 1]
w_sample = np.linspace(0, 3, 6)
b_sample = np.linspace(0, 2, 5)

for w_s in w_sample:
    for b_s in b_sample:
        # Compute gradient at this point
        y_p = w_s * x_val + b_s
        err = y_p - y_val
        dw = 2 * x_val * err
        db = 2 * err
        
        # Normalize for visualization
        mag = np.sqrt(dw**2 + db**2)
        if mag > 0:
            dw_norm = dw / mag * 0.3
            db_norm = db / mag * 0.3
            ax.arrow(w_s, b_s, -dw_norm, -db_norm, 
                    head_width=0.1, head_length=0.1, 
                    fc='blue', ec='blue', alpha=0.6)

ax.scatter([2.5], [0], color='red', s=200, marker='*', 
          edgecolors='white', linewidths=2, label='Optimal', zorder=5)
ax.set_xlabel('w', fontsize=12)
ax.set_ylabel('b', fontsize=12)
ax.set_title('Gradient Field -∇L', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 3])
ax.set_ylim([0, 2])

# Plot 3: Gradient descent path
ax = axes[1, 0]
w_path, b_path = [0.0], [0.0]
lr_vis = 0.1

for _ in range(20):
    w_curr, b_curr = w_path[-1], b_path[-1]
    y_p = w_curr * x_val + b_curr
    err = y_p - y_val
    dw = 2 * x_val * err
    db = 2 * err
    
    w_new = w_curr - lr_vis * dw
    b_new = b_curr - lr_vis * db
    
    w_path.append(w_new)
    b_path.append(b_new)

ax.plot(w_path, b_path, 'o-', linewidth=2, markersize=6, 
       color='green', label='GD Path')
ax.scatter([w_path[0]], [b_path[0]], color='blue', s=150, 
          marker='o', label='Start', zorder=5, edgecolors='white', linewidths=2)
ax.scatter([w_path[-1]], [b_path[-1]], color='purple', s=150, 
          marker='s', label='End', zorder=5, edgecolors='white', linewidths=2)
ax.scatter([2.5], [0], color='red', s=200, marker='*', 
          label='Optimal', zorder=5, edgecolors='white', linewidths=2)
ax.set_xlabel('w', fontsize=12)
ax.set_ylabel('b', fontsize=12)
ax.set_title('Gradient Descent Trajectory', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Convergence
ax = axes[1, 1]
losses = []
for w_i, b_i in zip(w_path, b_path):
    y_p = w_i * x_val + b_i
    loss_i = (y_val - y_p)**2
    losses.append(loss_i)

ax.plot(losses, linewidth=2, color='orange', marker='o', markersize=4)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss L(w,b)', fontsize=12)
ax.set_title('Loss vs Iteration', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('mathematical_derivations.png', dpi=150, bbox_inches='tight')
plt.show()

