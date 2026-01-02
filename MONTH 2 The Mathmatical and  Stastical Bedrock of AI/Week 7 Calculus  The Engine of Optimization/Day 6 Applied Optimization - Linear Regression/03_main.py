
#Part 3: Gradient Descent Implementation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

np.random.seed(42)

print("="*60)
print("TOPIC 3: GRADIENT DESCENT - COMPLETE IMPLEMENTATION")
print("="*60)

# 1. Generate data
print("\n1. Data Setup")
print("-" * 50)

true_w, true_b = 2.5, 1.0
n_samples = 100
X = np.linspace(0, 10, n_samples)
y = true_w * X + true_b + np.random.randn(n_samples) * 1.5

print(f"Dataset: {n_samples} points")
print(f"True parameters: w={true_w}, b={true_b}")

# 2. Core functions
def predict(X, w, b):
    return w * X + b

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_gradients(X, y, w, b):
    n = len(X)
    y_pred = predict(X, w, b)
    errors = y_pred - y
    dw = (2/n) * np.sum(X * errors)
    db = (2/n) * np.sum(errors)
    return dw, db

# 3. Gradient Descent Algorithm
print("\n2. Gradient Descent Algorithm")
print("-" * 50)

def gradient_descent(X, y, learning_rate=0.01, n_iterations=100, verbose=True):
    """
    Train linear regression using gradient descent
    
    Returns:
        w, b: Final parameters
        history: Dictionary containing training history
    """
    # Initialize parameters
    w = 0.0
    b = 0.0
    
    # Store history
    history = {
        'w': [w],
        'b': [b],
        'loss': [mse_loss(y, predict(X, w, b))],
        'dw': [],
        'db': []
    }
    
    if verbose:
        print(f"{'Iter':<6} {'Loss':<12} {'w':<12} {'b':<12} {'dw':<12} {'db':<12}")
        print("-" * 72)
    
    for iteration in range(n_iterations):
        # Compute gradients
        dw, db = compute_gradients(X, y, w, b)
        
        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Compute loss
        loss = mse_loss(y, predict(X, w, b))
        
        # Store history
        history['w'].append(w)
        history['b'].append(b)
        history['loss'].append(loss)
        history['dw'].append(dw)
        history['db'].append(db)
        
        # Print progress
        if verbose and (iteration % 10 == 0 or iteration == n_iterations - 1):
            print(f"{iteration:<6} {loss:<12.4f} {w:<12.4f} {b:<12.4f} {dw:<12.4f} {db:<12.4f}")
    
    return w, b, history

# Train the model
w_final, b_final, history = gradient_descent(
    X, y, 
    learning_rate=0.01, 
    n_iterations=100,
    verbose=True
)

print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
print(f"True parameters:    w={true_w:.4f}, b={true_b:.4f}")
print(f"Learned parameters: w={w_final:.4f}, b={b_final:.4f}")
print(f"Parameter error:    Δw={abs(w_final-true_w):.4f}, Δb={abs(b_final-true_b):.4f}")
print(f"Final loss: {history['loss'][-1]:.4f}")

# 4. Experiment with different learning rates
print("\n3. Learning Rate Impact")
print("-" * 50)

learning_rates = [0.001, 0.01, 0.05, 0.1]
results = {}

for lr in learning_rates:
    w, b, hist = gradient_descent(X, y, learning_rate=lr, n_iterations=100, verbose=False)
    results[lr] = hist
    print(f"LR={lr:.3f} → Final loss: {hist['loss'][-1]:.4f}, w={w:.4f}, b={b:.4f}")

# 5. Visualization Suite
print("\n4. Creating Visualizations")
print("-" * 50)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training progress (loss curve)
ax1 = fig.add_subplot(gs[0, :])
for lr in learning_rates:
    ax1.plot(results[lr]['loss'], linewidth=2, label=f'LR={lr}')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('MSE Loss', fontsize=12)
ax1.set_title('Training Loss vs Iteration (Different Learning Rates)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Parameter evolution (w)
ax2 = fig.add_subplot(gs[1, 0])
for lr in learning_rates:
    ax2.plot(results[lr]['w'], linewidth=2, label=f'LR={lr}')
ax2.axhline(true_w, color='r', linestyle='--', linewidth=2, label='True w')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('w (weight)', fontsize=12)
ax2.set_title('Weight Evolution', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Parameter evolution (b)
ax3 = fig.add_subplot(gs[1, 1])
for lr in learning_rates:
    ax3.plot(results[lr]['b'], linewidth=2, label=f'LR={lr}')
ax3.axhline(true_b, color='r', linestyle='--', linewidth=2, label='True b')
ax3.set_xlabel('Iteration', fontsize=12)
ax3.set_ylabel('b (bias)', fontsize=12)
ax3.set_title('Bias Evolution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Gradient magnitude
ax4 = fig.add_subplot(gs[1, 2])
gradient_magnitudes = [np.sqrt(dw**2 + db**2) for dw, db in zip(history['dw'], history['db'])]
ax4.plot(gradient_magnitudes, linewidth=2, color='purple')
ax4.set_xlabel('Iteration', fontsize=12)
ax4.set_ylabel('||∇L||', fontsize=12)
ax4.set_title('Gradient Magnitude', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

# Plot 5: Final fit
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(X, y, alpha=0.4, s=20, label='Data', color='blue')
ax5.plot(X, predict(X, true_w, true_b), 'g--', linewidth=2, label='True model', alpha=0.7)
ax5.plot(X, predict(X, w_final, b_final), 'r-', linewidth=2, label='Learned model')
ax5.set_xlabel('X', fontsize=12)
ax5.set_ylabel('y', fontsize=12)
ax5.set_title('Final Model Fit', fontsize=13, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Residual plot
ax6 = fig.add_subplot(gs[2, 1])
residuals = y - predict(X, w_final, b_final)
ax6.scatter(predict(X, w_final, b_final), residuals, alpha=0.5, s=20, color='red')
ax6.axhline(0, color='k', linestyle='--', alpha=0.5)
ax6.set_xlabel('Predicted y', fontsize=12)
ax6.set_ylabel('Residuals', fontsize=12)
ax6.set_title('Residual Plot', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot 7: Parameter space trajectory
ax7 = fig.add_subplot(gs[2, 2])
w_vals = history['w']
b_vals = history['b']
colors = plt.cm.viridis(np.linspace(0, 1, len(w_vals)))
for i in range(len(w_vals)-1):
    ax7.plot(w_vals[i:i+2], b_vals[i:i+2], color=colors[i], linewidth=2, alpha=0.7)
ax7.scatter([true_w], [true_b], color='red', s=200, marker='*', label='True', zorder=5, edgecolors='black', linewidths=2)
ax7.scatter([w_vals[0]], [b_vals[0]], color='green', s=100, marker='o', label='Start', zorder=5)
ax7.scatter([w_vals[-1]], [b_vals[-1]], color='blue', s=100, marker='s', label='End', zorder=5)
ax7.set_xlabel('w', fontsize=12)
ax7.set_ylabel('b', fontsize=12)
ax7.set_title('Parameter Space Trajectory', fontsize=13, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.savefig('gradient_descent_complete.png', dpi=150, bbox_inches='tight')
plt.show()

