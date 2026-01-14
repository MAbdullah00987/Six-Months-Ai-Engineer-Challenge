
#Part 2 - Neural Network Functions

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sympy import symbols, diff, exp, log, lambdify, simplify

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("CHAIN RULE MASTERY - PART 2: NEURAL NETWORK ACTIVATION FUNCTIONS")
# TOPIC 6: Sigmoid Function and Its Derivative
print("TOPIC 6: Sigmoid Function - The Classic Neural Network Activation")


# Define sigmoid using SymPy
x = symbols('x')
sigmoid_sym = 1 / (1 + exp(-x))
sigmoid_derivative_sym = diff(sigmoid_sym, x)

print("Sigmoid function: σ(x) = 1 / (1 + e^(-x))")
print(f"Derivative: σ'(x) = {simplify(sigmoid_derivative_sym)}")
print("\nKey insight: σ'(x) = σ(x) * (1 - σ(x))")

# NumPy implementation
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def sigmoid_derivative(x):
    """Derivative of sigmoid using chain rule"""
    s = sigmoid(x)
    return s * (1 - s)

# Visualize
x_vals = np.linspace(-10, 10, 200)
y_sigmoid = sigmoid(x_vals)
dy_sigmoid = sigmoid_derivative(x_vals)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sigmoid function
axes[0, 0].plot(x_vals, y_sigmoid, 'b-', linewidth=2.5, label='σ(x)')
axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='y=0.5')
axes[0, 0].axvline(x=0, color='g', linestyle='--', alpha=0.5, label='x=0')
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('σ(x)', fontsize=12)
axes[0, 0].set_title('Sigmoid Activation Function', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)
axes[0, 0].set_ylim([-0.1, 1.1])

# Plot 2: Sigmoid derivative
axes[0, 1].plot(x_vals, dy_sigmoid, 'r-', linewidth=2.5, label="σ'(x)")
axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[0, 1].fill_between(x_vals, 0, dy_sigmoid, alpha=0.3, color='red')
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel("σ'(x)", fontsize=12)
axes[0, 1].set_title('Sigmoid Derivative (Gradient)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=10)

# Plot 3: Both on same plot
axes[1, 0].plot(x_vals, y_sigmoid, 'b-', linewidth=2.5, label='σ(x)')
axes[1, 0].plot(x_vals, dy_sigmoid, 'r-', linewidth=2.5, label="σ'(x)")
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('Value', fontsize=12)
axes[1, 0].set_title('Sigmoid and Its Derivative', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)

# Plot 4: Derivative property visualization
sigma_vals = np.linspace(0.01, 0.99, 100)
derivative_from_property = sigma_vals * (1 - sigma_vals)
axes[1, 1].plot(sigma_vals, derivative_from_property, 'purple', linewidth=2.5)
axes[1, 1].set_xlabel('σ(x)', fontsize=12)
axes[1, 1].set_ylabel("σ'(x) = σ(x)(1-σ(x))", fontsize=12)
axes[1, 1].set_title('Derivative Property: σ\'(x) = σ(x)(1-σ(x))', 
                      fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].fill_between(sigma_vals, 0, derivative_from_property, alpha=0.3, color='purple')

plt.tight_layout()
plt.savefig('sigmoid_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Sigmoid visualization saved as 'sigmoid_analysis.png'")
plt.show()


# TOPIC 7: Composite Function - sigmoid(linear(x))
print("TOPIC 7: Composite Function - sigmoid(Wx + b)")


# Define a linear function: z = Wx + b
W = 2.0  # Weight
b = -1.0  # Bias

def linear(x, W=2.0, b=-1.0):
    """Linear transformation: z = Wx + b"""
    return W * x + b

def sigmoid_of_linear(x, W=2.0, b=-1.0):
    """Composite: σ(Wx + b)"""
    z = linear(x, W, b)
    return sigmoid(z)

def derivative_sigmoid_of_linear(x, W=2.0, b=-1.0):
    """
    Chain rule: d/dx[σ(Wx + b)] = σ'(z) * W
    where z = Wx + b
    """
    z = linear(x, W, b)
    dsigmoid_dz = sigmoid_derivative(z)  # σ'(z)
    dz_dx = W  # d(Wx + b)/dx = W
    return dsigmoid_dz * dz_dx

print(f"Linear function: z(x) = {W}x + {b}")
print("Composite function: f(x) = σ(z(x)) = σ(Wx + b)")
print("\nChain Rule Breakdown:")
print("  1. Inner function: z(x) = Wx + b")
print("  2. dz/dx = W")
print("  3. Outer function: σ(z)")
print("  4. dσ/dz = σ(z)(1 - σ(z))")
print("  5. Final: df/dx = dσ/dz * dz/dx = σ(z)(1-σ(z)) * W")

# Visualize the transformation
x_vals = np.linspace(-3, 3, 200)
z_vals = linear(x_vals, W, b)
output_vals = sigmoid_of_linear(x_vals, W, b)
derivative_vals = derivative_sigmoid_of_linear(x_vals, W, b)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Step 1: Input x
axes[0, 0].plot(x_vals, x_vals, 'g-', linewidth=2.5, label='Input x')
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('x', fontsize=12)
axes[0, 0].set_title('Step 1: Input Layer', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)

# Step 2: Linear transformation
axes[0, 1].plot(x_vals, z_vals, 'b-', linewidth=2.5, label=f'z = {W}x + {b}')
axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel('z', fontsize=12)
axes[0, 1].set_title('Step 2: Linear Transformation', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=10)

# Step 3: Sigmoid activation
axes[1, 0].plot(x_vals, output_vals, 'purple', linewidth=2.5, label='σ(z)')
axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('σ(z)', fontsize=12)
axes[1, 0].set_title('Step 3: Sigmoid Activation (Output)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)

# Step 4: Derivative (Gradient)
axes[1, 1].plot(x_vals, derivative_vals, 'r-', linewidth=2.5, label="df/dx")
axes[1, 1].fill_between(x_vals, 0, derivative_vals, alpha=0.3, color='red')
axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[1, 1].set_xlabel('x', fontsize=12)
axes[1, 1].set_ylabel('df/dx', fontsize=12)
axes[1, 1].set_title('Step 4: Gradient (via Chain Rule)', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('sigmoid_linear_composite.png', dpi=150, bbox_inches='tight')
print("\n✓ Composite function visualization saved")
plt.show()


# TOPIC 8: Manual Step-by-Step Chain Rule Calculation
print("TOPIC 8: Manual Step-by-Step Chain Rule Calculation")

test_x = 1.0
print(f"\nCalculating derivative at x = {test_x}")

# Step 1: Compute z = Wx + b
z = linear(test_x, W, b)
print(f"Step 1: z = Wx + b = {W} * {test_x} + {b} = {z}")

# Step 2: Compute σ(z)
sigma_z = sigmoid(z)
print(f"Step 2: σ(z) = σ({z}) = {sigma_z:.6f}")

# Step 3: Compute dσ/dz
dsigma_dz = sigmoid_derivative(z)
print(f"Step 3: dσ/dz = σ(z)(1-σ(z)) = {sigma_z:.6f} * {1-sigma_z:.6f} = {dsigma_dz:.6f}")

# Step 4: Compute dz/dx
dz_dx = W
print(f"Step 4: dz/dx = W = {dz_dx}")

# Step 5: Apply chain rule
df_dx = dsigma_dz * dz_dx
print(f"Step 5: df/dx = dσ/dz * dz/dx = {dsigma_dz:.6f} * {dz_dx} = {df_dx:.6f}")

# Verify with function
df_dx_computed = derivative_sigmoid_of_linear(test_x, W, b)
print(f"\nVerification: Function output = {df_dx_computed:.6f}")
print(f"Match: {'✓ YES' if abs(df_dx - df_dx_computed) < 1e-10 else '✗ NO'}")

# TOPIC 9: Comparison of Different Activation Functions
print("TOPIC 9: Different Activation Functions and Their Derivatives")


def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU derivative: 1 if x > 0, else 0"""
    return (x > 0).astype(float)

def tanh(x):
    """Hyperbolic tangent"""
    return np.tanh(x)

def tanh_derivative(x):
    """tanh derivative: 1 - tanh²(x)"""
    t = np.tanh(x)
    return 1 - t**2

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: max(alpha*x, x)"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU derivative"""
    return np.where(x > 0, 1, alpha)

# Create comparison DataFrame
x_sample = np.array([-2, -1, 0, 1, 2])
comparison_data = {
    'x': x_sample,
    'sigmoid(x)': sigmoid(x_sample),
    "sigmoid'(x)": sigmoid_derivative(x_sample),
    'tanh(x)': tanh(x_sample),
    "tanh'(x)": tanh_derivative(x_sample),
    'ReLU(x)': relu(x_sample),
    "ReLU'(x)": relu_derivative(x_sample)
}

df_comparison = pd.DataFrame(comparison_data)
print("\n" + df_comparison.to_string(index=False))

# Visualize all activations
x_vals = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

activations = [
    ('Sigmoid', sigmoid, sigmoid_derivative),
    ('Tanh', tanh, tanh_derivative),
    ('ReLU', relu, relu_derivative),
    ('Leaky ReLU', leaky_relu, leaky_relu_derivative),
]

for idx, (name, func, deriv) in enumerate(activations):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col] if idx < 3 else axes[1, idx - 3]
    
    y_vals = func(x_vals)
    dy_vals = deriv(x_vals)
    
    ax.plot(x_vals, y_vals, 'b-', linewidth=2.5, label=f'{name}')
    ax.plot(x_vals, dy_vals, 'r--', linewidth=2, label=f"{name}'")
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Output', fontsize=11)
    ax.set_title(f'{name} and Derivative', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

# Remove empty subplots
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('activation_functions_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Activation functions comparison saved")
plt.show()
