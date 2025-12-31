
#Phase 2: Manual Calculation (Build Intuition)
#Let's work through a concrete example by hand first:
#Example: h(x) = sigmoid(3x + 2)

#Outer function: f(u) = sigmoid(u) = 1/(1 + e^(-u))
#Inner function: g(x) = 3x + 2

#Step-by-step:

#g'(x) = 3
#f'(u) = sigmoid(u) · (1 - sigmoid(u))
#h'(x) = f'(g(x)) · g'(x) = sigmoid(3x + 2) · (1 - sigmoid(3x + 2)) · 3

#Python Implementation (NumPy)

import numpy as np
import matplotlib.pyplot as plt

# Define functions
def linear(x, w=3, b=2):
    """Inner function: g(x) = wx + b"""
    return w * x + b

def sigmoid(z):
    """Outer function: f(z) = 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of sigmoid: f'(z) = sigmoid(z) * (1 - sigmoid(z))"""
    s = sigmoid(z)
    return s * (1 - s)

def composite_function(x):
    """h(x) = sigmoid(linear(x)) = sigmoid(3x + 2)"""
    return sigmoid(linear(x))

def composite_derivative_manual(x, w=3):
    """h'(x) using chain rule: f'(g(x)) * g'(x)"""
    inner_value = linear(x, w)  # g(x) = 3x + 2
    outer_derivative = sigmoid_derivative(inner_value)  # f'(g(x))
    inner_derivative = w  # g'(x) = 3
    return outer_derivative * inner_derivative

# Numerical derivative (for verification)
def numerical_derivative(f, x, h=1e-5):
    """Approximate derivative using finite differences"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test the implementation
x_test = 2.0
print("=" * 60)
print("CHAIN RULE VERIFICATION")
print("=" * 60)
print(f"Testing at x = {x_test}")
print(f"\n1. Inner function g(x) = 3x + 2:")
print(f"   g({x_test}) = {linear(x_test):.4f}")
print(f"\n2. Outer function f(z) = sigmoid(z):")
print(f"   f(g({x_test})) = {composite_function(x_test):.4f}")
print(f"\n3. Chain Rule Derivative (Analytical):")
print(f"   h'({x_test}) = {composite_derivative_manual(x_test):.6f}")
print(f"\n4. Numerical Derivative (Verification):")
num_deriv = numerical_derivative(composite_function, x_test)
print(f"   h'({x_test}) ≈ {num_deriv:.6f}")
print(f"\n5. Difference (should be tiny):")
print(f"   |analytical - numerical| = {abs(composite_derivative_manual(x_test) - num_deriv):.10f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Chain Rule Breakdown: h(x) = sigmoid(3x + 2)', fontsize=16, fontweight='bold')

x = np.linspace(-3, 3, 500)

# Plot 1: Inner function g(x) = 3x + 2
ax1 = axes[0, 0]
ax1.plot(x, linear(x), 'b-', linewidth=2, label='g(x) = 3x + 2')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax1.scatter([x_test], [linear(x_test)], color='red', s=100, zorder=5, label=f'g({x_test}) = {linear(x_test):.2f}')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('g(x)', fontsize=12)
ax1.set_title('Inner Function (Linear)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Outer function f(z) = sigmoid(z)
ax2 = axes[0, 1]
z = np.linspace(-10, 10, 500)
ax2.plot(z, sigmoid(z), 'g-', linewidth=2, label='f(z) = sigmoid(z)')
ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax2.scatter([linear(x_test)], [sigmoid(linear(x_test))], color='red', s=100, zorder=5, 
            label=f'f(g({x_test})) = {sigmoid(linear(x_test)):.3f}')
ax2.set_xlabel('z', fontsize=12)
ax2.set_ylabel('f(z)', fontsize=12)
ax2.set_title('Outer Function (Sigmoid)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Composite function h(x) = f(g(x))
ax3 = axes[1, 0]
ax3.plot(x, composite_function(x), 'purple', linewidth=2, label='h(x) = sigmoid(3x + 2)')
ax3.scatter([x_test], [composite_function(x_test)], color='red', s=100, zorder=5,
            label=f'h({x_test}) = {composite_function(x_test):.3f}')
ax3.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('h(x)', fontsize=12)
ax3.set_title('Composite Function h(x) = f(g(x))', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Derivative comparison
ax4 = axes[1, 1]
analytical_deriv = np.array([composite_derivative_manual(xi) for xi in x])
numerical_deriv = np.array([numerical_derivative(composite_function, xi) for xi in x])
ax4.plot(x, analytical_deriv, 'r-', linewidth=2, label="h'(x) Analytical (Chain Rule)")
ax4.plot(x, numerical_deriv, 'b--', linewidth=2, alpha=0.6, label="h'(x) Numerical")
ax4.scatter([x_test], [composite_derivative_manual(x_test)], color='red', s=100, zorder=5,
            label=f"h'({x_test}) = {composite_derivative_manual(x_test):.4f}")
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel("h'(x)", fontsize=12)
ax4.set_title('Derivative of Composite Function', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

