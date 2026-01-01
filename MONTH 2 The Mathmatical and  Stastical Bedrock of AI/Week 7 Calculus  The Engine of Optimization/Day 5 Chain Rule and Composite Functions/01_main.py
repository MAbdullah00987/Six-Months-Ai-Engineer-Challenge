

#Day 5: Chain Rule and Composite Functions
#Focus: Understanding backpropagation's mathematical foundation

#Read: Mathematics for Machine Learning - Chain rule sections
#Watch: Coursera lectures on chain rule for single and multivariable functions
#Study: Connection between chain rule and neural network backpropagation

#Project 1: Chain Rule by Hand (1.5 hours)
#Manually calculate derivatives of composite functions:
#f(g(x)) where f(u) = u³, g(x) = 2x + 1
#h(x) = sin(x²)
#y(x) = e^(cos(x))
#Verify with SymPy
#Document the step-by-step process
#Project 2: Jacobian Matrix (1.5 hours)
#For vector-valued function F: ℝ² → ℝ²
#Example: F(x,y) = [x² + y, xy]
#Compute the Jacobian matrix manually
#Verify with code
#Visualize the transformation

#Deliverable: Jupyter notebook with hand calculations and symbolic verification


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

#Understanding the mathematical foundation of backpropagation
#Topic 1: Single Variable Chain Rule


# PART 1: SYMBOLIC CHAIN RULE WITH SYMPY

print("PART 1: SINGLE VARIABLE CHAIN RULE - SYMBOLIC")

# Define symbolic variable
x = sp.Symbol('x')

# Example 1: f(g(x)) = (x^2 + 1)^3
g = x**2 + 1
f = g**3

print("\n1. Composite Function: f(g(x)) = (x² + 1)³")
print(f"   Inner function g(x) = {g}")
print(f"   Outer function f(u) = u³")
print(f"   Composite f(g(x)) = {f}")

# Compute derivatives
dg_dx = sp.diff(g, x)  # g'(x)
df_du = 3 * g**2  # f'(u) where u = g(x)
df_dx_chain = sp.diff(f, x)  # Direct derivative
df_dx_manual = df_du * dg_dx  # Chain rule: f'(g(x)) * g'(x)

print(f"\n   dg/dx = {dg_dx}")
print(f"   df/du = {df_du.subs(g, sp.Symbol('u'))}")
print(f"   df/dx (chain rule) = {df_dx_manual}")
print(f"   df/dx (direct) = {df_dx_chain}")
print(f"   Simplified: {sp.simplify(df_dx_chain)}")

# Example 2: f(g(x)) = sin(3x^2)
print("\n2. Composite Function: f(g(x)) = sin(3x²)")
g2 = 3*x**2
f2 = sp.sin(g2)

dg2_dx = sp.diff(g2, x)
df2_dg2 = sp.cos(g2)
df2_dx = sp.diff(f2, x)

print(f"   g(x) = {g2}")
print(f"   f(g(x)) = {f2}")
print(f"   dg/dx = {dg2_dx}")
print(f"   df/dg = {df2_dg2}")
print(f"   df/dx = {df2_dx}")


# PART 2: NUMERICAL CHAIN RULE WITH NUMPY
print("PART 2: NUMERICAL COMPUTATION OF CHAIN RULE")


def composite_function_1(x):
    """f(g(x)) = (x^2 + 1)^3"""
    g = x**2 + 1
    f = g**3
    return f

def derivative_chain_1(x):
    """Derivative using chain rule"""
    g = x**2 + 1
    dg_dx = 2*x
    df_dg = 3 * g**2
    return df_dg * dg_dx

# Test values
x_vals = np.array([-2, -1, 0, 1, 2])
y_vals = composite_function_1(x_vals)
dy_vals = derivative_chain_1(x_vals)

print("\n3. Numerical Evaluation at Different Points:")
print(f"{'x':>6} {'f(g(x))':>12} {'df/dx':>12}")
print("-"*32)
for xi, yi, dyi in zip(x_vals, y_vals, dy_vals):
    print(f"{xi:6.1f} {yi:12.1f} {dyi:12.1f}")



# PART 3: VISUALIZATION WITH MATPLOTLIB
print("PART 3: VISUALIZING CHAIN RULE")


# Create fine grid for plotting
x_plot = np.linspace(-2, 2, 1000)
y_plot = composite_function_1(x_plot)
dy_plot = derivative_chain_1(x_plot)

# Inner function g(x)
g_plot = x_plot**2 + 1

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Chain Rule Visualization: f(g(x)) = (x² + 1)³', 
             fontsize=16, fontweight='bold')

# Plot 1: Inner function g(x)
ax1 = axes[0, 0]
ax1.plot(x_plot, g_plot, 'b-', linewidth=2, label='g(x) = x² + 1')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('g(x)', fontsize=12)
ax1.set_title('Inner Function g(x)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Plot 2: Composite function f(g(x))
ax2 = axes[0, 1]
ax2.plot(x_plot, y_plot, 'r-', linewidth=2, label='f(g(x)) = (x² + 1)³')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('f(g(x))', fontsize=12)
ax2.set_title('Composite Function f(g(x))', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# Plot 3: Derivative df/dx
ax3 = axes[1, 0]
ax3.plot(x_plot, dy_plot, 'g-', linewidth=2, label="df/dx = 6x(x² + 1)²")
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('df/dx', fontsize=12)
ax3.set_title('Derivative via Chain Rule', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)

# Plot 4: Tangent line at x=1
ax4 = axes[1, 1]
x_point = 1.0
y_point = composite_function_1(x_point)
slope = derivative_chain_1(x_point)

# Tangent line
x_tangent = np.linspace(0, 2, 100)
y_tangent = y_point + slope * (x_tangent - x_point)

ax4.plot(x_plot, y_plot, 'r-', linewidth=2, label='f(g(x))', alpha=0.7)
ax4.plot(x_tangent, y_tangent, 'b--', linewidth=2, label=f'Tangent at x={x_point}')
ax4.plot(x_point, y_point, 'go', markersize=12, label=f'Point ({x_point}, {y_point:.1f})')
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.set_title(f'Tangent Line (slope = {slope:.1f})', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_xlim(0, 2)
ax4.set_ylim(0, 80)

plt.tight_layout()
plt.savefig('chain_rule_basics.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'chain_rule_basics.png'")
plt.show()

print("KEY INSIGHTS:")
print("="*70)
print("1. Chain rule: df/dx = (df/dg) × (dg/dx)")
print("2. Work from outside to inside")
print("3. Each layer contributes a multiplicative factor")
print("4. This is the foundation of backpropagation!")
print("="*70)