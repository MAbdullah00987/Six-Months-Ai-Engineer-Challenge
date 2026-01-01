#The Engine of Backpropagation (Chain Rule)
#Objective: Master the Chain Rule. This is exactly how deep neural networks learn 
#(Backpropagation is just the chain rule applied 50 times in a row).
#Concept: $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$.
#Resource:
#Read: Chapter 5.3 of Mathematics for Machine Learning.
#Task: "Chain Rule by Hand" Project. Create a composite function in Python (e.g., sigmoid(linear(x))) 
#and calculate its derivative step-by-step.


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import symbols, diff, exp, sin, cos, lambdify, simplify
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


#PART 1: UNDERSTANDING THE BASICS
# TOPIC 1: Simple Chain Rule with SymPy (Symbolic Computation)

print("TOPIC 1: Chain Rule with SymPy - Symbolic Differentiation")
x = symbols('x')
# Example 1: f(g(x)) = (x^2 + 1)^3
g = x**2 + 1  # Inner function
f = g**3      # Outer function applied to g

print("\nExample 1: f(g(x)) = (x² + 1)³")
print(f"Inner function g(x) = {g}")
print(f"Composite function f(g(x)) = {f}")

# Using SymPy to find derivative
df_dx = diff(f, x)
print(f"\nDerivative df/dx = {df_dx}")

# Manual chain rule: f'(g(x)) * g'(x)
dg_dx = diff(g, x)  # g'(x) = 2x
print(f"\nManual Chain Rule:")
print(f"  g'(x) = {dg_dx}")
print(f"  f'(g) = 3g² = 3(x² + 1)²")
print(f"  f'(g(x)) * g'(x) = 3(x² + 1)² * 2x = {simplify(df_dx)}")


# TOPIC 2: Visualizing the Chain Rule
print("TOPIC 2: Visualizing the Chain Rule")


# Convert symbolic to numeric function
f_numeric = lambdify(x, f, 'numpy')
df_numeric = lambdify(x, df_dx, 'numpy')

# Create x values
x_vals = np.linspace(-2, 2, 100)
y_vals = f_numeric(x_vals)
dy_vals = df_numeric(x_vals)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Original function
axes[0, 0].plot(x_vals, y_vals, 'b-', linewidth=2, label='f(g(x)) = (x² + 1)³')
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('f(g(x))', fontsize=12)
axes[0, 0].set_title('Composite Function', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)

# Plot 2: Derivative
axes[0, 1].plot(x_vals, dy_vals, 'r-', linewidth=2, label="f'(g(x)) * g'(x)")
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel("df/dx", fontsize=12)
axes[0, 1].set_title('Derivative via Chain Rule', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=10)

# Plot 3: Inner function g(x)
g_numeric = lambdify(x, g, 'numpy')
g_vals = g_numeric(x_vals)
axes[1, 0].plot(x_vals, g_vals, 'g-', linewidth=2, label='g(x) = x² + 1')
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('g(x)', fontsize=12)
axes[1, 0].set_title('Inner Function g(x)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)

# Plot 4: Tangent line demonstration at x=1
x_point = 1.0
y_point = f_numeric(x_point)
slope = df_numeric(x_point)
tangent_x = np.linspace(x_point - 0.5, x_point + 0.5, 50)
tangent_y = slope * (tangent_x - x_point) + y_point

axes[1, 1].plot(x_vals, y_vals, 'b-', linewidth=2, label='f(g(x))')
axes[1, 1].plot(tangent_x, tangent_y, 'r--', linewidth=2, label=f'Tangent (slope={slope:.2f})')
axes[1, 1].plot(x_point, y_point, 'ro', markersize=10, label=f'Point ({x_point}, {y_point:.2f})')
axes[1, 1].set_xlabel('x', fontsize=12)
axes[1, 1].set_ylabel('f(g(x))', fontsize=12)
axes[1, 1].set_title('Tangent Line at x=1', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=10)
axes[1, 1].set_xlim([0, 2])
axes[1, 1].set_ylim([0, 30])

plt.tight_layout()
plt.savefig('chain_rule_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'chain_rule_visualization.png'")
plt.show()

# TOPIC 3: Multiple Examples with Pandas DataFrame
print("TOPIC 3: Chain Rule Examples - Organized with Pandas")

# Create examples dataframe
examples = []

# Example 2: sin(x^2)
g2 = x**2
f2 = sin(g2)
df2 = diff(f2, x)
examples.append({
    'Function': 'sin(x²)',
    'Inner g(x)': 'x²',
    "g'(x)": '2x',
    'Outer f(u)': 'sin(u)',
    "f'(u)": 'cos(u)',
    'Chain Rule Result': str(simplify(df2))
})

# Example 3: e^(3x)
g3 = 3*x
f3 = exp(g3)
df3 = diff(f3, x)
examples.append({
    'Function': 'e^(3x)',
    'Inner g(x)': '3x',
    "g'(x)": '3',
    'Outer f(u)': 'e^u',
    "f'(u)": 'e^u',
    'Chain Rule Result': str(simplify(df3))
})

# Example 4: (cos(x))^2
g4 = cos(x)
f4 = g4**2
df4 = diff(f4, x)
examples.append({
    'Function': 'cos²(x)',
    'Inner g(x)': 'cos(x)',
    "g'(x)": '-sin(x)',
    'Outer f(u)': 'u²',
    "f'(u)": '2u',
    'Chain Rule Result': str(simplify(df4))
})

df_examples = pd.DataFrame(examples)
print("\n" + df_examples.to_string(index=False))

# TOPIC 4: Numerical Verification using NumPy
print("TOPIC 4: Numerical Verification - Finite Differences")


def numerical_derivative(f, x, h=1e-5):
    """Compute derivative using finite difference method"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test on Example 1: (x^2 + 1)^3
test_x = 1.5

# Analytical derivative
analytical = df_numeric(test_x)

# Numerical derivative
numerical = numerical_derivative(f_numeric, test_x)

print(f"\nAt x = {test_x}:")
print(f"Analytical derivative (Chain Rule): {analytical:.10f}")
print(f"Numerical derivative (Finite Diff): {numerical:.10f}")
print(f"Difference: {abs(analytical - numerical):.2e}")
print(f"Match: {'✓ YES' if abs(analytical - numerical) < 1e-6 else '✗ NO'}")


# TOPIC 5: Visualization with Seaborn - Heatmap of Derivatives
print("TOPIC 5: Heatmap of Derivative Values")


# Create a grid of x values and different composite functions
x_range = np.linspace(-3, 3, 50)
functions_dict = {
    '(x²+1)³': lambda x: 6*x*(x**2 + 1)**2,
    'sin(x²)': lambda x: 2*x*np.cos(x**2),
    'e^(x²)': lambda x: 2*x*np.exp(x**2),
    'cos²(x)': lambda x: -2*np.cos(x)*np.sin(x)
}

# Calculate derivatives for all functions
derivative_data = np.zeros((len(functions_dict), len(x_range)))
for i, (name, func) in enumerate(functions_dict.items()):
    derivative_data[i, :] = func(x_range)

# Create heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(derivative_data, 
            xticklabels=[f'{x:.1f}' if i % 5 == 0 else '' for i, x in enumerate(x_range)],
            yticklabels=list(functions_dict.keys()),
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Derivative Value'},
            annot=False)
plt.xlabel('x value', fontsize=12)
plt.ylabel('Function', fontsize=12)
plt.title('Derivative Values Across Different Composite Functions', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('derivative_heatmap.png', dpi=150, bbox_inches='tight')
print("\n✓ Heatmap saved as 'derivative_heatmap.png'")
plt.show()



