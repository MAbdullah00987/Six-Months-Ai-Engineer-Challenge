
#Project 1: Chain Rule by Hand (1.5 hours)
#Manually calculate derivatives of composite functions:
#f(g(x)) where f(u) = u³, g(x) = 2x + 1
#h(x) = sin(x²)
#y(x) = e^(cos(x))
#Verify with SymPy
#Document the step-by-step process

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import *
from sympy.abc import x, u

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

print("="*80)
print("CHAIN RULE DERIVATIVE CALCULATIONS - STEP BY STEP")
print("="*80)

# ============================================================================
# PROBLEM 1: f(g(x)) where f(u) = u³, g(x) = 2x + 1
# ============================================================================
print("\n" + "="*80)
print("PROBLEM 1: f(g(x)) where f(u) = u³, g(x) = 2x + 1")
print("="*80)

print("\n MANUAL CALCULATION:")
print("-" * 40)
print("Step 1: Identify inner and outer functions")
print("   Inner function: g(x) = 2x + 1")
print("   Outer function: f(u) = u³")
print("\nStep 2: Find derivatives")
print("   g'(x) = d/dx(2x + 1) = 2")
print("   f'(u) = d/du(u³) = 3u²")
print("\nStep 3: Apply chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x)")
print("   f'(g(x)) = 3(g(x))² = 3(2x + 1)²")
print("   g'(x) = 2")
print("\nStep 4: Multiply")
print("   d/dx[(2x + 1)³] = 3(2x + 1)² · 2 = 6(2x + 1)²")
print("\nStep 5: Expand (optional)")
print("   = 6(4x² + 4x + 1)")
print("   = 24x² + 24x + 6")

print("\n SYMPY VERIFICATION:")
print("-" * 40)
x_sym = symbols('x')
f1 = (2*x_sym + 1)**3
f1_derivative = diff(f1, x_sym)
print(f"   Function: f(x) = (2x + 1)³")
print(f"   Derivative: f'(x) = {f1_derivative}")
print(f"   Expanded: f'(x) = {expand(f1_derivative)}")

# ============================================================================
# PROBLEM 2: h(x) = sin(x²)
# ============================================================================
print("\n" + "="*80)
print("PROBLEM 2: h(x) = sin(x²)")
print("="*80)

print("\n MANUAL CALCULATION:")
print("-" * 40)
print("Step 1: Identify inner and outer functions")
print("   Inner function: g(x) = x²")
print("   Outer function: f(u) = sin(u)")
print("\nStep 2: Find derivatives")
print("   g'(x) = d/dx(x²) = 2x")
print("   f'(u) = d/du(sin(u)) = cos(u)")
print("\nStep 3: Apply chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x)")
print("   f'(g(x)) = cos(x²)")
print("   g'(x) = 2x")
print("\nStep 4: Multiply")
print("   d/dx[sin(x²)] = cos(x²) · 2x = 2x·cos(x²)")

print("\n SYMPY VERIFICATION:")
print("-" * 40)
f2 = sin(x_sym**2)
f2_derivative = diff(f2, x_sym)
print(f"   Function: h(x) = sin(x²)")
print(f"   Derivative: h'(x) = {f2_derivative}")

# ============================================================================
# PROBLEM 3: y(x) = e^(cos(x))
# ============================================================================
print("\n" + "="*80)
print("PROBLEM 3: y(x) = e^(cos(x))")
print("="*80)

print("\n MANUAL CALCULATION:")
print("-" * 40)
print("Step 1: Identify inner and outer functions")
print("   Inner function: g(x) = cos(x)")
print("   Outer function: f(u) = e^u")
print("\nStep 2: Find derivatives")
print("   g'(x) = d/dx(cos(x)) = -sin(x)")
print("   f'(u) = d/du(e^u) = e^u")
print("\nStep 3: Apply chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x)")
print("   f'(g(x)) = e^(cos(x))")
print("   g'(x) = -sin(x)")
print("\nStep 4: Multiply")
print("   d/dx[e^(cos(x))] = e^(cos(x)) · (-sin(x)) = -sin(x)·e^(cos(x))")

print("\n SYMPY VERIFICATION:")
print("-" * 40)
f3 = exp(cos(x_sym))
f3_derivative = diff(f3, x_sym)
print(f"   Function: y(x) = e^(cos(x))")
print(f"   Derivative: y'(x) = {f3_derivative}")

# ============================================================================
# NUMERICAL VERIFICATION AND VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("NUMERICAL VERIFICATION USING NUMPY")
print("="*80)

# Define x values for plotting
x_vals = np.linspace(-3, 3, 1000)

# Problem 1: (2x + 1)³
def f1_func(x):
    return (2*x + 1)**3

def f1_deriv_func(x):
    return 6 * (2*x + 1)**2

# Problem 2: sin(x²)
def f2_func(x):
    return np.sin(x**2)

def f2_deriv_func(x):
    return 2*x * np.cos(x**2)

# Problem 3: e^(cos(x))
def f3_func(x):
    return np.exp(np.cos(x))

def f3_deriv_func(x):
    return -np.sin(x) * np.exp(np.cos(x))

# Numerical derivative approximation (for verification)
def numerical_derivative(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2 * h)

# Calculate values
y1 = f1_func(x_vals)
y1_deriv = f1_deriv_func(x_vals)
y1_deriv_num = numerical_derivative(f1_func, x_vals)

y2 = f2_func(x_vals)
y2_deriv = f2_deriv_func(x_vals)
y2_deriv_num = numerical_derivative(f2_func, x_vals)

y3 = f3_func(x_vals)
y3_deriv = f3_deriv_func(x_vals)
y3_deriv_num = numerical_derivative(f3_func, x_vals)

# Create comparison DataFrame
comparison_data = {
    'x': [0, 1, 2, -1, -2],
}

# Add analytical derivatives
comparison_data['f1_analytical'] = [f1_deriv_func(xi) for xi in comparison_data['x']]
comparison_data['f2_analytical'] = [f2_deriv_func(xi) for xi in comparison_data['x']]
comparison_data['f3_analytical'] = [f3_deriv_func(xi) for xi in comparison_data['x']]

# Add numerical derivatives
comparison_data['f1_numerical'] = [numerical_derivative(f1_func, xi) for xi in comparison_data['x']]
comparison_data['f2_numerical'] = [numerical_derivative(f2_func, xi) for xi in comparison_data['x']]
comparison_data['f3_numerical'] = [numerical_derivative(f3_func, xi) for xi in comparison_data['x']]

df_comparison = pd.DataFrame(comparison_data)

print("\n📊 COMPARISON TABLE: Analytical vs Numerical Derivatives")
print("-" * 80)
print("\nProblem 1: f(x) = (2x + 1)³")
print(df_comparison[['x', 'f1_analytical', 'f1_numerical']])
print("\nProblem 2: h(x) = sin(x²)")
print(df_comparison[['x', 'f2_analytical', 'f2_numerical']])
print("\nProblem 3: y(x) = e^(cos(x))")
print(df_comparison[['x', 'f3_analytical', 'f3_numerical']])

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Chain Rule Derivatives: Functions and Their Derivatives', fontsize=16, fontweight='bold')

# Problem 1
axes[0, 0].plot(x_vals, y1, 'b-', linewidth=2, label='f(x) = (2x + 1)³')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('f(x)', fontsize=12)
axes[0, 0].set_title('Problem 1: Original Function', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)

axes[0, 1].plot(x_vals, y1_deriv, 'r-', linewidth=2, label="f'(x) = 6(2x + 1)²")
axes[0, 1].plot(x_vals, y1_deriv_num, 'g--', linewidth=1, alpha=0.7, label="Numerical derivative")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel("f'(x)", fontsize=12)
axes[0, 1].set_title('Problem 1: Derivative', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)

# Problem 2
axes[1, 0].plot(x_vals, y2, 'b-', linewidth=2, label='h(x) = sin(x²)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('h(x)', fontsize=12)
axes[1, 0].set_title('Problem 2: Original Function', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linewidth=0.5)

axes[1, 1].plot(x_vals, y2_deriv, 'r-', linewidth=2, label="h'(x) = 2x·cos(x²)")
axes[1, 1].plot(x_vals, y2_deriv_num, 'g--', linewidth=1, alpha=0.7, label="Numerical derivative")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlabel('x', fontsize=12)
axes[1, 1].set_ylabel("h'(x)", fontsize=12)
axes[1, 1].set_title('Problem 2: Derivative', fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
axes[1, 1].axvline(x=0, color='k', linewidth=0.5)

# Problem 3
axes[2, 0].plot(x_vals, y3, 'b-', linewidth=2, label='y(x) = e^(cos(x))')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_xlabel('x', fontsize=12)
axes[2, 0].set_ylabel('y(x)', fontsize=12)
axes[2, 0].set_title('Problem 3: Original Function', fontsize=13, fontweight='bold')
axes[2, 0].legend(fontsize=11)
axes[2, 0].axhline(y=0, color='k', linewidth=0.5)
axes[2, 0].axvline(x=0, color='k', linewidth=0.5)

axes[2, 1].plot(x_vals, y3_deriv, 'r-', linewidth=2, label="y'(x) = -sin(x)·e^(cos(x))")
axes[2, 1].plot(x_vals, y3_deriv_num, 'g--', linewidth=1, alpha=0.7, label="Numerical derivative")
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].set_xlabel('x', fontsize=12)
axes[2, 1].set_ylabel("y'(x)", fontsize=12)
axes[2, 1].set_title('Problem 3: Derivative', fontsize=13, fontweight='bold')
axes[2, 1].legend(fontsize=11)
axes[2, 1].axhline(y=0, color='k', linewidth=0.5)
axes[2, 1].axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

# ============================================================================
# ERROR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ERROR ANALYSIS: Analytical vs Numerical Derivatives")
print("="*80)

error1 = np.abs(y1_deriv - y1_deriv_num)
error2 = np.abs(y2_deriv - y2_deriv_num)
error3 = np.abs(y3_deriv - y3_deriv_num)

print(f"\nProblem 1 - Max Error: {np.max(error1):.2e}")
print(f"Problem 2 - Max Error: {np.max(error2):.2e}")
print(f"Problem 3 - Max Error: {np.max(error3):.2e}")

# Create error visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Error Analysis: Analytical vs Numerical Derivatives', fontsize=14, fontweight='bold')

axes[0].semilogy(x_vals, error1, 'purple', linewidth=2)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('Absolute Error (log scale)', fontsize=12)
axes[0].set_title('Problem 1 Error', fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(x_vals, error2, 'purple', linewidth=2)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('Absolute Error (log scale)', fontsize=12)
axes[1].set_title('Problem 2 Error', fontsize=12)
axes[1].grid(True, alpha=0.3)

axes[2].semilogy(x_vals, error3, 'purple', linewidth=2)
axes[2].set_xlabel('x', fontsize=12)
axes[2].set_ylabel('Absolute Error (log scale)', fontsize=12)
axes[2].set_title('Problem 3 Error', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
