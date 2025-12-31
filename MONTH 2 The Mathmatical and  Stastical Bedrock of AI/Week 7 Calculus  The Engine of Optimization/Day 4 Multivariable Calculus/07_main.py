#Phase 2: Manual Calculation (Build Intuition)
#Let's work through a concrete example by hand first:
#Example: h(x) = sigmoid(3x + 2)

#Outer function: f(u) = sigmoid(u) = 1/(1 + e^(-u))
#Inner function: g(x) = 3x + 2

#Step-by-step:

#g'(x) = 3
#f'(u) = sigmoid(u) · (1 - sigmoid(u))
#h'(x) = f'(g(x)) · g'(x) = sigmoid(3x + 2) · (1 - sigmoid(3x + 2)) · 3

#Comprehensive Chain Rule Examples with Pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define functions and derivatives
def f1(x): return x**2
def f1_prime(x): return 2*x

def f2(x): return np.sin(x)
def f2_prime(x): return np.cos(x)

def f3(x): return np.exp(x)
def f3_prime(x): return np.exp(x)

def f4(x): return np.log(x + 1e-10)
def f4_prime(x): return 1 / (x + 1e-10)

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_prime(x): 
    s = sigmoid(x)
    return s * (1 - s)

# Numerical derivative for verification
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

print("=" * 80)
print("COMPREHENSIVE CHAIN RULE EXAMPLES")
print("=" * 80)

# Example 1: Simple composition
print("\n" + "=" * 80)
print("EXAMPLE 1: h(x) = sin(x²)")
print("=" * 80)

x_vals = np.linspace(-2, 2, 11)
results = []

for x in x_vals:
    # g(x) = x²
    g_x = f1(x)
    g_prime_x = f1_prime(x)
    
    # f(u) = sin(u)
    f_g_x = f2(g_x)
    f_prime_g_x = f2_prime(g_x)
    
    # h(x) = f(g(x))
    h_x = f2(f1(x))
    
    # Chain rule: h'(x) = f'(g(x)) * g'(x)
    h_prime_analytical = f_prime_g_x * g_prime_x
    
    # Numerical verification
    h = lambda t: f2(f1(t))
    h_prime_numerical = numerical_derivative(h, x)
    
    results.append({
        'x': x,
        'g(x)=x²': g_x,
        "g'(x)=2x": g_prime_x,
        'f(g(x))=sin(x²)': f_g_x,
        "f'(g(x))=cos(x²)": f_prime_g_x,
        "h'(x) Chain": h_prime_analytical,
        "h'(x) Numerical": h_prime_numerical,
        'Error': abs(h_prime_analytical - h_prime_numerical)
    })

df1 = pd.DataFrame(results)
print("\nDetailed Calculation:")
print(df1.to_string(index=False))
print(f"\nMax Error: {df1['Error'].max():.2e}")

# Example 2: Triple composition
print("\n" + "=" * 80)
print("EXAMPLE 2: h(x) = e^(sin(x²))")
print("=" * 80)

x_vals2 = np.linspace(-1.5, 1.5, 9)
results2 = []

for x in x_vals2:
    # Layer 1: g1(x) = x²
    g1_x = f1(x)
    g1_prime = f1_prime(x)
    
    # Layer 2: g2(u) = sin(u)
    g2_g1 = f2(g1_x)
    g2_prime = f2_prime(g1_x)
    
    # Layer 3: f(v) = e^v
    h_x = f3(g2_g1)
    f_prime = f3_prime(g2_g1)
    
    # Chain rule: h'(x) = f'(g2(g1(x))) * g2'(g1(x)) * g1'(x)
    h_prime_chain = f_prime * g2_prime * g1_prime
    
    # Numerical
    h = lambda t: f3(f2(f1(t)))
    h_prime_num = numerical_derivative(h, x)
    
    results2.append({
        'x': x,
        'x²': g1_x,
        'sin(x²)': g2_g1,
        'e^(sin(x²))': h_x,
        "Chain Rule h'": h_prime_chain,
        "Numerical h'": h_prime_num,
        'Error': abs(h_prime_chain - h_prime_num)
    })

df2 = pd.DataFrame(results2)
print("\nTriple Composition Breakdown:")
print(df2.to_string(index=False))
print(f"\nMax Error: {df2['Error'].max():.2e}")

# Example 3: Product with chain rule
print("\n" + "=" * 80)
print("EXAMPLE 3: h(x) = x · sin(x²)  [Product Rule + Chain Rule]")
print("=" * 80)

x_vals3 = np.linspace(0, 2, 9)
results3 = []

for x in x_vals3:
    # h(x) = u(x) · v(x) where u(x) = x, v(x) = sin(x²)
    u_x = x
    u_prime = 1
    
    v_x = f2(f1(x))
    # v'(x) = cos(x²) * 2x (chain rule)
    v_prime = f2_prime(f1(x)) * f1_prime(x)
    
    h_x = u_x * v_x
    
    # Product rule: h'(x) = u'(x)v(x) + u(x)v'(x)
    h_prime_product = u_prime * v_x + u_x * v_prime
    
    # Numerical
    h = lambda t: t * f2(f1(t))
    h_prime_num = numerical_derivative(h, x)
    
    results3.append({
        'x': x,
        'u(x)=x': u_x,
        'v(x)=sin(x²)': v_x,
        "v'(x) Chain": v_prime,
        "h'(x) Product": h_prime_product,
        "Numerical": h_prime_num,
        'Error': abs(h_prime_product - h_prime_num)
    })

df3 = pd.DataFrame(results3)
print("\nProduct Rule + Chain Rule:")
print(df3.to_string(index=False))

# Example 4: Deep composition (like neural networks)
print("\n" + "=" * 80)
print("EXAMPLE 4: Deep Chain h(x) = σ(σ(σ(x))) [3-layer sigmoid]")
print("=" * 80)

x_vals4 = np.linspace(-2, 2, 7)
results4 = []

for x in x_vals4:
    # Layer by layer forward
    z1 = sigmoid(x)
    z2 = sigmoid(z1)
    z3 = sigmoid(z2)
    
    # Backward (chain rule)
    dz3_dz2 = sigmoid_prime(z2)
    dz2_dz1 = sigmoid_prime(z1)
    dz1_dx = sigmoid_prime(x)
    
    # Full chain
    dz3_dx_chain = dz3_dz2 * dz2_dz1 * dz1_dx
    
    # Numerical
    h = lambda t: sigmoid(sigmoid(sigmoid(t)))
    dz3_dx_num = numerical_derivative(h, x)
    
    results4.append({
        'x': x,
        'Layer1': z1,
        'Layer2': z2,
        'Layer3': z3,
        'dL3/dL2': dz3_dz2,
        'dL2/dL1': dz2_dz1,
        'dL1/dx': dz1_dx,
        'Full Chain': dz3_dx_chain,
        'Numerical': dz3_dx_num,
        'Error': abs(dz3_dx_chain - dz3_dx_num)  # FIXED: Added Error column
    })

df4 = pd.DataFrame(results4)
print("\nDeep Sigmoid Chain (Backpropagation):")
print(df4.to_string(index=False))

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY: ACCURACY OF CHAIN RULE VS NUMERICAL")
print("=" * 80)

summary = pd.DataFrame({
    'Example': [
        'sin(x²)',
        'e^(sin(x²))',
        'x·sin(x²)',
        'σ(σ(σ(x)))'
    ],
    'Max Error': [
        df1['Error'].max(),
        df2['Error'].max(),
        df3['Error'].max(),
        df4['Error'].max()  # FIXED: Now this column exists
    ],
    'Mean Error': [
        df1['Error'].mean(),
        df2['Error'].mean(),
        df3['Error'].mean(),
        df4['Error'].mean()  # FIXED: Now this column exists
    ]
})

