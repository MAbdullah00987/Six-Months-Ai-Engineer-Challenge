#Phase 2: Manual Calculation (Build Intuition)
#Let's work through a concrete example by hand first:
#Example: h(x) = sigmoid(3x + 2)

#Outer function: f(u) = sigmoid(u) = 1/(1 + e^(-u))
#Inner function: g(x) = 3x + 2

#Step-by-step:

#g'(x) = 3
#f'(u) = sigmoid(u) · (1 - sigmoid(u))
#h'(x) = f'(g(x)) · g'(x) = sigmoid(3x + 2) · (1 - sigmoid(3x + 2)) · 3

#Symbolic Mathematics with SymPy

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variable
x = sp.Symbol('x')

# Define inner function g(x) = 3x + 2
g = 3*x + 2

# Define outer function f(u) = sigmoid(u) = 1/(1 + e^(-u))
u = sp.Symbol('u')
f = 1 / (1 + sp.exp(-u))

# Create composite function h(x) = f(g(x))
h = f.subs(u, g)

print("=" * 70)
print("CHAIN RULE ANALYSIS WITH SYMPY")
print("=" * 70)

print("\n1. FUNCTIONS DEFINED:")
print("-" * 70)
print(f"   Inner function:     g(x) = {g}")
print(f"   Outer function:     f(u) = {f}")
print(f"   Composite function: h(x) = {sp.simplify(h)}")

# Calculate derivatives symbolically
g_prime = sp.diff(g, x)
f_prime = sp.diff(f, u)
h_prime = sp.diff(h, x)

print("\n2. DERIVATIVES:")
print("-" * 70)
print(f"   g'(x) = {g_prime}")
print(f"   f'(u) = {sp.simplify(f_prime)}")
print(f"   h'(x) = {sp.simplify(h_prime)}")

# Verify chain rule manually
chain_rule_result = f_prime.subs(u, g) * g_prime
print("\n3. CHAIN RULE VERIFICATION:")
print("-" * 70)
print(f"   f'(g(x)) · g'(x) = {sp.simplify(chain_rule_result)}")
print(f"   Direct h'(x)     = {sp.simplify(h_prime)}")
print(f"   Are they equal?  {sp.simplify(chain_rule_result - h_prime) == 0}")

# Evaluate at specific point
x_val = 2
h_prime_at_2 = h_prime.subs(x, x_val).evalf()
print(f"\n4. NUMERICAL EVALUATION:")
print("-" * 70)
print(f"   h'({x_val}) = {h_prime_at_2}")

# Example 2: More complex composition
print("\n" + "=" * 70)
print("EXAMPLE 2: h(x) = (2x² + 1)³")
print("=" * 70)

g2 = 2*x**2 + 1
f2 = u**3
h2 = f2.subs(u, g2)

g2_prime = sp.diff(g2, x)
f2_prime = sp.diff(f2, u)
h2_prime = sp.diff(h2, x)

print(f"\n   g(x)  = {g2}")
print(f"   f(u)  = {f2}")
print(f"   h(x)  = {sp.expand(h2)}")
print(f"\n   g'(x) = {g2_prime}")
print(f"   f'(u) = {f2_prime}")
print(f"   h'(x) = {sp.expand(h2_prime)}")
print(f"\n   Chain Rule: f'(g(x)) · g'(x) = {sp.expand(f2_prime.subs(u, g2) * g2_prime)}")

# Example 3: Triple composition
print("\n" + "=" * 70)
print("EXAMPLE 3: Triple Chain Rule - h(x) = sin(e^(x²))")
print("=" * 70)

g3 = x**2
f3_inner = sp.exp(u)
v = sp.Symbol('v')
f3_outer = sp.sin(v)

# Build composition step by step
step1 = g3
step2 = f3_inner.subs(u, step1)
h3 = f3_outer.subs(v, step2)

print(f"\n   Layer 1: g(x)   = {g3}")
print(f"   Layer 2: f(u)   = {f3_inner}")
print(f"   Layer 3: k(v)   = {f3_outer}")
print(f"   Composite: h(x) = {h3}")

h3_prime = sp.diff(h3, x)
print(f"\n   h'(x) = {h3_prime}")

# Show step-by-step chain rule
print(f"\n   Step-by-step:")
print(f"   1. d/dv[sin(v)]     = cos(v)")
print(f"   2. d/du[e^u]        = e^u")
print(f"   3. d/dx[x²]         = 2x")
print(f"   4. Chain: cos(e^(x²)) · e^(x²) · 2x = {sp.simplify(h3_prime)}")

