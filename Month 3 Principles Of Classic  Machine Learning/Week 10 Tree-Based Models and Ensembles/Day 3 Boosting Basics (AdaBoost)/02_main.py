
#PART 2: SMOTE (Synthetic Minority Over-sampling Technique)

import numpy as np
from scipy.spatial.distance import euclidean
import sympy as sp

# Mathematical Foundation of SMOTE
print("=" * 60)
print("SMOTE MATHEMATICAL FOUNDATION")
print("=" * 60)

# Define symbols
x1, x2, y1, y2, lambda_sym = sp.symbols('x1 x2 y1 y2 lambda')

# SMOTE formula: x_new = x + λ(x_neighbor - x)
# where λ ∈ [0, 1] is random
x_synthetic = x1 + lambda_sym * (x2 - x1)
y_synthetic = y1 + lambda_sym * (y2 - y1)

print("\nSMOTE Synthetic Sample Formula:")
print(f"x_new = x + λ(x_neighbor - x)")
print(f"x_new = {x_synthetic}")
print(f"y_new = {y_synthetic}")
print(f"\nWhere λ (lambda) is randomly selected from [0, 1]")

# Example calculation
print("\n" + "=" * 60)
print("NUMERICAL EXAMPLE")
print("=" * 60)
x_original = np.array([2, 3])
x_neighbor = np.array([5, 7])
lambda_val = 0.6

x_new = x_original + lambda_val * (x_neighbor - x_original)
print(f"Original point:  {x_original}")
print(f"Neighbor point:  {x_neighbor}")
print(f"Lambda value:    {lambda_val}")
print(f"Synthetic point: {x_new}")