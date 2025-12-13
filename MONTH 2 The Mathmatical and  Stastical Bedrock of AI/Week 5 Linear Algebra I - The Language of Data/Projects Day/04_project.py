
#4: Solving Linear Equations: Set up and solve a system of linear equations using np.linalg.solve().

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("="*70)
print("LINEAR EQUATIONS SOLVER USING NumPy")
print("="*70)

# =============================================================================
# EXAMPLE 1: Simple 2x2 System
# =============================================================================
print("\n" + "="*70)
print("EXAMPLE 1: 2x2 System of Linear Equations")
print("="*70)

# System of equations:
# 2x + 3y = 8
# 4x - y = 2

print("\nEquations:")
print("  2x + 3y = 8")
print("  4x - y = 2")

# Coefficient matrix A and constants vector b
A1 = np.array([[2, 3], 
               [4, -1]])
b1 = np.array([8, 2])

print("\nCoefficient Matrix A:")
print(A1)
print("\nConstants Vector b:")
print(b1)

# Solve using np.linalg.solve()
solution1 = np.linalg.solve(A1, b1)

print("\n" + "-"*50)
print("SOLUTION:")
print("-"*50)
print(f"x = {solution1[0]:.4f}")
print(f"y = {solution1[1]:.4f}")

# Verification
verification1 = np.dot(A1, solution1)
print(f"\nVerification (A @ solution = b):")
print(f"Expected: {b1}")
print(f"Got:      {verification1}")
print(f"Correct:  {np.allclose(verification1, b1)}")

# Visualize the solution
plt.figure(figsize=(10, 6))
x_vals = np.linspace(-2, 6, 100)
y1_vals = (b1[0] - A1[0, 0] * x_vals) / A1[0, 1]  # 2x + 3y = 8
y2_vals = (b1[1] - A1[1, 0] * x_vals) / A1[1, 1]  # 4x - y = 2

plt.plot(x_vals, y1_vals, 'b-', linewidth=2, label='2x + 3y = 8')
plt.plot(x_vals, y2_vals, 'r-', linewidth=2, label='4x - y = 2')
plt.plot(solution1[0], solution1[1], 'go', markersize=15, 
         label=f'Solution: ({solution1[0]:.2f}, {solution1[1]:.2f})', zorder=5)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('2x2 System: Intersection Point is the Solution', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('example1_2x2_system.png', dpi=300, bbox_inches='tight')
print("\n✓ Graph saved as 'example1_2x2_system.png'")

# =============================================================================
# EXAMPLE 2: 3x3 System
# =============================================================================
print("\n\n" + "="*70)
print("EXAMPLE 2: 3x3 System of Linear Equations")
print("="*70)

# System of equations:
# x + 2y + z = 9
# 2x - y + 3z = 8
# 3x + y - z = 3

print("\nEquations:")
print("  x + 2y + z = 9")
print("  2x - y + 3z = 8")
print("  3x + y - z = 3")

A2 = np.array([[1, 2, 1], 
               [2, -1, 3], 
               [3, 1, -1]])
b2 = np.array([9, 8, 3])

print("\nCoefficient Matrix A:")
print(A2)
print("\nConstants Vector b:")
print(b2)

# Solve the system
solution2 = np.linalg.solve(A2, b2)

print("\n" + "-"*50)
print("SOLUTION:")
print("-"*50)
print(f"x = {solution2[0]:.4f}")
print(f"y = {solution2[1]:.4f}")
print(f"z = {solution2[2]:.4f}")

# Verification
verification2 = np.dot(A2, solution2)
print(f"\nVerification (A @ solution = b):")
print(f"Expected: {b2}")
print(f"Got:      {verification2}")
print(f"Correct:  {np.allclose(verification2, b2)}")

# =============================================================================
# EXAMPLE 3: Real-World Problem - Store Pricing
# =============================================================================
print("\n\n" + "="*70)
print("EXAMPLE 3: Real-World Problem - Store Pricing")
print("="*70)

print("\nProblem:")
print("A store sells apples, bananas, and oranges.")
print("  Day 1: 2 apples + 3 bananas + 1 orange = $8")
print("  Day 2: 1 apple + 2 bananas + 3 oranges = $10")
print("  Day 3: 3 apples + 1 banana + 2 oranges = $9")
print("\nFind the price of each fruit.")

A3 = np.array([[2, 3, 1], 
               [1, 2, 3], 
               [3, 1, 2]])
b3 = np.array([8, 10, 9])

print("\nCoefficient Matrix A:")
print(A3)
print("\nConstants Vector b:")
print(b3)

# Solve
solution3 = np.linalg.solve(A3, b3)

print("\n" + "-"*50)
print("SOLUTION:")
print("-"*50)
print(f"Apple price:  ${solution3[0]:.2f}")
print(f"Banana price: ${solution3[1]:.2f}")
print(f"Orange price: ${solution3[2]:.2f}")

# Create DataFrame for better display
df_results = pd.DataFrame({
    'Item': ['Apple', 'Banana', 'Orange'],
    'Price ($)': solution3
})

print("\n" + "-"*50)
print("Results as Pandas DataFrame:")
print("-"*50)
print(df_results.to_string(index=False))

# Verification
verification3 = np.dot(A3, solution3)
print(f"\nVerification:")
print(f"Day 1: 2×{solution3[0]:.2f} + 3×{solution3[1]:.2f} + 1×{solution3[2]:.2f} = ${verification3[0]:.2f} (Expected: $8)")
print(f"Day 2: 1×{solution3[0]:.2f} + 2×{solution3[1]:.2f} + 3×{solution3[2]:.2f} = ${verification3[1]:.2f} (Expected: $10)")
print(f"Day 3: 3×{solution3[0]:.2f} + 1×{solution3[1]:.2f} + 2×{solution3[2]:.2f} = ${verification3[2]:.2f} (Expected: $9)")

# Visualize pricing
plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#FFD93D', '#FF8C42']
bars = plt.bar(df_results['Item'], df_results['Price ($)'], color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('Price ($)', fontsize=12)
plt.xlabel('Fruit', fontsize=12)
plt.title('Fruit Prices Solved Using Linear Equations', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:.2f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('example3_fruit_prices.png', dpi=300, bbox_inches='tight')
print("\n✓ Graph saved as 'example3_fruit_prices.png'")

# =============================================================================
# EXAMPLE 4: Checking Matrix Properties
# =============================================================================
print("\n\n" + "="*70)
print("EXAMPLE 4: Matrix Properties and Analysis")
print("="*70)

print("\nFor the store pricing matrix:")
print(A3)

# Calculate determinant
det = np.linalg.det(A3)
print(f"\nDeterminant: {det:.4f}")
print(f"  → Matrix is {'invertible' if det != 0 else 'NOT invertible'}")

# Calculate matrix inverse
if det != 0:
    A_inv = np.linalg.inv(A3)
    print("\nInverse Matrix:")
    print(A_inv)
    
    # Verify: A @ A_inv = Identity
    identity_check = np.dot(A3, A_inv)
    print("\nVerification (A @ A_inv should be Identity):")
    print(identity_check)

# Calculate matrix rank
rank = np.linalg.matrix_rank(A3)
print(f"\nMatrix Rank: {rank}")
print(f"  → System has a {'unique solution' if rank == A3.shape[0] else 'no unique solution'}")

# =============================================================================
# EXAMPLE 5: Multiple Solutions Using Pandas
# =============================================================================
print("\n\n" + "="*70)
print("EXAMPLE 5: Solving Multiple Systems with Pandas")
print("="*70)

# Create multiple systems to solve
systems = {
    'System 1': {'A': np.array([[1, 1], [1, -1]]), 'b': np.array([5, 1])},
    'System 2': {'A': np.array([[3, 2], [1, 4]]), 'b': np.array([7, 10])},
    'System 3': {'A': np.array([[2, 1], [5, 3]]), 'b': np.array([4, 11])}
}

results_list = []

for name, system in systems.items():
    sol = np.linalg.solve(system['A'], system['b'])
    results_list.append({
        'System': name,
        'x': sol[0],
        'y': sol[1]
    })

df_multiple = pd.DataFrame(results_list)
print("\nMultiple Systems Solved:")
print(df_multiple.to_string(index=False))

# Visualize all solutions
plt.figure(figsize=(12, 8))

for idx, (name, system) in enumerate(systems.items(), 1):
    plt.subplot(2, 2, idx)
    
    A = system['A']
    b = system['b']
    sol = np.linalg.solve(A, b)
    
    x_range = np.linspace(sol[0] - 5, sol[0] + 5, 100)
    
    # Plot both lines
    y1 = (b[0] - A[0, 0] * x_range) / A[0, 1]
    y2 = (b[1] - A[1, 0] * x_range) / A[1, 1]
    
    plt.plot(x_range, y1, 'b-', linewidth=2, label='Equation 1')
    plt.plot(x_range, y2, 'r-', linewidth=2, label='Equation 2')
    plt.plot(sol[0], sol[1], 'go', markersize=12, label=f'Solution', zorder=5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{name}: ({sol[0]:.2f}, {sol[1]:.2f})', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('example5_multiple_systems.png', dpi=300, bbox_inches='tight')
print("\n✓ Graph saved as 'example5_multiple_systems.png'")

# =============================================================================
# Summary
# =============================================================================
print("\n\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\n✓ All examples completed successfully!")
print("✓ Graphs saved:")
print("  - example1_2x2_system.png")
print("  - example3_fruit_prices.png")
print("  - example5_multiple_systems.png")
print("\nKey NumPy Functions Used:")
print("  • np.linalg.solve(A, b)  - Solve linear system")
print("  • np.linalg.det(A)       - Calculate determinant")
print("  • np.linalg.inv(A)       - Calculate inverse matrix")
print("  • np.linalg.matrix_rank(A) - Calculate matrix rank")
print("  • np.dot(A, x)           - Matrix multiplication")
print("  • np.allclose(a, b)      - Check if arrays are equal (with tolerance)")

plt.show()