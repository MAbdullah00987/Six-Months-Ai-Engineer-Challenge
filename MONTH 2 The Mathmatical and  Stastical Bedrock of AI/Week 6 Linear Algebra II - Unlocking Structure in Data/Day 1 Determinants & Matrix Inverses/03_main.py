#Hands-On Practice Exercises with Solutions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("HANDS-ON PRACTICE EXERCISES")
print("Determinants and Matrix Inverses")
print("="*70)

# ============================================================================
# EXERCISE 1: COMPUTE DETERMINANTS BY HAND AND VERIFY
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 1: Computing Determinants")
print("="*70)

print("\n--- Problem 1.1: 2×2 Matrix ---")
A = np.array([[5, 3],
              [2, 1]])
print("Matrix A =")
print(A)
print("\nCompute det(A) by hand using: ad - bc")
print("Your calculation: (5)(1) - (3)(2) = 5 - 6 = -1")
print(f"NumPy verification: {np.linalg.det(A):.4f}")

print("\n--- Problem 1.2: 3×3 Matrix ---")
B = np.array([[2, 1, 3],
              [1, 0, 1],
              [0, 2, 1]])
print("Matrix B =")
print(B)
print("\nUsing cofactor expansion along first row:")
print("det(B) = 2*|0 1| - 1*|1 1| + 3*|1 0|")
print("           |2 1|     |0 1|     |0 2|")
print("       = 2*(0-2) - 1*(1-0) + 3*(2-0)")
print("       = 2*(-2) - 1 + 3*2")
print("       = -4 - 1 + 6 = 1")
print(f"NumPy verification: {np.linalg.det(B):.4f}")

# ============================================================================
# EXERCISE 2: DETERMINE INVERTIBILITY
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 2: Check Invertibility")
print("="*70)

test_matrices = [
    ("Matrix C", np.array([[1, 2], [3, 4]])),
    ("Matrix D", np.array([[2, 6], [1, 3]])),
    ("Matrix E", np.array([[1, 0, 2], [2, 1, 0], [3, 2, 1]])),
    ("Matrix F", np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])),
]

for name, matrix in test_matrices:
    print(f"\n--- {name} ---")
    print(matrix)
    det = np.linalg.det(matrix)
    print(f"det = {det:.6f}")
    
    if abs(det) < 1e-10:
        print("❌ NOT INVERTIBLE (det ≈ 0)")
        print("Why? The rows/columns are linearly dependent.")
    else:
        print("✓ INVERTIBLE (det ≠ 0)")
        print("You can find the inverse!")

# ============================================================================
# EXERCISE 3: COMPUTE MATRIX INVERSES
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 3: Computing Matrix Inverses")
print("="*70)

print("\n--- Problem 3.1: Simple 2×2 Inverse ---")
G = np.array([[4, 7],
              [2, 6]])
print("Matrix G =")
print(G)

det_G = np.linalg.det(G)
print(f"\nStep 1: Check determinant = {det_G:.4f}")

if det_G != 0:
    print("Step 2: Matrix is invertible, compute inverse")
    G_inv = np.linalg.inv(G)
    print("\nG^(-1) =")
    print(G_inv)
    
    print("\nStep 3: Verify G × G^(-1) = I")
    identity = G @ G_inv
    print(identity)
    print("✓ Verification successful!")
    
    # Manual calculation for 2x2
    print("\n--- Manual 2×2 Inverse Formula ---")
    print("For A = [a b]")
    print("        [c d]")
    print("\nA^(-1) = (1/det) × [ d  -b]")
    print("                    [-c   a]")
    print(f"\nFor our matrix: det = {det_G}")
    a, b = G[0]
    c, d = G[1]
    print(f"G^(-1) = (1/{det_G:.1f}) × [{d:2.0f} {-b:3.0f}]")
    print(f"                        [{-c:2.0f}  {a:2.0f}]")
    manual_inv = (1/det_G) * np.array([[d, -b], [-c, a]])
    print("\nManual calculation result:")
    print(manual_inv)
    print(f"Matches NumPy? {np.allclose(manual_inv, G_inv)}")

print("\n--- Problem 3.2: 3×3 Matrix Inverse ---")
H = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])
print("Matrix H =")
print(H)

det_H = np.linalg.det(H)
print(f"\nDeterminant = {det_H:.4f}")

if det_H != 0:
    H_inv = np.linalg.inv(H)
    print("\nH^(-1) =")
    print(H_inv)
    
    print("\nVerification: H × H^(-1) =")
    print(H @ H_inv)

# ============================================================================
# EXERCISE 4: SOLVING LINEAR SYSTEMS
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 4: Solving Linear Systems Using Inverses")
print("="*70)

print("\nSolve: Ax = b")
print("\nWhere A and b are:")

A = np.array([[2, 1],
              [1, 3]])
b = np.array([5, 7])

print("A =")
print(A)
print(f"\nb = {b}")

print("\nMethod 1: Using matrix inverse")
print("x = A^(-1) × b")

det_A = np.linalg.det(A)
print(f"\ndet(A) = {det_A:.4f} (invertible!)")

A_inv = np.linalg.inv(A)
x_inv = A_inv @ b

print("\nA^(-1) =")
print(A_inv)
print(f"\nx = A^(-1) × b = {x_inv}")

print("\nMethod 2: Using np.linalg.solve (more efficient)")
x_solve = np.linalg.solve(A, b)
print(f"x = {x_solve}")

print("\nVerification: A × x = b?")
print(f"A × x = {A @ x_inv}")
print(f"b = {b}")
print(f"Match? {np.allclose(A @ x_inv, b)}")

# ============================================================================
# EXERCISE 5: AREA TRANSFORMATION CHALLENGE
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 5: Area Transformation Challenge")
print("="*70)

transformations = {
    "Scaling by 2": np.array([[2, 0], [0, 2]]),
    "Horizontal shear": np.array([[1, 2], [0, 1]]),
    "Vertical compression": np.array([[1, 0], [0, 0.5]]),
}

fig, axes = plt.subplots(1, len(transformations), figsize=(15, 5))

for idx, (name, T) in enumerate(transformations.items()):
    # Unit square
    square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])
    
    # Transform
    transformed = T @ square
    
    # Calculate areas
    det = np.linalg.det(T)
    
    # Plot
    ax = axes[idx]
    ax.fill(square[0], square[1], alpha=0.3, color='blue', label='Original')
    ax.fill(transformed[0], transformed[1], alpha=0.3, color='red', label='Transformed')
    ax.plot(square[0], square[1], 'b-', linewidth=2)
    ax.plot(transformed[0], transformed[1], 'r-', linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 3)
    ax.set_title(f'{name}\ndet = {det:.2f}\nArea × {abs(det):.2f}', fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.savefig('area_transformations.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFor each transformation:")
for name, T in transformations.items():
    det = np.linalg.det(T)
    print(f"\n{name}:")
    print(f"  Matrix: {T.tolist()}")
    print(f"  Determinant: {det:.4f}")
    print(f"  Original area: 1")
    print(f"  Transformed area: {abs(det):.4f}")
    print(f"  Area multiplied by: {abs(det):.4f}")

# ============================================================================
# EXERCISE 6: TRUE/FALSE QUIZ
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 6: True/False Quiz (Test Your Understanding)")
print("="*70)

quiz = [
    ("If det(A) = 0, then A is invertible", False, 
     "False. det(A) = 0 means A is SINGULAR (not invertible)"),
    
    ("If det(A) = 5, then A is invertible", True, 
     "True. Any non-zero determinant means the matrix is invertible"),
    
    ("det(AB) = det(A) + det(B)", False, 
     "False. det(AB) = det(A) × det(B) (multiplication, not addition)"),
    
    ("det(A^T) = det(A)", True, 
     "True. The determinant of a transpose equals the original determinant"),
    
    ("If det(A) = -2, the transformation includes a reflection", True, 
     "True. Negative determinant indicates reflection"),
    
    ("All diagonal matrices are invertible", False, 
     "False. Only if all diagonal elements are non-zero"),
    
    ("The identity matrix has determinant 1", True, 
     "True. det(I) = 1 always"),
    
    ("If A is 3×3 and det(A) = 2, then det(2A) = 4", False, 
     "False. det(2A) = 2³ × det(A) = 8 × 2 = 16"),
]

correct = 0
for i, (question, answer, explanation) in enumerate(quiz, 1):
    print(f"\n{i}. {question}")
    print(f"   Answer: {answer}")
    print(f"   Explanation: {explanation}")
    correct += 1

print(f"\n{'='*70}")
print(f"Quiz complete! Review the explanations above.")
print(f"{'='*70}")

# ============================================================================
# EXERCISE 7: PRACTICE PROBLEMS (DO YOURSELF)
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 7: Additional Practice Problems")
print("="*70)

practice_problems = """
Try these problems yourself, then verify with NumPy:

1. Compute the determinant of:
   A = [[3, 1],
        [2, 4]]
   
   Answer: _______

2. Is this matrix invertible? Why or why not?
   B = [[1, 2],
        [2, 4]]
   
   Answer: _______

3. Find the inverse of:
   C = [[1, 2],
        [3, 5]]
   
   Answer: C^(-1) = _______

4. What is the area scaling factor for:
   D = [[2, 0],
        [0, 3]]
   
   Answer: _______

5. If det(A) = 5 and det(B) = 3, what is det(AB)?
   
   Answer: _______
"""

print(practice_problems)

# Provide answers for verification
print("\n--- SOLUTIONS FOR VERIFICATION ---")
print("\n1. det(A) = 3*4 - 1*2 = 10")
A1 = np.array([[3, 1], [2, 4]])
print(f"   NumPy: {np.linalg.det(A1):.4f}")

print("\n2. B is NOT invertible because det(B) = 0")
B1 = np.array([[1, 2], [2, 4]])
print(f"   NumPy: {np.linalg.det(B1):.6f}")
print("   Note: 2nd row = 2 × 1st row (linearly dependent)")

print("\n3. C^(-1):")
C1 = np.array([[1, 2], [3, 5]])
C1_inv = np.linalg.inv(C1)
print(C1_inv)

print("\n4. Area scaling = |det(D)| = |6| = 6")
D1 = np.array([[2, 0], [0, 3]])
print(f"   NumPy: {abs(np.linalg.det(D1)):.4f}")

print("\n5. det(AB) = det(A) × det(B) = 5 × 3 = 15")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("🎓 CONGRATULATIONS! You've completed all exercises!")
print("="*70)

summary_table = pd.DataFrame({
    'Concept': [
        'Determinant = 0',
        'Determinant ≠ 0',
        'det(AB)',
        'det(A^-1)',
        'det(A^T)',
        '|det(A)|',
        'Negative det'
    ],
    'Meaning': [
        'Matrix is SINGULAR (not invertible)',
        'Matrix is INVERTIBLE',
        'det(A) × det(B)',
        '1 / det(A)',
        'det(A)',
        'Area/volume scaling factor',
        'Includes reflection'
    ],
    'Example': [
        '[[1,2],[2,4]]',
        '[[1,2],[3,4]]',
        '-',
        '-',
        '-',
        '2×2 → area × |det|',
        'Mirror transformation'
    ]
})

print("\n📋 Quick Reference Table:")
print(summary_table.to_string(index=False))

print("\n✅ Key Skills Mastered:")
print("  • Computing determinants (2×2, 3×3)")
print("  • Checking invertibility")
print("  • Finding matrix inverses")
print("  • Understanding geometric meaning")
print("  • Solving linear systems")
print("  • Visualizing transformations")

print("\n🎯 Next Steps:")
print("  1. Review the BITS Pilani lectures on determinants")
print("  2. Practice more problems from your textbook")
print("  3. Move on to eigenvalues and eigenvectors")
print("  4. Apply to real machine learning problems")

print("\n" + "="*70)