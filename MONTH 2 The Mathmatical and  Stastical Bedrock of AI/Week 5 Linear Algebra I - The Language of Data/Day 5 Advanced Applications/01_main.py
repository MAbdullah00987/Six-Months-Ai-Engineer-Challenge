
#Day 5: Advanced Applications
#Read: Mathematics for ML Chapter 3 (Intro) - Eigenvalues preview
#Review Coursera Week 4 materials - Matrix inverses and special matrices

#Project 1: Affine Transformations
#Implement translation + rotation + scaling
#Use homogeneous coordinates (3×3 matrices for 2D)
#Create animation of shape moving and rotating
#Apply to letter shapes or simple graphics


#Project 2: PageRank Algorithm (Simplified)
#Create 5-6 node web graph
##Build adjacency matrix
#Implement power iteration: vₖ₊₁ = M × vₖ
#Iterate until convergence
#Rank pages by final scores

#Exercise Set:
#Compute matrix inverses for 3×3 matrices
#Verify A × A⁻¹ = I for 5 examples

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


print("PART 1: EIGENVALUES AND EIGENVECTORS")

# Topic 1: Understanding Eigenvalues
print("\n--- Topic 1: What are Eigenvalues? ---")
print("When A*v = λ*v, λ is eigenvalue, v is eigenvector")
print("The vector v only gets scaled, not rotated!\n")

A1 = np.array([[4, 1],
               [2, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A1)

print(f"Matrix A:\n{A1}\n")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}\n")

# Verify: A*v = λ*v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = A1 @ v
    lam_v = lam * v
    print(f"Eigenvalue {i+1}: λ = {lam:.4f}")
    print(f"  A*v = {Av}")
    print(f"  λ*v = {lam_v}")
    print(f"  Match: {np.allclose(Av, lam_v)}\n")

# Topic 2: Geometric Interpretation
print("\n--- Topic 2: Geometric Visualization ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original vectors
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 1]
random_v = np.array([1, 0.5])

# After transformation
Av1 = A1 @ v1
Av2 = A1 @ v2
A_random = A1 @ random_v

# Plot 1: Before transformation
ax = axes[0]
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
          color='blue', width=0.01, label='Eigenvector 1')
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
          color='red', width=0.01, label='Eigenvector 2')
ax.quiver(0, 0, random_v[0], random_v[1], angles='xy', scale_units='xy', 
          scale=1, color='green', width=0.01, label='Random vector')
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Before Transformation')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Plot 2: After transformation
ax = axes[1]
ax.quiver(0, 0, Av1[0], Av1[1], angles='xy', scale_units='xy', scale=1, 
          color='blue', width=0.01, label=f'λ₁v₁ (λ={eigenvalues[0]:.2f})')
ax.quiver(0, 0, Av2[0], Av2[1], angles='xy', scale_units='xy', scale=1, 
          color='red', width=0.01, label=f'λ₂v₂ (λ={eigenvalues[1]:.2f})')
ax.quiver(0, 0, A_random[0], A_random[1], angles='xy', scale_units='xy', 
          scale=1, color='green', width=0.01, label='Random (rotated)')
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('After Transformation A*v')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('eigenvalue_geometry.png', dpi=150, bbox_inches='tight')
print("Saved: eigenvalue_geometry.png")
plt.show()

# Topic 3: Computing Eigenvalues (3x3 matrices)
print("\n--- Topic 3: Eigenvalues for 3×3 Matrices ---")

matrices_3x3 = [
    np.array([[6, -1, 0],
              [-1, 5, -1],
              [0, -1, 4]]),
    
    np.array([[1, 2, 3],
              [0, 4, 5],
              [0, 0, 6]]),
    
    np.array([[2, 1, 0],
              [1, 2, 1],
              [0, 1, 2]])
]

for idx, A in enumerate(matrices_3x3, 1):
    print(f"\nMatrix {idx}:\n{A}")
    eig_vals, eig_vecs = np.linalg.eig(A)
    print(f"Eigenvalues: {eig_vals}")
    
    # Check if matrix is symmetric
    is_symmetric = np.allclose(A, A.T)
    print(f"Symmetric: {is_symmetric}")
    
    if is_symmetric:
        print("→ Real eigenvalues guaranteed!")

# Topic 4: Applications - Power Iteration
print("\n--- Topic 4: Dominant Eigenvalue (Power Iteration) ---")

def power_iteration(A, num_iterations=10):
    """Find dominant eigenvalue using power iteration"""
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    eigenvalues = []
    for i in range(num_iterations):
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)
        eigenvalue = v.T @ A @ v
        eigenvalues.append(eigenvalue)
        v = v_new
    
    return eigenvalues

A_test = np.array([[4, 1],
                   [2, 3]])

iterations = power_iteration(A_test, 20)
true_max_eig = max(np.abs(np.linalg.eig(A_test)[0]))

plt.figure(figsize=(10, 5))
plt.plot(iterations, 'b-o', label='Power Iteration')
plt.axhline(true_max_eig, color='r', linestyle='--', 
            label=f'True Value: {true_max_eig:.4f}')
plt.xlabel('Iteration')
plt.ylabel('Eigenvalue Estimate')
plt.title('Power Iteration Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('power_iteration.png', dpi=150, bbox_inches='tight')
print("Saved: power_iteration.png")
plt.show()


print("PART 2: MATRIX INVERSES")


# Topic 5: Computing Matrix Inverses (3×3)
print("\n--- Topic 5: Computing 3×3 Matrix Inverses ---")

test_matrices = [
    np.array([[2, 1, 0],
              [1, 2, 1],
              [0, 1, 2]]),
    
    np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]]),
    
    np.array([[4, 7, 2],
              [3, 6, 1],
              [2, 5, 9]]),
    
    np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 0, 1]]),  # Singular matrix
    
    np.array([[3, 0, 2],
              [2, 0, -2],
              [0, 1, 1]])
]

for idx, A in enumerate(test_matrices, 1):
    print(f"\n{'='*50}")
    print(f"Example {idx}:")
    print(f"Matrix A:\n{A}")
    
    # Check determinant
    det = np.linalg.det(A)
    print(f"\nDeterminant: {det:.6f}")
    
    if abs(det) < 1e-10:
        print("⚠ Matrix is SINGULAR (not invertible)!")
        print("Determinant ≈ 0, no inverse exists")
        continue
    
    # Compute inverse
    A_inv = np.linalg.inv(A)
    print(f"\nInverse A⁻¹:\n{A_inv}")
    
    # Verify A × A⁻¹ = I
    product = A @ A_inv
    identity = np.eye(3)
    
    print(f"\nA × A⁻¹:\n{product}")
    print(f"\nIdentity I:\n{identity}")
    
    is_identity = np.allclose(product, identity)
    print(f"\n✓ Verification: A × A⁻¹ = I? {is_identity}")
    
    if is_identity:
        max_error = np.max(np.abs(product - identity))
        print(f"  Maximum error: {max_error:.2e}")

# Topic 6: Manual Inverse Computation (2×2)
print("\n--- Topic 6: Manual 2×2 Inverse Formula ---")
print("For 2×2 matrix: A = [[a, b], [c, d]]")
print("A⁻¹ = (1/det) × [[d, -b], [-c, a]]")

A_2x2 = np.array([[3, 4],
                  [2, 5]])

a, b = A_2x2[0, 0], A_2x2[0, 1]
c, d = A_2x2[1, 0], A_2x2[1, 1]

det = a*d - b*c
print(f"\nMatrix: {A_2x2.tolist()}")
print(f"det = ad - bc = {a}×{d} - {b}×{c} = {det}")

A_inv_manual = (1/det) * np.array([[d, -b],
                                    [-c, a]])
A_inv_numpy = np.linalg.inv(A_2x2)

print(f"\nManual inverse:\n{A_inv_manual}")
print(f"\nNumPy inverse:\n{A_inv_numpy}")
print(f"\nMatch: {np.allclose(A_inv_manual, A_inv_numpy)}")

# Topic 7: Condition Number
print("\n--- Topic 7: Matrix Condition Number ---")
print("Condition number = ||A|| × ||A⁻¹||")
print("High condition number → ill-conditioned → numerical instability\n")

test_cond_matrices = {
    "Well-conditioned": np.array([[4, 1], [1, 3]]),
    "Moderate": np.array([[10, 7], [8, 7]]),
    "Ill-conditioned": np.array([[1, 1], [1, 1.0001]])
}

cond_numbers = []
labels = []

for name, A in test_cond_matrices.items():
    cond = np.linalg.cond(A)
    cond_numbers.append(cond)
    labels.append(name)
    print(f"{name:20s}: {cond:.2e}")

plt.figure(figsize=(10, 5))
colors = ['green', 'orange', 'red']
bars = plt.bar(labels, cond_numbers, color=colors, alpha=0.7)
plt.yscale('log')
plt.ylabel('Condition Number (log scale)')
plt.title('Matrix Condition Numbers')
plt.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, cond_numbers):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2e}', ha='center', va='bottom')

plt.savefig('condition_numbers.png', dpi=150, bbox_inches='tight')
print("\nSaved: condition_numbers.png")
plt.show()

# Topic 8: Special Matrices
print("\n--- Topic 8: Special Matrices ---")

# Orthogonal matrix
print("\n1. ORTHOGONAL MATRIX")
print("   Q^T × Q = I, Q⁻¹ = Q^T")
theta = np.pi / 4
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
print(f"Rotation matrix (45°):\n{Q}")
print(f"Q^T:\n{Q.T}")
print(f"Q^T × Q:\n{Q.T @ Q}")
print(f"Is orthogonal: {np.allclose(Q.T @ Q, np.eye(2))}")

# Symmetric matrix
print("\n2. SYMMETRIC MATRIX")
print("   A = A^T")
S = np.array([[4, 1, 2],
              [1, 5, 3],
              [2, 3, 6]])
print(f"Matrix S:\n{S}")
print(f"S^T:\n{S.T}")
print(f"Is symmetric: {np.allclose(S, S.T)}")
eig_s = np.linalg.eig(S)[0]
print(f"Eigenvalues (all real): {eig_s}")

# Diagonal matrix
print("\n3. DIAGONAL MATRIX")
D = np.diag([2, 5, 3])
print(f"Diagonal matrix:\n{D}")
D_inv = np.diag(1/np.diag(D))
print(f"Inverse (flip diagonal):\n{D_inv}")
print(f"D × D⁻¹:\n{D @ D_inv}")

# Topic 9: Solving Linear Systems
print("\n--- Topic 9: Solving Ax = b using Inverse ---")

A = np.array([[2, 1, 1],
              [1, 3, 2],
              [1, 0, 0]])
b = np.array([4, 5, 6])

print(f"System: Ax = b")
print(f"A:\n{A}")
print(f"b: {b}")

# Method 1: Using inverse
x_inv = np.linalg.inv(A) @ b
print(f"\nSolution using A⁻¹: x = A⁻¹b")
print(f"x = {x_inv}")

# Method 2: Using solve (more efficient)
x_solve = np.linalg.solve(A, b)
print(f"\nSolution using solve: {x_solve}")

# Verify
print(f"\nVerification Ax = {A @ x_inv}")
print(f"Original b = {b}")
print(f"Match: {np.allclose(A @ x_inv, b)}")

# Topic 10: Eigenvalue Heatmap
print("\n--- Topic 10: Eigenvalue Spectrum Visualization ---")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

matrices_viz = {
    "Symmetric": np.array([[4, 1, 2], [1, 5, 3], [2, 3, 6]]),
    "Upper Triangular": np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]]),
    "Random": np.random.randn(3, 3),
    "Positive Definite": None
}

# Create positive definite
A_pd = np.random.randn(3, 3)
matrices_viz["Positive Definite"] = A_pd.T @ A_pd

for ax, (name, A) in zip(axes.flat, matrices_viz.items()):
    eig_vals = np.linalg.eig(A)[0]
    
    # Plot eigenvalues in complex plane
    ax.scatter(eig_vals.real, eig_vals.imag, s=100, c='red', 
               edgecolors='black', linewidths=2, zorder=3)
    
    for i, ev in enumerate(eig_vals):
        ax.annotate(f'λ{i+1}', (ev.real, ev.imag), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(f'{name}\nEigenvalues: {eig_vals}')

plt.tight_layout()
plt.savefig('eigenvalue_spectrum.png', dpi=150, bbox_inches='tight')
print("Saved: eigenvalue_spectrum.png")
plt.show()

# Topic 11: Matrix Inverse Heatmap
print("\n--- Topic 11: Matrix and Inverse Heatmaps ---")

A_heat = np.array([[4, 1, 2],
                   [1, 5, 3],
                   [2, 3, 6]])
A_inv_heat = np.linalg.inv(A_heat)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original matrix
sns.heatmap(A_heat, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[0], cbar_kws={'label': 'Value'})
axes[0].set_title('Matrix A')

# Inverse
sns.heatmap(A_inv_heat, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, ax=axes[1], cbar_kws={'label': 'Value'})
axes[1].set_title('Inverse A⁻¹')

# Product
product = A_heat @ A_inv_heat
sns.heatmap(product, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, ax=axes[2], cbar_kws={'label': 'Value'})
axes[2].set_title('A × A⁻¹ (Identity)')

plt.tight_layout()
plt.savefig('matrix_inverse_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: matrix_inverse_heatmap.png")
plt.show()

'''
KEY CONCEPTS COVERED:

EIGENVALUES & EIGENVECTORS:
 Definition: Av = λv
 Geometric meaning: direction-preserving vectors
 Computing for 2×2 and 3×3 matrices
 Power iteration algorithm
 Visualization in 2D space

MATRIX INVERSES:
 Definition: A × A⁻¹ = I
 Computing 3×3 inverses
 5 verified examples
 Determinant check
 Condition numbers

SPECIAL MATRICES:
 Orthogonal matrices
 Symmetric matrices
 Diagonal matrices

APPLICATIONS:
 Solving linear systems
 Numerical stability
 Transformation visualization

'''