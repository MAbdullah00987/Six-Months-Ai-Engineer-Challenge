
#Part 3: Deep Dive - Building Intuition
#SPECIAL MATRICES
#SPECIAL MATRICES AND THEIR PROPERTIES


special_matrices = {
    'Identity': np.eye(2),
    'Diagonal': np.array([[3, 0], [0, 2]]),
    'Symmetric': np.array([[2, 1], [1, 2]]),
    'Orthogonal': np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                            [np.sin(np.pi/6), np.cos(np.pi/6)]]),
    'Singular': np.array([[2, 4], [1, 2]]),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

unit_square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])

for idx, (name, matrix) in enumerate(special_matrices.items()):
    ax = axes[idx]
    
    # Plot original
    ax.plot(unit_square[0], unit_square[1], 'k--', linewidth=2, alpha=0.5)
    ax.fill(unit_square[0], unit_square[1], alpha=0.2, color='gray')
    
    # Plot transformed
    transformed = matrix @ unit_square
    ax.plot(transformed[0], transformed[1], linewidth=2.5, color='purple')
    ax.fill(transformed[0], transformed[1], alpha=0.3, color='purple')
    
    # Calculate properties
    det = np.linalg.det(matrix)
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        trace = np.trace(matrix)
    except:
        eigenvalues = None
        trace = None
    
    # Formatting
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    properties = f"det = {det:.3f}"
    if eigenvalues is not None:
        properties += f"\nλ = {eigenvalues.real}"
    
    ax.set_title(f'{name} Matrix\n{properties}', fontsize=11, fontweight='bold')
    
    # Add matrix text
    matrix_str = np.array2string(matrix, precision=2, suppress_small=True)
    ax.text(0.02, 0.02, matrix_str, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

# Remove extra subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

print("\nSPECIAL MATRIX PROPERTIES:")
print("-" * 70)

for name, matrix in special_matrices.items():
    print(f"\n{name} Matrix:")
    print(matrix)
    
    det = np.linalg.det(matrix)
    print(f"  Determinant: {det:.4f}")
    
    if name == 'Identity':
        print("  → Does NOTHING (leaves everything unchanged)")
        print("  → det = 1, all eigenvalues = 1")
    
    elif name == 'Diagonal':
        print("  → Scales each axis independently")
        print("  → Eigenvalues are the diagonal entries")
        print("  → Very easy to work with!")
    
    elif name == 'Symmetric':
        print("  → A = A^T (symmetric across diagonal)")
        print("  → Always has REAL eigenvalues")
        print("  → Eigenvectors are ORTHOGONAL")
    
    elif name == 'Orthogonal':
        print("  → Represents rotation/reflection")
        print("  → Preserves lengths and angles")
        print("  → A^T @ A = I (inverse = transpose)")
        print(f"  → det = ±1 (this one: {det:.4f})")
    
    elif name == 'Singular':
        print("  → det = 0, NOT invertible")
        print("  → Collapses space to lower dimension")
        print("  → Information is LOST")


# Another Example
# Example: Image transformation
print("\n1. IMAGE TRANSFORMATIONS")
print("   - Rotation: Use rotation matrices")
print("   - Scaling: Use diagonal matrices")
print("   - Shearing: Use shear matrices")

# Example: Solve linear systems
print("\n2. SOLVING LINEAR SYSTEMS")
A_system = np.array([[2, 1], [1, 3]])
b = np.array([5, 6])
x = np.linalg.solve(A_system, b)
print(f"   Solve Ax = b:")
print(f"   A = \n{A_system}")
print(f"   b = {b}")
print(f"   x = {x}")
print(f"   Verification: Ax = {A_system @ x} ✓")

# Example: Principal Component Analysis (PCA)
print("\n3. PCA (Principal Component Analysis)")
print("   - Find eigenvectors of covariance matrix")
print("   - Eigenvectors = principal components")
print("   - Eigenvalues = variance explained")
print("   - Used for dimensionality reduction!")

# Generate sample data
np.random.seed(42)
data = np.random.randn(100, 2) @ np.array([[2, 0.5], [0.5, 1]])

# Compute covariance
cov_matrix = np.cov(data.T)
eigenvalues_pca, eigenvectors_pca = np.linalg.eig(cov_matrix)

print(f"\n   Data covariance matrix:\n{cov_matrix}")
print(f"   Eigenvalues (variance): {eigenvalues_pca}")
print(f"   → First PC explains {eigenvalues_pca[0]/sum(eigenvalues_pca)*100:.1f}% variance")

# Visualize PCA
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=30)

# Plot principal components
for i, (val, vec) in enumerate(zip(eigenvalues_pca, eigenvectors_pca.T)):
    ax.quiver(0, 0, vec[0]*np.sqrt(val)*3, vec[1]*np.sqrt(val)*3,
              angles='xy', scale_units='xy', scale=1,
              color=['red', 'blue'][i], width=0.01,
              label=f'PC{i+1} (λ={val:.2f})')

ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.legend()
ax.set_title('PCA: Finding Principal Directions in Data', fontsize=14, fontweight='bold')
plt.show()

"""
 MATRIX MULTIPLICATION = TRANSFORMATION APPLICATION
   result = matrix @ vector

 DETERMINANT = AREA/VOLUME SCALING
   det = 0 → singular (loses dimension)
   |det| = scaling factor

 EIGENVALUES/EIGENVECTORS = SPECIAL DIRECTIONS
   Matrix @ eigenvector = eigenvalue * eigenvector
   These are the "axis" of the transformation

 INVERSE = UNDO THE TRANSFORMATION
   A @ inv(A) = I (if det ≠ 0)

 KEY NUMPY FUNCTIONS:
   - np.linalg.det()        → determinant
   - np.linalg.inv()        → inverse
   - np.linalg.eig()        → eigenvalues/eigenvectors
   - np.linalg.solve()      → solve Ax = b
   - @ operator             → matrix multiplication
"""