#Day 6: Integration & Review
#Goal: Consolidate all Week 6 concepts and see the big picture.

#Create a concept map connecting: determinants → invertibility → eigenvalues → PCA → SVD
#Work through practice problems from the textbook

#Afternoon: Comprehensive Project (3-4 hours)
#Build a "Linear Algebra Toolkit" - Create a single Jupyter notebook or web app that includes:

#All 10 mini-projects from the week
#Interactive widgets to adjust parameters
#Visualizations for each concept
#Written explanations connecting theory to code

#Write a summary of what you learned
#Test yourself: Can you explain each concept to someone else?
#Identify any weak areas to revisit
#Preview Week 7 materials

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, make_blobs

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("=" * 80)
print("WEEK 6 INTEGRATION: CONNECTING LINEAR ALGEBRA CONCEPTS")
print("=" * 80)
print("\nConcept Chain: Determinants → Invertibility → Eigenvalues → PCA → SVD")
print("=" * 80)

# ============================================================================
# CONCEPT 1: DETERMINANTS
# ============================================================================
print("\n" + "="*80)
print("CONCEPT 1: DETERMINANTS")
print("="*80)
print("\nWhat is a Determinant?")
print("- Scalar value that encodes geometric properties of a matrix")
print("- Represents the scaling factor of transformation")
print("- For 2D: area scaling; For 3D: volume scaling")

# Example 1: 2x2 Matrix
print("\n--- Example 1.1: 2x2 Matrix ---")
A = np.array([[3, 1],
              [2, 4]])
det_A = np.linalg.det(A)
print(f"Matrix A:\n{A}")
print(f"Determinant: {det_A:.4f}")
print(f"Interpretation: Transforms area by factor of {det_A:.4f}")

# Example 2: 3x3 Matrix
print("\n--- Example 1.2: 3x3 Matrix ---")
B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
det_B = np.linalg.det(B)
print(f"Matrix B:\n{B}")
print(f"Determinant: {det_B:.4f}")
print(f"Note: det ≈ 0 means matrix is singular (not invertible)")

# Example 3: Effect of scaling
print("\n--- Example 1.3: Determinant Properties ---")
C = np.array([[2, 0],
              [0, 3]])
det_C = np.linalg.det(C)
print(f"Diagonal Matrix C:\n{C}")
print(f"Determinant: {det_C:.4f}")
print(f"For diagonal matrix: det = product of diagonal = {2*3}")

# Visualization 1: Determinant as Area Scaling
fig1 = plt.figure(figsize=(15, 5))

# Original unit square
ax1 = fig1.add_subplot(131)
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])
ax1.plot(square[0], square[1], 'b-', linewidth=2, label='Unit Square')
ax1.fill(square[0], square[1], alpha=0.3, color='blue')
ax1.set_xlim(-0.5, 5)
ax1.set_ylim(-0.5, 5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Original Unit Square\nArea = 1', fontweight='bold')
ax1.legend()

# Transformed by matrix A
ax2 = fig1.add_subplot(132)
transformed = A @ square
ax2.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed')
ax2.fill(transformed[0], transformed[1], alpha=0.3, color='red')
ax2.set_xlim(-0.5, 5)
ax2.set_ylim(-0.5, 5)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title(f'After Transformation by A\nArea = {det_A:.2f}', fontweight='bold')
ax2.legend()

# Comparison
ax3 = fig1.add_subplot(133)
ax3.plot(square[0], square[1], 'b-', linewidth=2, alpha=0.5, label='Original')
ax3.fill(square[0], square[1], alpha=0.2, color='blue')
ax3.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed')
ax3.fill(transformed[0], transformed[1], alpha=0.3, color='red')
ax3.set_xlim(-0.5, 5)
ax3.set_ylim(-0.5, 5)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.set_title(f'Overlay\nScaling Factor = {det_A:.2f}', fontweight='bold')
ax3.legend()

plt.tight_layout()
plt.savefig('1_determinants.png', dpi=300, bbox_inches='tight')
print("\n✓ Determinant visualization saved")
plt.close()

# ============================================================================
# CONCEPT 2: INVERTIBILITY
# ============================================================================
print("\n" + "="*80)
print("CONCEPT 2: INVERTIBILITY")
print("="*80)
print("\nConnection to Determinants:")
print("- Matrix is invertible ⟺ det(A) ≠ 0")
print("- Inverse undoes the transformation: A @ A⁻¹ = I")
print("- If det(A) = 0, matrix is singular (no inverse)")

# Example 1: Invertible Matrix
print("\n--- Example 2.1: Invertible Matrix ---")
D = np.array([[4, 7],
              [2, 6]])
det_D = np.linalg.det(D)
print(f"Matrix D:\n{D}")
print(f"Determinant: {det_D:.4f}")

if det_D != 0:
    D_inv = np.linalg.inv(D)
    print(f"\nInverse D⁻¹:\n{D_inv}")
    
    # Verify: D @ D_inv = I
    identity = D @ D_inv
    print(f"\nVerification D @ D⁻¹:\n{identity}")
    print(f"Is identity? {np.allclose(identity, np.eye(2))}")

# Example 2: Singular Matrix
print("\n--- Example 2.2: Singular Matrix (Not Invertible) ---")
E = np.array([[1, 2],
              [2, 4]])
det_E = np.linalg.det(E)
print(f"Matrix E:\n{E}")
print(f"Determinant: {det_E:.10f}")
print("Note: Rows are linearly dependent (row2 = 2 * row1)")
print("→ Matrix is singular, cannot be inverted")

try:
    E_inv = np.linalg.inv(E)
except np.linalg.LinAlgError:
    print("✗ Cannot compute inverse (as expected)")

# Example 3: Solving Linear Systems
print("\n--- Example 2.3: Solving Linear Systems ---")
print("System: Ax = b")
F = np.array([[3, 1],
              [1, 2]])
b = np.array([9, 8])
print(f"Matrix F:\n{F}")
print(f"Vector b: {b}")

# Method 1: Using inverse
x_inv = np.linalg.inv(F) @ b
print(f"\nSolution (using inverse): x = {x_inv}")

# Method 2: Using solve (more efficient)
x_solve = np.linalg.solve(F, b)
print(f"Solution (using solve): x = {x_solve}")

# Verify
print(f"Verification F @ x = {F @ x_solve}")

# Visualization 2: Invertibility
fig2 = plt.figure(figsize=(15, 5))

# Invertible matrix transformation
ax1 = fig2.add_subplot(131)
original = np.array([[0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 0]])
transformed_inv = D @ original
inverse_transformed = D_inv @ transformed_inv

ax1.plot(original[0], original[1], 'b-', linewidth=2, label='Original', marker='o')
ax1.plot(transformed_inv[0], transformed_inv[1], 'r-', linewidth=2, label='D @ original', marker='s')
ax1.plot(inverse_transformed[0], inverse_transformed[1], 'g--', linewidth=2, label='D⁻¹ @ (D @ original)', marker='^')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_title('Invertible Matrix\nTransform → Inverse → Original', fontweight='bold')

# Singular matrix compression
ax2 = fig2.add_subplot(132)
transformed_sing = E @ original
ax2.plot(original[0], original[1], 'b-', linewidth=2, label='Original', marker='o')
ax2.plot(transformed_sing[0], transformed_sing[1], 'r-', linewidth=2, label='E @ original (collapsed)', marker='s')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_title('Singular Matrix\nInformation Loss (No Inverse)', fontweight='bold')

# Determinant vs Invertibility
ax3 = fig2.add_subplot(133)
matrices = ['D\n(Invertible)', 'E\n(Singular)']
dets = [abs(det_D), abs(det_E)]
colors = ['green', 'red']
bars = ax3.bar(matrices, dets, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.axhline(y=0.001, color='orange', linestyle='--', linewidth=2, label='Near-zero threshold')
ax3.set_ylabel('|Determinant|', fontweight='bold')
ax3.set_title('Determinant Comparison', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('2_invertibility.png', dpi=300, bbox_inches='tight')
print("\n✓ Invertibility visualization saved")
plt.close()

# ============================================================================
# CONCEPT 3: EIGENVALUES & EIGENVECTORS
# ============================================================================
print("\n" + "="*80)
print("CONCEPT 3: EIGENVALUES & EIGENVECTORS")
print("="*80)
print("\nConnection to Previous Concepts:")
print("- Eigenvalues relate to determinant: det(A) = product of eigenvalues")
print("- Eigenvectors are special directions that don't rotate under transformation")
print("- Equation: Av = λv (v is eigenvector, λ is eigenvalue)")

# Example 1: Computing Eigenvalues
print("\n--- Example 3.1: Computing Eigenvalues & Eigenvectors ---")
G = np.array([[4, 2],
              [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(G)
print(f"Matrix G:\n{G}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: A @ v = λ @ v
print("\n--- Verification ---")
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = G @ v
    lam_v = lam * v
    print(f"\nEigenvector {i+1}: {v}")
    print(f"G @ v = {Av}")
    print(f"λ * v = {lam_v}")
    print(f"Equal? {np.allclose(Av, lam_v)}")

# Verify determinant relationship
det_G = np.linalg.det(G)
prod_eigenvalues = np.prod(eigenvalues)
print(f"\ndet(G) = {det_G:.4f}")
print(f"Product of eigenvalues = {prod_eigenvalues:.4f}")
print(f"Equal? {np.allclose(det_G, prod_eigenvalues)}")

# Example 2: Symmetric Matrix (Real Eigenvalues)
print("\n--- Example 3.2: Symmetric Matrix ---")
H = np.array([[3, 1],
              [1, 3]])
eigenvalues_H, eigenvectors_H = np.linalg.eig(H)
print(f"Symmetric Matrix H:\n{H}")
print(f"Eigenvalues: {eigenvalues_H}")
print("Note: Symmetric matrices always have real eigenvalues")
print("      and orthogonal eigenvectors")

# Check orthogonality
v1 = eigenvectors_H[:, 0]
v2 = eigenvectors_H[:, 1]
dot_product = np.dot(v1, v2)
print(f"\nDot product of eigenvectors: {dot_product:.10f}")
print(f"Orthogonal? {np.allclose(dot_product, 0)}")

# Visualization 3: Eigenvectors
fig3 = plt.figure(figsize=(15, 5))

# Original vectors
ax1 = fig3.add_subplot(131)
# Create grid of vectors
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)
UV = np.array([X.flatten(), Y.flatten()])

ax1.quiver(X, Y, X, Y, alpha=0.3, color='gray', scale=30)
# Plot eigenvectors
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    ax1.arrow(0, 0, v[0]*2, v[1]*2, head_width=0.2, head_length=0.2, 
              fc=f'C{i}', ec=f'C{i}', linewidth=3, label=f'Eigenvector {i+1}')
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_title('Eigenvectors (Special Directions)', fontweight='bold')

# After transformation
ax2 = fig3.add_subplot(132)
UV_transformed = G @ UV
X_t = UV_transformed[0].reshape(X.shape)
Y_t = UV_transformed[1].reshape(Y.shape)
ax2.quiver(X, Y, X_t, Y_t, alpha=0.3, color='gray', scale=30)

# Transformed eigenvectors
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    v_transformed = G @ v
    ax2.arrow(0, 0, v[0]*2, v[1]*2, head_width=0.2, head_length=0.2, 
              fc=f'C{i}', ec=f'C{i}', linewidth=2, alpha=0.3, linestyle='--')
    ax2.arrow(0, 0, v_transformed[0]*2, v_transformed[1]*2, head_width=0.2, head_length=0.2, 
              fc=f'C{i}', ec=f'C{i}', linewidth=3, label=f'λ{i+1}={eigenvalues[i]:.2f}')
ax2.set_xlim(-8, 8)
ax2.set_ylim(-8, 8)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_title('After Transformation\n(Eigenvectors only scaled)', fontweight='bold')

# Eigenvalue spectrum
ax3 = fig3.add_subplot(133)
indices = np.arange(len(eigenvalues))
ax3.bar(indices, eigenvalues.real, alpha=0.7, color=['C0', 'C1'], edgecolor='black', linewidth=2)
ax3.set_xlabel('Eigenvalue Index', fontweight='bold')
ax3.set_ylabel('Eigenvalue', fontweight='bold')
ax3.set_title('Eigenvalue Spectrum', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(indices)

plt.tight_layout()
plt.savefig('3_eigenvalues.png', dpi=300, bbox_inches='tight')
print("\n✓ Eigenvalue visualization saved")
plt.close()

# ============================================================================
# CONCEPT 4: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================
print("\n" + "="*80)
print("CONCEPT 4: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*80)
print("\nConnection to Eigenvalues:")
print("- PCA finds directions of maximum variance")
print("- Principal components = eigenvectors of covariance matrix")
print("- Eigenvalues = variance explained by each component")

# Load dataset
print("\n--- Example 4.1: PCA on Iris Dataset ---")
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"Original data shape: {X_iris.shape}")
print(f"Features: {iris.feature_names}")

# Standardize data
X_mean = np.mean(X_iris, axis=0)
X_std = np.std(X_iris, axis=0)
X_standardized = (X_iris - X_mean) / X_std

# Compute covariance matrix
cov_matrix = np.cov(X_standardized.T)
print(f"\nCovariance matrix shape: {cov_matrix.shape}")

# Compute eigenvalues of covariance matrix
cov_eigenvalues, cov_eigenvectors = np.linalg.eig(cov_matrix)
print(f"\nEigenvalues of covariance matrix: {cov_eigenvalues}")
print(f"These represent variance along each principal component")

# Sort eigenvalues
sorted_indices = np.argsort(cov_eigenvalues)[::-1]
cov_eigenvalues = cov_eigenvalues[sorted_indices]
cov_eigenvectors = cov_eigenvectors[:, sorted_indices]

# Compute explained variance
explained_variance_ratio = cov_eigenvalues / np.sum(cov_eigenvalues)
print(f"\nExplained variance ratio: {explained_variance_ratio}")
print(f"First 2 components explain: {np.sum(explained_variance_ratio[:2])*100:.2f}% of variance")

# Apply PCA using sklearn
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)
print(f"\nPCA-transformed data shape: {X_pca.shape}")
print(f"Explained variance by sklearn: {pca.explained_variance_ratio_}")

# Visualization 4: PCA
fig4 = plt.figure(figsize=(18, 5))

# Original data (first 2 features)
ax1 = fig4.add_subplot(131)
for i, target_name in enumerate(iris.target_names):
    mask = y_iris == i
    ax1.scatter(X_standardized[mask, 0], X_standardized[mask, 1], 
                label=target_name, alpha=0.6, s=50)
ax1.set_xlabel(iris.feature_names[0], fontweight='bold')
ax1.set_ylabel(iris.feature_names[1], fontweight='bold')
ax1.set_title('Original Features\n(First 2 of 4)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# PCA projection
ax2 = fig4.add_subplot(132)
for i, target_name in enumerate(iris.target_names):
    mask = y_iris == i
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                label=target_name, alpha=0.6, s=50)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
ax2.set_title('PCA Projection\n(Maximum Variance Directions)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Explained variance
ax3 = fig4.add_subplot(133)
components = np.arange(1, len(pca.explained_variance_ratio_)+1)
cumsum = np.cumsum(pca.explained_variance_ratio_)
ax3.bar(components, pca.explained_variance_ratio_, alpha=0.7, color='skyblue', 
        edgecolor='black', linewidth=2, label='Individual')
ax3.plot(components, cumsum, 'ro-', linewidth=2, markersize=8, label='Cumulative')
ax3.set_xlabel('Principal Component', fontweight='bold')
ax3.set_ylabel('Explained Variance Ratio', fontweight='bold')
ax3.set_title('Variance Explained by PCs', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(components)

plt.tight_layout()
plt.savefig('4_pca.png', dpi=300, bbox_inches='tight')
print("\n✓ PCA visualization saved")
plt.close()

# ============================================================================
# CONCEPT 5: SINGULAR VALUE DECOMPOSITION (SVD)
# ============================================================================
print("\n" + "="*80)
print("CONCEPT 5: SINGULAR VALUE DECOMPOSITION (SVD)")
print("="*80)
print("\nConnection to PCA:")
print("- SVD is more general than eigendecomposition")
print("- Works on any matrix (not just square)")
print("- A = U Σ Vᵀ")
print("- Σ contains singular values (related to eigenvalues)")
print("- PCA can be computed via SVD")

# Example 1: SVD on rectangular matrix
print("\n--- Example 5.1: SVD Decomposition ---")
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
print(f"Matrix M shape: {M.shape}")
print(f"Matrix M:\n{M}")

# Compute SVD
U, S, Vt = np.linalg.svd(M, full_matrices=False)
print(f"\nU shape: {U.shape} (left singular vectors)")
print(f"S shape: {S.shape} (singular values)")
print(f"Vᵀ shape: {Vt.shape} (right singular vectors)")
print(f"\nSingular values: {S}")

# Reconstruct matrix
Sigma = np.diag(S)
M_reconstructed = U @ Sigma @ Vt
print(f"\nReconstructed M:\n{M_reconstructed}")
print(f"Reconstruction error: {np.linalg.norm(M - M_reconstructed):.10f}")

# Example 2: Low-rank approximation
print("\n--- Example 5.2: Low-Rank Approximation ---")
print("Using only top-k singular values for compression")

for k in [1, 2, 3]:
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    M_k = U_k @ np.diag(S_k) @ Vt_k
    error = np.linalg.norm(M - M_k)
    print(f"\nRank-{k} approximation error: {error:.4f}")
    print(f"Matrix M_{k}:\n{M_k}")

# Example 3: SVD for image compression
print("\n--- Example 5.3: Image Compression ---")
# Create a simple image
np.random.seed(42)
image = np.random.rand(50, 50)

# Add some structure
x, y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
image = np.sin(np.sqrt(x**2 + y**2)) + 0.1 * np.random.randn(50, 50)

U_img, S_img, Vt_img = np.linalg.svd(image, full_matrices=False)
print(f"Image shape: {image.shape}")
print(f"Number of singular values: {len(S_img)}")

# Visualization 5: SVD
fig5 = plt.figure(figsize=(18, 10))

# Original image
ax1 = fig5.add_subplot(2, 4, 1)
im1 = ax1.imshow(image, cmap='viridis')
ax1.set_title('Original Image', fontweight='bold')
ax1.axis('off')
plt.colorbar(im1, ax=ax1)

# Approximations with different ranks
ranks = [1, 5, 10, 20]
for idx, k in enumerate(ranks):
    ax = fig5.add_subplot(2, 4, idx + 2)
    U_k = U_img[:, :k]
    S_k = S_img[:k]
    Vt_k = Vt_img[:k, :]
    img_k = U_k @ np.diag(S_k) @ Vt_k
    
    im = ax.imshow(img_k, cmap='viridis')
    compression_ratio = (k * (image.shape[0] + image.shape[1]) + k) / (image.shape[0] * image.shape[1])
    ax.set_title(f'Rank {k}\nCompression: {compression_ratio*100:.1f}%', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

# Singular values
ax6 = fig5.add_subplot(2, 4, 6)
ax6.plot(S_img, 'bo-', linewidth=2, markersize=4)
ax6.set_xlabel('Index', fontweight='bold')
ax6.set_ylabel('Singular Value', fontweight='bold')
ax6.set_title('Singular Value Spectrum', fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_yscale('log')

# Cumulative energy
ax7 = fig5.add_subplot(2, 4, 7)
energy = np.cumsum(S_img**2) / np.sum(S_img**2)
ax7.plot(energy, 'ro-', linewidth=2, markersize=4)
ax7.axhline(y=0.9, color='green', linestyle='--', label='90% energy')
ax7.axhline(y=0.95, color='orange', linestyle='--', label='95% energy')
ax7.set_xlabel('Number of Components', fontweight='bold')
ax7.set_ylabel('Cumulative Energy', fontweight='bold')
ax7.set_title('Energy Retention', fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Reconstruction error
ax8 = fig5.add_subplot(2, 4, 8)
errors = []
ranks_test = range(1, min(image.shape) + 1)
for k in ranks_test:
    U_k = U_img[:, :k]
    S_k = S_img[:k]
    Vt_k = Vt_img[:k, :]
    img_k = U_k @ np.diag(S_k) @ Vt_k
    error = np.linalg.norm(image - img_k)
    errors.append(error)

ax8.plot(ranks_test, errors, 'mo-', linewidth=2, markersize=4)
ax8.set_xlabel('Rank', fontweight='bold')
ax8.set_ylabel('Reconstruction Error', fontweight='bold')
ax8.set_title('Error vs Rank', fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.set_yscale('log')

plt.tight_layout()
plt.savefig('5_svd.png', dpi=300, bbox_inches='tight')
print("\n✓ SVD visualization saved")
plt.close()

# ============================================================================
# CONCEPT MAP: CONNECTING ALL CONCEPTS
# ============================================================================
print("\n" + "="*80)
print("CREATING CONCEPT MAP")
print("="*80)

fig6 = plt.figure(figsize=(20, 12))
ax = fig6.add_subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define positions for concepts
concepts = {
    'Determinants': (2, 8),
    'Invertibility': (5, 8),
    'Eigenvalues': (2, 5),
    'PCA': (5, 5),
    'SVD': (8, 5),
    'Covariance': (5, 2)
}

# Draw concept boxes
for concept, (x, y) in concepts.items():
    bbox = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', 
                           facecolor='lightblue', 
                           linewidth=3)
    ax.add_patch(bbox)
    ax.text(x, y, concept, ha='center', va='center', 
            fontsize=14, fontweight='bold')

# Draw connections with arrows
connections = [
    ('Determinants', 'Invertibility', 'det(A) ≠ 0 ⟺ Invertible'),
    ('Determinants', 'Eigenvalues', 'det(A) = ∏λᵢ'),
    ('Invertibility', 'Eigenvalues', 'A invertible ⟺ no zero λ'),
    ('Eigenvalues', 'PCA', 'PCs = eigenvectors of Cov'),
    ('PCA', 'SVD', 'PCA via SVD'),
    ('PCA', 'Covariance', 'PCA of covariance'),
    ('SVD', 'Eigenvalues', 'σᵢ² = λᵢ for AᵀA')
]

for concept1, concept2, label in connections:
    x1, y1 = concepts[concept1]
    x2, y2 = concepts[concept2]
    
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           connectionstyle="arc3,rad=0.1",
                           arrowstyle='->,head_width=0.4,head_length=0.8',
                           color='darkred', linewidth=2.5, alpha=0.7)
    ax.add_patch(arrow)
    
    # Add label at midpoint
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mid_x, mid_y, label, ha='center', va='bottom',
            fontsize=9, style='italic', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Add title
ax.text(5, 9.5, 'WEEK 6 CONCEPT MAP: LINEAR ALGEBRA INTEGRATION', 
        ha='center', va='center', fontsize=18, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=0.5))

# Add key insights
insights = [
    "Key Insights:",
    "1. Determinant tells if matrix is invertible",
    "2. Eigenvalues are roots of det(A - λI) = 0",
    "3. PCA finds eigenvectors of covariance matrix",
    "4. SVD generalizes eigendecomposition",
    "5. All concepts connected through matrix theory"
]
y_pos = 0.8
for insight in insights:
    ax.text(0.5, y_pos, insight, ha='left', va='top',
            fontsize=10, fontweight='bold' if 'Key' in insight else 'normal')
    y_pos -= 0.15

plt.tight_layout()
plt.savefig('6_concept_map.png', dpi=300, bbox_inches='tight')
print("\n✓ Concept map saved")
plt.close()

# ============================================================================
# COMPREHENSIVE EXAMPLE: ALL CONCEPTS TOGETHER
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE EXAMPLE: ALL CONCEPTS TOGETHER")
print("="*80)
print("\nScenario: Analyzing a dataset with all learned techniques")

# Generate sample data
np.random.seed(42)
n_samples = 200
X_data = np.random.randn(n_samples, 3)
X_data[:, 1] = X_data[:, 0] * 2 + np.random.randn(n_samples) * 0.5
X_data[:, 2] = X_data[:, 0] * 0.5 - X_data[:, 1] * 0.3 + np.random.randn(n_samples) * 0.8

print(f"Generated data shape: {X_data.shape}")

# Step 1: Compute covariance matrix
print("\n--- Step 1: Covariance Matrix ---")
X_centered = X_data - np.mean(X_data, axis=0)
cov = np.cov(X_centered.T)
print(f"Covariance matrix:\n{cov}")

# Step 2: Determinant
det_cov = np.linalg.det(cov)
print(f"\nDeterminant of covariance: {det_cov:.4f}")
print(f"Matrix is {'invertible' if det_cov != 0 else 'singular'}")

# Step 3: Eigendecomposition
eigenvalues_cov, eigenvectors_cov = np.linalg.eig(cov)
print(f"\n--- Step 2: Eigendecomposition ---")
print(f"Eigenvalues: {eigenvalues_cov}")
print(f"Product of eigenvalues: {np.prod(eigenvalues_cov):.4f}")
print(f"Matches determinant? {np.allclose(det_cov, np.prod(eigenvalues_cov))}")

# Step 4: PCA
pca_comprehensive = PCA()
X_pca_comp = pca_comprehensive.fit_transform(X_centered)
print(f"\n--- Step 3: PCA ---")
print(f"Explained variance ratios: {pca_comprehensive.explained_variance_ratio_}")
print(f"PCA components (first 2):\n{pca_comprehensive.components_[:2]}")

# Step 5: SVD
U_comp, S_comp, Vt_comp = np.linalg.svd(X_centered, full_matrices=False)
print(f"\n--- Step 4: SVD ---")
print(f"Singular values: {S_comp}")
print(f"Relationship: σ² / (n-1) = λ (PCA eigenvalue)")
print(f"σ² / (n-1) = {(S_comp**2) / (n_samples - 1)}")
print(f"PCA eigenvalues: {pca_comprehensive.explained_variance_}")

# Final comprehensive visualization
fig7 = plt.figure(figsize=(20, 12))

# Original data (3D)
ax1 = fig7.add_subplot(2, 3, 1, projection='3d')
ax1.scatter(X_data[:, 0], X_data[:, 1], X_data[:, 2], 
            c=X_data[:, 0], cmap='viridis', alpha=0.6, s=20)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Feature 3')
ax1.set_title('Original 3D Data', fontweight='bold', fontsize=12)

# Covariance matrix
ax2 = fig7.add_subplot(2, 3, 2)
sns.heatmap(cov, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, ax=ax2, cbar_kws={'label': 'Covariance'})
ax2.set_title(f'Covariance Matrix\ndet = {det_cov:.2f}', fontweight='bold', fontsize=12)

# Eigenvalues
ax3 = fig7.add_subplot(2, 3, 3)
indices = np.arange(len(eigenvalues_cov))
ax3.bar(indices, eigenvalues_cov, color=['red', 'orange', 'yellow'], 
        alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_xlabel('Component', fontweight='bold')
ax3.set_ylabel('Eigenvalue (Variance)', fontweight='bold')
ax3.set_title('Eigenvalue Spectrum', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')

# PCA projection (2D)
ax4 = fig7.add_subplot(2, 3, 4)
scatter = ax4.scatter(X_pca_comp[:, 0], X_pca_comp[:, 1], 
                     c=X_data[:, 0], cmap='viridis', alpha=0.6, s=20)
ax4.set_xlabel(f'PC1 ({pca_comprehensive.explained_variance_ratio_[0]*100:.1f}%)', 
               fontweight='bold')
ax4.set_ylabel(f'PC2 ({pca_comprehensive.explained_variance_ratio_[1]*100:.1f}%)', 
               fontweight='bold')
ax4.set_title('PCA Projection (2D)', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Original Feature 1')

# Singular values
ax5 = fig7.add_subplot(2, 3, 5)
ax5.plot(S_comp, 'bo-', linewidth=2, markersize=10, label='Singular values')
ax5.set_xlabel('Index', fontweight='bold')
ax5.set_ylabel('Singular Value', fontweight='bold')
ax5.set_title('SVD: Singular Value Spectrum', fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Variance explained comparison
ax6 = fig7.add_subplot(2, 3, 6)
components = np.arange(1, len(pca_comprehensive.explained_variance_ratio_) + 1)
cumsum = np.cumsum(pca_comprehensive.explained_variance_ratio_)
ax6.bar(components, pca_comprehensive.explained_variance_ratio_, 
        alpha=0.7, color='skyblue', edgecolor='black', linewidth=2, label='Individual')
ax6.plot(components, cumsum, 'ro-', linewidth=3, markersize=10, label='Cumulative')
ax6.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% threshold')
ax6.set_xlabel('Component', fontweight='bold')
ax6.set_ylabel('Variance Explained', fontweight='bold')
ax6.set_title('Total Variance Explained', fontweight='bold', fontsize=12)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('7_comprehensive_example.png', dpi=300, bbox_inches='tight')
print("\n✓ Comprehensive example visualization saved")
plt.close()

# ============================================================================
# SUMMARY & KEY TAKEAWAYS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & KEY TAKEAWAYS")
print("="*80)

summary_data = {
    'Concept': ['Determinants', 'Invertibility', 'Eigenvalues', 'PCA', 'SVD'],
    'Key Formula': [
        'det(A) = scalar',
        'A⁻¹ exists ⟺ det(A)≠0',
        'Av = λv',
        'Cov(X) = VΛVᵀ',
        'A = UΣVᵀ'
    ],
    'Purpose': [
        'Volume scaling',
        'Solve Ax=b',
        'Find invariant directions',
        'Dimensionality reduction',
        'Matrix decomposition'
    ],
    'Connection': [
        'Foundation',
        '→ Determinants',
        '→ Invertibility',
        '→ Eigenvalues',
        '→ PCA generalization'
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

print("\n" + "="*80)
print("PRACTICAL APPLICATIONS")
print("="*80)
applications = """
1. DETERMINANTS
   - Check if system of equations has unique solution
   - Compute area/volume transformations
   - Numerical stability checks

2. INVERTIBILITY
   - Solve linear systems (Ax = b)
   - Find matrix inverse for transformations
   - Least squares regression

3. EIGENVALUES & EIGENVECTORS
   - Stability analysis (control systems)
   - Markov chains (PageRank algorithm)
   - Quantum mechanics (energy states)
   - Vibration analysis

4. PCA
   - Feature extraction
   - Data compression
   - Noise reduction
   - Visualization of high-dimensional data
   - Face recognition (Eigenfaces)

5. SVD
   - Recommender systems (Netflix, Amazon)
   - Image compression
   - Natural language processing (LSA)
   - Pseudoinverse computation
   - Signal processing
"""
print(applications)

print("\n" + "="*80)
print("NUMPY OPERATIONS MASTERED")
print("="*80)
operations = """
Matrix Operations:
- np.linalg.det(A)          # Determinant
- np.linalg.inv(A)          # Inverse
- np.linalg.eig(A)          # Eigendecomposition
- np.linalg.svd(A)          # SVD
- np.linalg.solve(A, b)     # Solve linear system
- A @ B                     # Matrix multiplication
- A.T                       # Transpose
- np.cov(X.T)              # Covariance matrix

Statistical Operations:
- np.mean(X, axis=0)        # Column-wise mean
- np.std(X, axis=0)         # Column-wise std
- (X - mean) / std          # Broadcasting for standardization

Array Operations:
- np.allclose(A, B)         # Check equality with tolerance
- np.diag(v)               # Create diagonal matrix
- np.eye(n)                # Identity matrix
- np.cumsum(arr)           # Cumulative sum
"""
print(operations)

print("\n" + "="*80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nGenerated Visualizations:")
print("  1. 1_determinants.png - Determinant as area scaling")
print("  2. 2_invertibility.png - Invertible vs singular matrices")
print("  3. 3_eigenvalues.png - Eigenvectors and transformations")
print("  4. 4_pca.png - Principal component analysis")
print("  5. 5_svd.png - Singular value decomposition & compression")
print("  6. 6_concept_map.png - Connecting all concepts")
print("  7. 7_comprehensive_example.png - Complete workflow")
print("\nYou now have a strong foundation in linear algebra with Python!")
