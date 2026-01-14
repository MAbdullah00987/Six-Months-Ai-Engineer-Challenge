

#Linear Independence Check: Use matrix rank to determine if a set of vectors is linearly independent. 
# Create test cases with dependent and independent vectors.


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def check_linear_independence(vectors, tolerance=1e-10):
    """
    Check if a set of vectors is linearly independent using matrix rank.
    
    Parameters:
    -----------
    vectors : numpy.ndarray
        Matrix where each column is a vector to check
    tolerance : float
        Numerical tolerance for rank calculation
    
    Returns:
    --------
    dict : Contains independence results and detailed analysis
    """
    n_vectors = vectors.shape[1]
    vector_dim = vectors.shape[0]
    
    # Calculate rank
    rank = np.linalg.matrix_rank(vectors, tol=tolerance)
    
    # Check if linearly independent
    is_independent = (rank == n_vectors)
    
    print(f"\n{'='*70}")
    print(f"LINEAR INDEPENDENCE CHECK")
    print(f"{'='*70}")
    print(f"Number of vectors: {n_vectors}")
    print(f"Vector dimension: {vector_dim}")
    print(f"Matrix rank: {rank}")
    print(f"Maximum possible rank: {min(n_vectors, vector_dim)}")
    print(f"\n{'-'*70}")
    
    # Display vectors
    print("Vectors:")
    for i in range(n_vectors):
        vec = vectors[:, i]
        print(f"  v{i} = {vec}")
    
    print(f"\n{'-'*70}")
    print(f"RESULT: ", end="")
    
    if is_independent:
        print("✓ LINEARLY INDEPENDENT")
        print(f"\nExplanation: Rank ({rank}) equals number of vectors ({n_vectors})")
        print("No vector can be written as a linear combination of others.")
    else:
        print("✗ LINEARLY DEPENDENT")
        print(f"\nExplanation: Rank ({rank}) < number of vectors ({n_vectors})")
        print(f"At least {n_vectors - rank} vector(s) can be expressed as")
        print("linear combinations of the others.")
        
        # Try to find dependency relationship
        if n_vectors <= 4:  # Only for small sets
            find_dependency(vectors)
    
    print(f"{'='*70}\n")
    
    return {
        'is_independent': is_independent,
        'rank': rank,
        'n_vectors': n_vectors,
        'dimension': vector_dim
    }


def find_dependency(vectors):
    """
    Attempt to find linear dependency relationships.
    Uses reduced row echelon form approach.
    """
    print(f"\n{'-'*70}")
    print("DEPENDENCY ANALYSIS:")
    print(f"{'-'*70}")
    
    n_vectors = vectors.shape[1]
    
    # Check if any vector is zero
    for i in range(n_vectors):
        if np.allclose(vectors[:, i], 0):
            print(f"v{i} is the zero vector (linearly dependent)")
            return
    
    # Check if any vector is a scalar multiple of another
    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            v1 = vectors[:, i]
            v2 = vectors[:, j]
            
            # Check if v2 = c * v1
            if not np.allclose(v1, 0):
                ratio = v2 / (v1 + 1e-10)
                if np.allclose(ratio, ratio[0]):
                    c = ratio[0]
                    print(f"Found: v{j} = {c:.3f} × v{i}")
                    return
    
    # For 3 vectors, check if one is a linear combination of others
    if n_vectors == 3:
        # Try to express v2 as combination of v0 and v1
        try:
            A = vectors[:, [0, 1]]
            b = vectors[:, 2]
            coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            if residuals.size > 0 and residuals[0] < 1e-10:
                print(f"Found: v2 = {coeffs[0]:.3f} × v0 + {coeffs[1]:.3f} × v1")
                return
        except:
            pass
    
    print("Dependency relationship is complex (not easily expressible)")


def visualize_2d_independence(vectors, title="2D Linear Independence"):
    """Visualize 2D vectors to show independence/dependence"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    n_vectors = vectors.shape[1]
    
    # Plot vectors
    for i in range(n_vectors):
        vector = vectors[:, i]
        ax.quiver(0, 0, vector[0], vector[1], 
                 angles='xy', scale_units='xy', scale=1,
                 color=colors[i % len(colors)], width=0.012,
                 label=f'v{i} = [{vector[0]:.2f}, {vector[1]:.2f}]',
                 alpha=0.8, linewidth=2.5)
        
        # Add vector label
        ax.text(vector[0]*1.15, vector[1]*1.15, f'v{i}',
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Draw span visualization for dependent vectors
    rank = np.linalg.matrix_rank(vectors)
    if rank < n_vectors and rank == 1:
        # All vectors lie on the same line
        v = vectors[:, 0]
        t = np.linspace(-3, 3, 100)
        span_x = t * v[0]
        span_y = t * v[1]
        ax.plot(span_x, span_y, 'k--', alpha=0.3, linewidth=2,
               label='Span (1D line)')
    
    # Set equal aspect and grid
    max_val = np.max(np.abs(vectors)) * 1.4
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def visualize_3d_independence(vectors, title="3D Linear Independence"):
    """Visualize 3D vectors to show independence/dependence"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    n_vectors = vectors.shape[1]
    
    # Plot vectors
    for i in range(n_vectors):
        vector = vectors[:, i]
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2],
                 color=colors[i % len(colors)], 
                 arrow_length_ratio=0.15, linewidth=3.5,
                 label=f'v{i} = [{vector[0]:.1f}, {vector[1]:.1f}, {vector[2]:.1f}]',
                 alpha=0.8)
        
        # Add vector label
        ax.text(vector[0]*1.15, vector[1]*1.15, vector[2]*1.15, 
               f'v{i}', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Visualize span for dependent vectors
    rank = np.linalg.matrix_rank(vectors)
    if rank == 2 and n_vectors >= 2:
        # Vectors span a plane
        v1 = vectors[:, 0]
        v2 = vectors[:, 1]
        
        # Create mesh for the plane
        t1 = np.linspace(-1.5, 1.5, 10)
        t2 = np.linspace(-1.5, 1.5, 10)
        T1, T2 = np.meshgrid(t1, t2)
        
        X = T1 * v1[0] + T2 * v2[0]
        Y = T1 * v1[1] + T2 * v2[1]
        Z = T1 * v1[2] + T2 * v2[2]
        
        ax.plot_surface(X, Y, Z, alpha=0.2, color='yellow',
                       edgecolor='none')
        ax.text(0, 0, 0, 'Span (2D plane)', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Set labels and limits
    max_val = np.max(np.abs(vectors)) * 1.4
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()



#Independent 2D vectors

print("\n" + "█"*70)
print("TEST CASE 1: LINEARLY INDEPENDENT 2D VECTORS")
print("█"*70)

independent_2d = np.array([
    [1, 0],
    [0, 1]
])

print("\nVectors form a basis for R²")
result1 = check_linear_independence(independent_2d)
visualize_2d_independence(independent_2d, 
    "Test 1: Independent 2D Vectors (Basis for R²)")



#Dependent 2D vectors (scalar multiple)

print("\n" + "█"*70)
print("TEST CASE 2: DEPENDENT 2D VECTORS (Scalar Multiple)")
print("█"*70)

dependent_2d_scalar = np.array([
    [2, 4],
    [1, 2]
])

print("\nv1 is a scalar multiple of v0")
result2 = check_linear_independence(dependent_2d_scalar)
visualize_2d_independence(dependent_2d_scalar, 
    "Test 2: Dependent Vectors (v1 = 2×v0)")



#Dependent 2D vectors (linear combination)

print("\n" + "█"*70)
print("TEST CASE 3: DEPENDENT 2D VECTORS (Linear Combination)")
print("█"*70)

dependent_2d_combo = np.array([
    [1, 2, 3],
    [2, 1, 3]
])

print("\nThree vectors in 2D space (impossible to be independent)")
result3 = check_linear_independence(dependent_2d_combo)
visualize_2d_independence(dependent_2d_combo, 
    "Test 3: Three Vectors in 2D (v2 = v0 + v1)")



#Independent 3D vectors (standard basis)

print("\n" + "█"*70)
print("TEST CASE 4: INDEPENDENT 3D VECTORS (Standard Basis)")
print("█"*70)

independent_3d_basis = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

print("\nStandard basis vectors for R³")
result4 = check_linear_independence(independent_3d_basis)
visualize_3d_independence(independent_3d_basis, 
    "Test 4: Independent 3D Basis Vectors")



#Independent 3D vectors (non-standard)

print("\n" + "█"*70)
print("TEST CASE 5: INDEPENDENT 3D VECTORS (Non-Standard)")
print("█"*70)

independent_3d = np.array([
    [1, 2, 1],
    [2, 1, 0],
    [1, 0, 3]
])

print("\nArbitrary independent vectors in R³")
result5 = check_linear_independence(independent_3d)
visualize_3d_independence(independent_3d, 
    "Test 5: Independent Non-Standard 3D Vectors")



#Dependent 3D vectors (one in plane of others)

print("\n" + "█"*70)
print("TEST CASE 6: DEPENDENT 3D VECTORS (Planar)")
print("█"*70)

dependent_3d_planar = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 2]
])

print("\nThird vector lies in plane of first two")
result6 = check_linear_independence(dependent_3d_planar)
visualize_3d_independence(dependent_3d_planar, 
    "Test 6: Dependent Vectors (v2 = v0 + v1)")


#Dependent 3D vectors (all collinear)

print("\n" + "█"*70)
print("TEST CASE 7: DEPENDENT 3D VECTORS (Collinear)")
print("█"*70)

dependent_3d_collinear = np.array([
    [1, 2, 3],
    [2, 4, 6],
    [3, 6, 9]
])

print("\nAll vectors on the same line")
result7 = check_linear_independence(dependent_3d_collinear)
visualize_3d_independence(dependent_3d_collinear, 
    "Test 7: Collinear Vectors (All on Same Line)")



#Four vectors in 3D space

print("\n" + "█"*70)
print("TEST CASE 8: FOUR VECTORS IN 3D SPACE")
print("█"*70)

four_vectors_3d = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1]
])

print("\nFour vectors in 3D (max rank = 3, so dependent)")
result8 = check_linear_independence(four_vectors_3d)
visualize_3d_independence(four_vectors_3d, 
    "Test 8: Four Vectors in 3D Space")



'''
DEFINITION:
Vectors v₁, v₂, ..., vₙ are LINEARLY INDEPENDENT if:
  c₁v₁ + c₂v₂ + ... + cₙvₙ = 0  implies  c₁ = c₂ = ... = cₙ = 0

RANK METHOD:
• Arrange vectors as columns in a matrix
• Calculate rank = number of linearly independent columns
• If rank = number of vectors → INDEPENDENT
• If rank < number of vectors → DEPENDENT

IMPORTANT FACTS:
1. In n-dimensional space, max n linearly independent vectors
2. If more than n vectors in Rⁿ → always dependent
3. Any set containing the zero vector is dependent
4. Two vectors are dependent ↔ one is scalar multiple of other
5. Linearly independent vectors span their space efficiently

GEOMETRIC INTERPRETATION:
• 2D Independent: Vectors point in different directions (not collinear)
• 2D Dependent: Vectors lie on the same line
• 3D Independent: Vectors span full 3D space
• 3D Dependent: Vectors lie in a plane (rank 2) or line (rank 1)

APPLICATIONS:
• Determining basis for vector spaces
• Solving systems of linear equations
• Understanding dimensionality in data analysis
• Computer graphics transformations
• Machine learning feature selection
'''
