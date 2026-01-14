

#Check for Orthogonality: Write a function that checks if the columns of a matrix are orthogonal 
# (using dot products). Visualize orthogonal vs non-orthogonal vectors.


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def check_orthogonality(matrix, tolerance=1e-10):
    """
    Check if the columns of a matrix are orthogonal.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Input matrix to check
    tolerance : float
        Numerical tolerance for dot product (default: 1e-10)
    
    Returns:
    --------
    dict : Contains orthogonality results and dot products
    """
    n_cols = matrix.shape[1]
    is_orthogonal = True
    dot_products = {}
    results = []
    
    print(f"\n{'='*60}")
    print(f"ORTHOGONALITY CHECK")
    print(f"{'='*60}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Number of column vectors: {n_cols}\n")
    
    # Check all pairs of columns
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            col_i = matrix[:, i]
            col_j = matrix[:, j]
            
            # Calculate dot product
            dot_prod = np.dot(col_i, col_j)
            dot_products[(i, j)] = dot_prod
            
            # Check if orthogonal (dot product ≈ 0)
            is_pair_orthogonal = abs(dot_prod) < tolerance
            
            print(f"Column {i} · Column {j} = {dot_prod:.6f}", end="")
            if is_pair_orthogonal:
                print(" ✓ Orthogonal")
            else:
                print(" ✗ NOT Orthogonal")
                is_orthogonal = False
            
            results.append({
                'pair': (i, j),
                'dot_product': dot_prod,
                'is_orthogonal': is_pair_orthogonal
            })
    
    # Check if columns are also unit vectors (orthonormal)
    print(f"\n{'-'*60}")
    print("MAGNITUDE CHECK (For Orthonormality)")
    print(f"{'-'*60}")
    
    is_orthonormal = is_orthogonal
    for i in range(n_cols):
        col = matrix[:, i]
        magnitude = np.linalg.norm(col)
        is_unit = abs(magnitude - 1.0) < tolerance
        
        print(f"||Column {i}|| = {magnitude:.6f}", end="")
        if is_unit:
            print(" ✓ Unit vector")
        else:
            print(" ✗ NOT unit vector")
            is_orthonormal = False
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"{'='*60}")
    print(f"Orthogonal: {is_orthogonal}")
    print(f"Orthonormal: {is_orthonormal}")
    print(f"{'='*60}\n")
    
    return {
        'is_orthogonal': is_orthogonal,
        'is_orthonormal': is_orthonormal,
        'dot_products': dot_products,
        'results': results
    }


def visualize_2d_vectors(matrix, title="2D Vector Visualization"):
    """Visualize 2D vectors to show orthogonality"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    n_cols = matrix.shape[1]
    
    # Plot vectors
    for i in range(n_cols):
        vector = matrix[:, i]
        ax.quiver(0, 0, vector[0], vector[1], 
                 angles='xy', scale_units='xy', scale=1,
                 color=colors[i % len(colors)], width=0.01,
                 label=f'v{i} = [{vector[0]:.2f}, {vector[1]:.2f}]',
                 alpha=0.7, linewidth=2)
        
        # Add vector label at endpoint
        ax.text(vector[0]*1.1, vector[1]*1.1, f'v{i}',
               fontsize=12, fontweight='bold')
    
    # Calculate angles between vectors
    if n_cols >= 2:
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                v1 = matrix[:, i]
                v2 = matrix[:, j]
                
                # Calculate angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)
                
                # Add angle annotation
                mid_x = (v1[0] + v2[0]) / 4
                mid_y = (v1[1] + v2[1]) / 4
                ax.text(mid_x, mid_y, f'∠ = {angle_deg:.1f}°',
                       fontsize=10, bbox=dict(boxstyle='round', 
                       facecolor='yellow', alpha=0.5))
    
    # Set equal aspect and grid
    max_val = np.max(np.abs(matrix)) * 1.3
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def visualize_3d_vectors(matrix, title="3D Vector Visualization"):
    """Visualize 3D vectors to show orthogonality"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    n_cols = matrix.shape[1]
    
    # Plot vectors
    for i in range(n_cols):
        vector = matrix[:, i]
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2],
                 color=colors[i % len(colors)], 
                 arrow_length_ratio=0.15, linewidth=3,
                 label=f'v{i} = [{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}]',
                 alpha=0.8)
        
        # Add vector label at endpoint
        ax.text(vector[0]*1.15, vector[1]*1.15, vector[2]*1.15, f'v{i}',
               fontsize=12, fontweight='bold')
    
    # Set equal aspect and labels
    max_val = np.max(np.abs(matrix)) * 1.3
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



#Orthogonal 2D vectors (perpendicular)

print("\n" + "="*70)
print("EXAMPLE 1: ORTHOGONAL 2D VECTORS")
print("="*70)

orthogonal_2d = np.array([
    [1, 0],
    [0, 1]
])

print("\nMatrix:")
print(orthogonal_2d)
result1 = check_orthogonality(orthogonal_2d)
visualize_2d_vectors(orthogonal_2d, "Example 1: Orthogonal Vectors (90° angle)")



#Non-orthogonal 2D vectors

print("\n" + "="*70)
print("EXAMPLE 2: NON-ORTHOGONAL 2D VECTORS")
print("="*70)

non_orthogonal_2d = np.array([
    [1, 1],
    [0, 1]
])

print("\nMatrix:")
print(non_orthogonal_2d)
result2 = check_orthogonality(non_orthogonal_2d)
visualize_2d_vectors(non_orthogonal_2d, "Example 2: Non-Orthogonal Vectors (45° angle)")



#Orthonormal 3D vectors (standard basis)

print("\n" + "="*70)
print("EXAMPLE 3: ORTHONORMAL 3D VECTORS (Standard Basis)")
print("="*70)

orthonormal_3d = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

print("\nMatrix:")
print(orthonormal_3d)
result3 = check_orthogonality(orthonormal_3d)
visualize_3d_vectors(orthonormal_3d, "Example 3: Orthonormal 3D Vectors")



#Orthogonal but not orthonormal 3D vectors

print("\n" + "="*70)
print("EXAMPLE 4: ORTHOGONAL BUT NOT ORTHONORMAL 3D VECTORS")
print("="*70)

orthogonal_not_normal_3d = np.array([
    [2, 0, 0],
    [0, 3, 0],
    [0, 0, 1]
])

print("\nMatrix:")
print(orthogonal_not_normal_3d)
result4 = check_orthogonality(orthogonal_not_normal_3d)
visualize_3d_vectors(orthogonal_not_normal_3d, 
                     "Example 4: Orthogonal but NOT Orthonormal")



#Non-orthogonal 3D vectors



non_orthogonal_3d = np.array([
    [1, 1, 1],
    [1, 0, 0],
    [0, 1, 0]
])

print("\nMatrix:")
print(non_orthogonal_3d)
result5 = check_orthogonality(non_orthogonal_3d)
visualize_3d_vectors(non_orthogonal_3d, "Example 5: Non-Orthogonal 3D Vectors")


#Main points 

#1. ORTHOGONAL VECTORS:
 #  - Dot product = 0
 #  - Angle = 90° between vectors
 #  - Perpendicular to each other

#. ORTHONORMAL VECTORS:
#   - Orthogonal (dot product = 0)
#   - Each vector has magnitude = 1 (unit vectors)
#   - Most useful in linear algebra

#3. DOT PRODUCT TEST:
#   - v₁ · v₂ = 0  →  Orthogonal
#   - v₁ · v₂ ≠ 0  →  NOT Orthogonal

#4. ANGLE CALCULATION:
#   - cos(θ) = (v₁ · v₂) / (||v₁|| × ||v₂||)
#   - θ = 90° for orthogonal vectors
