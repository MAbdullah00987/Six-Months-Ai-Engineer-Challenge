#Day 1: Determinants & Matrix Inverses
#Goal: Understand determinants and when/how matrices can be inverted.

#Mathematics for Machine Learning, Chapter 4: Sections on determinants and matrix inverses
#BITS Pilani Course: Lectures on determinants and invertibility

#Matrix Inverse Calculator: Write a script that checks if a matrix is invertible (det ≠ 0) and calculates its inverse if it exists. Test with various matrices.
#Determinant and Area: Create a visualization showing how the determinant of a 2×2 matrix relates to area transformation. Plot a unit square and its transformed version.
#Key Concepts to Master
#Computing determinants (2×2, 3×3, and general)
#Geometric interpretation of determinants
#Conditions for invertibility
#Computing matrix inverses



#Matrix Determinants and Inverses 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

print("="*60)
print("DAY 1: DETERMINANTS & MATRIX INVERSES")
print("="*60)

# ============================================================================
# SECTION 1: COMPUTING DETERMINANTS
# ============================================================================

print("\n" + "="*60)
print("SECTION 1: COMPUTING DETERMINANTS")
print("="*60)

# Manual computation for 2x2 matrix
def det_2x2_manual(matrix):
    """
    Compute determinant of 2x2 matrix manually
    Formula: ad - bc
    """
    a, b = matrix[0]
    c, d = matrix[1]
    return a * d - b * c

# Manual computation for 3x3 matrix
def det_3x3_manual(matrix):
    """
    Compute determinant of 3x3 matrix using cofactor expansion
    """
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    
    det = a * (e*i - f*h) - b * (d*i - f*g) + c * (d*h - e*g)
    return det

# Example 1: 2x2 Matrix
print("\n--- Example 1: 2×2 Matrix ---")
A_2x2 = np.array([[3, 2],
                   [1, 4]])
print("Matrix A:")
print(A_2x2)
print(f"\nManual calculation: det(A) = (3)(4) - (2)(1) = {det_2x2_manual(A_2x2)}")
print(f"NumPy calculation: det(A) = {np.linalg.det(A_2x2):.4f}")

# Example 2: 3x3 Matrix
print("\n--- Example 2: 3×3 Matrix ---")
A_3x3 = np.array([[1, 2, 3],
                   [0, 1, 4],
                   [5, 6, 0]])
print("Matrix A:")
print(A_3x3)
print(f"\nManual calculation: det(A) = {det_3x3_manual(A_3x3)}")
print(f"NumPy calculation: det(A) = {np.linalg.det(A_3x3):.4f}")

# Example 3: Singular matrix (det = 0)
print("\n--- Example 3: Singular Matrix (Non-invertible) ---")
A_singular = np.array([[2, 4],
                        [1, 2]])
print("Matrix A:")
print(A_singular)
print(f"det(A) = {np.linalg.det(A_singular):.4f}")
print("This matrix is SINGULAR (not invertible) because det = 0")

# ============================================================================
# SECTION 2: MATRIX INVERSE CALCULATOR
# ============================================================================

print("\n" + "="*60)
print("SECTION 2: MATRIX INVERSE CALCULATOR")
print("="*60)

def check_and_invert(matrix, name="Matrix"):
    """
    Check if a matrix is invertible and compute its inverse
    """
    print(f"\n--- Analyzing {name} ---")
    print(f"{name}:")
    print(matrix)
    
    # Compute determinant
    det = np.linalg.det(matrix)
    print(f"\nDeterminant: {det:.6f}")
    
    # Check invertibility
    if abs(det) < 1e-10:  # Using small threshold for numerical stability
        print(f"❌ {name} is NOT INVERTIBLE (det ≈ 0)")
        return None
    else:
        print(f"✓ {name} is INVERTIBLE (det ≠ 0)")
        
        # Compute inverse
        A_inv = np.linalg.inv(matrix)
        print(f"\nInverse of {name}:")
        print(A_inv)
        
        # Verify: A × A^(-1) = I
        identity = matrix @ A_inv
        print(f"\nVerification (A × A⁻¹ = I):")
        print(identity)
        
        return A_inv

# Test with various matrices
test_matrices = {
    "Invertible 2×2": np.array([[4, 7], [2, 6]]),
    "Singular 2×2": np.array([[2, 4], [1, 2]]),
    "Invertible 3×3": np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]]),
    "Identity 3×3": np.eye(3),
    "Diagonal Matrix": np.diag([2, 3, 4])
}

for name, matrix in test_matrices.items():
    check_and_invert(matrix, name)

# ============================================================================
# SECTION 3: GEOMETRIC INTERPRETATION - AREA TRANSFORMATION
# ============================================================================

print("\n" + "="*60)
print("SECTION 3: GEOMETRIC INTERPRETATION")
print("="*60)

def visualize_transformation(matrix, title="Matrix Transformation"):
    """
    Visualize how a matrix transforms a unit square
    Shows the geometric meaning of determinant as area scaling
    """
    # Define unit square vertices
    unit_square = np.array([[0, 0],
                            [1, 0],
                            [1, 1],
                            [0, 1],
                            [0, 0]])  # Close the square
    
    # Transform the square
    transformed = (matrix @ unit_square.T).T
    
    # Compute areas
    original_area = 1.0
    det = np.linalg.det(matrix)
    transformed_area = abs(det) * original_area
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original square
    ax1.fill(unit_square[:, 0], unit_square[:, 1], 
             alpha=0.3, color='blue', edgecolor='blue', linewidth=2)
    ax1.plot(unit_square[:, 0], unit_square[:, 1], 'bo-', linewidth=2)
    ax1.set_xlim(-0.5, 2)
    ax1.set_ylim(-0.5, 2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Original Unit Square\nArea = {original_area}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Transformed square
    ax2.fill(transformed[:, 0], transformed[:, 1], 
             alpha=0.3, color='red', edgecolor='red', linewidth=2)
    ax2.plot(transformed[:, 0], transformed[:, 1], 'ro-', linewidth=2)
    
    # Also show original square faintly for comparison
    ax2.plot(unit_square[:, 0], unit_square[:, 1], 'b--', alpha=0.3, linewidth=1)
    
    ax2.set_xlim(-1, max(4, transformed[:, 0].max() + 1))
    ax2.set_ylim(-1, max(4, transformed[:, 1].max() + 1))
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Transformed Square\nArea = |det(A)| × 1 = {transformed_area:.2f}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.suptitle(f'{title}\nMatrix A = {matrix.tolist()}\ndet(A) = {det:.2f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'transformation_{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n{title}")
    print(f"Matrix:\n{matrix}")
    print(f"Determinant: {det:.4f}")
    print(f"Original area: {original_area}")
    print(f"Transformed area: {transformed_area:.4f}")
    print(f"Area scaling factor: {abs(det):.4f}")

# Visualization 1: Scaling transformation
print("\n--- Visualization 1: Scaling (det > 1) ---")
A1 = np.array([[2, 0],
               [0, 3]])
visualize_transformation(A1, "Scaling Transformation")

# Visualization 2: Rotation + Scaling
print("\n--- Visualization 2: Rotation + Scaling ---")
theta = np.pi / 6  # 30 degrees
A2 = np.array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]]) * 1.5
visualize_transformation(A2, "Rotation and Scaling")

# Visualization 3: Shear transformation
print("\n--- Visualization 3: Shear Transformation ---")
A3 = np.array([[1, 0.5],
               [0, 1]])
visualize_transformation(A3, "Shear Transformation")

# Visualization 4: Reflection (det < 0)
print("\n--- Visualization 4: Reflection (det < 0) ---")
A4 = np.array([[-1, 0],
               [0, 1]])
visualize_transformation(A4, "Reflection (Negative Det)")

# ============================================================================
# SECTION 4: PROPERTIES OF DETERMINANTS
# ============================================================================

print("\n" + "="*60)
print("SECTION 4: IMPORTANT PROPERTIES OF DETERMINANTS")
print("="*60)

A = np.array([[2, 3], [1, 4]])
B = np.array([[5, 2], [3, 1]])

print("\nMatrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Property 1: det(AB) = det(A) × det(B)
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
det_AB = np.linalg.det(A @ B)

print(f"\n1. Multiplicative Property:")
print(f"   det(A) = {det_A:.4f}")
print(f"   det(B) = {det_B:.4f}")
print(f"   det(A) × det(B) = {det_A * det_B:.4f}")
print(f"   det(AB) = {det_AB:.4f}")
print(f"   ✓ They match!")

# Property 2: det(A^T) = det(A)
det_A_T = np.linalg.det(A.T)
print(f"\n2. Transpose Property:")
print(f"   det(A) = {det_A:.4f}")
print(f"   det(A^T) = {det_A_T:.4f}")
print(f"   ✓ They match!")

# Property 3: det(A^(-1)) = 1/det(A)
if det_A != 0:
    A_inv = np.linalg.inv(A)
    det_A_inv = np.linalg.det(A_inv)
    print(f"\n3. Inverse Property:")
    print(f"   det(A) = {det_A:.4f}")
    print(f"   1/det(A) = {1/det_A:.4f}")
    print(f"   det(A^(-1)) = {det_A_inv:.4f}")
    print(f"   ✓ They match!")

# Property 4: det(cA) = c^n × det(A) for n×n matrix
c = 2
A_scaled = c * A
det_A_scaled = np.linalg.det(A_scaled)
n = A.shape[0]
print(f"\n4. Scalar Multiplication Property:")
print(f"   c = {c}")
print(f"   n = {n} (matrix dimension)")
print(f"   det(A) = {det_A:.4f}")
print(f"   c^n × det(A) = {c**n * det_A:.4f}")
print(f"   det(cA) = {det_A_scaled:.4f}")
print(f"   ✓ They match!")

# ============================================================================
# SECTION 5: SUMMARY TABLE
# ============================================================================

print("\n" + "="*60)
print("SECTION 5: SUMMARY")
print("="*60)

# Create summary DataFrame
summary_data = {
    'Matrix Type': ['Invertible', 'Singular', 'Identity', 'Diagonal', 'Orthogonal'],
    'Determinant': ['det ≠ 0', 'det = 0', 'det = 1', 'Product of diagonal', '|det| = 1'],
    'Has Inverse?': ['Yes', 'No', 'Yes', 'Yes (if no zeros)', 'Yes'],
    'Geometric Meaning': ['Changes area/volume', 'Collapses dimension', 'No change', 'Scales axes', 'Preserves length']
}

summary_df = pd.DataFrame(summary_data)
print("\nMatrix Types and Their Properties:")
print(summary_df.to_string(index=False))

print("\n" + "="*60)
print("KEY TAKEAWAYS")
print("="*60)
print("""
1. Determinant = 0 → Matrix is SINGULAR (not invertible)
2. Determinant ≠ 0 → Matrix is INVERTIBLE
3. |det(A)| = Area/Volume scaling factor
4. det(A) < 0 → Transformation includes reflection
5. det(AB) = det(A) × det(B)
6. det(A^(-1)) = 1/det(A)
7. To find inverse: A^(-1) exists only if det(A) ≠ 0
""")

print("\n✓ Tutorial Complete! Check the generated visualizations.")
print("="*60)