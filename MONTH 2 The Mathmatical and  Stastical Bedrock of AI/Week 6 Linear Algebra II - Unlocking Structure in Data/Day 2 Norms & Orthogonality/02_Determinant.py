
#What is a Determinant?A determinant is a scalar value that encodes important properties of a matrix:
#Geometric interpretation: How much a transformation scales area/volume
#Algebraic interpretation: Whether a matrix is invertible
#Computational interpretation: Solution existence for linear systems

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

'''
# Basic determinant computation
A_2x2 = np.array([
    [3, 2],
    [1, 4]
])

# For 2×2 matrix: det = ad - bc
det_manual = A_2x2[0,0] * A_2x2[1,1] - A_2x2[0,1] * A_2x2[1,0]
det_numpy = np.linalg.det(A_2x2)

print("Matrix A:")
print(A_2x2)
print(f"\nDeterminant (manual): {det_manual}")
print(f"Determinant (NumPy): {det_numpy:.2f}")
print(f"\nInterpretation: This transformation scales area by {abs(det_numpy):.2f}x")
'''

'''
#2. Determinant Computation Methods
#A. 2×2 Matrix Determinant

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def det_2x2(matrix):
    """
    Compute determinant of 2×2 matrix
    Formula: det([[a,b],[c,d]]) = ad - bc
    
    Parameters:
    -----------
    matrix : np.ndarray or list
        2×2 matrix
        
    Returns:
    --------
    float : determinant value
    """
    # Convert to numpy array if needed
    matrix = np.array(matrix)
    
    # Validate dimensions
    if matrix.shape != (2, 2):
        raise ValueError(f"Matrix must be 2×2, got shape {matrix.shape}")
    
    # Extract elements - CORRECT SYNTAX
    a, b = matrix[0, 0], matrix[0, 1]  # First row
    c, d = matrix[1, 0], matrix[1, 1]  # Second row
    
    # Calculate determinant: ad - bc
    return a * d - b * c


def det_3x3(matrix):
    """
    Compute determinant of 3×3 matrix using cofactor expansion
    Formula: det = a(ei−fh) − b(di−fg) + c(dh−eg)
    """
    matrix = np.array(matrix)
    
    if matrix.shape != (3, 3):
        raise ValueError(f"Matrix must be 3×3, got shape {matrix.shape}")
    
    # Extract elements
    a, b, c = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    d, e, f = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    g, h, i = matrix[2, 0], matrix[2, 1], matrix[2, 2]
    
    # Cofactor expansion along first row
    det = a * (e*i - f*h) - b * (d*i - f*g) + c * (d*h - e*g)
    
    return det


def visualize_determinant(matrix):
    """Visualize matrix and its determinant geometrically"""
    matrix = np.array(matrix)
    
    if matrix.shape == (2, 2):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Matrix Heatmap
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    ax=axes[0], cbar_kws={'label': 'Value'},
                    linewidths=2, linecolor='black', square=True)
        axes[0].set_title('Matrix Visualization', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        
        # Plot 2: Geometric Interpretation (2D)
        ax = axes[1]
        
        # Original vectors
        v1 = matrix[:, 0]  # First column
        v2 = matrix[:, 1]  # Second column
        
        # Plot vectors
        ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', 
                  scale=1, color='blue', width=0.01, label=f'v1 = [{v1[0]:.1f}, {v1[1]:.1f}]')
        ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', 
                  scale=1, color='red', width=0.01, label=f'v2 = [{v2[0]:.1f}, {v2[1]:.1f}]')
        
        # Plot parallelogram
        parallelogram = plt.Polygon([[0, 0], v1, v1+v2, v2], 
                                     alpha=0.3, color='green', 
                                     label=f'Area = |det| = {abs(det_2x2(matrix)):.2f}')
        ax.add_patch(parallelogram)
        
        # Set limits and grid
        max_val = max(abs(matrix).max(), 1) * 1.5
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_title('Geometric Interpretation\n(Parallelogram Area)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        
        # Plot 3: Calculation Breakdown
        ax = axes[2]
        ax.axis('off')
        
        det_value = det_2x2(matrix)
        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
        
        calculation_text = f"""
DETERMINANT CALCULATION

Matrix:
┌         ┐
│ {a:>6.2f}  {b:>6.2f} │
│ {c:>6.2f}  {d:>6.2f} │
└         ┘

Formula: det = ad - bc

Step 1: a × d = {a:.2f} × {d:.2f} = {a*d:.2f}
Step 2: b × c = {b:.2f} × {c:.2f} = {b*c:.2f}
Step 3: det = {a*d:.2f} - {b*c:.2f}

Result: det = {det_value:.2f}

Properties:
• If det = 0: Matrix is SINGULAR (not invertible)
• If det ≠ 0: Matrix is INVERTIBLE
• |det| = Area of parallelogram
        """
        
        ax.text(0.1, 0.5, calculation_text, fontsize=11, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Calculation Breakdown', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    elif matrix.shape == (3, 3):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Matrix Heatmap
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    ax=axes[0], cbar_kws={'label': 'Value'},
                    linewidths=2, linecolor='black', square=True)
        axes[0].set_title('3×3 Matrix Visualization', fontsize=14, fontweight='bold')
        
        # Plot 2: Calculation Breakdown
        ax = axes[1]
        ax.axis('off')
        
        det_value = det_3x3(matrix)
        
        calculation_text = f"""
3×3 DETERMINANT CALCULATION

Matrix:
┌                      ┐
│ {matrix[0,0]:>6.2f}  {matrix[0,1]:>6.2f}  {matrix[0,2]:>6.2f} │
│ {matrix[1,0]:>6.2f}  {matrix[1,1]:>6.2f}  {matrix[1,2]:>6.2f} │
│ {matrix[2,0]:>6.2f}  {matrix[2,1]:>6.2f}  {matrix[2,2]:>6.2f} │
└                      ┘

Cofactor Expansion (Row 1):

det = a(ei - fh) - b(di - fg) + c(dh - eg)

Result: det = {det_value:.2f}

NumPy Verification: {np.linalg.det(matrix):.2f}

Properties:
• If det = 0: Matrix is SINGULAR
• If det ≠ 0: Matrix is INVERTIBLE
• |det| = Volume of parallelepiped
        """
        
        ax.text(0.1, 0.5, calculation_text, fontsize=10, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.set_title('Calculation Breakdown', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()


def test_determinants():
    """Test determinant calculations with various matrices"""
    
    print("="*70)
    print("DETERMINANT CALCULATOR - TESTING SUITE")
    print("="*70)
    
    # Test 1: Basic 2×2 Matrix
    print("\n" + "─"*70)
    print("TEST 1: Basic 2×2 Matrix")
    print("─"*70)
    M1 = np.array([[5, 3], [2, 7]])
    det_custom = det_2x2(M1)
    det_numpy = np.linalg.det(M1)
    
    print(f"Matrix:\n{M1}")
    print(f"Custom determinant: {det_custom:.2f}")
    print(f"NumPy verification: {det_numpy:.2f}")
    print(f"Match: {np.isclose(det_custom, det_numpy)} ✓" if np.isclose(det_custom, det_numpy) else "Match: False ✗")
    
    visualize_determinant(M1)
    
    # Test 2: Identity Matrix
    print("\n" + "─"*70)
    print("TEST 2: Identity Matrix (det should be 1)")
    print("─"*70)
    M2 = np.eye(2)
    det_custom = det_2x2(M2)
    print(f"Matrix:\n{M2}")
    print(f"Determinant: {det_custom:.2f} (Expected: 1.00) ✓" if np.isclose(det_custom, 1) else f"Determinant: {det_custom:.2f} ✗")
    
    # Test 3: Singular Matrix (det = 0)
    print("\n" + "─"*70)
    print("TEST 3: Singular Matrix (det should be 0)")
    print("─"*70)
    M3 = np.array([[2, 4], [1, 2]])
    det_custom = det_2x2(M3)
    print(f"Matrix:\n{M3}")
    print(f"Determinant: {det_custom:.2f} (Expected: 0.00)")
    print(f"This matrix is SINGULAR (not invertible) ✓" if np.isclose(det_custom, 0) else "This matrix is INVERTIBLE")
    
    visualize_determinant(M3)
    
    # Test 4: Negative Determinant
    print("\n" + "─"*70)
    print("TEST 4: Matrix with Negative Determinant")
    print("─"*70)
    M4 = np.array([[1, 2], [3, 4]])
    det_custom = det_2x2(M4)
    det_numpy = np.linalg.det(M4)
    print(f"Matrix:\n{M4}")
    print(f"Custom determinant: {det_custom:.2f}")
    print(f"NumPy verification: {det_numpy:.2f}")
    print("Note: Negative determinant means transformation flips orientation")
    
    # Test 5: 3×3 Matrix
    print("\n" + "─"*70)
    print("TEST 5: 3×3 Matrix")
    print("─"*70)
    M5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    det_custom = det_3x3(M5)
    det_numpy = np.linalg.det(M5)
    
    print(f"Matrix:\n{M5}")
    print(f"Custom determinant: {det_custom:.2f}")
    print(f"NumPy verification: {det_numpy:.2f}")
    print(f"Match: {np.isclose(det_custom, det_numpy)} ✓" if np.isclose(det_custom, det_numpy) else "Match: False ✗")
    
    visualize_determinant(M5)
    
    # Test 6: Random matrices
    print("\n" + "─"*70)
    print("TEST 6: Random Matrices (Stress Test)")
    print("─"*70)
    
    n_tests = 100
    matches = 0
    for i in range(n_tests):
        random_matrix = np.random.randint(-10, 10, size=(2, 2))
        det_custom = det_2x2(random_matrix)
        det_numpy = np.linalg.det(random_matrix)
        if np.isclose(det_custom, det_numpy):
            matches += 1
    
    print(f"Tested {n_tests} random 2×2 matrices")
    print(f"Matches: {matches}/{n_tests}")
    print(f"Success Rate: {matches/n_tests*100:.1f}% ✓" if matches == n_tests else f"Success Rate: {matches/n_tests*100:.1f}% ✗")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70)


# Run the tests
if __name__ == "__main__":
    test_determinants()
'''

'''
#General N×N Determinant (LU Decomposition)

def det_lu_decomposition(matrix):
    """
    Compute determinant using LU decomposition
    det(A) = det(P) * det(L) * det(U)
    """
    from scipy.linalg import lu
    
    P, L, U = lu(matrix)
    
    # det(L) = 1 (unit triangular)
    # det(U) = product of diagonal elements
    # det(P) = ±1 (permutation matrix)
    
    det_U = np.prod(np.diag(U))
    det_P = np.linalg.det(P)
    
    return det_P * det_U

# Example with larger matrix
M = np.random.rand(5, 5)
print(f"5×5 Random Matrix determinant: {det_lu_decomposition(M):.6f}")
print(f"NumPy verification: {np.linalg.det(M):.6f}")
'''

'''
#Geometric Interpretation of Determinants
#A. Visualizing Area Scaling (2D)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

def visualize_determinant_2d(matrix, title="Determinant as Area Scaling"):
    """
    Visualize how determinant represents area scaling
    """
    # Unit square vertices (creating a closed path)
    unit_square = np.array([
        [0, 1, 1, 0, 0],  # x-coordinates
        [0, 0, 1, 1, 0]   # y-coordinates
    ])
    
    # Transform the square using matrix multiplication
    transformed = matrix @ unit_square
    
    # Calculate determinant
    det = np.linalg.det(matrix)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # FIXED: Correct array indexing with [row, :] syntax
    # Original unit square (area = 1)
    ax1.fill(unit_square[0, :], unit_square[1, :], alpha=0.3, color='blue', 
             edgecolor='blue', linewidth=2)
    ax1.plot(unit_square[0, :], unit_square[1, :], 'bo-', linewidth=2, markersize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 4)
    ax1.set_ylim(-0.5, 4)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_title('Original Square\nArea = 1', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x-axis')
    ax1.set_ylabel('y-axis')
    ax1.text(0.5, 0.5, 'Area = 1', fontsize=12, ha='center', 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # FIXED: Correct array indexing
    # Transformed parallelogram
    ax2.fill(transformed[0, :], transformed[1, :], alpha=0.3, color='red', 
             edgecolor='red', linewidth=2)
    ax2.plot(transformed[0, :], transformed[1, :], 'ro-', linewidth=2, markersize=8)
    ax2.grid(True, alpha=0.3)
    
    # Auto-scale axes based on transformed shape
    x_min, x_max = transformed[0, :].min(), transformed[0, :].max()
    y_min, y_max = transformed[1, :].min(), transformed[1, :].max()
    margin = 0.5
    ax2.set_xlim(min(-0.5, x_min - margin), max(4, x_max + margin))
    ax2.set_ylim(min(-0.5, y_min - margin), max(4, y_max + margin))
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.set_title(f'Transformed Shape\nArea = |det| = {abs(det):.2f}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('x-axis')
    ax2.set_ylabel('y-axis')
    
    # Calculate center for text label (excluding last point which is duplicate)
    center_x = np.mean(transformed[0, :4])
    center_y = np.mean(transformed[1, :4])
    ax2.text(center_x, center_y, f'Area = {abs(det):.2f}', 
             fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle(f'{title}\nMatrix = {matrix.tolist()}\ndet = {det:.2f}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"Determinant: {det:.2f}")
    print(f"Area scaling factor: {abs(det):.2f}x")
    if det < 0:
        print("⚠️  Negative determinant: Orientation is REVERSED (reflection)")
    else:
        print("✓ Positive determinant: Orientation is PRESERVED")


def visualize_transformation_vectors(matrix, title="Vector Transformation"):
    """
    Visualize how matrix transforms basis vectors
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Standard basis vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    # Transformed basis vectors
    v1 = matrix @ e1
    v2 = matrix @ e2
    
    # Plot original basis vectors
    ax.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.008, alpha=0.5, label='Original e₁ = [1, 0]')
    ax.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1,
              color='green', width=0.008, alpha=0.5, label='Original e₂ = [0, 1]')
    
    # Plot transformed basis vectors
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
              color='red', width=0.012, label=f'Transformed e₁ → [{v1[0]:.2f}, {v1[1]:.2f}]')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
              color='orange', width=0.012, label=f'Transformed e₂ → [{v2[0]:.2f}, {v2[1]:.2f}]')
    
    # Plot original unit square
    unit_square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    ax.fill(unit_square[0, :], unit_square[1, :], alpha=0.15, color='blue', 
            edgecolor='blue', linewidth=1.5, linestyle='--')
    
    # Plot transformed parallelogram
    transformed = matrix @ unit_square
    ax.fill(transformed[0, :], transformed[1, :], alpha=0.25, color='red', 
            edgecolor='red', linewidth=2)
    
    # Calculate determinant
    det = np.linalg.det(matrix)
    
    # Set equal aspect and limits
    max_val = max(abs(transformed).max(), 2) * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'{title}\ndet = {det:.2f}', fontsize=14, fontweight='bold')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    
    plt.tight_layout()
    plt.show()


def compare_transformations():
    """
    Compare multiple transformation types side-by-side
    """
    transformations = {
        'Identity': np.eye(2),
        'Scaling (2×3)': np.array([[2, 0], [0, 3]]),
        'Shear (horizontal)': np.array([[1, 1], [0, 1]]),
        'Rotation (45°)': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                     [np.sin(np.pi/4), np.cos(np.pi/4)]]),
        'Reflection (x-axis)': np.array([[1, 0], [0, -1]]),
        'Compression': np.array([[0.5, 0], [0, 0.5]])
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    unit_square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    
    for idx, (name, matrix) in enumerate(transformations.items()):
        ax = axes[idx]
        
        # Transform square
        transformed = matrix @ unit_square
        det = np.linalg.det(matrix)
        
        # Plot original
        ax.fill(unit_square[0, :], unit_square[1, :], alpha=0.3, color='blue',
                edgecolor='blue', linewidth=1.5, linestyle='--', label='Original')
        
        # Plot transformed
        ax.fill(transformed[0, :], transformed[1, :], alpha=0.5, color='red',
                edgecolor='red', linewidth=2, label='Transformed')
        
        # Styling
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 4)
        ax.set_title(f'{name}\ndet = {det:.2f}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Comparison of Linear Transformations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def interactive_determinant_explorer():
    """
    Create multiple examples showing different determinant properties
    """
    print("="*70)
    print("DETERMINANT AS AREA SCALING - INTERACTIVE EXPLORER")
    print("="*70)
    
    examples = [
        {
            'name': 'Identity (No Change)',
            'matrix': np.eye(2),
            'description': 'Identity matrix leaves everything unchanged. det = 1'
        },
        {
            'name': 'Uniform Scaling',
            'matrix': np.array([[2, 0], [0, 2]]),
            'description': 'Scales by 2 in all directions. Area increases by 2² = 4'
        },
        {
            'name': 'Non-uniform Scaling',
            'matrix': np.array([[2, 0], [0, 3]]),
            'description': 'Scales by 2 in x, 3 in y. Area increases by 2×3 = 6'
        },
        {
            'name': 'Horizontal Shear',
            'matrix': np.array([[1, 1.5], [0, 1]]),
            'description': 'Shears horizontally. Area preserved (det = 1)'
        },
        {
            'name': 'Rotation 60°',
            'matrix': np.array([[np.cos(np.pi/3), -np.sin(np.pi/3)],
                               [np.sin(np.pi/3), np.cos(np.pi/3)]]),
            'description': 'Rotation preserves area (det = 1)'
        },
        {
            'name': 'Reflection over x-axis',
            'matrix': np.array([[1, 0], [0, -1]]),
            'description': 'Reflection gives negative determinant (det = -1)'
        },
        {
            'name': 'Singular Matrix (Collapse)',
            'matrix': np.array([[1, 2], [2, 4]]),
            'description': 'Collapses to a line. No area (det = 0)'
        }
    ]
    
    for i, example in enumerate(examples):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i+1}: {example['name']}")
        print(f"{'='*70}")
        print(f"Description: {example['description']}")
        print(f"Matrix:\n{example['matrix']}")
        
        visualize_determinant_2d(example['matrix'], example['name'])
        
        if i < len(examples) - 1:
            print("\n" + "-"*70)


# Main execution
if __name__ == "__main__":
    # Example 1: Scaling Transformation
    print("=" * 70)
    print("EXAMPLE 1: Scaling Transformation")
    print("=" * 70)
    scaling = np.array([[2, 0], [0, 3]])
    visualize_determinant_2d(scaling, "Scaling: 2x in x-direction, 3x in y-direction")
    
    # Example 2: Shear Transformation
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Shear Transformation")
    print("=" * 70)
    shear = np.array([[1, 1], [0, 1]])
    visualize_determinant_2d(shear, "Shear Transformation")
    
    # Example 3: Rotation (Preserves Area)
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Rotation (Preserves Area)")
    print("=" * 70)
    angle = np.pi / 4  # 45 degrees
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    visualize_determinant_2d(rotation, "Rotation by 45° (det = 1)")
    
    # Bonus: Vector transformation visualization
    print("\n" + "=" * 70)
    print("BONUS: Vector Transformation Visualization")
    print("=" * 70)
    visualize_transformation_vectors(scaling, "How Scaling Transforms Basis Vectors")
    
    # Bonus: Compare multiple transformations
    print("\n" + "=" * 70)
    print("BONUS: Comparing 6 Different Transformations")
    print("=" * 70)
    compare_transformations()
    
    # Uncomment to run the interactive explorer
    # interactive_determinant_explorer()

'''

#B. Volume Scaling in 3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_determinant_3d(matrix):
    """
    Visualize determinant as volume scaling in 3D
    """
    # Unit cube vertices
    unit_cube = np.array([
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]
    ])
    
    # Transform the cube
    transformed = matrix @ unit_cube
    
    det = np.linalg.det(matrix)
    
    # Create 3D visualization
    fig = plt.figure(figsize=(16, 7))
    
    # Original cube
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(unit_cube[0, :], unit_cube[1, :], unit_cube[2, :], 
                c='blue', s=100, alpha=0.6)
    
    # Draw edges
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), 
             (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    for edge in edges:
        points = unit_cube[:, edge]
        ax1.plot3D(points[0, :], points[1, :], points[2, :], 'b-', linewidth=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Unit Cube\nVolume = 1', fontsize=14, fontweight='bold')
    ax1.set_box_aspect([1,1,1])
    
    # Transformed parallelepiped
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(transformed[0, :], transformed[1, :], transformed[2, :], 
                c='red', s=100, alpha=0.6)
    
    for edge in edges:
        points = transformed[:, edge]
        ax2.plot3D(points[0, :], points[1, :], points[2, :], 'r-', linewidth=2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Transformed Shape\nVolume = |det| = {abs(det):.2f}', 
                  fontsize=14, fontweight='bold')
    ax2.set_box_aspect([1,1,1])
    
    plt.suptitle(f'3D Determinant Visualization\ndet = {det:.2f}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Example 3D transformation
M_3d = np.array([
    [2, 0, 0],
    [0, 1, 0],
    [0, 0, 0.5]
])

print("3D Scaling Matrix:")
print(M_3d)
print(f"Determinant: {np.linalg.det(M_3d):.2f}")
print("Volume scaling: 2 × 1 × 0.5 = 1.0")
print()

visualize_determinant_3d(M_3d)