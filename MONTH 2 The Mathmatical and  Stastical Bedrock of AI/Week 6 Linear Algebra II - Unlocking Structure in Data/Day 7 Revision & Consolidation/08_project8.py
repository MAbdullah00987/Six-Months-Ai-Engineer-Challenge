
#8: Check for Orthogonality: Write a function to check if the columns of a matrix are orthogonal.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)

class OrthogonalityChecker:
    """Complete orthogonality checker for matrix columns"""
    
    def __init__(self, matrix, tolerance=1e-10):
        """
        Initialize with matrix
        
        Parameters:
        - matrix: numpy array (n x m) where columns are vectors
        - tolerance: numerical tolerance for checking orthogonality
        """
        self.matrix = np.array(matrix, dtype=float)
        self.tolerance = tolerance
        self.n_rows, self.n_cols = self.matrix.shape
        self.gram_matrix = None
        self.is_orthogonal = False
        self.is_orthonormal = False
        
    def check_orthogonality(self, verbose=True):
        """
        Check if columns are orthogonal
        
        Two vectors u and v are orthogonal if u·v = 0
        """
        if verbose:
            print("="*80)
            print("CHECKING ORTHOGONALITY")
            print("="*80)
            print(f"\nMatrix shape: {self.matrix.shape}")
            print(f"Number of column vectors: {self.n_cols}")
        
        # Calculate Gram matrix (inner products of all column pairs)
        self.gram_matrix = self.matrix.T @ self.matrix
        
        if verbose:
            print("\nGram Matrix (A^T @ A):")
            print("(Each element [i,j] = dot product of column i and column j)")
            self._print_matrix(self.gram_matrix)
        
        # Check if off-diagonal elements are zero
        off_diagonal_mask = ~np.eye(self.n_cols, dtype=bool)
        off_diagonal_values = self.gram_matrix[off_diagonal_mask]
        max_off_diagonal = np.max(np.abs(off_diagonal_values)) if len(off_diagonal_values) > 0 else 0
        
        self.is_orthogonal = max_off_diagonal < self.tolerance
        
        if verbose:
            print(f"\n{'='*80}")
            print("ORTHOGONALITY TEST RESULTS")
            print("="*80)
            print(f"Maximum off-diagonal value: {max_off_diagonal:.2e}")
            print(f"Tolerance: {self.tolerance:.2e}")
            
            if self.is_orthogonal:
                print("✓ Columns ARE ORTHOGONAL")
            else:
                print("✗ Columns ARE NOT ORTHOGONAL")
        
        # Check each pair individually
        if verbose:
            print("\nPairwise Dot Products:")
            for i in range(self.n_cols):
                for j in range(i+1, self.n_cols):
                    dot_product = self.gram_matrix[i, j]
                    is_ortho = abs(dot_product) < self.tolerance
                    status = "✓ Orthogonal" if is_ortho else "✗ Not orthogonal"
                    print(f"  Column {i+1} · Column {j+1} = {dot_product:.6f} [{status}]")
        
        return self.is_orthogonal
    
    def check_orthonormality(self, verbose=True):
        """
        Check if columns are orthonormal
        
        Vectors are orthonormal if:
        1. They are orthogonal (u·v = 0 for i≠j)
        2. They are unit vectors (||u|| = 1)
        """
        if self.gram_matrix is None:
            self.check_orthogonality(verbose=False)
        
        if verbose:
            print("\n" + "="*80)
            print("CHECKING ORTHONORMALITY")
            print("="*80)
        
        # Check if Gram matrix is identity
        identity = np.eye(self.n_cols)
        self.is_orthonormal = np.allclose(self.gram_matrix, identity, 
                                          atol=self.tolerance)
        
        if verbose:
            # Check norms
            print("\nVector Norms:")
            for i in range(self.n_cols):
                norm = np.linalg.norm(self.matrix[:, i])
                is_unit = np.isclose(norm, 1.0, atol=self.tolerance)
                status = "✓ Unit vector" if is_unit else "✗ Not unit"
                print(f"  ||Column {i+1}|| = {norm:.6f} [{status}]")
            
            print(f"\n{'='*80}")
            if self.is_orthonormal:
                print("✓ Columns ARE ORTHONORMAL (orthogonal + unit vectors)")
            else:
                if self.is_orthogonal:
                    print("✗ Columns are ORTHOGONAL but NOT ORTHONORMAL")
                    print("   (vectors are perpendicular but not unit length)")
                else:
                    print("✗ Columns are NOT ORTHONORMAL")
        
        return self.is_orthonormal
    
    def normalize_columns(self):
        """Normalize columns to create orthonormal matrix"""
        print("\n" + "="*80)
        print("NORMALIZING COLUMNS")
        print("="*80)
        
        normalized = np.zeros_like(self.matrix)
        for i in range(self.n_cols):
            col = self.matrix[:, i]
            norm = np.linalg.norm(col)
            if norm > self.tolerance:
                normalized[:, i] = col / norm
                print(f"Column {i+1}: ||v|| = {norm:.4f} → normalized to ||v|| = 1.0")
            else:
                print(f"Column {i+1}: Zero vector, cannot normalize")
        
        return normalized
    
    def gram_schmidt(self):
        """
        Apply Gram-Schmidt process to orthogonalize columns
        """
        print("\n" + "="*80)
        print("GRAM-SCHMIDT ORTHOGONALIZATION")
        print("="*80)
        
        orthogonal = np.zeros_like(self.matrix)
        
        for i in range(self.n_cols):
            # Start with current column
            vec = self.matrix[:, i].copy()
            
            # Subtract projections onto previous orthogonal vectors
            for j in range(i):
                projection = np.dot(vec, orthogonal[:, j]) / np.dot(orthogonal[:, j], orthogonal[:, j])
                vec = vec - projection * orthogonal[:, j]
                print(f"  Step {i+1}.{j+1}: Remove component parallel to vector {j+1}")
            
            orthogonal[:, i] = vec
            print(f"  Vector {i+1} orthogonalized, norm = {np.linalg.norm(vec):.4f}")
        
        return orthogonal
    
    def _print_matrix(self, matrix):
        """Print matrix nicely"""
        df = pd.DataFrame(matrix)
        print(df.to_string())
    
    def get_angles(self):
        """Calculate angles between all column pairs (in degrees)"""
        angles = np.zeros((self.n_cols, self.n_cols))
        
        for i in range(self.n_cols):
            for j in range(self.n_cols):
                if i == j:
                    angles[i, j] = 0
                else:
                    vi = self.matrix[:, i]
                    vj = self.matrix[:, j]
                    
                    # cos(θ) = (u·v) / (||u|| ||v||)
                    cos_angle = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
                    # Clip to handle numerical errors
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_rad = np.arccos(cos_angle)
                    angles[i, j] = np.degrees(angle_rad)
        
        return angles

def visualize_orthogonality(checker, title=""):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Original Matrix Heatmap
    ax1 = plt.subplot(2, 4, 1)
    sns.heatmap(checker.matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, cbar_kws={'label': 'Value'}, ax=ax1)
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Row Index')
    ax1.set_title(f'Original Matrix\n{title}', fontsize=12, fontweight='bold')
    
    # Plot 2: Gram Matrix
    ax2 = plt.subplot(2, 4, 2)
    sns.heatmap(checker.gram_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                center=0, square=True, cbar_kws={'label': 'Dot Product'}, ax=ax2)
    ax2.set_xlabel('Column Index')
    ax2.set_ylabel('Column Index')
    ax2.set_title('Gram Matrix (A^T @ A)\nDiagonal=norms², Off-diag=dot products', 
                  fontsize=11, fontweight='bold')
    
    # Plot 3: Column Norms
    ax3 = plt.subplot(2, 4, 3)
    norms = [np.linalg.norm(checker.matrix[:, i]) for i in range(checker.n_cols)]
    bars = ax3.bar(range(1, checker.n_cols+1), norms, 
                   color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                label='Unit length', alpha=0.7)
    ax3.set_xlabel('Column Index')
    ax3.set_ylabel('Norm (Length)')
    ax3.set_title('Column Vector Norms', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    # Add value labels
    for i, (bar, norm) in enumerate(zip(bars, norms)):
        ax3.text(bar.get_x() + bar.get_width()/2, norm, f'{norm:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Angles Between Vectors
    ax4 = plt.subplot(2, 4, 4)
    angles = checker.get_angles()
    mask = np.triu(np.ones_like(angles, dtype=bool), k=1)
    sns.heatmap(angles, annot=True, fmt='.1f', cmap='coolwarm',
                mask=~mask, square=True, cbar_kws={'label': 'Angle (degrees)'}, 
                ax=ax4, vmin=0, vmax=180)
    ax4.set_xlabel('Column Index')
    ax4.set_ylabel('Column Index')
    ax4.set_title('Angles Between Columns\n(90° = orthogonal)', 
                  fontsize=11, fontweight='bold')
    
    # Plot 5-6: 2D Vector Visualization (if 2D or 3D)
    if checker.n_rows == 2 and checker.n_cols <= 4:
        ax5 = plt.subplot(2, 4, 5)
        colors = ['red', 'blue', 'green', 'orange']
        
        for i in range(checker.n_cols):
            vec = checker.matrix[:, i]
            ax5.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.2,
                     fc=colors[i], ec=colors[i], linewidth=2, alpha=0.7,
                     label=f'Col {i+1}')
            
            # Add label at tip
            ax5.text(vec[0]*1.1, vec[1]*1.1, f'v{i+1}', 
                    fontsize=12, fontweight='bold')
        
        # Set equal aspect and limits
        max_val = np.max(np.abs(checker.matrix)) * 1.3
        ax5.set_xlim(-max_val, max_val)
        ax5.set_ylim(-max_val, max_val)
        ax5.set_aspect('equal')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='k', linewidth=0.5)
        ax5.axvline(x=0, color='k', linewidth=0.5)
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
        ax5.set_title('2D Vector Visualization', fontsize=12, fontweight='bold')
        ax5.legend()
        
        # Add angle arcs
        for i in range(checker.n_cols):
            for j in range(i+1, checker.n_cols):
                angle = angles[i, j]
                if abs(angle - 90) < 5:  # Close to orthogonal
                    # Draw right angle marker
                    v1 = checker.matrix[:, i]
                    v2 = checker.matrix[:, j]
                    scale = min(np.linalg.norm(v1), np.linalg.norm(v2)) * 0.2
                    
                    v1_norm = v1 / np.linalg.norm(v1) * scale
                    v2_norm = v2 / np.linalg.norm(v2) * scale
                    
                    corner = v1_norm + v2_norm
                    square = np.array([[0, 0], v1_norm, corner, v2_norm, [0, 0]])
                    ax5.plot(square[:, 0], square[:, 1], 'k-', linewidth=1.5, alpha=0.5)
    
    # Plot 7: 3D Visualization (if 3D)
    if checker.n_rows == 3 and checker.n_cols <= 4:
        ax7 = fig.add_subplot(2, 4, 7, projection='3d')
        colors = ['red', 'blue', 'green', 'orange']
        
        for i in range(checker.n_cols):
            vec = checker.matrix[:, i]
            ax7.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                      color=colors[i], arrow_length_ratio=0.1, linewidth=3,
                      alpha=0.7, label=f'Col {i+1}')
            
            # Add label
            ax7.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, f'v{i+1}',
                    fontsize=12, fontweight='bold')
        
        max_val = np.max(np.abs(checker.matrix)) * 1.3
        ax7.set_xlim(-max_val, max_val)
        ax7.set_ylim(-max_val, max_val)
        ax7.set_zlim(-max_val, max_val)
        ax7.set_xlabel('x')
        ax7.set_ylabel('y')
        ax7.set_zlabel('z')
        ax7.set_title('3D Vector Visualization', fontsize=12, fontweight='bold')
        ax7.legend()
    
    # Plot 8: Summary Info
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    summary = f"""
    ORTHOGONALITY SUMMARY
    {'='*40}
    
    Matrix Shape: {checker.n_rows} × {checker.n_cols}
    
    Orthogonal: {'✓ YES' if checker.is_orthogonal else '✗ NO'}
    Orthonormal: {'✓ YES' if checker.is_orthonormal else '✗ NO'}
    
    Column Norms:
    """
    
    for i in range(checker.n_cols):
        norm = np.linalg.norm(checker.matrix[:, i])
        summary += f"\n    ||v{i+1}|| = {norm:.4f}"
    
    summary += "\n\n    Dot Products (off-diagonal):\n"
    for i in range(checker.n_cols):
        for j in range(i+1, checker.n_cols):
            dot = checker.gram_matrix[i, j]
            summary += f"\n    v{i+1}·v{j+1} = {dot:.4f}"
    
    if checker.is_orthogonal:
        summary += "\n\n    ✓ All dot products ≈ 0"
    
    if checker.is_orthonormal:
        summary += "\n    ✓ All norms = 1"
        summary += "\n    ✓ Matrix columns form"
        summary += "\n      orthonormal basis"
    
    ax8.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# TEST CASES
# ============================================================================

print("="*80)
print("ORTHOGONALITY CHECKER - TEST CASES")
print("="*80)

# Test Case 1: Orthonormal vectors (standard basis)
print("\n" + "="*80)
print("TEST CASE 1: Orthonormal Vectors (Standard Basis)")
print("="*80)

A1 = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

checker1 = OrthogonalityChecker(A1)
checker1.check_orthogonality()
checker1.check_orthonormality()
visualize_orthogonality(checker1, "Standard Basis (Orthonormal)")

# Test Case 2: Orthogonal but not orthonormal
print("\n" + "="*80)
print("TEST CASE 2: Orthogonal but NOT Orthonormal")
print("="*80)

A2 = np.array([[2, 0],
               [0, 3]])

checker2 = OrthogonalityChecker(A2)
checker2.check_orthogonality()
checker2.check_orthonormality()

# Normalize
print("\nNormalizing columns...")
A2_normalized = checker2.normalize_columns()
print("\nNormalized Matrix:")
print(A2_normalized)

checker2_norm = OrthogonalityChecker(A2_normalized)
checker2_norm.check_orthogonality(verbose=False)
checker2_norm.check_orthonormality()

visualize_orthogonality(checker2, "Orthogonal (not normalized)")

# Test Case 3: NOT orthogonal
print("\n" + "="*80)
print("TEST CASE 3: NOT Orthogonal")
print("="*80)

A3 = np.array([[1, 1],
               [0, 1]])

checker3 = OrthogonalityChecker(A3)
checker3.check_orthogonality()
checker3.check_orthonormality()
visualize_orthogonality(checker3, "Not Orthogonal")

# Test Case 4: Apply Gram-Schmidt
print("\n" + "="*80)
print("TEST CASE 4: Gram-Schmidt Orthogonalization")
print("="*80)

A4 = np.array([[1, 1, 0],
               [1, 0, 1],
               [0, 1, 1]], dtype=float)

print("Original Matrix:")
print(A4)

checker4 = OrthogonalityChecker(A4)
checker4.check_orthogonality()

# Apply Gram-Schmidt
A4_ortho = checker4.gram_schmidt()
print("\nOrthogonalized Matrix (Gram-Schmidt):")
print(A4_ortho)

# Check result
checker4_ortho = OrthogonalityChecker(A4_ortho)
checker4_ortho.check_orthogonality()
checker4_ortho.check_orthonormality()

# Normalize to get orthonormal
A4_orthonormal = checker4_ortho.normalize_columns()
checker4_orthonormal = OrthogonalityChecker(A4_orthonormal)
checker4_orthonormal.check_orthogonality(verbose=False)
checker4_orthonormal.check_orthonormality()

visualize_orthogonality(checker4_orthonormal, "After Gram-Schmidt + Normalization")

# Test Case 5: 2D Perpendicular Vectors
print("\n" + "="*80)
print("TEST CASE 5: 2D Perpendicular Vectors")
print("="*80)

# 45-degree rotation
angle = np.pi / 4
A5 = np.array([[np.cos(angle), -np.sin(angle)],
               [np.sin(angle), np.cos(angle)]])

checker5 = OrthogonalityChecker(A5)
checker5.check_orthogonality()
checker5.check_orthonormality()
visualize_orthogonality(checker5, "Rotation Matrix (Orthonormal)")

# ============================================================================
# SUMMARY AND KEY CONCEPTS
# ============================================================================

print("\n" + "="*80)
print("KEY CONCEPTS - ORTHOGONALITY")
print("="*80)

print("""
1. ORTHOGONAL VECTORS:
   • Two vectors u and v are orthogonal if u·v = 0
   • Geometrically: perpendicular (90° angle)
   • Example: [1,0] and [0,1]

2. ORTHOGONAL MATRIX:
   • All column pairs are orthogonal
   • Gram matrix A^T @ A has zeros off-diagonal
   • Columns don't have to be unit length

3. ORTHONORMAL VECTORS:
   • Orthogonal AND unit length (||v|| = 1)
   • Example: standard basis vectors

4. ORTHONORMAL MATRIX:
   • All columns are orthonormal
   • A^T @ A = I (identity matrix)
   • Also called "unitary" matrix
   • Properties: A^T = A^(-1)

5. GRAM MATRIX:
   • G = A^T @ A
   • Element [i,j] = dot product of column i and j
   • Diagonal = squared norms
   • Off-diagonal = dot products

6. CHECKING ORTHOGONALITY:
   Method 1: Check if u·v ≈ 0 for all pairs
   Method 2: Check if A^T @ A is diagonal
   Method 3: Check angles (should be 90°)

7. GRAM-SCHMIDT PROCESS:
   • Algorithm to orthogonalize vectors
   • Takes linearly independent vectors
   • Produces orthogonal (or orthonormal) vectors

8. APPLICATIONS:
   • QR decomposition
   • Principal Component Analysis
   • Computer graphics (rotations)
   • Signal processing
   • Least squares problems

9. IMPORTANT PROPERTIES:
   • Orthogonal matrices preserve length
   • Orthogonal matrices preserve angles
   • det(A) = ±1 for orthogonal matrices
   • Easy to invert: A^(-1) = A^T
""")

print("="*80)

# Create a quick reference function
def quick_check_orthogonal(matrix, tolerance=1e-10):
    """
    Quick function to check orthogonality
    
    Returns:
    - is_orthogonal: bool
    - is_orthonormal: bool
    """
    A = np.array(matrix, dtype=float)
    gram = A.T @ A
    
    # Check orthogonality (off-diagonal near zero)
    off_diag = gram - np.diag(np.diag(gram))
    is_orthogonal = np.allclose(off_diag, 0, atol=tolerance)
    
    # Check orthonormality (identity matrix)
    is_orthonormal = np.allclose(gram, np.eye(A.shape[1]), atol=tolerance)
    
    return is_orthogonal, is_orthonormal

print("\n" + "="*80)
print("QUICK CHECK FUNCTION EXAMPLE")
print("="*80)

test_matrix = np.array([[1, 0], [0, 1]])
is_orth, is_orthonorm = quick_check_orthogonal(test_matrix)
print(f"Matrix:\n{test_matrix}")
print(f"Orthogonal: {is_orth}")
print(f"Orthonormal: {is_orthonorm}")

print("\n" + "="*80)