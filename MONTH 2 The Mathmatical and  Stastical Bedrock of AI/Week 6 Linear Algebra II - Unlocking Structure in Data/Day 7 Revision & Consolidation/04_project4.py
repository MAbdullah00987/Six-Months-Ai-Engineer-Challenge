
#4: Matrix Inverse Calculator: Write a script that checks if a matrix is invertible and calculates its inverse
# if it is.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

class MatrixInverseCalculator:
    """Complete matrix inverse calculator with verification"""
    
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)
        self.inverse = None
        self.determinant = None
        self.is_invertible = False
        self.condition_number = None
        self.rank = None
        
    def check_square(self):
        """Check if matrix is square"""
        rows, cols = self.matrix.shape
        if rows != cols:
            print(f"❌ Matrix is NOT square: {rows}x{cols}")
            print("   Only square matrices can have inverses!")
            return False
        print(f"✓ Matrix is square: {rows}x{cols}")
        return True
    
    def calculate_determinant(self):
        """Calculate determinant"""
        self.determinant = np.linalg.det(self.matrix)
        print(f"\nDeterminant: {self.determinant:.6f}")
        
        if abs(self.determinant) < 1e-10:
            print("❌ Determinant ≈ 0: Matrix is SINGULAR (not invertible)")
            self.is_invertible = False
            return False
        else:
            print("✓ Determinant ≠ 0: Matrix is INVERTIBLE")
            self.is_invertible = True
            return True
    
    def calculate_rank(self):
        """Calculate matrix rank"""
        self.rank = np.linalg.matrix_rank(self.matrix)
        n = self.matrix.shape[0]
        print(f"\nRank: {self.rank}/{n}")
        
        if self.rank == n:
            print("✓ Full rank: Matrix is invertible")
            return True
        else:
            print(f"❌ Rank deficient: Matrix is singular")
            return False
    
    def calculate_condition_number(self):
        """Calculate condition number (measures numerical stability)"""
        self.condition_number = np.linalg.cond(self.matrix)
        print(f"\nCondition Number: {self.condition_number:.2f}")
        
        if self.condition_number < 10:
            print("✓ Well-conditioned: Numerically stable")
        elif self.condition_number < 100:
            print("⚠ Moderately conditioned: Some numerical sensitivity")
        elif self.condition_number < 1000:
            print("⚠ Ill-conditioned: Numerically sensitive")
        else:
            print("❌ Very ill-conditioned: Numerical instability likely")
    
    def calculate_inverse(self):
        """Calculate matrix inverse"""
        if not self.is_invertible:
            print("\n❌ Cannot calculate inverse: Matrix is not invertible")
            return None
        
        try:
            self.inverse = np.linalg.inv(self.matrix)
            print("\n✓ Inverse calculated successfully!")
            return self.inverse
        except np.linalg.LinAlgError as e:
            print(f"\n❌ Error calculating inverse: {e}")
            return None
    
    def verify_inverse(self):
        """Verify that A @ A_inv = I"""
        if self.inverse is None:
            return False
        
        # Calculate A @ A_inv
        product = self.matrix @ self.inverse
        identity = np.eye(self.matrix.shape[0])
        
        # Check if result is close to identity
        is_correct = np.allclose(product, identity, atol=1e-10)
        
        print("\n" + "="*70)
        print("VERIFICATION: A @ A⁻¹ = I")
        print("="*70)
        
        if is_correct:
            print("✓ Verification PASSED: A @ A⁻¹ = I (within tolerance)")
        else:
            print("⚠ Verification shows numerical errors")
            max_error = np.max(np.abs(product - identity))
            print(f"   Maximum error: {max_error:.2e}")
        
        return is_correct, product
    
    def analyze(self):
        """Complete analysis of the matrix"""
        print("="*70)
        print("MATRIX INVERSE ANALYSIS")
        print("="*70)
        
        print("\nOriginal Matrix A:")
        print(self.matrix)
        
        # Step 1: Check if square
        if not self.check_square():
            return
        
        # Step 2: Calculate determinant
        if not self.calculate_determinant():
            self.calculate_rank()
            return
        
        # Step 3: Calculate rank
        self.calculate_rank()
        
        # Step 4: Calculate condition number
        self.calculate_condition_number()
        
        # Step 5: Calculate inverse
        self.calculate_inverse()
        
        if self.inverse is not None:
            print("\nInverse Matrix A⁻¹:")
            print(self.inverse)
            
            # Step 6: Verify inverse
            self.verify_inverse()
        
        return self.inverse

def visualize_matrix_properties(calc):
    """Create comprehensive visualizations"""
    
    if calc.inverse is None:
        # Only show original matrix if not invertible
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original matrix heatmap
        sns.heatmap(calc.matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, cbar_kws={'label': 'Value'},
                    ax=axes[0])
        axes[0].set_title('Original Matrix A', fontsize=14, fontweight='bold')
        
        # Information panel
        axes[1].axis('off')
        info_text = f"""
        MATRIX PROPERTIES
        ═══════════════════════
        
        Shape: {calc.matrix.shape[0]}×{calc.matrix.shape[1]}
        
        Determinant: {calc.determinant:.6f}
        
        Rank: {calc.rank}
        
        Condition Number: {calc.condition_number:.2f}
        
        STATUS: NOT INVERTIBLE
        
        Reason: {'Not square' if calc.matrix.shape[0] != calc.matrix.shape[1] 
                 else 'Determinant = 0 (Singular)'}
        """
        axes[1].text(0.1, 0.5, info_text, fontsize=12, 
                     verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        plt.show()
        return
    
    # Full visualization for invertible matrices
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Original Matrix Heatmap
    ax1 = plt.subplot(3, 3, 1)
    sns.heatmap(calc.matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, cbar_kws={'label': 'Value'}, ax=ax1)
    ax1.set_title('Original Matrix A', fontsize=12, fontweight='bold')
    
    # Plot 2: Inverse Matrix Heatmap
    ax2 = plt.subplot(3, 3, 2)
    sns.heatmap(calc.inverse, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, cbar_kws={'label': 'Value'}, ax=ax2)
    ax2.set_title('Inverse Matrix A⁻¹', fontsize=12, fontweight='bold')
    
    # Plot 3: Product A @ A⁻¹ (should be identity)
    ax3 = plt.subplot(3, 3, 3)
    product = calc.matrix @ calc.inverse
    sns.heatmap(product, annot=True, fmt='.2f', cmap='RdGy',
                center=0, square=True, cbar_kws={'label': 'Value'}, ax=ax3)
    ax3.set_title('Verification: A @ A⁻¹', fontsize=12, fontweight='bold')
    
    # Plot 4: Eigenvalues visualization
    ax4 = plt.subplot(3, 3, 4)
    eigenvalues = np.linalg.eigvals(calc.matrix)
    ax4.scatter(eigenvalues.real, eigenvalues.imag, s=200, c='red', 
                edgecolors='black', linewidth=2, alpha=0.7)
    ax4.axhline(y=0, color='k', linewidth=0.5)
    ax4.axvline(x=0, color='k', linewidth=0.5)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Real Part')
    ax4.set_ylabel('Imaginary Part')
    ax4.set_title('Eigenvalues of A', fontsize=12, fontweight='bold')
    
    # Add eigenvalue labels
    for i, ev in enumerate(eigenvalues):
        ax4.annotate(f'λ{i+1}\n{ev:.2f}', 
                     (ev.real, ev.imag),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, bbox=dict(boxstyle='round', 
                                          facecolor='wheat', alpha=0.5))
    
    # Plot 5: Singular values
    ax5 = plt.subplot(3, 3, 5)
    U, S, Vt = np.linalg.svd(calc.matrix)
    ax5.bar(range(1, len(S)+1), S, color='steelblue', alpha=0.7)
    ax5.set_xlabel('Index')
    ax5.set_ylabel('Singular Value')
    ax5.set_title('Singular Values', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, val in enumerate(S):
        ax5.text(i+1, val, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Condition number interpretation
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis('off')
    
    # Color code based on condition number
    if calc.condition_number < 10:
        cond_status = "Excellent"
        cond_color = "green"
    elif calc.condition_number < 100:
        cond_status = "Good"
        cond_color = "orange"
    elif calc.condition_number < 1000:
        cond_status = "Poor"
        cond_color = "red"
    else:
        cond_status = "Very Poor"
        cond_color = "darkred"
    
    info_text = f"""
    NUMERICAL PROPERTIES
    ══════════════════════
    
    Determinant: {calc.determinant:.6f}
    
    Condition Number: {calc.condition_number:.2f}
    Status: {cond_status}
    
    Rank: {calc.rank}/{calc.matrix.shape[0]}
    
    Matrix Norm: {np.linalg.norm(calc.matrix):.2f}
    
    Inverse Norm: {np.linalg.norm(calc.inverse):.2f}
    
    Trace(A): {np.trace(calc.matrix):.2f}
    
    Trace(A⁻¹): {np.trace(calc.inverse):.2f}
    """
    
    ax6.text(0.1, 0.5, info_text, fontsize=11, 
             verticalalignment='center', family='monospace')
    
    # Plot 7: Error in verification
    ax7 = plt.subplot(3, 3, 7)
    identity = np.eye(calc.matrix.shape[0])
    error_matrix = product - identity
    sns.heatmap(error_matrix, annot=True, fmt='.2e', cmap='Reds',
                square=True, cbar_kws={'label': 'Error'}, ax=ax7)
    ax7.set_title('Error: (A @ A⁻¹) - I', fontsize=12, fontweight='bold')
    
    # Plot 8: Effect of transformation
    if calc.matrix.shape[0] == 2:
        ax8 = plt.subplot(3, 3, 8)
        
        # Create unit vectors
        unit_vectors = np.array([[1, 0], [0, 1]]).T
        transformed = calc.matrix @ unit_vectors
        
        # Plot original vectors
        ax8.quiver([0, 0], [0, 0], unit_vectors[0], unit_vectors[1],
                   angles='xy', scale_units='xy', scale=1,
                   color=['red', 'blue'], alpha=0.5, width=0.01,
                   label='Original')
        
        # Plot transformed vectors
        ax8.quiver([0, 0], [0, 0], transformed[0], transformed[1],
                   angles='xy', scale_units='xy', scale=1,
                   color=['darkred', 'darkblue'], width=0.01,
                   label='Transformed')
        
        ax8.set_xlim(-3, 3)
        ax8.set_ylim(-3, 3)
        ax8.axhline(y=0, color='k', linewidth=0.5)
        ax8.axvline(x=0, color='k', linewidth=0.5)
        ax8.grid(True, alpha=0.3)
        ax8.set_aspect('equal')
        ax8.set_xlabel('x')
        ax8.set_ylabel('y')
        ax8.set_title('Transformation Effect (2D)', fontsize=12, fontweight='bold')
        ax8.legend()
    
    # Plot 9: Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary = f"""
    VERIFICATION SUMMARY
    ═══════════════════════
    
    ✓ Matrix is invertible
    
    ✓ Determinant ≠ 0
    
    ✓ Full rank
    
    ✓ Inverse calculated
    
    ✓ A @ A⁻¹ ≈ I
    
    Max verification error:
    {np.max(np.abs(error_matrix)):.2e}
    
    All eigenvalues non-zero:
    {np.all(np.abs(eigenvalues) > 1e-10)}
    """
    
    ax9.text(0.1, 0.5, summary, fontsize=11, 
             verticalalignment='center', family='monospace',
             color='darkgreen')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# TEST CASES
# ============================================================================

print("="*70)
print("MATRIX INVERSE CALCULATOR - TEST CASES")
print("="*70)

# Test Case 1: Simple 2x2 invertible matrix
print("\n" + "="*70)
print("TEST CASE 1: Simple 2×2 Invertible Matrix")
print("="*70)

A1 = [[4, 7],
      [2, 6]]

calc1 = MatrixInverseCalculator(A1)
calc1.analyze()
visualize_matrix_properties(calc1)

# Test Case 2: 3x3 invertible matrix
print("\n" + "="*70)
print("TEST CASE 2: 3×3 Invertible Matrix")
print("="*70)

A2 = [[2, -1, 0],
      [-1, 2, -1],
      [0, -1, 2]]

calc2 = MatrixInverseCalculator(A2)
calc2.analyze()
visualize_matrix_properties(calc2)

# Test Case 3: Singular matrix (not invertible)
print("\n" + "="*70)
print("TEST CASE 3: Singular Matrix (NOT Invertible)")
print("="*70)

A3 = [[1, 2, 3],
      [2, 4, 6],
      [3, 6, 9]]

calc3 = MatrixInverseCalculator(A3)
calc3.analyze()
visualize_matrix_properties(calc3)

# Test Case 4: Identity matrix
print("\n" + "="*70)
print("TEST CASE 4: Identity Matrix (Self-Inverse)")
print("="*70)

A4 = np.eye(3)

calc4 = MatrixInverseCalculator(A4)
calc4.analyze()
visualize_matrix_properties(calc4)

# Test Case 5: Nearly singular matrix (ill-conditioned)
print("\n" + "="*70)
print("TEST CASE 5: Ill-Conditioned Matrix")
print("="*70)

A5 = [[1, 1],
      [1, 1.0001]]

calc5 = MatrixInverseCalculator(A5)
calc5.analyze()
visualize_matrix_properties(calc5)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("KEY CONCEPTS - MATRIX INVERTIBILITY")
print("="*70)
print("""
1. INVERTIBILITY CONDITIONS:
   • Matrix must be SQUARE (n×n)
   • Determinant must be NON-ZERO
   • Matrix must have FULL RANK
   • All eigenvalues must be NON-ZERO

2. CONDITION NUMBER:
   • Measures numerical stability
   • κ(A) = ||A|| × ||A⁻¹||
   • Low (< 10): Well-conditioned
   • High (> 1000): Ill-conditioned

3. VERIFICATION:
   • A @ A⁻¹ = I (Identity matrix)
   • A⁻¹ @ A = I
   • Used to check calculation accuracy

4. SINGULAR VALUE DECOMPOSITION:
   • All singular values must be non-zero
   • Smallest singular value indicates stability
   • SVD is more numerically stable than direct inversion
""")
