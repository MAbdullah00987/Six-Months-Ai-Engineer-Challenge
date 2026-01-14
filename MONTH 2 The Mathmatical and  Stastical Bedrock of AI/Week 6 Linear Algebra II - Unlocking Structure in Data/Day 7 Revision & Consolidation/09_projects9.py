
# Project 9:
# Linear Independence Check: Use matrix rank to determine if a set of vectors is linearly independent.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import null_space

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)

class LinearIndependenceChecker:
    """Complete linear independence checker using matrix rank"""
    
    def __init__(self, vectors, tolerance=1e-10):
        """
        Initialize with vectors
        
        Parameters:
        - vectors: list of vectors or matrix where each column is a vector
        - tolerance: numerical tolerance for rank calculation
        """
        if isinstance(vectors, list):
            # Convert list of vectors to matrix (vectors as columns)
            self.matrix = np.column_stack(vectors)
        else:
            self.matrix = np.array(vectors, dtype=float)
        
        self.tolerance = tolerance
        self.n_rows, self.n_cols = self.matrix.shape
        self.rank = None
        self.is_independent = None
        self.dependent_columns = []
        self.null_space_basis = None
        
    def check_independence(self, verbose=True):
        """
        Check linear independence using matrix rank
        
        Vectors are linearly independent if rank equals number of vectors
        """
        if verbose:
            print("="*80)
            print("LINEAR INDEPENDENCE CHECK")
            print("="*80)
            print(f"\nMatrix shape: {self.matrix.shape}")
            print(f"Number of vectors: {self.n_cols}")
            print(f"Vector dimension: {self.n_rows}")
            print("\nMatrix (vectors as columns):")
            self._print_matrix(self.matrix)
        
        # Calculate rank
        self.rank = np.linalg.matrix_rank(self.matrix, tol=self.tolerance)
        
        # Check independence
        self.is_independent = (self.rank == self.n_cols)
        
        if verbose:
            print("RANK ANALYSIS")
            print(f"Matrix rank: {self.rank}")
            print(f"Number of vectors: {self.n_cols}")
            print(f"Expected rank for independence: {self.n_cols}")
            
            if self.is_independent:
                print("\nVectors are LINEARLY INDEPENDENT")
                print("  → No vector can be written as combination of others")
                print("  → All vectors contribute unique information")
            else:
                print("\n Vectors are LINEARLY DEPENDENT")
                print(f"  → {self.n_cols - self.rank} redundant vector(s)")
                print("  → Some vectors can be written as combinations of others")
        
        return self.is_independent
    
    def find_dependent_vectors(self, verbose=True):
        """
        Identify which vectors are dependent
        Uses reduced row echelon form (RREF) concept
        """
        if self.rank is None:
            self.check_independence(verbose=False)
        
        if self.is_independent:
            if verbose:
                print("\nAll vectors are independent")
            return []
        
        if verbose:
            print("IDENTIFYING DEPENDENT VECTORS")
        
        # Use SVD to find dependent columns
        U, s, Vt = np.linalg.svd(self.matrix, full_matrices=False)
        
        # Small singular values indicate dependencies
        dependent_indices = np.where(s < self.tolerance)[0]
        self.dependent_columns = dependent_indices.tolist()
        
        if verbose:
            print("\nSingular values:")
            for i, sv in enumerate(s):
                status = "small (dependent)" if sv < self.tolerance else "significant"
                print(f"  σ{i+1} = {sv:.2e} [{status}]")
            
            if len(self.dependent_columns) > 0:
                print(f"\nDependent column indices: {self.dependent_columns}")
        
        return self.dependent_columns
    
    def find_dependency_relationships(self, verbose=True):
        """
        Find the actual linear combinations showing dependencies
        """
        if self.is_independent:
            if verbose:
                print("\n No dependencies to find")
            return []
        
        if verbose:
            
            print("DEPENDENCY RELATIONSHIPS")
        
        # Calculate null space (solutions to Ax = 0)
        self.null_space_basis = null_space(self.matrix)
        
        if verbose:
            print(f"\nNull space dimension: {self.null_space_basis.shape[1]}")
            print(f"(Number of linear dependencies)")
            
            print("\nNull space vectors (each shows a dependency):")
            for i in range(self.null_space_basis.shape[1]):
                null_vec = self.null_space_basis[:, i]
                print(f"\nDependency {i+1}:")
                
                # Create readable equation
                terms = []
                for j, coef in enumerate(null_vec):
                    if abs(coef) > self.tolerance:
                        sign = "+" if coef > 0 else "-"
                        if len(terms) == 0 and sign == "+":
                            sign = ""
                        terms.append(f"{sign} {abs(coef):.3f}*v{j+1}")
                
                equation = " ".join(terms) + " = 0"
                print(f"  {equation}")
                
                # Verify
                result = self.matrix @ null_vec
                print(f"  Verification: ||result|| = {np.linalg.norm(result):.2e} (should be ≈0)")
        
        return self.null_space_basis
    
    def find_maximal_independent_subset(self, verbose=True):
        """
        Find a maximal set of independent vectors
        """
        if verbose:
            print("MAXIMAL INDEPENDENT SUBSET")
        
        # Use QR decomposition with column pivoting
        Q, R, P = scipy.linalg.qr(self.matrix, pivoting=True)
        
        # Independent columns are those with non-zero diagonal in R
        independent_cols = []
        for i in range(min(self.n_rows, self.n_cols)):
            if abs(R[i, i]) > self.tolerance:
                independent_cols.append(P[i])
        
        if verbose:
            print(f"\nMaximal independent subset (column indices): {independent_cols}")
            print(f"Size: {len(independent_cols)} out of {self.n_cols} vectors")
            
            if len(independent_cols) < self.n_cols:
                dependent = [i for i in range(self.n_cols) if i not in independent_cols]
                print(f"Dependent vectors (can be removed): {dependent}")
        
        return independent_cols
    
    def visualize_span(self, verbose=True):
        """
        Describe the span (subspace) of the vectors
        """
        if verbose:
            print("\n" + "="*80)
            print("SPAN ANALYSIS")
            print("="*80)
            print(f"\nThe vectors span a {self.rank}-dimensional subspace")
            print(f"  in {self.n_rows}-dimensional space")
            
            if self.rank == self.n_rows:
                print("\n✓ Vectors span the entire space")
                print("  → Any vector in R^{} can be written as their combination".format(self.n_rows))
            else:
                print(f"\n→ Vectors span only a {self.rank}-dimensional subspace")
                print(f"  → Cannot reach all of R^{self.n_rows}")
    
    def test_vector_in_span(self, test_vector, verbose=True):
        """
        Test if a vector is in the span of the given vectors
        """
        test_vector = np.array(test_vector).flatten()
        
        # Augment matrix with test vector
        augmented = np.column_stack([self.matrix, test_vector])
        rank_augmented = np.linalg.matrix_rank(augmented, tol=self.tolerance)
        
        in_span = (rank_augmented == self.rank)
        
        if verbose:
            print("TESTING VECTOR IN SPAN")
            print(f"Test vector: {test_vector}")
            print(f"\nOriginal rank: {self.rank}")
            print(f"Augmented rank: {rank_augmented}")
            
            if in_span:
                print("Vector IS in the span")
                print("  → Can be written as linear combination of given vectors")
                
                # Try to find coefficients
                try:
                    coeffs = np.linalg.lstsq(self.matrix, test_vector, rcond=None)[0]
                    print(f"\n  Coefficients: {coeffs}")
                    reconstruction = self.matrix @ coeffs
                    error = np.linalg.norm(test_vector - reconstruction)
                    print(f"  Reconstruction error: {error:.2e}")
                except:
                    pass
            else:
                print(" Vector is NOT in the span")
                print("  → Cannot be written as linear combination")
        
        return in_span
    
    def _print_matrix(self, matrix):
        """Print matrix nicely"""
        df = pd.DataFrame(matrix, 
                         columns=[f'v{i+1}' for i in range(matrix.shape[1])])
        print(df.to_string())
    
    def analyze(self):
        """Complete analysis"""
        self.check_independence()
        self.find_dependent_vectors()
        
        if not self.is_independent:
            self.find_dependency_relationships()
            self.find_maximal_independent_subset()
        
        self.visualize_span()
        
        return self.is_independent, self.rank

def visualize_vectors(checker, title=""):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Matrix Heatmap
    ax1 = plt.subplot(2, 4, 1)
    sns.heatmap(checker.matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, cbar_kws={'label': 'Value'}, ax=ax1,
                xticklabels=[f'v{i+1}' for i in range(checker.n_cols)],
                yticklabels=[f'dim{i+1}' for i in range(checker.n_rows)])
    ax1.set_title(f'Vector Matrix\n{title}', fontsize=12, fontweight='bold')
    
    # Plot 2: Rank visualization
    ax2 = plt.subplot(2, 4, 2)
    
    categories = ['Rank', 'Num Vectors', 'Dimension']
    values = [checker.rank, checker.n_cols, checker.n_rows]
    colors = ['green' if checker.is_independent else 'orange', 'blue', 'purple']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Count')
    ax2.set_title('Rank Analysis', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, val, str(val),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add independence status
    status_text = "✓ Independent" if checker.is_independent else "✗ Dependent"
    status_color = 'green' if checker.is_independent else 'red'
    ax2.text(0.5, 0.95, status_text, transform=ax2.transAxes,
            fontsize=14, fontweight='bold', color=status_color,
            ha='center', va='top')
    
    # Plot 3: Singular values
    ax3 = plt.subplot(2, 4, 3)
    U, s, Vt = np.linalg.svd(checker.matrix, full_matrices=False)
    
    colors_sv = ['green' if sv > checker.tolerance else 'red' for sv in s]
    bars = ax3.bar(range(1, len(s)+1), s, color=colors_sv, 
                   alpha=0.7, edgecolor='black')
    ax3.axhline(y=checker.tolerance, color='red', linestyle='--', 
                linewidth=2, label=f'Tolerance ({checker.tolerance:.0e})', alpha=0.7)
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Singular Value')
    ax3.set_title('Singular Values\n(small values → dependence)', 
                  fontsize=11, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    # Plot 4: Vector norms
    ax4 = plt.subplot(2, 4, 4)
    norms = [np.linalg.norm(checker.matrix[:, i]) for i in range(checker.n_cols)]
    bars = ax4.bar(range(1, checker.n_cols+1), norms, 
                   color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Vector Index')
    ax4.set_ylabel('Norm (Length)')
    ax4.set_title('Vector Norms', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, norm) in enumerate(zip(bars, norms)):
        ax4.text(bar.get_x() + bar.get_width()/2, norm, f'{norm:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 5: 2D Visualization (if 2D vectors)
    if checker.n_rows == 2:
        ax5 = plt.subplot(2, 4, 5)
        colors_vec = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i in range(min(checker.n_cols, 6)):
            vec = checker.matrix[:, i]
            color = colors_vec[i % len(colors_vec)]
            ax5.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.2,
                     fc=color, ec=color, linewidth=2.5, alpha=0.7,
                     label=f'v{i+1}')
            
            # Add label
            ax5.text(vec[0]*1.15, vec[1]*1.15, f'v{i+1}', 
                    fontsize=12, fontweight='bold')
        
        max_val = np.max(np.abs(checker.matrix)) * 1.4
        ax5.set_xlim(-max_val, max_val)
        ax5.set_ylim(-max_val, max_val)
        ax5.set_aspect('equal')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='k', linewidth=0.5)
        ax5.axvline(x=0, color='k', linewidth=0.5)
        ax5.set_xlabel('x', fontsize=11)
        ax5.set_ylabel('y', fontsize=11)
        ax5.set_title('2D Vector Visualization', fontsize=12, fontweight='bold')
        ax5.legend()
        
        # Highlight if dependent
        if not checker.is_independent and checker.n_cols == 2:
            # Show that one vector is multiple of another
            ratio = checker.matrix[:, 1] / checker.matrix[:, 0]
            if np.allclose(ratio, ratio[0]):
                ax5.text(0.5, 0.05, f'v2 ≈ {ratio[0]:.2f} × v1', 
                        transform=ax5.transAxes, fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Plot 6: 3D Visualization (if 3D vectors)
    if checker.n_rows == 3:
        ax6 = fig.add_subplot(2, 4, 6, projection='3d')
        colors_vec = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i in range(min(checker.n_cols, 6)):
            vec = checker.matrix[:, i]
            color = colors_vec[i % len(colors_vec)]
            ax6.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                      color=color, arrow_length_ratio=0.15, linewidth=3,
                      alpha=0.7, label=f'v{i+1}')
            
            # Add label
            ax6.text(vec[0]*1.15, vec[1]*1.15, vec[2]*1.15, f'v{i+1}',
                    fontsize=12, fontweight='bold')
        
        max_val = np.max(np.abs(checker.matrix)) * 1.4
        ax6.set_xlim(-max_val, max_val)
        ax6.set_ylim(-max_val, max_val)
        ax6.set_zlim(-max_val, max_val)
        ax6.set_xlabel('x', fontsize=10)
        ax6.set_ylabel('y', fontsize=10)
        ax6.set_zlabel('z', fontsize=10)
        ax6.set_title('3D Vector Visualization', fontsize=12, fontweight='bold')
        ax6.legend()
    
    # Plot 7: Gram Matrix (inner products)
    ax7 = plt.subplot(2, 4, 7)
    gram = checker.matrix.T @ checker.matrix
    sns.heatmap(gram, annot=True, fmt='.2f', cmap='YlOrRd',
                square=True, cbar_kws={'label': 'Inner Product'}, ax=ax7,
                xticklabels=[f'v{i+1}' for i in range(checker.n_cols)],
                yticklabels=[f'v{i+1}' for i in range(checker.n_cols)])
    ax7.set_title('Gram Matrix (V^T V)\nLarge off-diag → correlation', 
                  fontsize=11, fontweight='bold')
    
    # Plot 8: Summary
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    summary = f"""
    LINEAR INDEPENDENCE SUMMARY
    {'='*45}
    
    Vectors: {checker.n_cols}
    Dimension: {checker.n_rows}
    Rank: {checker.rank}
    
    Status: {'✓ INDEPENDENT' if checker.is_independent else '✗ DEPENDENT'}
    
    """
    
    if checker.is_independent:
        summary += """
    ✓ All vectors are independent
    ✓ No redundancy
    ✓ Each vector adds new information
    ✓ Cannot remove any vector
    """
    else:
        redundancy = checker.n_cols - checker.rank
        summary += f"""
    ✗ {redundancy} redundant vector(s)
    ✗ Some vectors are combinations
      of others
    
    Effective dimension: {checker.rank}
    Redundancy: {redundancy}
    """
    
    summary += f"""
    
    SPAN INFORMATION:
    {'='*45}
    
    Vectors span a {checker.rank}D subspace
    in {checker.n_rows}D space
    """
    
    if checker.rank == checker.n_rows:
        summary += "\n    ✓ Span entire space"
    else:
        summary += f"\n    → Limited to {checker.rank}D subspace"
    
    color = 'darkgreen' if checker.is_independent else 'darkred'
    ax8.text(0.05, 0.5, summary, fontsize=10, verticalalignment='center',
            family='monospace', color=color,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    plt.show()


# IMPORT SCIPY

import scipy.linalg

print("LINEAR INDEPENDENCE CHECKER - TEST CASES")

# Test Case 1: Independent vectors (standard basis)
print("TEST CASE 1: Independent Vectors (Standard Basis)")


v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [0, 0, 1]

checker1 = LinearIndependenceChecker([v1, v2, v3])
checker1.analyze()
visualize_vectors(checker1, "Independent (Standard Basis)")

# Test Case 2: Dependent vectors (one is multiple of another)
print("\n" + "="*80)
print("TEST CASE 2: Dependent Vectors (Scalar Multiple)")
print("="*80)

v1 = [1, 2]
v2 = [2, 4]  # v2 = 2*v1

checker2 = LinearIndependenceChecker([v1, v2])
checker2.analyze()
visualize_vectors(checker2, "Dependent (Scalar Multiple)")

# Test Case 3: Dependent vectors (linear combination)
print("\n" + "="*80)
print("TEST CASE 3: Dependent Vectors (Linear Combination)")
print("="*80)

v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [1, 1, 0]  # v3 = v1 + v2

checker3 = LinearIndependenceChecker([v1, v2, v3])
checker3.analyze()
visualize_vectors(checker3, "Dependent (Linear Combination)")

# Test Case 4: More vectors than dimensions
print("\n" + "="*80)
print("TEST CASE 4: More Vectors than Dimensions (Always Dependent)")
print("="*80)

v1 = [1, 0]
v2 = [0, 1]
v3 = [1, 1]
v4 = [2, 3]

checker4 = LinearIndependenceChecker([v1, v2, v3, v4])
checker4.analyze()
print("\nNote: In 2D space, cannot have more than 2 independent vectors!")
visualize_vectors(checker4, "4 vectors in 2D (Must be dependent)")

# Test Case 5: Complex dependency
print("\n" + "="*80)
print("TEST CASE 5: Complex Dependency")
print("="*80)

v1 = [1, 2, 3]
v2 = [2, 4, 6]  # 2*v1
v3 = [1, 1, 1]
v4 = [3, 5, 7]  # 2*v1 + v3

checker5 = LinearIndependenceChecker([v1, v2, v3, v4])
checker5.analyze()
visualize_vectors(checker5, "Complex Dependencies")

# Test Case 6: Test vector in span
print("\n" + "="*80)
print("TEST CASE 6: Testing Vector in Span")
print("="*80)

v1 = [1, 0]
v2 = [0, 1]

checker6 = LinearIndependenceChecker([v1, v2])
checker6.check_independence()

# Test vectors
test_vec1 = [3, 4]  # Should be in span (3*v1 + 4*v2)
test_vec2 = [2, -1]  # Should be in span (2*v1 - 1*v2)

checker6.test_vector_in_span(test_vec1)
checker6.test_vector_in_span(test_vec2)

# Test Case 7: Nearly dependent (numerical issues)
print("\n" + "="*80)
print("TEST CASE 7: Nearly Dependent Vectors (Numerical Sensitivity)")
print("="*80)

v1 = [1, 0]
v2 = [1, 1e-11]  # Almost parallel to v1

checker7 = LinearIndependenceChecker([v1, v2], tolerance=1e-10)
checker7.analyze()
print("\nNote: With tolerance=1e-10, these are considered dependent")

checker7b = LinearIndependenceChecker([v1, v2], tolerance=1e-12)
checker7b.check_independence()
print("\nWith tolerance=1e-12, these are considered independent")


# KEY CONCEPTS AND FORMULAS



print("KEY CONCEPTS - LINEAR INDEPENDENCE")


print("""
1. DEFINITION:
   Vectors v1, v2, ..., vn are linearly independent if:
   c1*v1 + c2*v2 + ... + cn*vn = 0
   implies c1 = c2 = ... = cn = 0
   
   (The only way to get zero is with all zero coefficients)

2. RANK METHOD:
   • Arrange vectors as columns of matrix A
   • Calculate rank(A)
   • Independent if rank(A) = number of vectors
   • Dependent if rank(A) < number of vectors

3. KEY RELATIONSHIPS:
   rank(A) = number of independent columns
   rank(A) ≤ min(rows, columns)
   rank(A) = dimension of column space (span)

4. DEPENDENCY DETECTION:
   Method 1: Check if rank < number of vectors
   Method 2: Find null space (solutions to Ax = 0)
   Method 3: Check if determinant = 0 (square matrices)
   Method 4: Row reduce to echelon form

5. SPECIAL CASES:
   • n vectors in R^m with n > m → ALWAYS dependent
   • Zero vector in set → ALWAYS dependent
   • Any vector repeated → ALWAYS dependent
   • Proportional vectors → ALWAYS dependent

6. GEOMETRIC INTERPRETATION:
   2D: Independent if not collinear (different directions)
   3D: Independent if not coplanar (span volume)
   nD: Independent if span full n-dimensional space

7. NULL SPACE:
   • Solutions to Ax = 0
   • Each null space vector shows a dependency
   • Dimension = n - rank(A)

8. APPLICATIONS:
   • Basis selection for vector spaces
   • Feature selection in machine learning
   • Checking if system has unique solution
   • Dimensionality reduction
   • Determining coordinate systems

9. PROPERTIES:
   • Subset of independent vectors is independent
   • Adding vector to dependent set keeps it dependent
   • Any n+1 vectors in R^n are dependent

10. COMMON TESTS:
    Determinant test (square matrices):
      det(A) ≠ 0 → columns independent
      det(A) = 0 → columns dependent
    
    Rank test (general):
      rank(A) = n → n columns independent
      rank(A) < n → columns dependent
    
    Gram matrix test:
      det(A^T A) ≠ 0 → columns independent
      det(A^T A) = 0 → columns dependent
""")

print("="*80)

# Quick reference function
def quick_check_independence(vectors, tolerance=1e-10):
    """
    Quick function to check linear independence
    
    Parameters:
    - vectors: list of vectors or matrix
    
    Returns:
    - is_independent: bool
    - rank: int
    """
    if isinstance(vectors, list):
        A = np.column_stack(vectors)
    else:
        A = np.array(vectors)
    
    rank = np.linalg.matrix_rank(A, tol=tolerance)
    is_independent = (rank == A.shape[1])
    
    return is_independent, rank

print("\n" + "="*80)
print("QUICK CHECK FUNCTION EXAMPLES")
print("="*80)

# Example 1
vectors1 = [[1, 0], [0, 1]]
indep1, rank1 = quick_check_independence(vectors1)
print(f"\nVectors: {vectors1}")
print(f"Independent: {indep1}, Rank: {rank1}")

# Example 2
vectors2 = [[1, 2], [2, 4]]
indep2, rank2 = quick_check_independence(vectors2)
print(f"\nVectors: {vectors2}")
print(f"Independent: {indep2}, Rank: {rank2}")

print("\n" + "="*80)