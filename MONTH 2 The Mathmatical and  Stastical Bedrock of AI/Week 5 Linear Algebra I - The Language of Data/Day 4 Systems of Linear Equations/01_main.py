#Day 4: Systems of Linear Equations

#Read: Mathematics for ML Chapter 2 (Section 2.3) - Solving linear systems
#Watch: Coursera Week 3 - Gaussian elimination, matrix inverses
#Focus: Row reduction, rank, linear independence

#Project 1: Solving Linear Equations
#Set up 3-4 real-world problems as Ax = b
#Solve using np.linalg.solve()
#Verify solutions by substitution
#Handle singular matrices (discuss why no solution exists)


#Project 2: Network Flow Model
#Model a 4-intersection traffic network
#Set up conservation equations: flow in = flow out
#Solve for unknown flows
#Visualize network with node/edge diagram

#Exercise Set:
#Solve 10 systems by hand using Gaussian elimination
#dentify inconsistent and underdetermined systems


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# PART 1: SOLVING LINEAR SYSTEMS - MULTIPLE METHODS


class LinearSystemSolver:
    """Complete toolkit for solving linear systems"""
    
    def __init__(self, A, b):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.n = len(b)
        
    def method_1_numpy_solve(self):
        """Method 1: Using numpy's built-in solver (fastest)"""
        try:
            x = np.linalg.solve(self.A, self.b)
            return x, "Success"
        except np.linalg.LinAlgError:
            return None, "Singular matrix - no unique solution"
    
    def method_2_inverse(self):
        """Method 2: Using matrix inverse (x = A^-1 * b)"""
        try:
            A_inv = np.linalg.inv(self.A)
            x = A_inv @ self.b
            return x, "Success"
        except np.linalg.LinAlgError:
            return None, "Matrix not invertible"
    
    def method_3_gaussian_elimination(self):
        """Method 3: Manual Gaussian elimination"""
        # Create augmented matrix [A|b]
        Ab = np.column_stack([self.A.copy(), self.b.copy()])
        n = self.n
        
        # Forward elimination with partial pivoting
        for i in range(n):
            # Find pivot (largest element in column)
            max_row = i + np.argmax(np.abs(Ab[i:, i]))
            if abs(Ab[max_row, i]) < 1e-10:
                return None, "Singular matrix"
            
            # Swap rows
            Ab[[i, max_row]] = Ab[[max_row, i]]
            
            # Eliminate below pivot
            for j in range(i + 1, n):
                factor = Ab[j, i] / Ab[i, i]
                Ab[j, i:] -= factor * Ab[i, i:]
        
        # Back substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (Ab[i, -1] - np.sum(Ab[i, i+1:n] * x[i+1:n])) / Ab[i, i]
        
        return x, "Success"
    
    def method_4_lu_decomposition(self):
        """Method 4: LU decomposition"""
        try:
            P, L, U = linalg.lu(self.A)
            # Solve Ly = Pb, then Ux = y
            y = linalg.solve_triangular(L, P.T @ self.b, lower=True)
            x = linalg.solve_triangular(U, y, lower=False)
            return x, "Success"
        except:
            return None, "LU decomposition failed"
    
    def method_5_cramer_rule(self):
        """Method 5: Cramer's rule (only for small systems)"""
        det_A = np.linalg.det(self.A)
        if abs(det_A) < 1e-10:
            return None, "Determinant is zero"
        
        x = np.zeros(self.n)
        for i in range(self.n):
            A_i = self.A.copy()
            A_i[:, i] = self.b
            x[i] = np.linalg.det(A_i) / det_A
        
        return x, "Success"
    
    def compare_all_methods(self):
        """Compare all methods and their performance"""
        methods = [
            ("NumPy solve", self.method_1_numpy_solve),
            ("Matrix inverse", self.method_2_inverse),
            ("Gaussian elimination", self.method_3_gaussian_elimination),
            ("LU decomposition", self.method_4_lu_decomposition),
            ("Cramer's rule", self.method_5_cramer_rule)
        ]
        
        print("="*70)
        print("COMPARING ALL SOLUTION METHODS")
        print("="*70)
        
        for name, method in methods:
            import time
            start = time.time()
            result, status = method()
            elapsed = time.time() - start
            
            print(f"\n{name}:")
            print(f"  Status: {status}")
            if result is not None:
                print(f"  Solution: {result}")
                print(f"  Time: {elapsed*1000:.4f} ms")
                print(f"  Verification (Ax): {self.A @ result}")
                print(f"  Error: {np.linalg.norm(self.A @ result - self.b):.2e}")
            print("-"*70)


# Example usage
print("\n" + "="*70)
print("EXAMPLE 1: Simple 2x2 System")
print("="*70)
A = [[2, 3], [4, -1]]
b = [8, 2]
solver = LinearSystemSolver(A, b)
solver.compare_all_methods()



# PART 2: GAUSSIAN ELIMINATION - DETAILED IMPLEMENTATION

def gaussian_elimination_verbose(A, b):
    """Gaussian elimination with step-by-step output"""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # Augmented matrix
    Ab = np.column_stack([A, b])
    
    print("\n" + "="*70)
    print("GAUSSIAN ELIMINATION - STEP BY STEP")
    print("="*70)
    print("\nInitial augmented matrix [A|b]:")
    print(Ab)
    
    # Forward elimination
    print("\n--- FORWARD ELIMINATION ---")
    for i in range(n):
        print(f"\nStep {i+1}: Eliminate column {i} below row {i}")
        
        # Partial pivoting
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
            print(f"  Swapped row {i} with row {max_row}")
        
        pivot = Ab[i, i]
        print(f"  Pivot element: {pivot:.4f}")
        
        # Eliminate
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
            print(f"  R{j} = R{j} - ({factor:.4f}) * R{i}")
        
        print(f"\nMatrix after step {i+1}:")
        print(Ab)
    
    # Back substitution
    print("\n--- BACK SUBSTITUTION ---")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.sum(Ab[i, i+1:n] * x[i+1:n])) / Ab[i, i]
        print(f"x[{i}] = {x[i]:.4f}")
    
    return x

# Example
A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
b = [8, -11, -3]
solution = gaussian_elimination_verbose(A, b)



# PART 3: MATRIX RANK AND ROW REDUCTION


def analyze_matrix_rank(A):
    """Comprehensive rank analysis"""
    A = np.array(A, dtype=float)
    
    print("\n" + "="*70)
    print("MATRIX RANK ANALYSIS")
    print("="*70)
    print("\nOriginal matrix:")
    print(A)
    print(f"Shape: {A.shape}")
    
    # Compute rank
    rank = np.linalg.matrix_rank(A)
    print(f"\nRank: {rank}")
    print(f"Is full rank? {rank == min(A.shape)}")
    
    # Row reduction to echelon form
    U = A.copy()
    m, n = U.shape
    
    print("\nRow reduction to echelon form:")
    pivot_count = 0
    for i in range(min(m, n)):
        # Find pivot
        pivot_row = i + np.argmax(np.abs(U[i:, i]))
        if abs(U[pivot_row, i]) > 1e-10:
            U[[i, pivot_row]] = U[[pivot_row, i]]
            U[i] /= U[i, i]  # Normalize
            
            # Eliminate below
            for j in range(i + 1, m):
                U[j] -= U[j, i] * U[i]
            
            pivot_count += 1
    
    print(U)
    print(f"\nNumber of non-zero rows: {pivot_count}")
    
    # SVD for detailed analysis
    U_svd, s, Vt = np.linalg.svd(A)
    print(f"\nSingular values: {s}")
    print(f"Condition number: {np.max(s)/np.min(s):.2e}")
    
    return rank

# Examples
print("\nExample 1: Full rank matrix")
A1 = [[1, 2], [3, 4]]
analyze_matrix_rank(A1)

print("\nExample 2: Rank deficient matrix")
A2 = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]
analyze_matrix_rank(A2)



# PART 4: LINEAR INDEPENDENCE


def test_linear_independence(vectors):
    """Test if vectors are linearly independent"""
    A = np.array(vectors)
    
    print("\n" + "="*70)
    print("LINEAR INDEPENDENCE TEST")
    print("="*70)
    print("\nVectors:")
    for i, v in enumerate(vectors):
        print(f"v{i+1} = {v}")
    
    # Method 1: Rank test
    rank = np.linalg.matrix_rank(A)
    n_vectors = A.shape[0]
    
    print(f"\nNumber of vectors: {n_vectors}")
    print(f"Rank of matrix: {rank}")
    
    if rank == n_vectors:
        print("[OK] LINEARLY INDEPENDENT")
        independent = True
    else:
        print("[FAIL] LINEARLY DEPENDENT")
        independent = False
    
    # Method 2: Find linear combination (if dependent)
    if not independent:
        print("\nFinding linear combination:")
        # Solve homogeneous system
        null_space = linalg.null_space(A.T)
        if null_space.size > 0:
            coeffs = null_space[:, 0]
            print(f"Coefficients: {coeffs}")
            
            # Normalize
            coeffs = coeffs / coeffs[np.argmax(np.abs(coeffs))]
            print(f"Normalized: {coeffs}")
            
            # Show combination
            print("\nLinear combination (approximately zero):")
            result = np.sum([c * np.array(v) for c, v in zip(coeffs, vectors)], axis=0)
            print(f"Result: {result}")
    
    # Method 3: Determinant (square matrices only)
    if A.shape[0] == A.shape[1]:
        det = np.linalg.det(A)
        print(f"\nDeterminant: {det:.6f}")
        print(f"Non-singular: {abs(det) > 1e-10}")
    
    return independent

# Examples
print("\nExample 1: Independent vectors")
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [0, 0, 1]
test_linear_independence([v1, v2, v3])

print("\nExample 2: Dependent vectors")
v1 = [1, 2, 3]
v2 = [2, 4, 6]  # 2 * v1
v3 = [1, 1, 1]
test_linear_independence([v1, v2, v3])



# PART 5: COMPREHENSIVE VISUALIZATIONS


def create_comprehensive_visualization():
    """Create all visualizations for linear systems"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. System of equations (2D)
    ax1 = plt.subplot(2, 3, 1)
    x = np.linspace(-5, 5, 100)
    y1 = (8 - 2*x) / 3  # 2x + 3y = 8
    y2 = (2 - 4*x) / (-1)  # 4x - y = 2
    ax1.plot(x, y1, 'b-', linewidth=2, label='2x + 3y = 8')
    ax1.plot(x, y2, 'r-', linewidth=2, label='4x - y = 2')
    ax1.plot(1, 2, 'go', markersize=12, label='Solution (1, 2)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('2D Linear System', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3, 5)
    ax1.set_ylim(-2, 6)
    
    # 2. Coefficient matrix heatmap
    ax2 = plt.subplot(2, 3, 2)
    A = np.array([[2, 3], [4, -1]])
    sns.heatmap(A, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                center=0, cbar_kws={'label': 'Coefficient Value'},
                linewidths=2, linecolor='black', ax=ax2)
    ax2.set_title('Coefficient Matrix A', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # 3. Gaussian elimination animation
    ax3 = plt.subplot(2, 3, 3)
    steps = [
        np.array([[2, 3, 8], [4, -1, 2]]),
        np.array([[2, 3, 8], [0, -7, -14]])
    ]
    colors = ['lightblue', 'lightgreen']
    for i, step in enumerate(steps):
        y_offset = i * 3
        for j, row in enumerate(step):
            text = f"[{row[0]:4.1f} {row[1]:4.1f} | {row[2]:4.1f}]"
            ax3.text(0.1, 0.7 - y_offset*0.1 - j*0.08, text, 
                    fontsize=10, family='monospace',
                    bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.5))
    ax3.text(0.1, 0.85, 'Step 1: Original', fontsize=11, fontweight='bold')
    ax3.text(0.1, 0.45, 'Step 2: After R2-2R1', fontsize=11, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Gaussian Elimination', fontsize=14, fontweight='bold')
    
    # 4. Vector space visualization
    ax4 = plt.subplot(2, 3, 4)
    origin = [0, 0]
    v1 = [2, 3]
    v2 = [4, -1]
    ax4.quiver(*origin, v1[0], v1[1], angles='xy', scale_units='xy', 
               scale=1, color='blue', width=0.015, label='Column 1')
    ax4.quiver(*origin, v2[0], v2[1], angles='xy', scale_units='xy', 
               scale=1, color='red', width=0.015, label='Column 2')
    # Show span
    for a in np.linspace(-2, 2, 9):
        for b in np.linspace(-2, 2, 9):
            result = a * np.array(v1) + b * np.array(v2)
            ax4.plot(result[0], result[1], 'k.', alpha=0.1, markersize=2)
    ax4.set_xlim(-10, 10)
    ax4.set_ylim(-10, 10)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('y', fontsize=12)
    ax4.set_title('Column Space Span', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Rank vs Matrix Size
    ax5 = plt.subplot(2, 3, 5)
    sizes = range(1, 8)
    full_ranks = []
    deficient_ranks = []
    for n in sizes:
        # Full rank
        M1 = np.random.randn(n, n)
        full_ranks.append(np.linalg.matrix_rank(M1))
        
        # Rank deficient
        M2 = np.random.randn(n, n)
        M2[-1] = M2[0]  # Make last row equal to first
        deficient_ranks.append(np.linalg.matrix_rank(M2))
    
    x_pos = np.arange(len(sizes))
    width = 0.35
    ax5.bar(x_pos - width/2, full_ranks, width, label='Full Rank', 
            color='green', alpha=0.7)
    ax5.bar(x_pos + width/2, deficient_ranks, width, label='Rank Deficient',
            color='orange', alpha=0.7)
    ax5.set_xlabel('Matrix Size (n×n)', fontsize=12)
    ax5.set_ylabel('Rank', fontsize=12)
    ax5.set_title('Matrix Rank Comparison', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(sizes)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Condition number vs Error
    ax6 = plt.subplot(2, 3, 6)
    cond_numbers = []
    errors = []
    for _ in range(50):
        n = 3
        A = np.random.randn(n, n)
        cond = np.linalg.cond(A)
        x_true = np.random.randn(n)
        b = A @ x_true
        b_noisy = b + 1e-5 * np.random.randn(n)
        x_computed = np.linalg.solve(A, b_noisy)
        error = np.linalg.norm(x_computed - x_true)
        
        cond_numbers.append(cond)
        errors.append(error)
    
    ax6.scatter(cond_numbers, errors, alpha=0.6, s=50)
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.set_xlabel('Condition Number', fontsize=12)
    ax6.set_ylabel('Solution Error', fontsize=12)
    ax6.set_title('Conditioning vs Numerical Error', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('linear_systems_complete_visualization.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Visualization saved as 'linear_systems_complete_visualization.png'")
    # plt.show()

# Create all visualizations
create_comprehensive_visualization()



# PART 6: REAL-WORLD APPLICATION EXAMPLES


def application_circuit_analysis():
    """Example: Kirchhoff's laws in circuit analysis"""
    print("\n" + "="*70)
    print("APPLICATION 1: ELECTRICAL CIRCUIT ANALYSIS")
    print("="*70)
    print("""
    Circuit with 3 loops:
    Loop 1: 10V - I1*2Ohm - I2*3Ohm = 0
    Loop 2: I2*3Ohm - I3*4Ohm = 0
    Current conservation: I1 = I2 + I3
    """)
    
    # System of equations
    A = np.array([
        [2, 3, 0],    # Loop 1
        [0, 3, -4],   # Loop 2
        [1, -1, -1]   # Current conservation
    ])
    b = np.array([10, 0, 0])
    
    currents = np.linalg.solve(A, b)
    print(f"\nSolution:")
    print(f"I1 = {currents[0]:.3f} A")
    print(f"I2 = {currents[1]:.3f} A")
    print(f"I3 = {currents[2]:.3f} A")

def application_portfolio_optimization():
    """Example: Portfolio allocation"""
    print("\n" + "="*70)
    print("APPLICATION 2: PORTFOLIO OPTIMIZATION")
    print("="*70)
    print("""
    Invest in 3 assets with constraints:
    - Total investment = $10,000
    - Expected return = $500
    - Risk constraint
    """)
    
    # Constraints matrix
    A = np.array([
        [1, 1, 1],      # Total money
        [0.05, 0.03, 0.07],  # Expected returns
        [2, 1, 4]       # Risk weights
    ])
    b = np.array([10000, 500, 15000])
    
    allocation = np.linalg.solve(A, b)
    print(f"\nOptimal allocation:")
    for i, amount in enumerate(allocation):
        print(f"Asset {i+1}: ${amount:.2f}")

# Run applications
application_circuit_analysis()
application_portfolio_optimization()



# Main Points 
'''
1. CHOOSING THE RIGHT METHOD:
   - Small systems (n < 10): Any method works
   - Large systems: Use np.linalg.solve (uses LU internally)
   - Multiple right-hand sides: Use LU decomposition once
   - Sparse systems: Use scipy.sparse.linalg
   
2. NUMERICAL STABILITY:
   - Always check condition number
   - Use partial pivoting in Gaussian elimination
   - Be aware of floating-point errors
   
3. CHECKING SOLUTIONS:
   - Verify: ||Ax - b|| should be small
   - Check residual norm
   - Test edge cases
   
4. RANK AND INDEPENDENCE:
   - Use rank to determine solvability
   - Check determinant for square matrices
   - Use SVD for numerical rank with tolerance
   
5. VISUALIZATION TIPS:
   - Plot 2D systems to understand geometry
   - Use heatmaps for matrix structure
   - Visualize column space for intuition

6. COMMON PITFALLS:
   - Don't invert matrices unless necessary (slow and unstable)
   - Check for singular matrices before solving
   - Be careful with ill-conditioned systems
   - Always use appropriate tolerances
'''
