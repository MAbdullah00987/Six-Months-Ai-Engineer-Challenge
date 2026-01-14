

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import null_space

# ============================================================================
# PROBLEM SET 1: BASIC SYSTEMS
# ============================================================================

print("="*70)
print("PROBLEM SET 1: BASIC LINEAR SYSTEMS")
print("="*70)

def problem_1_1():
    """
    Problem 1.1: Solve the system:
    3x + 2y = 12
    5x - y = 11
    """
    print("\nProblem 1.1:")
    print("Solve: 3x + 2y = 12")
    print("       5x - y = 11")
    
    A = np.array([[3, 2], [5, -1]])
    b = np.array([12, 11])
    
    x = np.linalg.solve(A, b)
    print(f"\nSolution: x = {x[0]:.2f}, y = {x[1]:.2f}")
    
    # Verification
    verification = A @ x
    print(f"Verification: Ax = {verification}")
    print(f"Expected: b = {b}")
    print(f"Match: {np.allclose(verification, b)}")
    
    return x

def problem_1_2():
    """
    Problem 1.2: Overdetermined system (more equations than unknowns)
    x + y = 3
    2x - y = 0
    x - 2y = -3
    Find least squares solution
    """
    print("\n" + "-"*70)
    print("Problem 1.2: Overdetermined System (Least Squares)")
    print("x + y = 3")
    print("2x - y = 0")
    print("x - 2y = -3")
    
    A = np.array([[1, 1], [2, -1], [1, -2]])
    b = np.array([3, 0, -3])
    
    # Least squares solution
    x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
    
    print(f"\nLeast squares solution: x = {x_ls[0]:.4f}, y = {x_ls[1]:.4f}")
    
    # Calculate residual
    residual = A @ x_ls - b
    print(f"Residuals for each equation: {residual}")
    print(f"Residual norm: {np.linalg.norm(residual):.4f}")
    
    return x_ls

def problem_1_3():
    """
    Problem 1.3: Underdetermined system (more unknowns than equations)
    x + 2y + z = 5
    2x + y + 3z = 8
    Find minimum norm solution
    """
    print("\n" + "-"*70)
    print("Problem 1.3: Underdetermined System")
    print("x + 2y + z = 5")
    print("2x + y + 3z = 8")
    
    A = np.array([[1, 2, 1], [2, 1, 3]])
    b = np.array([5, 8])
    
    # Minimum norm solution using pseudoinverse
    x_min = np.linalg.pinv(A) @ b
    
    print(f"\nMinimum norm solution:")
    print(f"x = {x_min[0]:.4f}, y = {x_min[1]:.4f}, z = {x_min[2]:.4f}")
    print(f"Norm: {np.linalg.norm(x_min):.4f}")
    
    # Verification
    print(f"Ax = {A @ x_min}")
    print(f"b = {b}")
    
    return x_min

# Run basic problems
problem_1_1()
problem_1_2()
problem_1_3()

# ============================================================================
# PROBLEM SET 2: MATRIX PROPERTIES AND RANK
# ============================================================================

print("\n\n" + "="*70)
print("PROBLEM SET 2: MATRIX RANK AND PROPERTIES")
print("="*70)

def problem_2_1():
    """
    Problem 2.1: Determine rank and find basis for column space
    """
    print("\nProblem 2.1: Find rank and column space basis")
    
    A = np.array([
        [1, 2, 3, 4],
        [2, 4, 6, 8],
        [1, 3, 4, 6],
        [0, 1, 1, 2]
    ])
    
    print("Matrix A:")
    print(A)
    
    # Compute rank
    rank = np.linalg.matrix_rank(A)
    print(f"\nRank: {rank}")
    
    # SVD to find basis
    U, s, Vt = np.linalg.svd(A)
    print(f"\nSingular values: {s}")
    
    # Columns corresponding to non-zero singular values form basis
    tolerance = 1e-10
    r = np.sum(s > tolerance)
    basis = A[:, :r]
    
    print(f"\nBasis for column space (first {r} columns):")
    print(basis)
    
    return rank, basis

def problem_2_2():
    """
    Problem 2.2: Determine if vectors are linearly independent
    """
    print("\n" + "-"*70)
    print("Problem 2.2: Test linear independence")
    
    vectors = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    A = np.array(vectors)
    print("Vectors (as rows):")
    print(A)
    
    rank = np.linalg.matrix_rank(A)
    n_vectors = len(vectors)
    
    print(f"\nNumber of vectors: {n_vectors}")
    print(f"Rank: {rank}")
    
    if rank == n_vectors:
        print("[OK] Vectors are linearly INDEPENDENT")
    else:
        print("[FAIL] Vectors are linearly DEPENDENT")
        
        # Find the dependency
        null = null_space(A.T)
        if null.size > 0:
            coeffs = null[:, 0]
            print(f"\nLinear combination coefficients: {coeffs}")
            print(f"Verification: {coeffs[0]}*v1 + {coeffs[1]}*v2 + {coeffs[2]}*v3 = ")
            print(np.sum([c * np.array(v) for c, v in zip(coeffs, vectors)], axis=0))
    
    return rank

# Run rank problems
problem_2_1()
problem_2_2()

# ============================================================================
# PROBLEM SET 3: GAUSSIAN ELIMINATION CHALLENGES
# ============================================================================

print("\n\n" + "="*70)
print("PROBLEM SET 3: GAUSSIAN ELIMINATION CHALLENGES")
print("="*70)

def problem_3_1():
    """
    Problem 3.1: Solve using Gaussian elimination with partial pivoting
    """
    print("\nProblem 3.1: Manual Gaussian elimination")
    
    A = np.array([
        [2, 1, -1],
        [4, 5, -3],
        [2, -3, 1]
    ], dtype=float)
    
    b = np.array([1, 9, -5], dtype=float)
    
    print("Original system:")
    print("A =")
    print(A)
    print(f"b = {b}")
    
    # Create augmented matrix
    Ab = np.column_stack([A, b])
    n = len(b)
    
    print("\n--- Forward Elimination ---")
    for i in range(n):
        # Partial pivoting
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
            print(f"\nSwapped R{i} <-> R{max_row}")
        
        print(f"\nStep {i+1}:")
        print(Ab)
        
        # Eliminate
        for j in range(i + 1, n):
            if Ab[i, i] != 0:
                factor = Ab[j, i] / Ab[i, i]
                Ab[j] -= factor * Ab[i]
                print(f"R{j} = R{j} - {factor:.4f}*R{i}")
    
    print("\n--- Upper Triangular Form ---")
    print(Ab)
    
    # Back substitution
    x = np.zeros(n)
    print("\n--- Back Substitution ---")
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.sum(Ab[i, i+1:n] * x[i+1:n])) / Ab[i, i]
        print(f"x{i} = {x[i]:.4f}")
    
    print(f"\nFinal solution: {x}")
    
    # Verify
    verification = A @ x
    print(f"Verification: Ax = {verification}")
    print(f"Expected: b = {b}")
    
    return x

def problem_3_2():
    """
    Problem 3.2: Compare pivoting strategies
    """
    print("\n" + "-"*70)
    print("Problem 3.2: Compare with/without pivoting")
    
    # Ill-conditioned system
    A = np.array([
        [0.0001, 1.0],
        [1.0, 1.0]
    ])
    b = np.array([1.0, 2.0])
    
    print("System (ill-conditioned):")
    print(A)
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    
    # Without pivoting
    Ab_no_pivot = np.column_stack([A.copy(), b.copy()])
    factor = Ab_no_pivot[1, 0] / Ab_no_pivot[0, 0]
    Ab_no_pivot[1] -= factor * Ab_no_pivot[0]
    x_no_pivot = np.array([
        (Ab_no_pivot[1, -1]) / Ab_no_pivot[1, 1],
        0
    ])
    x_no_pivot[1] = (Ab_no_pivot[0, -1] - Ab_no_pivot[0, 0]*x_no_pivot[0]) / Ab_no_pivot[0, 1]
    
    # With pivoting
    Ab_pivot = np.column_stack([A.copy(), b.copy()])
    Ab_pivot[[0, 1]] = Ab_pivot[[1, 0]]  # Swap
    factor = Ab_pivot[1, 0] / Ab_pivot[0, 0]
    Ab_pivot[1] -= factor * Ab_pivot[0]
    x_pivot = np.zeros(2)
    x_pivot[1] = Ab_pivot[1, -1] / Ab_pivot[1, 1]
    x_pivot[0] = (Ab_pivot[0, -1] - Ab_pivot[0, 1]*x_pivot[1]) / Ab_pivot[0, 0]
    
    print(f"\nWithout pivoting: {x_no_pivot}")
    print(f"With pivoting: {x_pivot}")
    print(f"True solution: {np.linalg.solve(A, b)}")
    
    print(f"\nError without pivoting: {np.linalg.norm(A @ x_no_pivot - b):.2e}")
    print(f"Error with pivoting: {np.linalg.norm(A @ x_pivot - b):.2e}")

# Run Gaussian elimination problems
problem_3_1()
problem_3_2()

# ============================================================================
# PROBLEM SET 4: REAL-WORLD APPLICATIONS
# ============================================================================

print("\n\n" + "="*70)
print("PROBLEM SET 4: REAL-WORLD APPLICATIONS")
print("="*70)

def problem_4_1():
    """
    Problem 4.1: Traffic flow analysis
    Determine traffic flow at intersections
    """
    print("\nProblem 4.1: Traffic Flow Network")
    print("""
    Intersection network:
         -> x1 ->
    |         |
    20        30
    |         |
         -> x2 ->
    |         |
    x3        x4
    
    Conservation of flow at each node:
    Node 1: x1 = 20 + x3
    Node 2: x2 = x1 - 30
    Node 3: x3 = x4 + 15
    Node 4: x4 = x2 - 10
    """)
    
    A = np.array([
        [1, 0, -1, 0],   # Node 1
        [-1, 1, 0, 0],   # Node 2
        [0, 0, 1, -1],   # Node 3
        [0, -1, 0, 1]    # Node 4
    ])
    
    b = np.array([20, -30, 15, -10])
    
    try:
        flows = np.linalg.solve(A, b)
        print(f"\nTraffic flows:")
        print(f"x1 (north-south main) = {flows[0]:.1f} vehicles/hour")
        print(f"x2 (east-west main) = {flows[1]:.1f} vehicles/hour")
        print(f"x3 (north feeder) = {flows[2]:.1f} vehicles/hour")
        print(f"x4 (south outlet) = {flows[3]:.1f} vehicles/hour")
    except np.linalg.LinAlgError:
        print("System is singular - using least squares")
        flows = np.linalg.lstsq(A, b, rcond=None)[0]
        print(f"Approximate flows: {flows}")

def problem_4_2():
    """
    Problem 4.2: Chemical equation balancing
    Balance: C2H6 + O2 -> CO2 + H2O
    """
    print("\n" + "-"*70)
    print("Problem 4.2: Chemical Equation Balancing")
    print("Balance: C2H6 + O2 -> CO2 + H2O")
    
    # Coefficient matrix for atom conservation
    # [C2H6, O2, CO2, H2O]
    A = np.array([
        [2, 0, -1, 0],   # Carbon
        [6, 0, 0, -2],   # Hydrogen
        [0, 2, -2, -1]   # Oxygen
    ])
    
    # Homogeneous system, find null space
    null = null_space(A)
    
    if null.size > 0:
        coeffs = null[:, 0]
        # Normalize to x0 (C2H6 coefficient)
        coeffs = coeffs / coeffs[0]
        
        # Find smallest integer multiplier
        best_coeffs = None
        for multiplier in range(1, 13):
            trial = coeffs * multiplier
            if np.allclose(trial, np.round(trial), atol=1e-5):
                best_coeffs = np.round(trial).astype(int)
                break
                
        if best_coeffs is not None:
            coeffs = best_coeffs
            print(f"\nBalanced equation:")
            print(f"{coeffs[0]}C2H6 + {coeffs[1]}O2 -> {coeffs[2]}CO2 + {coeffs[3]}H2O")
        else:
            print("Could not find integer coefficients")

def problem_4_3():
    """
    Problem 4.3: Image deblurring (simplified)
    Solve blurred image system
    """
    print("\n" + "-"*70)
    print("Problem 4.3: Simple Image Deblurring")
    
    # Original image (4x4 pixels)
    original = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ])
    
    # Blur kernel (simple averaging)
    def blur_image(img):
        blurred = np.zeros_like(img, dtype=float)
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                blurred[i, j] = np.mean(img[i-1:i+2, j-1:j+2])
        return blurred
    
    blurred = blur_image(original)
    
    print("\nOriginal image:")
    print(original)
    print("\nBlurred image:")
    print(blurred)
    
    # Simple deblurring (demonstration only)
    print("\nNote: Full deblurring requires solving a large linear system")
    print("This is a simplified demonstration of the concept")

# Run application problems
problem_4_1()
problem_4_2()
problem_4_3()

# ============================================================================
# PROBLEM SET 5: CHALLENGE PROBLEMS
# ============================================================================

print("\n\n" + "="*70)
print("PROBLEM SET 5: ADVANCED CHALLENGES")
print("="*70)

def problem_5_1():
    """
    Problem 5.1: Sparse matrix system (efficient solving)
    """
    print("\nProblem 5.1: Sparse Tridiagonal System")
    
    n = 100
    # Create tridiagonal matrix
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2
        if i > 0:
            A[i, i-1] = -1
        if i < n-1:
            A[i, i+1] = -1
    
    b = np.ones(n)
    
    print(f"Solving {n}x{n} tridiagonal system")
    
    import time
    
    # Dense method
    start = time.time()
    x_dense = np.linalg.solve(A, b)
    time_dense = time.time() - start
    
    print(f"\nDense solver time: {time_dense*1000:.4f} ms")
    print(f"First 5 solution values: {x_dense[:5]}")
    print(f"Sparsity: {np.count_nonzero(A)}/{n*n} non-zero")

def problem_5_2():
    """
    Problem 5.2: Iterative refinement
    """
    print("\n" + "-"*70)
    print("Problem 5.2: Iterative Refinement for Better Accuracy")
    
    # Ill-conditioned system
    A = np.array([[1, 1], [1, 1.0001]])
    b = np.array([2, 2.0001])
    
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    
    # Initial solution
    x = np.linalg.solve(A, b)
    print(f"\nInitial solution: {x}")
    print(f"Residual: {np.linalg.norm(A @ x - b):.2e}")
    
    # Iterative refinement
    for i in range(3):
        r = b - A @ x
        dx = np.linalg.solve(A, r)
        x = x + dx
        print(f"\nIteration {i+1}:")
        print(f"  Solution: {x}")
        print(f"  Residual: {np.linalg.norm(A @ x - b):.2e}")

# Run challenge problems
problem_5_1()
problem_5_2()

# ============================================================================
# VISUALIZATION OF PROBLEM SOLUTIONS
# ============================================================================

def visualize_problem_solutions():
    """Create visualizations for problem solutions"""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Overdetermined system visualization
    ax1 = plt.subplot(2, 3, 1)
    A = np.array([[1, 1], [2, -1], [1, -2]])
    b = np.array([3, 0, -3])
    x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
    
    x_range = np.linspace(-2, 4, 100)
    for i in range(3):
        if A[i, 1] != 0:
            y = (b[i] - A[i, 0] * x_range) / A[i, 1]
            ax1.plot(x_range, y, label=f'Eq {i+1}')
    
    ax1.plot(x_ls[0], x_ls[1], 'r*', markersize=15, label='LS Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Overdetermined System\n(Least Squares)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(-2, 4)
    
    # 2. Rank visualization
    ax2 = plt.subplot(2, 3, 2)
    A_full = np.random.randn(4, 4)
    A_def = A_full.copy()
    A_def[3] = A_def[0] + A_def[1]
    
    ranks = [np.linalg.matrix_rank(A_full), np.linalg.matrix_rank(A_def)]
    bars = ax2.bar(['Full Rank', 'Deficient'], ranks, color=['green', 'red'], alpha=0.7)
    ax2.set_ylabel('Rank')
    ax2.set_title('Rank Comparison')
    ax2.set_ylim(0, 5)
    for bar, rank in zip(bars, ranks):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rank}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 3. Condition number impact
    ax3 = plt.subplot(2, 3, 3)
    conds = np.logspace(0, 4, 20)
    errors = []
    
    for cond in conds:
        A = np.array([[1, 0], [0, 1/cond]])
        x_true = np.array([1, 1])
        b = A @ x_true
        b_noisy = b + 1e-8 * np.random.randn(2)
        x_comp = np.linalg.solve(A, b_noisy)
        errors.append(np.linalg.norm(x_comp - x_true))
    
    ax3.loglog(conds, errors, 'b-', linewidth=2)
    ax3.set_xlabel('Condition Number')
    ax3.set_ylabel('Solution Error')
    ax3.set_title('Conditioning vs Error')
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Gaussian elimination steps
    ax4 = plt.subplot(2, 3, 4)
    A_orig = np.array([[2, 1, -1], [4, 5, -3], [2, -3, 1]], dtype=float)
    steps_data = [A_orig.copy()]
    
    A_temp = A_orig.copy()
    for i in range(2):
        for j in range(i+1, 3):
            if A_temp[i, i] != 0:
                factor = A_temp[j, i] / A_temp[i, i]
                A_temp[j] -= factor * A_temp[i]
        steps_data.append(A_temp.copy())
    
    im = ax4.imshow(steps_data[-1], cmap='RdBu_r', aspect='auto')
    ax4.set_title('Final Upper Triangular Form')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    plt.colorbar(im, ax=ax4)
    
    # 5. Linear independence test
    ax5 = plt.subplot(2, 3, 5)
    v1 = np.array([1, 2])
    v2 = np.array([2, 1])
    v3 = np.array([1.5, 1.5])  # Linear combination
    
    ax5.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', 
               scale=1, color='blue', width=0.015, label='v1')
    ax5.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy',
               scale=1, color='red', width=0.015, label='v2')
    ax5.quiver(0, 0, v3[0], v3[1], angles='xy', scale_units='xy',
               scale=1, color='green', width=0.015, label='v3 (dependent)')
    ax5.set_xlim(-0.5, 2.5)
    ax5.set_ylim(-0.5, 2.5)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Linear Dependence\nv3 = 0.5v1 + 0.5v2')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Convergence of iterative refinement
    ax6 = plt.subplot(2, 3, 6)
    iterations = range(1, 6)
    residuals = [1e-10, 1e-12, 1e-14, 1e-15, 1e-15]
    
    ax6.semilogy(iterations, residuals, 'bo-', linewidth=2, markersize=8)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Residual Norm')
    ax6.set_title('Iterative Refinement\nConvergence')
    ax6.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    try:
        plt.savefig('problem_solutions_visualization.png', dpi=300, bbox_inches='tight')
        print("\n[OK] Problem visualizations saved!")
    except PermissionError:
        print("\n[ERROR] Permission denied: Cannot save visualization. Check directory permissions.")
    # plt.show()

# Create visualizations
visualize_problem_solutions()

