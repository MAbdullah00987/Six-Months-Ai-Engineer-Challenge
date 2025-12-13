
#Interactive Matrix Properties Explorer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

# Set style
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)

print("="*70)
print("INTERACTIVE MATRIX PROPERTIES EXPLORER")
print("="*70)

# ============================================================================
# FUNCTION LIBRARY
# ============================================================================

def analyze_matrix_complete(matrix, name="Matrix"):
    """
    Complete analysis of a matrix including all properties
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {name}")
    print(f"{'='*70}")
    
    print(f"\nMatrix ({matrix.shape[0]}×{matrix.shape[1]}):")
    print(matrix)
    
    # Check if square
    if matrix.shape[0] != matrix.shape[1]:
        print("\n⚠ This is NOT a square matrix. No determinant or inverse exists.")
        return
    
    # Determinant
    det = np.linalg.det(matrix)
    print(f"\n📊 Determinant: {det:.6f}")
    
    # Rank
    rank = np.linalg.matrix_rank(matrix)
    print(f"📊 Rank: {rank}")
    print(f"📊 Full Rank: {'Yes' if rank == matrix.shape[0] else 'No'}")
    
    # Invertibility
    is_invertible = abs(det) > 1e-10
    print(f"\n{'✓' if is_invertible else '✗'} Invertible: {'YES' if is_invertible else 'NO'}")
    
    if is_invertible:
        # Compute inverse
        inv_matrix = np.linalg.inv(matrix)
        print(f"\nInverse Matrix:")
        print(inv_matrix)
        
        # Verify A × A^(-1) = I
        identity_check = matrix @ inv_matrix
        print(f"\nVerification (A × A⁻¹):")
        print(identity_check)
        
        # Condition number
        cond = np.linalg.cond(matrix)
        print(f"\n📊 Condition Number: {cond:.4f}")
        if cond < 10:
            print("   → Well-conditioned (stable for computations)")
        elif cond < 1000:
            print("   → Moderately conditioned")
        else:
            print("   → Ill-conditioned (numerical instability risk)")
    
    # Eigenvalues
    try:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        print(f"\n📊 Eigenvalues:")
        for i, ev in enumerate(eigenvalues):
            if np.isreal(ev):
                print(f"   λ{i+1} = {ev.real:.4f}")
            else:
                print(f"   λ{i+1} = {ev.real:.4f} + {ev.imag:.4f}i")
    except:
        print("\n⚠ Could not compute eigenvalues")
    
    # Trace
    trace = np.trace(matrix)
    print(f"\n📊 Trace: {trace:.4f}")
    
    # Norm
    fro_norm = np.linalg.norm(matrix, 'fro')
    print(f"📊 Frobenius Norm: {fro_norm:.4f}")
    
    # Special properties
    print(f"\n🔍 Special Properties:")
    print(f"   - Symmetric: {np.allclose(matrix, matrix.T)}")
    print(f"   - Diagonal: {np.allclose(matrix, np.diag(np.diag(matrix)))}")
    print(f"   - Identity: {np.allclose(matrix, np.eye(matrix.shape[0]))}")
    
    # Orthogonal check (for square matrices)
    if is_invertible:
        is_orthogonal = np.allclose(matrix @ matrix.T, np.eye(matrix.shape[0]))
        print(f"   - Orthogonal: {is_orthogonal}")
    
    return {
        'determinant': det,
        'invertible': is_invertible,
        'rank': rank,
        'condition_number': cond if is_invertible else None
    }

def create_comparison_visualization(matrices_dict):
    """
    Create a visual comparison of multiple matrices
    """
    n_matrices = len(matrices_dict)
    fig, axes = plt.subplots(2, n_matrices, figsize=(5*n_matrices, 10))
    
    if n_matrices == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (name, matrix) in enumerate(matrices_dict.items()):
        # Top row: Matrix heatmap
        ax1 = axes[0, idx]
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, cbar=True, ax=ax1, square=True)
        ax1.set_title(f'{name}\n{matrix.shape[0]}×{matrix.shape[1]}', 
                      fontweight='bold', fontsize=12)
        
        # Bottom row: Properties
        ax2 = axes[1, idx]
        ax2.axis('off')
        
        if matrix.shape[0] == matrix.shape[1]:
            det = np.linalg.det(matrix)
            rank = np.linalg.matrix_rank(matrix)
            is_inv = abs(det) > 1e-10
            
            props_text = f"Determinant: {det:.4f}\n"
            props_text += f"Rank: {rank}\n"
            props_text += f"Invertible: {'YES' if is_inv else 'NO'}\n"
            
            if is_inv:
                cond = np.linalg.cond(matrix)
                props_text += f"Condition #: {cond:.2f}\n"
            
            props_text += f"Trace: {np.trace(matrix):.4f}\n"
            props_text += f"Symmetric: {np.allclose(matrix, matrix.T)}"
            
            color = 'lightgreen' if is_inv else 'lightcoral'
        else:
            props_text = "Non-square matrix\nNo determinant"
            color = 'lightgray'
        
        bbox = FancyBboxPatch((0.1, 0.3), 0.8, 0.6, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='black', facecolor=color, 
                              transform=ax2.transAxes)
        ax2.add_patch(bbox)
        ax2.text(0.5, 0.6, props_text, transform=ax2.transAxes,
                ha='center', va='center', fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig('matrix_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# EXPLORE DIFFERENT MATRIX TYPES
# ============================================================================

print("\n" + "="*70)
print("PART 1: EXPLORING DIFFERENT MATRIX TYPES")
print("="*70)

# Collection of interesting matrices
matrices_to_explore = {
    "Identity 3×3": np.eye(3),
    
    "Diagonal Matrix": np.diag([2, 3, 4]),
    
    "Symmetric Matrix": np.array([[4, 1, 2],
                                   [1, 5, 3],
                                   [2, 3, 6]]),
    
    "Singular Matrix": np.array([[1, 2, 3],
                                  [2, 4, 6],
                                  [1, 2, 3]]),
    
    "Rotation Matrix": np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                  [np.sin(np.pi/4), np.cos(np.pi/4)]]),
    
    "Upper Triangular": np.array([[2, 3, 1],
                                   [0, 4, 2],
                                   [0, 0, 5]]),
    
    "Ill-Conditioned": np.array([[1, 1],
                                  [1, 1.0001]]),
}

# Analyze each matrix
results = {}
for name, matrix in matrices_to_explore.items():
    results[name] = analyze_matrix_complete(matrix, name)

# ============================================================================
# PART 2: RELATIONSHIP BETWEEN DETERMINANT AND INVERTIBILITY
# ============================================================================

print("\n" + "="*70)
print("PART 2: DETERMINANT vs INVERTIBILITY RELATIONSHIP")
print("="*70)

# Generate random matrices and check relationship
n_samples = 20
det_values = []
invertible_status = []
condition_numbers = []

print("\nGenerating 20 random 3×3 matrices...\n")

for i in range(n_samples):
    # Random matrix
    if i < 15:
        # Most should be invertible
        A = np.random.randn(3, 3)
    else:
        # Make some singular
        v = np.random.randn(3, 1)
        A = v @ v.T  # Rank-1 matrix
    
    det_val = np.linalg.det(A)
    is_inv = abs(det_val) > 1e-10
    
    det_values.append(det_val)
    invertible_status.append(is_inv)
    
    if is_inv:
        condition_numbers.append(np.linalg.cond(A))
    else:
        condition_numbers.append(np.inf)
    
    print(f"Matrix {i+1}: det = {det_val:10.6f} → {'Invertible' if is_inv else 'Singular'}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Determinant values
colors = ['green' if inv else 'red' for inv in invertible_status]
axes[0].bar(range(n_samples), det_values, color=colors, alpha=0.7, edgecolor='black')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[0].set_xlabel('Matrix Index', fontsize=12)
axes[0].set_ylabel('Determinant Value', fontsize=12)
axes[0].set_title('Determinant Values\n(Green = Invertible, Red = Singular)', 
                  fontweight='bold', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Plot 2: Condition numbers (log scale)
valid_cond = [c for c in condition_numbers if c != np.inf]
axes[1].hist(valid_cond, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Condition Number', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of Condition Numbers\n(Lower = Better)', 
                  fontweight='bold', fontsize=12)
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('determinant_invertibility.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# PART 3: VISUAL COMPARISON
# ============================================================================

print("\n" + "="*70)
print("PART 3: VISUAL COMPARISON OF MATRIX TYPES")
print("="*70)

comparison_matrices = {
    "Invertible\n(det≠0)": np.array([[2, 1], [1, 3]]),
    "Singular\n(det=0)": np.array([[2, 4], [1, 2]]),
    "Identity\n(det=1)": np.eye(2),
}

create_comparison_visualization(comparison_matrices)

# ============================================================================
# PART 4: SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("PART 4: SUMMARY STATISTICS")
print("="*70)

# Create summary DataFrame
summary_df = pd.DataFrame({
    'Matrix Type': list(matrices_to_explore.keys()),
    'Dimensions': [f"{m.shape[0]}×{m.shape[1]}" for m in matrices_to_explore.values()],
    'Determinant': [np.linalg.det(m) if m.shape[0]==m.shape[1] else np.nan 
                    for m in matrices_to_explore.values()],
    'Invertible': ['Yes' if (m.shape[0]==m.shape[1] and abs(np.linalg.det(m))>1e-10) else 'No' 
                   for m in matrices_to_explore.values()],
    'Rank': [np.linalg.matrix_rank(m) for m in matrices_to_explore.values()],
})

summary_df['Determinant'] = summary_df['Determinant'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")

print("\n" + summary_df.to_string(index=False))

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("KEY CONCEPTS SUMMARY")
print("="*70)

summary_points = """
1. DETERMINANT TELLS US:
   • det = 0 → Matrix is SINGULAR (rows/columns linearly dependent)
   • det ≠ 0 → Matrix is INVERTIBLE (full rank)
   • |det| > 1 → Transformation expands volume
   • |det| < 1 → Transformation contracts volume
   • det < 0 → Transformation includes reflection

2. MATRIX INVERSE:
   • Exists ONLY when det(A) ≠ 0
   • A × A⁻¹ = A⁻¹ × A = I (Identity matrix)
   • (AB)⁻¹ = B⁻¹A⁻¹ (reverse order)
   • (A⁻¹)⁻¹ = A

3. GEOMETRIC INTERPRETATION:
   • Determinant = signed area/volume scaling factor
   • Unit square → transformed parallelogram
   • |det| = area of parallelogram

4. CONDITION NUMBER:
   • Measures numerical stability
   • Low condition number (< 10) = well-conditioned
   • High condition number = ill-conditioned (avoid if possible)

5. PRACTICAL TIPS:
   • Always check det before computing inverse
   • Use np.linalg.det() for determinant
   • Use np.linalg.inv() for inverse
   • Check condition number for numerical stability
"""

print(summary_points)

print("\n" + "="*70)
print("✓ EXPLORATION COMPLETE!")
print("="*70)
print("\nYou now understand:")
print("  ✓ How to compute determinants")
print("  ✓ When matrices are invertible")
print("  ✓ How to find matrix inverses")
print("  ✓ Geometric meaning of determinants")
print("  ✓ Different types of matrices and their properties")
print("="*70)