
#Day 3: Eigenvalues & Eigenvectors
#Goal: Understand the special vectors that only get scaled by matrix transformations.


#Mathematics for Machine Learning, Chapter 4: Sections on eigenvalues and eigenvectors
#BITS Pilani Course: Lectures on eigendecomposition

#Eigenvector Visualizer: For a given 2×2 matrix, calculate and plot its eigenvectors. Show how they represent directions of pure stretch/compression. Animate the transformation.
#Change of Basis: Write a script to transform vector coordinates from one basis to another, using eigenvectors as the new basis.

#Key Concepts to Master
#Computing eigenvalues (characteristic polynomial)
#Finding eigenvectors
#Geometric interpretation
#Diagonalization


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# SECTION 1: THE FUNDAMENTAL CONCEPT

def section_1_concept():
    print("\n" + "="*80)
    print("SECTION 1: WHAT ARE EIGENVECTORS? THE BIG IDEA")
    print("="*80)
    print("""
    Definition: For matrix A, if Av = λv (where v ≠ 0):
    • v is an EIGENVECTOR (special direction)
    • λ is the EIGENVALUE (scaling factor)
    
    Key Insight: Eigenvectors are directions that DON'T ROTATE,
    they only get SCALED when transformed by matrix A!
    """)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    A = np.array([[2, 0], [0, 0.5]])
    
    vectors = [
        (np.array([1, 1]), 'Regular Vector', 'blue', False),
        (np.array([1, 0]), 'Eigenvector (λ=2)', 'green', True),
        (np.array([0, 1]), 'Eigenvector (λ=0.5)', 'purple', True)
    ]
    
    for idx, (v, title, color, is_eigen) in enumerate(vectors):
        ax = axes[idx]
        v_trans = A @ v
        
        # Original vector
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', 
                 scale=1, color=color, width=0.015, alpha=0.6, 
                 label='Original')
        
        # Transformed vector
        ax.quiver(0, 0, v_trans[0], v_trans[1], angles='xy', 
                 scale_units='xy', scale=1, color='red' if not is_eigen else 'darkgreen', 
                 width=0.015, label='Transformed', linewidth=2)
        
        if is_eigen:
            # Draw line showing direction preservation
            t = np.linspace(-1, 3, 100)
            direction = v / np.linalg.norm(v)
            ax.plot(t * direction[0], t * direction[1], 'k--', 
                   alpha=0.3, linewidth=1)
        
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        
        if is_eigen:
            ax.text(0.5, -0.3, '✓ Same direction!', fontsize=10, 
                   color='green', fontweight='bold')
        else:
            ax.text(0.5, -0.3, '✗ Direction changed!', fontsize=10, 
                   color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('01_concept.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Regular vectors change direction when transformed")
    print("✓ Eigenvectors stay on the same line (only scaled)")
    print("✓ This makes them SPECIAL and USEFUL!")


# SECTION 2: COMPUTING EIGENVALUES - CHARACTERISTIC POLYNOMIAL

def section_2_eigenvalues():
    print("\n" + "="*80)
    print("SECTION 2: COMPUTING EIGENVALUES - THE CHARACTERISTIC EQUATION")
    print("="*80)
    print("""
    Method: Solve det(A - λI) = 0
    
    Steps:
    1. Form matrix (A - λI) by subtracting λ from diagonal
    2. Calculate the determinant
    3. Solve the resulting polynomial equation
    4. Solutions are your eigenvalues!
    """)
    
    A = np.array([[4, 2], [1, 3]], dtype=float)
    
    print("\nExample Matrix A:")
    print(A)
    print("\nStep-by-step calculation:")
    print("-" * 60)
    
    print("\n1. Form (A - λI):")
    print("   [[4-λ,  2  ]")
    print("    [ 1,  3-λ ]]")
    
    print("\n2. Calculate det(A - λI):")
    print("   det = (4-λ)(3-λ) - (2)(1)")
    print("       = 12 - 4λ - 3λ + λ² - 2")
    print("       = λ² - 7λ + 10")
    print("   This is the CHARACTERISTIC POLYNOMIAL!")
    
    print("\n3. Solve λ² - 7λ + 10 = 0")
    a, b, c = 1, -7, 10
    disc = b**2 - 4*a*c
    l1 = (-b + np.sqrt(disc)) / (2*a)
    l2 = (-b - np.sqrt(disc)) / (2*a)
    
    print(f"   Using quadratic formula:")
    print(f"   λ₁ = {l1:.4f}")
    print(f"   λ₂ = {l2:.4f}")
    
    # Verify with NumPy
    eigenvalues = np.linalg.eigvals(A)
    print(f"\n4. NumPy verification: {eigenvalues}")
    
    # Visualize characteristic polynomial
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    lambda_vals = np.linspace(-1, 8, 1000)
    char_poly = lambda_vals**2 - 7*lambda_vals + 10
    
    ax1.plot(lambda_vals, char_poly, 'b-', linewidth=2.5, 
            label='p(λ) = λ² - 7λ + 10')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.plot(eigenvalues, [0, 0], 'ro', markersize=15, 
            label='Eigenvalues (roots)', zorder=5)
    
    for i, ev in enumerate(eigenvalues):
        ax1.annotate(f'λ{i+1} = {ev:.1f}', xy=(ev, 0), 
                    xytext=(ev, 8), fontsize=12, ha='center',
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('λ', fontsize=14, fontweight='bold')
    ax1.set_ylabel('det(A - λI)', fontsize=14, fontweight='bold')
    ax1.set_title('Characteristic Polynomial', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11)
    
    # Show matrix with eigenvalues
    ax2.text(0.5, 0.7, 'Matrix A', ha='center', fontsize=18, 
            fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.5, f'{A}', ha='center', fontsize=14, 
            family='monospace', transform=ax2.transAxes)
    ax2.text(0.5, 0.3, f'Eigenvalues:', ha='center', fontsize=14, 
            fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.2, f'λ₁ = {eigenvalues[0]:.2f}', ha='center', 
            fontsize=16, color='red', transform=ax2.transAxes)
    ax2.text(0.5, 0.1, f'λ₂ = {eigenvalues[1]:.2f}', ha='center', 
            fontsize=16, color='red', transform=ax2.transAxes)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('02_eigenvalues.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return A, eigenvalues


# SECTION 3: FINDING EIGENVECTORS

def section_3_eigenvectors(A, eigenvalues):
    print("\n" + "="*80)
    print("SECTION 3: FINDING EIGENVECTORS")
    print("="*80)
    print("""
    Once you have eigenvalue λ, find eigenvector v by solving:
    (A - λI)v = 0
    
    This gives a system of linear equations!
    """)
    
    print(f"\nFor λ₁ = {eigenvalues[0]}:")
    print("-" * 60)
    
    lambda1 = eigenvalues[0]
    A_lambda = A - lambda1 * np.eye(2)
    
    print(f"\n(A - {lambda1}I) =")
    print(A_lambda)
    
    print(f"\nSolving (A - {lambda1}I)v = 0:")
    print(f"  {A_lambda[0,0]:.1f}v₁ + {A_lambda[0,1]:.1f}v₂ = 0")
    print(f"  {A_lambda[1,0]:.1f}v₁ + {A_lambda[1,1]:.1f}v₂ = 0")
    
    print(f"\nFrom first equation: v₁ = {-A_lambda[0,1]/A_lambda[0,0]:.2f}v₂")
    print(f"Choosing v₂ = 1 → v₁ = {-A_lambda[0,1]/A_lambda[0,0]:.2f}")
    print(f"Eigenvector v₁ ≈ [{-A_lambda[0,1]/A_lambda[0,0]:.2f}, 1]ᵀ")
    
    # Complete solution with NumPy
    eigenvals_full, eigenvecs = np.linalg.eig(A)
    
    print("\n" + "="*80)
    print("COMPLETE EIGEN-DECOMPOSITION")
    print("="*80)
    
    for i, (val, vec) in enumerate(zip(eigenvals_full, eigenvecs.T)):
        print(f"\nEigenvalue λ_{i+1} = {val:.4f}")
        print(f"Eigenvector v_{i+1} = {vec}")
        
        # Verify Av = λv
        Av = A @ vec
        lambda_v = val * vec
        print(f"\nVerification:")
        print(f"  Av    = {Av}")
        print(f"  λv    = {lambda_v}")
        print(f"  Match? {np.allclose(Av, lambda_v)} ✓")
    
    # Visualize eigenvectors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show on unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    ax1.plot(circle[0], circle[1], 'b--', alpha=0.4, linewidth=2, 
            label='Unit circle')
    
    colors = ['red', 'green']
    for i, (val, vec) in enumerate(zip(eigenvals_full, eigenvecs.T)):
        vec_norm = vec / np.linalg.norm(vec)
        ax1.arrow(0, 0, vec_norm[0], vec_norm[1], head_width=0.1, 
                 head_length=0.08, fc=colors[i], ec=colors[i], 
                 linewidth=3, label=f'v_{i+1} (λ={val:.2f})')
        
        # Draw extended line
        t = np.linspace(-1.5, 1.5, 100)
        ax1.plot(t * vec_norm[0], t * vec_norm[1], 
                colors[i], alpha=0.3, linewidth=1)
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_title('Eigenvectors (Unit Length)', fontweight='bold', fontsize=14)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    
    # Show transformation effect
    ellipse = A @ circle
    ax2.plot(circle[0], circle[1], 'b--', alpha=0.4, linewidth=2)
    ax2.plot(ellipse[0], ellipse[1], 'b-', linewidth=2.5, 
            label='Transformed circle')
    
    for i, (val, vec) in enumerate(zip(eigenvals_full, eigenvecs.T)):
        vec_trans = A @ vec
        ax2.arrow(0, 0, vec_trans[0], vec_trans[1], head_width=0.15, 
                 head_length=0.12, fc=colors[i], ec=colors[i], 
                 linewidth=3, label=f'Av_{i+1} = {val:.2f}v_{i+1}')
    
    ax2.set_xlim(-3, 7)
    ax2.set_ylim(-3, 7)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_title('After Transformation', fontweight='bold', fontsize=14)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('03_eigenvectors.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return eigenvecs


# SECTION 4: GEOMETRIC INTERPRETATION

def section_4_geometric():
    print("\n" + "="*80)
    print("SECTION 4: GEOMETRIC INTERPRETATION")
    print("="*80)
    print("""
    What do eigenvalues mean geometrically?
    
    • Eigenvectors: Principal axes of transformation
    • Eigenvalues: Stretching/compression factors
    • |λ| > 1: Stretching along that direction
    • |λ| < 1: Compression along that direction
    • λ < 0: Reflection + scaling
    • λ = 0: Collapse to lower dimension
    """)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    matrices = [
        (np.array([[3, 0], [0, 1]]), "Pure Scaling\nλ₁=3, λ₂=1"),
        (np.array([[2, 1], [1, 2]]), "Symmetric\nλ₁=3, λ₂=1"),
        (np.array([[2, 0], [0, -1]]), "Scaling + Reflection\nλ₁=2, λ₂=-1"),
        (np.array([[1, 1], [0, 1]]), "Shear\nλ₁=λ₂=1"),
        (np.array([[0, -1], [1, 0]]), "Rotation 90°\nComplex λ"),
        (np.array([[2, 1], [0, 0.5]]), "Triangular\nλ₁=2, λ₂=0.5")
    ]
    
    for idx, (M, title) in enumerate(matrices):
        ax = axes[idx]
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle = np.array([np.cos(theta), np.sin(theta)])
        
        # Transform
        ellipse = M @ circle
        
        ax.plot(circle[0], circle[1], 'b--', alpha=0.5, linewidth=2, 
               label='Original')
        ax.plot(ellipse[0], ellipse[1], 'r-', linewidth=2.5, 
               label='Transformed')
        
        # Eigenvectors
        try:
            vals, vecs = np.linalg.eig(M)
            if np.all(np.isreal(vals)):
                colors = ['green', 'purple']
                for i, (val, vec) in enumerate(zip(vals, vecs.T)):
                    vec = vec.real
                    vec_norm = vec / np.linalg.norm(vec)
                    scale = abs(val.real) * 0.8
                    
                    ax.arrow(0, 0, vec_norm[0]*scale, vec_norm[1]*scale,
                            head_width=0.15, head_length=0.1, 
                            fc=colors[i], ec=colors[i], linewidth=2.5,
                            label=f'λ={val.real:.1f}', zorder=5)
            else:
                ax.text(0, 0, 'Complex\nEigenvalues', ha='center', 
                       fontsize=11, bbox=dict(boxstyle='round', 
                       facecolor='yellow', alpha=0.8))
        except:
            pass
        
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('04_geometric.png', dpi=150, bbox_inches='tight')
    plt.show()



def section_5_eigendecomposition():
    print("\n" + "="*80)
    print("SECTION 5: EIGENDECOMPOSITION - THE POWER FORMULA")
    print("="*80)
    print("""
    The BIG FORMULA: A = PDP⁻¹
    
    Where:
    • P: Matrix with eigenvectors as columns
    • D: Diagonal matrix with eigenvalues
    • P⁻¹: Inverse of P
    
    This factorization unlocks INCREDIBLE computational power!
    """)
    
    A = np.array([[4, 2], [1, 3]], dtype=float)
    
    print("Original Matrix A:")
    print(A)
    print()
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    P = eigenvectors
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)
    
    print("Eigenvectors matrix P (columns are eigenvectors):")
    print(P)
    print("\nDiagonal matrix D (eigenvalues on diagonal):")
    print(D)
    print("\nInverse P⁻¹:")
    print(P_inv)
    
    # Reconstruct
    A_reconstructed = P @ D @ P_inv
    
    print("\nReconstruction: A = PDP⁻¹")
    print(A_reconstructed)
    print(f"\nVerification: A ≈ PDP⁻¹? {np.allclose(A, A_reconstructed)} ✓")
    
    # Visualize decomposition
    fig = plt.figure(figsize=(16, 10))
    
    # Create custom layout
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
    
    # Top row: The decomposition
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    
    matrices = [(A, 'A', 'viridis'), (P, 'P', 'plasma'), 
                (D, 'D', 'coolwarm'), (P_inv, 'P⁻¹', 'cividis')]
    
    for ax, (mat, label, cmap) in zip([ax1, ax2, ax3, ax4], matrices):
        sns.heatmap(mat, annot=True, fmt='.2f', cmap=cmap, 
                   center=0, ax=ax, cbar=True, square=True, 
                   linewidths=1, cbar_kws={'shrink': 0.8})
        ax.set_title(label, fontweight='bold', fontsize=16)
    
    # Middle: Show the multiplication
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis('off')
    formula = "A = P × D × P⁻¹"
    ax_text.text(0.5, 0.6, formula, ha='center', fontsize=24, 
                fontweight='bold', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_text.text(0.5, 0.3, 'Eigenvectors × Eigenvalues × Eigenvectors⁻¹', 
                ha='center', fontsize=14, style='italic')
    
    # Bottom: Verification
    ax5 = fig.add_subplot(gs[2, 1:3])
    diff = np.abs(A - A_reconstructed)
    sns.heatmap(diff, annot=True, fmt='.2e', cmap='Reds', 
               ax=ax5, cbar=True, square=True, linewidths=1,
               cbar_kws={'label': 'Error', 'shrink': 0.8})
    ax5.set_title('Reconstruction Error: |A - PDP⁻¹|', 
                 fontweight='bold', fontsize=14)
    
    plt.savefig('05_eigendecomposition.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return A, P, D, P_inv

# SECTION 6: MATRIX POWERS - WHY DIAGONALIZATION MATTERS

def section_6_powers(A, P, D, P_inv):
    print("\n" + "="*80)
    print("SECTION 6: MATRIX POWERS - THE KILLER APPLICATION")
    print("="*80)
    print("""
    The MAGIC formula: A^n = PD^nP⁻¹
    
    Why this is AMAZING:
    • Computing A^100 directly: ~100 matrix multiplications (SLOW!)
    • Using diagonalization: Just raise diagonal elements to power (FAST!)
    
    D^n is trivial:
    [[λ₁, 0 ]]^n   [[λ₁ⁿ,  0  ]]
    [[ 0, λ₂]]   = [[ 0,  λ₂ⁿ]]
    """)
    
    import time
    
    n = 50
    
    # Method 1: Direct
    start = time.time()
    A_power_direct = np.linalg.matrix_power(A.astype(int), n)
    time_direct = time.time() - start
    
    # Method 2: Diagonalization
    start = time.time()
    eigenvals = np.linalg.eigvals(A)
    D_power = np.diag(eigenvals ** n)
    A_power_diag = P @ D_power @ P_inv
    time_diag = time.time() - start
    
    print(f"\nComputing A^{n}:")
    print(f"  Direct computation:  {time_direct*1000:.4f} ms")
    print(f"  Via diagonalization: {time_diag*1000:.4f} ms")
    print(f"  Speedup: {time_direct/time_diag:.1f}x faster! ⚡")
    
    print(f"\nResult A^{n} (first few elements):")
    print(f"  [[{A_power_direct[0,0]:.2e}, {A_power_direct[0,1]:.2e}]")
    print(f"   [{A_power_direct[1,0]:.2e}, {A_power_direct[1,1]:.2e}]]")
    
    # Visualize power growth
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    powers = np.arange(0, 21)
    eigenvals = np.linalg.eigvals(A)
    
    # Eigenvalue growth
    ax = axes[0, 0]
    for i, ev in enumerate(eigenvals):
        ev_powers = ev ** powers
        ax.plot(powers, ev_powers.real, 'o-', linewidth=2.5, 
               markersize=6, label=f'λ_{i+1} = {ev:.2f}')
    
    ax.set_xlabel('Power n', fontsize=13, fontweight='bold')
    ax.set_ylabel('λⁿ', fontsize=13, fontweight='bold')
    ax.set_title('Eigenvalue Growth', fontweight='bold', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Matrix norm growth
    ax = axes[0, 1]
    norms = []
    for p in powers:
        A_p = np.linalg.matrix_power(A.astype(int), p)
        norms.append(np.linalg.norm(A_p, 'fro'))
    
    ax.plot(powers, norms, 'ro-', linewidth=2.5, markersize=8)
    ax.set_xlabel('Power n', fontsize=13, fontweight='bold')
    ax.set_ylabel('||Aⁿ||', fontsize=13, fontweight='bold')
    ax.set_title('Matrix Norm Growth', fontweight='bold', fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Show pattern in powers
    ax = axes[1, 0]
    selected_powers = [0, 1, 2, 5, 10]
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_powers)))
    
    for i, p in enumerate(selected_powers):
        A_p = np.linalg.matrix_power(A.astype(int), p)
        theta = np.linspace(0, 2*np.pi, 100)
        circle = np.array([np.cos(theta), np.sin(theta)])
        transformed = A_p @ circle
        
        # Normalize for visualization
        scale = 1.0 / (i + 1) if i > 0 else 1.0
        ax.plot(transformed[0]*scale, transformed[1]*scale, 
               color=colors[i], linewidth=2, label=f'A^{p}', alpha=0.7)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title('Unit Circle Under A^n (scaled)', 
                fontweight='bold', fontsize=15)
    
    # Computation comparison
    ax = axes[1, 1]
    ax.axis('off')
    
    comparison_text = f"""
    COMPUTATION COMPARISON
    {'='*40}
    
    Task: Compute A^{n}
    
    Direct Method:
      • {n} matrix multiplications
      • Time: {time_direct*1000:.4f} ms
    
    Diagonalization Method:
      • Compute P, D, P⁻¹ once
      • Raise diagonal to power
      • Time: {time_diag*1000:.4f} ms
    
    Speedup: {time_direct/time_diag:.1f}x faster! ⚡
    
    For A^1000: Even more dramatic!
    """
    
    ax.text(0.1, 0.9, comparison_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('06_matrix_powers.png', dpi=150, bbox_inches='tight')
    plt.show()


# SECTION 7: SPECIAL PROPERTIES

def section_7_properties():
    print("\n" + "="*80)
    print("SECTION 7: IMPORTANT PROPERTIES & THEOREMS")
    print("="*80)
    
    cases = {
        'Diagonal': np.array([[3, 0], [0, 2]]),
        'Upper Triangular': np.array([[3, 1], [0, 2]]),
        'Lower Triangular': np.array([[3, 0], [1, 2]]),
        'Symmetric': np.array([[3, 1], [1, 2]]),
        'Skew-Symmetric': np.array([[0, -1], [1, 0]]),
        'Orthogonal': np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2)]])
    }