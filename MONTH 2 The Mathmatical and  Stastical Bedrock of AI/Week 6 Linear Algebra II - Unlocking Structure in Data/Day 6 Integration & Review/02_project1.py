"""
Linear Algebra Toolkit - Comprehensive Implementation
A complete toolkit covering 10 fundamental linear algebra concepts with
interactive visualizations and educational content.

Required libraries:
pip install numpy pandas matplotlib seaborn ipywidgets scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, VBox, HBox, Output
from IPython.display import display, Math, Latex
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class LinearAlgebraToolkit:
    """Main class for Linear Algebra Toolkit"""
    
    def __init__(self):
        self.output = Output()
        
    # ==================== PROJECT 1: MATRIX OPERATIONS ====================
    def matrix_operations_demo(self):
        """Interactive demo for basic matrix operations"""
        print("=" * 60)
        print("PROJECT 1: MATRIX OPERATIONS")
        print("=" * 60)
        
        def visualize_operations(rows=3, cols=3, operation='Addition'):
            with self.output:
                self.output.clear_output(wait=True)
                
                # Generate random matrices
                A = np.random.randint(-5, 6, size=(rows, cols))
                B = np.random.randint(-5, 6, size=(rows, cols))
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Display Matrix A
                axes[0].imshow(A, cmap='coolwarm', aspect='auto')
                axes[0].set_title('Matrix A', fontsize=14, fontweight='bold')
                for i in range(rows):
                    for j in range(cols):
                        axes[0].text(j, i, str(A[i, j]), ha='center', va='center')
                axes[0].set_xticks([])
                axes[0].set_yticks([])
                
                # Display Matrix B
                axes[1].imshow(B, cmap='coolwarm', aspect='auto')
                axes[1].set_title('Matrix B', fontsize=14, fontweight='bold')
                for i in range(rows):
                    for j in range(cols):
                        axes[1].text(j, i, str(B[i, j]), ha='center', va='center')
                axes[1].set_xticks([])
                axes[1].set_yticks([])
                
                # Perform operation
                if operation == 'Addition':
                    C = A + B
                    op_symbol = '+'
                elif operation == 'Subtraction':
                    C = A - B
                    op_symbol = '-'
                elif operation == 'Element-wise Multiplication':
                    C = A * B
                    op_symbol = '×'
                else:
                    C = A @ B.T  # Matrix multiplication with transpose
                    op_symbol = '@'
                
                # Display Result
                axes[2].imshow(C, cmap='viridis', aspect='auto')
                axes[2].set_title(f'Result (A {op_symbol} B)', fontsize=14, fontweight='bold')
                for i in range(C.shape[0]):
                    for j in range(C.shape[1]):
                        axes[2].text(j, i, str(C[i, j]), ha='center', va='center')
                axes[2].set_xticks([])
                axes[2].set_yticks([])
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n📊 Operation: {operation}")
                print(f"Matrix A shape: {A.shape}")
                print(f"Matrix B shape: {B.shape}")
                print(f"Result shape: {C.shape}")
        
        interact(visualize_operations,
                rows=IntSlider(min=2, max=5, step=1, value=3),
                cols=IntSlider(min=2, max=5, step=1, value=3),
                operation=Dropdown(options=['Addition', 'Subtraction', 
                                           'Element-wise Multiplication', 'Matrix Multiplication']))
        display(self.output)
    
    # ==================== PROJECT 2: VECTOR SPACES ====================
    def vector_spaces_demo(self):
        """Interactive demo for vector spaces and linear combinations"""
        print("=" * 60)
        print("PROJECT 2: VECTOR SPACES & LINEAR COMBINATIONS")
        print("=" * 60)
        
        def visualize_vectors(v1_x=1.0, v1_y=2.0, v2_x=2.0, v2_y=1.0, 
                            scalar1=1.0, scalar2=1.0):
            with self.output:
                self.output.clear_output(wait=True)
                
                v1 = np.array([v1_x, v1_y])
                v2 = np.array([v2_x, v2_y])
                result = scalar1 * v1 + scalar2 * v2
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Left plot: Original vectors
                ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', 
                          scale=1, color='blue', width=0.008, label=f'v1 = [{v1_x}, {v1_y}]')
                ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', 
                          scale=1, color='red', width=0.008, label=f'v2 = [{v2_x}, {v2_y}]')
                
                ax1.set_xlim(-10, 10)
                ax1.set_ylim(-10, 10)
                ax1.axhline(y=0, color='k', linewidth=0.5)
                ax1.axvline(x=0, color='k', linewidth=0.5)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlabel('x', fontsize=12)
                ax1.set_ylabel('y', fontsize=12)
                ax1.set_title('Base Vectors', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.set_aspect('equal')
                
                # Right plot: Linear combination
                ax2.quiver(0, 0, scalar1*v1[0], scalar1*v1[1], angles='xy', 
                          scale_units='xy', scale=1, color='blue', width=0.008, 
                          alpha=0.6, label=f'{scalar1}×v1')
                ax2.quiver(scalar1*v1[0], scalar1*v1[1], scalar2*v2[0], scalar2*v2[1], 
                          angles='xy', scale_units='xy', scale=1, color='red', 
                          width=0.008, alpha=0.6, label=f'{scalar2}×v2')
                ax2.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', 
                          scale=1, color='green', width=0.01, 
                          label=f'Result = [{result[0]:.2f}, {result[1]:.2f}]')
                
                ax2.set_xlim(-10, 10)
                ax2.set_ylim(-10, 10)
                ax2.axhline(y=0, color='k', linewidth=0.5)
                ax2.axvline(x=0, color='k', linewidth=0.5)
                ax2.grid(True, alpha=0.3)
                ax2.set_xlabel('x', fontsize=12)
                ax2.set_ylabel('y', fontsize=12)
                ax2.set_title(f'Linear Combination: {scalar1}×v1 + {scalar2}×v2', 
                            fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.set_aspect('equal')
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n📐 Linear Combination:")
                print(f"   {scalar1} × [{v1_x}, {v1_y}] + {scalar2} × [{v2_x}, {v2_y}]")
                print(f"   = [{result[0]:.2f}, {result[1]:.2f}]")
        
        interact(visualize_vectors,
                v1_x=FloatSlider(min=-5, max=5, step=0.5, value=1),
                v1_y=FloatSlider(min=-5, max=5, step=0.5, value=2),
                v2_x=FloatSlider(min=-5, max=5, step=0.5, value=2),
                v2_y=FloatSlider(min=-5, max=5, step=0.5, value=1),
                scalar1=FloatSlider(min=-3, max=3, step=0.5, value=1),
                scalar2=FloatSlider(min=-3, max=3, step=0.5, value=1))
        display(self.output)
    
    # ==================== PROJECT 3: EIGENVALUES & EIGENVECTORS ====================
    def eigenvalues_demo(self):
        """Interactive demo for eigenvalues and eigenvectors"""
        print("=" * 60)
        print("PROJECT 3: EIGENVALUES & EIGENVECTORS")
        print("=" * 60)
        
        def visualize_eigen(a=2.0, b=0.5, c=0.5, d=2.0):
            with self.output:
                self.output.clear_output(wait=True)
                
                A = np.array([[a, b], [c, d]])
                
                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(A)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Left plot: Eigenvectors
                ax1.quiver(0, 0, eigenvectors[0, 0], eigenvectors[1, 0], 
                          angles='xy', scale_units='xy', scale=1, color='blue', 
                          width=0.01, label=f'v1 (λ={eigenvalues[0]:.2f})')
                ax1.quiver(0, 0, eigenvectors[0, 1], eigenvectors[1, 1], 
                          angles='xy', scale_units='xy', scale=1, color='red', 
                          width=0.01, label=f'v2 (λ={eigenvalues[1]:.2f})')
                
                ax1.set_xlim(-2, 2)
                ax1.set_ylim(-2, 2)
                ax1.axhline(y=0, color='k', linewidth=0.5)
                ax1.axvline(x=0, color='k', linewidth=0.5)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlabel('x', fontsize=12)
                ax1.set_ylabel('y', fontsize=12)
                ax1.set_title('Eigenvectors', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.set_aspect('equal')
                
                # Right plot: Transformation effect
                # Create a circle of points
                theta = np.linspace(0, 2*np.pi, 50)
                circle = np.array([np.cos(theta), np.sin(theta)])
                transformed = A @ circle
                
                ax2.plot(circle[0], circle[1], 'b-', alpha=0.5, label='Original', linewidth=2)
                ax2.plot(transformed[0], transformed[1], 'r-', alpha=0.5, 
                        label='Transformed', linewidth=2)
                
                # Show eigenvector directions on transformed space
                ax2.quiver(0, 0, eigenvalues[0]*eigenvectors[0, 0], 
                          eigenvalues[0]*eigenvectors[1, 0], 
                          angles='xy', scale_units='xy', scale=1, color='blue', 
                          width=0.01, alpha=0.7)
                ax2.quiver(0, 0, eigenvalues[1]*eigenvectors[0, 1], 
                          eigenvalues[1]*eigenvectors[1, 1], 
                          angles='xy', scale_units='xy', scale=1, color='red', 
                          width=0.01, alpha=0.7)
                
                ax2.set_xlim(-5, 5)
                ax2.set_ylim(-5, 5)
                ax2.axhline(y=0, color='k', linewidth=0.5)
                ax2.axvline(x=0, color='k', linewidth=0.5)
                ax2.grid(True, alpha=0.3)
                ax2.set_xlabel('x', fontsize=12)
                ax2.set_ylabel('y', fontsize=12)
                ax2.set_title('Matrix Transformation Effect', fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.set_aspect('equal')
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n🔢 Matrix A:")
                print(f"   [[{a:.2f}, {b:.2f}]")
                print(f"    [{c:.2f}, {d:.2f}]]")
                print(f"\n📊 Eigenvalue 1: {eigenvalues[0]:.4f}")
                print(f"   Eigenvector 1: [{eigenvectors[0, 0]:.4f}, {eigenvectors[1, 0]:.4f}]")
                print(f"\n📊 Eigenvalue 2: {eigenvalues[1]:.4f}")
                print(f"   Eigenvector 2: [{eigenvectors[0, 1]:.4f}, {eigenvectors[1, 1]:.4f}]")
        
        interact(visualize_eigen,
                a=FloatSlider(min=-3, max=3, step=0.1, value=2),
                b=FloatSlider(min=-3, max=3, step=0.1, value=0.5),
                c=FloatSlider(min=-3, max=3, step=0.1, value=0.5),
                d=FloatSlider(min=-3, max=3, step=0.1, value=2))
        display(self.output)
    
    # ==================== PROJECT 4: LINEAR TRANSFORMATIONS ====================
    def transformations_demo(self):
        """Interactive demo for linear transformations"""
        print("=" * 60)
        print("PROJECT 4: LINEAR TRANSFORMATIONS")
        print("=" * 60)
        
        def visualize_transform(transform_type='Rotation', angle=45.0, 
                               scale_x=1.5, scale_y=1.5, shear=0.5):
            with self.output:
                self.output.clear_output(wait=True)
                
                # Create transformation matrix based on type
                if transform_type == 'Rotation':
                    theta = np.radians(angle)
                    T = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
                elif transform_type == 'Scaling':
                    T = np.array([[scale_x, 0],
                                 [0, scale_y]])
                elif transform_type == 'Shear':
                    T = np.array([[1, shear],
                                 [0, 1]])
                else:  # Reflection
                    T = np.array([[1, 0],
                                 [0, -1]])
                
                # Create a simple shape (house)
                house = np.array([[0, 0, 1, 1, 0.5, 0],
                                 [0, 1, 1, 0, 1.5, 0]])
                transformed_house = T @ house
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Original shape
                ax1.plot(house[0], house[1], 'b-', linewidth=2, marker='o')
                ax1.fill(house[0], house[1], alpha=0.3, color='blue')
                ax1.set_xlim(-3, 3)
                ax1.set_ylim(-3, 3)
                ax1.axhline(y=0, color='k', linewidth=0.5)
                ax1.axvline(x=0, color='k', linewidth=0.5)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlabel('x', fontsize=12)
                ax1.set_ylabel('y', fontsize=12)
                ax1.set_title('Original Shape', fontsize=14, fontweight='bold')
                ax1.set_aspect('equal')
                
                # Transformed shape
                ax2.plot(house[0], house[1], 'b--', alpha=0.3, linewidth=1, 
                        label='Original')
                ax2.plot(transformed_house[0], transformed_house[1], 'r-', 
                        linewidth=2, marker='o', label='Transformed')
                ax2.fill(transformed_house[0], transformed_house[1], alpha=0.3, color='red')
                ax2.set_xlim(-3, 3)
                ax2.set_ylim(-3, 3)
                ax2.axhline(y=0, color='k', linewidth=0.5)
                ax2.axvline(x=0, color='k', linewidth=0.5)
                ax2.grid(True, alpha=0.3)
                ax2.set_xlabel('x', fontsize=12)
                ax2.set_ylabel('y', fontsize=12)
                ax2.set_title(f'{transform_type} Transformation', 
                            fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.set_aspect('equal')
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n🔄 Transformation Matrix:")
                print(f"   [[{T[0,0]:.4f}, {T[0,1]:.4f}]")
                print(f"    [{T[1,0]:.4f}, {T[1,1]:.4f}]]")
        
        interact(visualize_transform,
                transform_type=Dropdown(options=['Rotation', 'Scaling', 'Shear', 'Reflection']),
                angle=FloatSlider(min=0, max=360, step=15, value=45),
                scale_x=FloatSlider(min=0.5, max=3, step=0.25, value=1.5),
                scale_y=FloatSlider(min=0.5, max=3, step=0.25, value=1.5),
                shear=FloatSlider(min=-2, max=2, step=0.25, value=0.5))
        display(self.output)
    
    # ==================== PROJECT 5: MATRIX DECOMPOSITIONS ====================
    def decomposition_demo(self):
        """Interactive demo for matrix decompositions (LU, QR, SVD)"""
        print("=" * 60)
        print("PROJECT 5: MATRIX DECOMPOSITIONS")
        print("=" * 60)
        
        def visualize_decomposition(decomp_type='SVD', matrix_size=3):
            with self.output:
                self.output.clear_output(wait=True)
                
                # Generate random matrix
                A = np.random.randn(matrix_size, matrix_size)
                
                fig = plt.figure(figsize=(16, 5))
                
                if decomp_type == 'SVD':
                    U, S, Vt = np.linalg.svd(A)
                    
                    # Visualize matrices
                    ax1 = plt.subplot(1, 5, 1)
                    im1 = ax1.imshow(A, cmap='coolwarm', aspect='auto')
                    ax1.set_title('A', fontsize=12, fontweight='bold')
                    plt.colorbar(im1, ax=ax1)
                    
                    ax2 = plt.subplot(1, 5, 2)
                    ax2.text(0.5, 0.5, '=', fontsize=30, ha='center', va='center')
                    ax2.axis('off')
                    
                    ax3 = plt.subplot(1, 5, 3)
                    im3 = ax3.imshow(U, cmap='coolwarm', aspect='auto')
                    ax3.set_title('U', fontsize=12, fontweight='bold')
                    plt.colorbar(im3, ax=ax3)
                    
                    ax4 = plt.subplot(1, 5, 4)
                    im4 = ax4.imshow(np.diag(S), cmap='coolwarm', aspect='auto')
                    ax4.set_title('Σ', fontsize=12, fontweight='bold')
                    plt.colorbar(im4, ax=ax4)
                    
                    ax5 = plt.subplot(1, 5, 5)
                    im5 = ax5.imshow(Vt, cmap='coolwarm', aspect='auto')
                    ax5.set_title('Vᵀ', fontsize=12, fontweight='bold')
                    plt.colorbar(im5, ax=ax5)
                    
                    print(f"\n📊 Singular Value Decomposition (SVD)")
                    print(f"   A = U × Σ × Vᵀ")
                    print(f"\n   Singular values: {S}")
                    
                elif decomp_type == 'QR':
                    Q, R = np.linalg.qr(A)
                    
                    ax1 = plt.subplot(1, 4, 1)
                    im1 = ax1.imshow(A, cmap='coolwarm', aspect='auto')
                    ax1.set_title('A', fontsize=12, fontweight='bold')
                    plt.colorbar(im1, ax=ax1)
                    
                    ax2 = plt.subplot(1, 4, 2)
                    ax2.text(0.5, 0.5, '=', fontsize=30, ha='center', va='center')
                    ax2.axis('off')
                    
                    ax3 = plt.subplot(1, 4, 3)
                    im3 = ax3.imshow(Q, cmap='coolwarm', aspect='auto')
                    ax3.set_title('Q (Orthogonal)', fontsize=12, fontweight='bold')
                    plt.colorbar(im3, ax=ax3)
                    
                    ax4 = plt.subplot(1, 4, 4)
                    im4 = ax4.imshow(R, cmap='coolwarm', aspect='auto')
                    ax4.set_title('R (Upper Triangular)', fontsize=12, fontweight='bold')
                    plt.colorbar(im4, ax=ax4)
                    
                    print(f"\n📊 QR Decomposition")
                    print(f"   A = Q × R")
                    print(f"   Q is orthogonal: QᵀQ = I")
                    
                else:  # LU
                    from scipy.linalg import lu
                    P, L, U = lu(A)
                    
                    ax1 = plt.subplot(1, 5, 1)
                    im1 = ax1.imshow(A, cmap='coolwarm', aspect='auto')
                    ax1.set_title('A', fontsize=12, fontweight='bold')
                    plt.colorbar(im1, ax=ax1)
                    
                    ax2 = plt.subplot(1, 5, 2)
                    ax2.text(0.5, 0.5, '=', fontsize=30, ha='center', va='center')
                    ax2.axis('off')
                    
                    ax3 = plt.subplot(1, 5, 3)
                    im3 = ax3.imshow(P, cmap='coolwarm', aspect='auto')
                    ax3.set_title('P (Permutation)', fontsize=12, fontweight='bold')
                    plt.colorbar(im3, ax=ax3)
                    
                    ax4 = plt.subplot(1, 5, 4)
                    im4 = ax4.imshow(L, cmap='coolwarm', aspect='auto')
                    ax4.set_title('L (Lower)', fontsize=12, fontweight='bold')
                    plt.colorbar(im4, ax=ax4)
                    
                    ax5 = plt.subplot(1, 5, 5)
                    im5 = ax5.imshow(U, cmap='coolwarm', aspect='auto')
                    ax5.set_title('U (Upper)', fontsize=12, fontweight='bold')
                    plt.colorbar(im5, ax=ax5)
                    
                    print(f"\n📊 LU Decomposition")
                    print(f"   PA = LU")
                
                plt.tight_layout()
                plt.show()
        
        interact(visualize_decomposition,
                decomp_type=Dropdown(options=['SVD', 'QR', 'LU']),
                matrix_size=IntSlider(min=3, max=6, step=1, value=3))
        display(self.output)
    
    # ==================== PROJECT 6: SOLVING LINEAR SYSTEMS ====================
    def linear_systems_demo(self):
        """Interactive demo for solving linear systems"""
        print("=" * 60)
        print("PROJECT 6: SOLVING LINEAR SYSTEMS")
        print("=" * 60)
        
        def solve_system(a11=2, a12=1, a21=1, a22=3, b1=5, b2=7):
            with self.output:
                self.output.clear_output(wait=True)
                
                A = np.array([[a11, a12], [a21, a22]], dtype=float)
                b = np.array([b1, b2], dtype=float)
                
                try:
                    x = np.linalg.solve(A, b)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Left: System visualization
                    x_range = np.linspace(-5, 5, 100)
                    
                    # Line 1: a11*x + a12*y = b1
                    if a12 != 0:
                        y1 = (b1 - a11*x_range) / a12
                        ax1.plot(x_range, y1, 'b-', linewidth=2, 
                                label=f'{a11}x + {a12}y = {b1}')
                    
                    # Line 2: a21*x + a22*y = b2
                    if a22 != 0:
                        y2 = (b2 - a21*x_range) / a22
                        ax1.plot(x_range, y2, 'r-', linewidth=2, 
                                label=f'{a21}x + {a22}y = {b2}')
                    
                    # Solution point
                    ax1.plot(x[0], x[1], 'go', markersize=15, 
                            label=f'Solution: ({x[0]:.2f}, {x[1]:.2f})')
                    
                    ax1.set_xlim(-5, 5)
                    ax1.set_ylim(-5, 5)
                    ax1.axhline(y=0, color='k', linewidth=0.5)
                    ax1.axvline(x=0, color='k', linewidth=0.5)
                    ax1.grid(True, alpha=0.3)
                    ax1.set_xlabel('x', fontsize=12)
                    ax1.set_ylabel('y', fontsize=12)
                    ax1.set_title('System of Equations', fontsize=14, fontweight='bold')
                    ax1.legend()
                    
                    # Right: Matrix representation
                    ax2.text(0.5, 0.8, 'Matrix Form: Ax = b', 
                            ha='center', fontsize=14, fontweight='bold',
                            transform=ax2.transAxes)
                    
                    matrix_text = f"""
                    [[{a11:5.1f}  {a12:5.1f}]   [x₁]   [{b1:5.1f}]
                     [{a21:5.1f}  {a22:5.1f}] × [x₂] = [{b2:5.1f}]
                    
                    Solution:
                    x₁ = {x[0]:7.3f}
                    x₂ = {x[1]:7.3f}
                    
                    Determinant: {np.linalg.det(A):.3f}
                    """
                    
                    ax2.text(0.5, 0.4, matrix_text, ha='center', va='center',
                            fontsize=12, family='monospace',
                            transform=ax2.transAxes)
                    ax2.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"\n✅ System solved successfully!")
                    print(f"   x₁ = {x[0]:.4f}")
                    print(f"   x₂ = {x[1]:.4f}")
                    print(f"\n   Verification:")
                    print(f"   Ax = {A @ x}")
                    print(f"   b  = {b}")
                    
                except np.linalg.LinAlgError:
                    print("\n❌ System is singular (no unique solution)")
                    print("   The determinant is zero or very close to zero.")
        
        interact(solve_system,
                a11=IntSlider(min=-5, max=5, step=1, value=2),
                a12=IntSlider(min=-5, max=5, step=1, value=1),
                a21=IntSlider(min=-5, max=5, step=1, value=1),
                a22=IntSlider(min=-5, max=5, step=1, value=3),
                b1=IntSlider(min=-10, max=10, step=1, value=5),
                b2=IntSlider(min=-10, max=10, step=1, value=7))
        display(self.output)
    
    # ==================== PROJECT 7: DETERMINANTS ====================
    def determinant_demo(self):
        """Interactive demo for determinants"""
        print("=" * 60)
        print("PROJECT 7: DETERMINANTS")
        print("=" * 60)
        
        def visualize_determinant(a=2, b=1, c=1, d=2):
            with self.output:
                self.output.clear_output(wait=True)
                
                A = np.array([[a, b], [c, d]], dtype=float)
                det = np.linalg.det(A)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Left: Unit square and transformation
                unit_square = np.array([[0, 1, 1, 0, 0],
                                       [0, 0, 1, 1, 0]])
                transformed = A @ unit_square
                
                ax1.fill(unit_square[0], unit_square[1], alpha=0.3, 
                        color='blue', label='Unit Square (Area=1)')
                ax1.plot(unit_square[0], unit_square[1], 'b-', linewidth=2)
                
                ax1.fill(transformed[0], transformed[1], alpha=0.3, 
                        color='red', label=f'Transformed (Area={abs(det):.2f})')
                ax1.plot(transformed[0], transformed[1], 'r-', linewidth=2)
                
                ax1.set_xlim(-5, 5)
                ax1.set_ylim(-5, 5)
                ax1.axhline(y=0, color='k', linewidth=0.5)
                ax1.axvline(x=0, color='k', linewidth=0.5)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlabel('x', fontsize=12)
                ax1.set_ylabel('y', fontsize=12)
                ax1.set_title('Area Scaling Visualization', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.set_aspect('equal')
                
                # Right: Determinant properties
                ax2.axis('off')
                info_text = f"""
                Matrix A:
                [[{a:5.1f}  {b:5.1f}]
                 [{c:5.1f}  {d:5.1f}]]
                
                Determinant = ad - bc
                           = ({a})×({d}) - ({b})×({c})
                           = {det:.4f}
                
                Properties:
                • |det| = Area scaling factor
                • det > 0: Preserves orientation
                • det < 0: Reverses orientation
                • det = 0: Collapses to lower dimension
                
                Status: {"✅ Invertible" if det != 0 else "❌ Singular (Not invertible)"}
                """
                
                ax2.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
                        verticalalignment='center', transform=ax2.transAxes)
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n📐 Determinant: {det:.4f}")
                print(f"   Area scaling: {abs(det):.4f}x")
                if det > 0:
                    print("   ↻ Orientation preserved")
                elif det < 0:
                    print("   ↺ Orientation reversed")
                else:
                    print("   ⚠ Dimension collapsed (singular)")
        
        interact(visualize_determinant,
                a=FloatSlider(min=-3, max=3, step=0.5, value=2),
                b=FloatSlider(min=-3, max=3, step=0.5, value=1),
                c=FloatSlider(min=-3, max=3, step=0.5, value=1),
                d=FloatSlider(min=-3, max=3, step=0.5, value=2))
        display(self.output)
    
    # ==================== PROJECT 8: ORTHOGONALITY ====================
    def orthogonality_demo(self):
        """Interactive demo for orthogonality and projections"""
        print("=" * 60)
        print("PROJECT 8: ORTHOGONALITY & PROJECTIONS")
        print("=" * 60)
        
        def visualize_projection(v1_x=3.0, v1_y=1.0, v2_x=1.0, v2_y=2.0):
            with self.output:
                self.output.clear_output(wait=True)
                
                v1 = np.array([v1_x, v1_y])
                v2 = np.array([v2_x, v2_y])
                
                # Project v2 onto v1
                proj = (np.dot(v2, v1) / np.dot(v1, v1)) * v1
                
                # Perpendicular component
                perp = v2 - proj
                
                # Angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Left: Projection visualization
                ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', 
                          scale=1, color='blue', width=0.01, label=f'v₁ = [{v1_x}, {v1_y}]')
                ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', 
                          scale=1, color='red', width=0.01, label=f'v₂ = [{v2_x}, {v2_y}]')
                ax1.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', 
                          scale=1, color='green', width=0.008, 
                          label=f'proj(v₂→v₁)')
                ax1.quiver(proj[0], proj[1], perp[0], perp[1], angles='xy', 
                          scale_units='xy', scale=1, color='purple', width=0.008,
                          linestyle='--', label='Perpendicular component')
                
                # Draw right angle indicator
                if np.linalg.norm(perp) > 0.1:
                    corner_size = 0.3
                    corner = proj + corner_size * perp / np.linalg.norm(perp)
                    corner2 = corner + corner_size * v1 / np.linalg.norm(v1)
                    ax1.plot([proj[0], corner[0], corner2[0]], 
                            [proj[1], corner[1], corner2[1]], 'k-', linewidth=1)
                
                ax1.set_xlim(-5, 5)
                ax1.set_ylim(-5, 5)
                ax1.axhline(y=0, color='k', linewidth=0.5)
                ax1.axvline(x=0, color='k', linewidth=0.5)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlabel('x', fontsize=12)
                ax1.set_ylabel('y', fontsize=12)
                ax1.set_title('Vector Projection', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.set_aspect('equal')
                
                # Right: Information panel
                ax2.axis('off')
                
                dot_product = np.dot(v1, v2)
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                orthogonal = "✅ Yes" if abs(dot_product) < 0.01 else "❌ No"
                
                info_text = f"""
                Vector Properties:
                
                v₁ · v₂ = {dot_product:.4f}
                |v₁| = {v1_norm:.4f}
                |v₂| = {v2_norm:.4f}
                
                Angle between vectors: {angle:.2f}°
                
                Orthogonal? {orthogonal}
                
                Projection formula:
                proj(v₂→v₁) = (v₂·v₁ / v₁·v₁) × v₁
                
                proj(v₂→v₁) = [{proj[0]:.3f}, {proj[1]:.3f}]
                
                Perpendicular part = [{perp[0]:.3f}, {perp[1]:.3f}]
                
                Verification:
                v₂ = proj + perp ✓
                """
                
                ax2.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                        verticalalignment='center', transform=ax2.transAxes)
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n📏 Dot Product: {dot_product:.4f}")
                print(f"   Angle: {angle:.2f}°")
                print(f"   Orthogonal: {orthogonal}")
        
        interact(visualize_projection,
                v1_x=FloatSlider(min=-5, max=5, step=0.5, value=3),
                v1_y=FloatSlider(min=-5, max=5, step=0.5, value=1),
                v2_x=FloatSlider(min=-5, max=5, step=0.5, value=1),
                v2_y=FloatSlider(min=-5, max=5, step=0.5, value=2))
        display(self.output)
    
    # ==================== PROJECT 9: LEAST SQUARES ====================
    def least_squares_demo(self):
        """Interactive demo for least squares regression"""
        print("=" * 60)
        print("PROJECT 9: LEAST SQUARES REGRESSION")
        print("=" * 60)
        
        def visualize_least_squares(n_points=20, noise_level=1.0, degree=1):
            with self.output:
                self.output.clear_output(wait=True)
                
                np.random.seed(42)
                
                # Generate data
                x = np.linspace(0, 10, n_points)
                true_y = 2 * x + 1
                noise = noise_level * np.random.randn(n_points)
                y = true_y + noise
                
                # Create design matrix for polynomial fitting
                X = np.vander(x, degree + 1, increasing=True)
                
                # Solve least squares: X^T X theta = X^T y
                theta = np.linalg.lstsq(X, y, rcond=None)[0]
                
                # Predictions
                x_plot = np.linspace(0, 10, 100)
                X_plot = np.vander(x_plot, degree + 1, increasing=True)
                y_pred = X_plot @ theta
                y_fit = X @ theta
                
                # Calculate residuals
                residuals = y - y_fit
                mse = np.mean(residuals**2)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Left: Data and fit
                ax1.scatter(x, y, color='blue', s=50, alpha=0.6, label='Data points')
                ax1.plot(x_plot, y_pred, 'r-', linewidth=2, label=f'Degree {degree} fit')
                
                # Draw residuals
                for i in range(len(x)):
                    ax1.plot([x[i], x[i]], [y[i], y_fit[i]], 'g--', alpha=0.3, linewidth=1)
                
                ax1.set_xlabel('x', fontsize=12)
                ax1.set_ylabel('y', fontsize=12)
                ax1.set_title('Least Squares Fit', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Right: Residual plot
                ax2.scatter(x, residuals, color='green', s=50, alpha=0.6)
                ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
                ax2.set_xlabel('x', fontsize=12)
                ax2.set_ylabel('Residuals', fontsize=12)
                ax2.set_title(f'Residual Plot (MSE={mse:.3f})', 
                            fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n📈 Least Squares Solution:")
                print(f"   Coefficients: {theta}")
                print(f"   Mean Squared Error: {mse:.4f}")
                print(f"\n   Equation: y = ", end="")
                for i, coef in enumerate(theta):
                    if i == 0:
                        print(f"{coef:.3f}", end="")
                    else:
                        print(f" + {coef:.3f}x^{i}", end="")
                print()
        
        interact(visualize_least_squares,
                n_points=IntSlider(min=10, max=50, step=5, value=20),
                noise_level=FloatSlider(min=0, max=3, step=0.5, value=1),
                degree=IntSlider(min=1, max=5, step=1, value=1))
        display(self.output)
    
    # ==================== PROJECT 10: PRINCIPAL COMPONENT ANALYSIS ====================
    def pca_demo(self):
        """Interactive demo for PCA"""
        print("=" * 60)
        print("PROJECT 10: PRINCIPAL COMPONENT ANALYSIS (PCA)")
        print("=" * 60)
        
        def visualize_pca(n_samples=100, correlation=0.8):
            with self.output:
                self.output.clear_output(wait=True)
                
                np.random.seed(42)
                
                # Generate correlated 2D data
                mean = [0, 0]
                cov = [[1, correlation], [correlation, 1]]
                data = np.random.multivariate_normal(mean, cov, n_samples)
                
                # Standardize
                data_centered = data - np.mean(data, axis=0)
                
                # Compute PCA using SVD
                U, S, Vt = np.linalg.svd(data_centered.T, full_matrices=False)
                
                # Principal components
                pc1 = U[:, 0]
                pc2 = U[:, 1]
                
                # Explained variance
                explained_var = (S**2) / (n_samples - 1)
                explained_var_ratio = explained_var / np.sum(explained_var)
                
                # Project data onto principal components
                projected = data_centered @ U
                
                fig = plt.figure(figsize=(16, 6))
                
                # Left: Original data with PC directions
                ax1 = plt.subplot(1, 3, 1)
                ax1.scatter(data_centered[:, 0], data_centered[:, 1], 
                           alpha=0.6, s=30)
                
                # Draw principal components
                scale = 3
                ax1.arrow(0, 0, scale*pc1[0], scale*pc1[1], 
                         head_width=0.2, head_length=0.2, fc='red', ec='red',
                         linewidth=3, label=f'PC1 ({explained_var_ratio[0]:.1%})')
                ax1.arrow(0, 0, scale*pc2[0], scale*pc2[1], 
                         head_width=0.2, head_length=0.2, fc='blue', ec='blue',
                         linewidth=3, label=f'PC2 ({explained_var_ratio[1]:.1%})')
                
                ax1.set_xlabel('Feature 1', fontsize=12)
                ax1.set_ylabel('Feature 2', fontsize=12)
                ax1.set_title('Original Data Space', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_aspect('equal')
                
                # Middle: Projected data
                ax2 = plt.subplot(1, 3, 2)
                ax2.scatter(projected[:, 0], projected[:, 1], alpha=0.6, s=30)
                ax2.axhline(y=0, color='k', linewidth=0.5)
                ax2.axvline(x=0, color='k', linewidth=0.5)
                ax2.set_xlabel('PC1', fontsize=12)
                ax2.set_ylabel('PC2', fontsize=12)
                ax2.set_title('Transformed Space', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_aspect('equal')
                
                # Right: Scree plot
                ax3 = plt.subplot(1, 3, 3)
                components = ['PC1', 'PC2']
                ax3.bar(components, explained_var_ratio * 100, color=['red', 'blue'])
                ax3.set_ylabel('Explained Variance (%)', fontsize=12)
                ax3.set_title('Scree Plot', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='y')
                
                for i, v in enumerate(explained_var_ratio * 100):
                    ax3.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n📊 PCA Results:")
                print(f"   PC1 direction: [{pc1[0]:.4f}, {pc1[1]:.4f}]")
                print(f"   PC2 direction: [{pc2[0]:.4f}, {pc2[1]:.4f}]")
                print(f"\n   Explained variance:")
                print(f"   PC1: {explained_var_ratio[0]:.2%}")
                print(f"   PC2: {explained_var_ratio[1]:.2%}")
                print(f"   Total: {sum(explained_var_ratio):.2%}")
        
        interact(visualize_pca,
                n_samples=IntSlider(min=50, max=200, step=25, value=100),
                correlation=FloatSlider(min=-0.9, max=0.9, step=0.1, value=0.8))
        display(self.output)
    
    # ==================== MAIN DASHBOARD ====================
    def run_all(self):
        """Run all demonstrations in sequence"""
        print("\n" + "="*80)
        print(" " * 20 + "LINEAR ALGEBRA TOOLKIT")
        print(" " * 15 + "Comprehensive Interactive Learning Tool")
        print("="*80 + "\n")
        
        demos = [
            ("Matrix Operations", self.matrix_operations_demo),
            ("Vector Spaces & Linear Combinations", self.vector_spaces_demo),
            ("Eigenvalues & Eigenvectors", self.eigenvalues_demo),
            ("Linear Transformations", self.transformations_demo),
            ("Matrix Decompositions", self.decomposition_demo),
            ("Solving Linear Systems", self.linear_systems_demo),
            ("Determinants", self.determinant_demo),
            ("Orthogonality & Projections", self.orthogonality_demo),
            ("Least Squares Regression", self.least_squares_demo),
            ("Principal Component Analysis", self.pca_demo)
        ]
        
        print("📚 Available Demonstrations:\n")
        for i, (name, _) in enumerate(demos, 1):
            print(f"   {i:2d}. {name}")
        
        print("\n" + "="*80)
        print("\n🎯 Use the dropdown below to select a demonstration:\n")
        
        def select_demo(demonstration):
            for name, func in demos:
                if name == demonstration:
                    func()
                    break
        
        demo_names = [name for name, _ in demos]
        interact(select_demo, 
                demonstration=Dropdown(options=demo_names, 
                                      description='Select:',
                                      style={'description_width': 'initial'}))


# ==================== USAGE INSTRUCTIONS ====================
if __name__ == "__main__":
    toolkit = LinearAlgebraToolkit()
    toolkit.run_all()
