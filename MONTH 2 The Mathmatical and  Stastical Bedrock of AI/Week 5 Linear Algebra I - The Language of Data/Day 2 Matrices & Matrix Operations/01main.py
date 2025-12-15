
#Day 2: Matrices & Matrix Operations
#Theory (2-3 hours):

#Read: Mathematics for ML Chapter 2 (Sections 2.4-2.5) - Matrices, matrix operations
#Watch: Coursera Week 1-2 - Matrix multiplication intuition
#Focus: Matrices as linear transformations, matrix multiplication mechanics

#Project 1: Matrix Multiplication from Scratch

#Implement matrix_multiply() with proper dimension checking
#Test with identity matrix, zero matrix, and random matrices
#Verify associativity: (AB)C = A(BC)


#Project 2: Image as a Matrix

#Create a 10×10 black & white image matrix
#Perform transpose, flip (using slicing)
#Apply brightness adjustment (scalar multiplication)
#Visualize with matplotlib



#Exercise Set:
#15 matrix multiplication problems (by hand for small matrices)
#Verify properties: (AB)ᵀ = BᵀAᵀ


#What Are Matrices?Matrices are 2D arrays that represent:
#Data tables (rows = samples, columns = features)
#Linear transformations (functions that map vectors to vectors)
#Systems of equations
#Relationships between variables


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


'''
#Creating matrices
matrix_2x3 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

matrix_identity = np.eye(3)  # 3x3 identity matrix
matrix_zeros = np.zeros((2, 4))  # 2x4 zero matrix
matrix_random = np.random.rand(3, 3)  # 3x3 random matrix

print("2x3 Matrix:")
print(matrix_2x3)
print(f"\nShape: {matrix_2x3.shape}")
print(f"Size: {matrix_2x3.size}")
'''

#Matrix Operations: Building Blocks
#A. Basic Matrix Operations

'''
# Element-wise operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Addition
C_add = A + B
print("A + B:")
print(C_add)

# Subtraction
C_sub = A - B
print("\nA - B:")
print(C_sub)

# Element-wise multiplication (Hadamard product)
C_elem = A * B  # NOT matrix multiplication!
print("\nElement-wise A * B:")
print(C_elem)

# Scalar multiplication
C_scalar = 3 * A
print("\n3 * A:")
print(C_scalar)

# Transpose
A_T = A.T
print("\nTranspose of A:")
print(A_T)
'''

'''
#B. Matrix Multiplication: The Core Concept

# Matrix multiplication (dot product)
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])  # 3x2

# Three ways to multiply matrices
C1 = np.dot(A, B)      # Traditional
C2 = A @ B             # Python 3.5+ operator (RECOMMENDED)
C3 = np.matmul(A, B)   # Explicit function

print("A @ B (2x3 × 3x2 = 2x2):")
print(C1)
print(f"\nResult shape: {C1.shape}")
'''

'''
#Understanding Matrix Multiplication Step-by-Step
def matrix_multiply_verbose(A, B):
    """
    Detailed matrix multiplication with explanation
    """
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")
    
    C = np.zeros((m, p))
    
    print(f"Multiplying {A.shape} × {B.shape} → {C.shape}\n")
    
    for i in range(m):
        for j in range(p):
            # Compute C[i,j] as dot product of row i of A and column j of B
            result = 0
            computation = []
            for k in range(n):
                result += A[i, k] * B[k, j]
                computation.append(f"{A[i,k]}×{B[k,j]}")
            
            C[i, j] = result
            print(f"C[{i},{j}] = {' + '.join(computation)} = {result}")
    
    return C
'''

'''
#3. Matrices as Linear Transformations: The Intuition
#A. Visualizing Transformations

def visualize_transformation(matrix, title="Linear Transformation"):
    """
    Visualize how a matrix transforms the unit square
    """
    # Original unit square
    original = np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0]
    ])
    
    # Transform the square
    transformed = matrix @ original
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original space
    ax1.plot(original[0, :], original[1, :], 'b-', linewidth=2, label='Unit Square')
    ax1.scatter([0], [0], color='red', s=100, zorder=5, label='Origin')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_title('Original Space', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # Transformed space
    ax2.plot(original[0, :], original[1, :], 'b--', linewidth=1, alpha=0.3, label='Original')
    ax2.plot(transformed[0, :], transformed[1, :], 'r-', linewidth=2, label='Transformed')
    ax2.scatter([0], [0], color='red', s=100, zorder=5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.set_title('Transformed Space', fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Example transformations

# 1. Scaling
scaling_matrix = np.array([
    [2, 0],
    [0, 1.5]
])
visualize_transformation(scaling_matrix, "Scaling Transformation")

# 2. Rotation (90 degrees counter-clockwise)
rotation_matrix = np.array([
    [0, -1],
    [1, 0]
])
visualize_transformation(rotation_matrix, "90° Rotation")

# 3. Shear
shear_matrix = np.array([
    [1, 0.5],
    [0, 1]
])
visualize_transformation(shear_matrix, "Shear Transformation")

# 4. Reflection (across y-axis)
reflection_matrix = np.array([
    [-1, 0],
    [0, 1]
])
visualize_transformation(reflection_matrix, "Reflection Across Y-axis")

# BONUS: More interesting transformations!

# 5. 45-degree rotation
angle = np.pi / 4
rotation_45 = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
visualize_transformation(rotation_45, "45° Rotation")

# 6. Non-uniform scaling
non_uniform = np.array([
    [3, 0],
    [0, 0.5]
])
visualize_transformation(non_uniform, "Non-uniform Scaling")

# 7. Combined transformation (rotate then scale)
combined = scaling_matrix @ rotation_matrix
visualize_transformation(combined, "Combined: Scale ∘ Rotate")

# 8. Projection onto x-axis
projection_x = np.array([
    [1, 0],
    [0, 0]
])
visualize_transformation(projection_x, "Projection onto X-axis")

print("All transformations visualized successfully!")
print("\n Key Insights:")
print("Scaling: Changes size along axes")
print("Rotation: Preserves shape and size, changes orientation")
print("Shear: Slants the shape")
print("Reflection: Flips across an axis")
print("Projection: Collapses onto a line/plane")
print("\n Try creating your own transformation matrices!")
'''


'''
#B. Composition of Transformations
# Multiple transformations = Matrix multiplication!
# Order matters: Read right to left

# First rotate, then scale
rotation = np.array([[0, -1], [1, 0]])
scaling = np.array([[2, 0], [0, 0.5]])

# Combined transformation
combined = scaling @ rotation  # First rotation, then scaling

print("Rotation Matrix:")
print(rotation)
print("\nScaling Matrix:")
print(scaling)
print("\nCombined Transformation (Scale ∘ Rotate):")
print(combined)

# Apply to a point
point = np.array([1, 0])
rotated = rotation @ point
final = scaling @ rotated

# Or directly
final_direct = combined @ point

print(f"\nOriginal point: {point}")
print(f"After rotation: {rotated}")
print(f"After scaling: {final}")
print(f"Direct application: {final_direct}")
'''

'''
#4. Practical Applications with NumPy
#A. Image Transformations

# Simulate a simple image as a matrix
image = np.array([
    [100, 150, 200],
    [120, 170, 210],
    [140, 180, 220]
])

# Flip vertically
flipped_vertical = np.flipud(image)

# Flip horizontally
flipped_horizontal = np.fliplr(image)

# Rotate 90 degrees
rotated_90 = np.rot90(image)

# Transpose (reflection along diagonal)
transposed = image.T

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

images = [
    (image, "Original"),
    (flipped_vertical, "Flipped Vertical"),
    (flipped_horizontal, "Flipped Horizontal"),
    (rotated_90, "Rotated 90°"),
    (transposed, "Transposed"),
    (image * 1.5, "Brightness +50%")
]

for ax, (img, title) in zip(axes.flat, images):
    sns.heatmap(img, ax=ax, cmap='viridis', annot=True, fmt='.0f', 
                cbar=False, square=True)
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
'''

'''
#B. Data Transformation Pipeline

# Real-world data transformation
np.random.seed(42)

# Original data: 100 samples, 3 features
data = np.random.randn(100, 3) * 10 + 50

print(f"Original data shape: {data.shape}")
print(f"Mean: {data.mean(axis=0)}")
print(f"Std: {data.std(axis=0)}")

# Centering (subtract mean)
data_centered = data - data.mean(axis=0)

# Standardization (center and scale)
data_standardized = (data - data.mean(axis=0)) / data.std(axis=0)

# Custom linear transformation
transformation_matrix = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
])

data_transformed = data_standardized @ transformation_matrix.T

# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

datasets = [
    (data, "Original Data"),
    (data_centered, "Centered Data"),
    (data_standardized, "Standardized Data"),
    (data_transformed, "Transformed Data")
]

for ax, (dataset, title) in zip(axes.flat, datasets):
    for i in range(3):
        ax.hist(dataset[:, i], bins=20, alpha=0.5, label=f'Feature {i+1}')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''

#Advanced Matrix Operations
'''
# Matrix inverse
A = np.array([
    [4, 7],
    [2, 6]
])

# Compute inverse
A_inv = np.linalg.inv(A)

print("Matrix A:")
print(A)
print("\nInverse of A:")
print(A_inv)

# Verify: A @ A_inv = I
identity_check = A @ A_inv
print("\nA @ A_inv (should be identity):")
print(identity_check)

# Determinant
det_A = np.linalg.det(A)
print(f"\nDeterminant of A: {det_A:.2f}")

# Geometric interpretation: determinant = scaling factor of area
print(f"Area scaling factor: {abs(det_A):.2f}x")
'''

'''
#B. Eigenvalues and Eigenvectors

# Eigenvalues and eigenvectors reveal fundamental properties
A = np.array([
    [4, -2],
    [1, 1]
])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors:")
print(eigenvectors)

# Verify: A @ v = λ @ v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_v = eigenvalues[i] * v
    A_v = A @ v
    
    print(f"\nEigenvalue {i+1}: λ = {eigenvalues[i]:.2f}")
    print(f"A @ v = {A_v}")
    print(f"λ @ v = {lambda_v}")
    print(f"Match: {np.allclose(A_v, lambda_v)}")
    '''

'''
#C. Matrix Decompositions
# SVD (Singular Value Decomposition)
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Decompose: A = U @ S @ V^T
U, s, VT = np.linalg.svd(A)

print("Original Matrix A:")
print(A)
print(f"\nU (left singular vectors):\n{U}")
print(f"\nSingular values: {s}")
print(f"\nV^T (right singular vectors):\n{VT}")

# Reconstruct
S = np.zeros((3, 3))
np.fill_diagonal(S, s)
A_reconstructed = U @ S @ VT

print(f"\nReconstructed A:")
print(A_reconstructed)
print(f"\nReconstruction error: {np.linalg.norm(A - A_reconstructed):.10f}")
'''


#6. Matrix Multiplication Intuition: Deep Dive

'''
#A. Three Ways to Think About Matrix Multiplication
def matrix_mult_perspectives(A, B):
    """
    Show three perspectives of matrix multiplication
    """
    print("=" * 60)
    print("THREE PERSPECTIVES ON MATRIX MULTIPLICATION")
    print("=" * 60)
    
    # Perspective 1: Dot products of rows and columns
    print("\n1️PERSPECTIVE 1: Row-Column Dot Products")
    print("C[i,j] = dot(A[row i], B[column j])")
    C1 = A @ B
    print(f"Result:\n{C1}\n")
    
    # Perspective 2: Linear combination of columns
    print("2️PERSPECTIVE 2: Linear Combination of Columns")
    print("Each column of C is a linear combination of columns of A")
    C2 = np.zeros((A.shape,[object Object],, B.shape,[object Object],))
    for j in range(B.shape,[object Object],):
        # Column j of C = A @ (column j of B)
        C2[:, j] = A @ B[:, j]
        print(f"Column {j}: A @ B[:,{j}] = A @ {B[:, j]}")
    print(f"Result:\n{C2}\n")
    
    # Perspective 3: Linear combination of rows
    print("3️PERSPECTIVE 3: Linear Combination of Rows")
    print("Each row of C is a linear combination of rows of B")
    C3 = np.zeros((A.shape,[object Object],, B.shape,[object Object],))
    for i in range(A.shape,[object Object],):
        # Row i of C = (row i of A) @ B
        C3[i, :] = A[i, :] @ B
        print(f"Row {i}: A[{i},:] @ B = {A[i, :]} @ B")
    print(f"Result:\n{C3}\n")
    
    # Verify all are the same
    print("All perspectives give the same result!")
    print(f"All equal: {np.allclose(C1, C2) and np.allclose(C2, C3)}")

# Example
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

matrix_mult_perspectives(A, B)
'''


#B. Visualizing Matrix Multiplication as Transformation

def visualize_matrix_mult_transformation():
    """
    Show how matrix multiplication transforms basis vectors
    """
    # Transformation matrix
    M = np.array([
        [2, 1],
        [1, 2]
    ])
    
    # Standard basis vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    # Transformed basis vectors
    Me1 = M @ e1
    Me2 = M @ e2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ============== Original basis ==============
    ax1.quiver(0, 0, e1[0], e1[1], 
               angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.015, label='e₁ = [1,0]')
    ax1.quiver(0, 0, e2[0], e2[1], 
               angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.015, label='e₂ = [0,1]')
    
    # Draw grid
    for i in range(-3, 4):
        ax1.plot([-3, 3], [i, i], 'k-', alpha=0.1, linewidth=0.5)
        ax1.plot([i, i], [-3, 3], 'k-', alpha=0.1, linewidth=0.5)
    
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=1)
    ax1.axvline(x=0, color='k', linewidth=1)
    ax1.set_title('Original Basis Vectors', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    
    # ============== Transformed basis ==============
    ax2.quiver(0, 0, Me1[0], Me1[1], 
               angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.015, label=f'M·e₁ = [{Me1[0]}, {Me1[1]}]')
    ax2.quiver(0, 0, Me2[0], Me2[1], 
               angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.015, label=f'M·e₂ = [{Me2[0]}, {Me2[1]}]')
    
    # Draw transformed grid
    for i in range(-3, 4):
        # Horizontal lines
        start = M @ np.array([-3, i])
        end = M @ np.array([3, i])
        ax2.plot([start[0], end[0]], [start[1], end[1]], 
                'k-', alpha=0.1, linewidth=0.5)
        
        # Vertical lines
        start = M @ np.array([i, -3])
        end = M @ np.array([i, 3])
        ax2.plot([start[0], end[0]], [start[1], end[1]], 
                'k-', alpha=0.1, linewidth=0.5)
    
    ax2.set_xlim(-10, 10)  # Wider limits for transformed space
    ax2.set_ylim(-10, 10)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=1)
    ax2.axvline(x=0, color='k', linewidth=1)
    ax2.set_title('Transformed Basis Vectors', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    
    plt.suptitle(f'Matrix M = {M.tolist()} transforms space', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("=" * 60)
    print("🎯 KEY INSIGHT:")
    print("=" * 60)
    print("The columns of M tell you where the basis vectors land!")
    print(f"  e₁ [1,0] → [{Me1[0]}, {Me1[1]}] (first column of M)")
    print(f"  e₂ [0,1] → [{Me2[0]}, {Me2[1]}] (second column of M)")
    print("\n💡 Matrix columns = where basis vectors go!")
    print("=" * 60)

# Run the visualization
visualize_matrix_mult_transformation()

#Interactive version with multiple transformations


def compare_transformations():
    """
    Compare different transformation matrices side by side
    """
    transformations = {
        'Scaling': np.array([[2, 0], [0, 2]]),
        'Rotation 90°': np.array([[0, -1], [1, 0]]),
        'Shear': np.array([[1, 1], [0, 1]]),
        'Reflection': np.array([[-1, 0], [0, 1]])
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (name, M) in enumerate(transformations.items()):
        ax = axes[idx]
        
        # Original basis
        e1, e2 = np.array([1, 0]), np.array([0, 1])
        Me1, Me2 = M @ e1, M @ e2
        
        # Draw original (faint)
        ax.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', 
                 scale=1, color='red', alpha=0.3, width=0.01)
        ax.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', 
                 scale=1, color='blue', alpha=0.3, width=0.01)
        
        # Draw transformed (bold)
        ax.quiver(0, 0, Me1[0], Me1[1], angles='xy', scale_units='xy', 
                 scale=1, color='red', width=0.015, 
                 label=f'e₁ → [{Me1[0]:.1f}, {Me1[1]:.1f}]')
        ax.quiver(0, 0, Me2[0], Me2[1], angles='xy', scale_units='xy', 
                 scale=1, color='blue', width=0.015,
                 label=f'e₂ → [{Me2[0]:.1f}, {Me2[1]:.1f}]')
        
        # Draw unit square and transformed square
        unit_square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        transformed_square = M @ unit_square
        
        ax.plot(unit_square[0], unit_square[1], 'k--', 
               alpha=0.3, linewidth=1, label='Original')
        ax.plot(transformed_square[0], transformed_square[1], 'g-', 
               linewidth=2, label='Transformed')
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(f'{name}\nM = {M.tolist()}', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
    
    plt.suptitle('How Different Matrices Transform Basis Vectors', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Run comparison
compare_transformations()


# ============================================================
# PRACTICE EXERCISES
# ============================================================


# Exercise 1
print("\n What does this matrix do?")
mystery = np.array([[3, 0], [0, 3]])
print(f"   M = {mystery.tolist()}")
print(f"   Hint: Where do [1,0] and [0,1] go?")
print(f"   Answer: [1,0] → {mystery @ np.array([1,0])}")
print(f"           [0,1] → {mystery @ np.array([0,1])}")

# Exercise 2
print("\n Rotation matrix for 45°:")
angle = np.pi / 4
rot45 = np.array([[np.cos(angle), -np.sin(angle)], 
                  [np.sin(angle), np.cos(angle)]])
print(f"   M = [[{rot45[0,0]:.3f}, {rot45[0,1]:.3f}]")
print(f"        [{rot45[1,0]:.3f}, {rot45[1,1]:.3f}]]")

# Exercise 3
print("\n Create your own transformation:")
print("   Try M = [[a, b], [c, d]] where:")
print("   - a, d control scaling")
print("   - b, c control shearing")
print("   - Determinant det(M) = ad - bc tells area scaling")

print("\n Challenge: Find a matrix that:")
print("   - Doubles the x-coordinate")
print("   - Keeps y-coordinate the same")
answer = np.array([[2, 0], [0, 1]])
print(f"   Answer: {answer.tolist()}")

print("\n Matrix composition:")
print("   If M₁ rotates 90° and M₂ scales by 2,")
print("   what does M₂ @ M₁ do? (Apply M₁ first, then M₂)")
M1 = np.array([[0, -1], [1, 0]])  # Rotate
M2 = np.array([[2, 0], [0, 2]])   # Scale
combined = M2 @ M1
print(f"   M₂ @ M₁ = {combined.tolist()}")
print(f"   Result: Rotate 90°, then scale by 2!")