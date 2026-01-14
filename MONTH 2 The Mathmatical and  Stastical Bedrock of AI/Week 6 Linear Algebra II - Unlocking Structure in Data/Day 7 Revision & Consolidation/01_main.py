
#Day 7: Revision & Consolidation

#1: Eigenvector Visualizer: For a given 2x2 matrix, calculate and plot its eigenvectors to see the directions of
# pure stretch/compression.
#2: Principal Component Analysis (PCA) from Scratch: Implement PCA using NumPy's eigenvalue decomposition to 
#perform dimensionality reduction on a dataset.
#3: Image Compression with SVD: Use SVD to compress an image by keeping only the top 'k' singular values and 
#reconstructing the image.
#4 Matrix Inverse Calculator: Write a script that checks if a matrix is invertible and calculates its inverse if it is.
#5: Determinant and Area: Show visually how the determinant of a 2x2 matrix relates to the change in area of a 
#transformed shape.
#6:Recommender System with SVD: Implement a basic movie recommender system using SVD on a user-item rating matrix.
#7:Covariance Matrix: Calculate and interpret the covariance matrix for a dataset.
#8: Check for Orthogonality: Write a function to check if the columns of a matrix are orthogonal.
#9: Linear Independence Check: Use matrix rank to determine if a set of vectors is linearly independent.
#10: Change of Basis: Write a script to transform the coordinates of a vector from one basis to another.

#Project 1:
# Eigenvector Visualizer: For a given 2x2 matrix, calculate and plot its eigenvectors to see the directions of
# pure stretch/compression.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 5))

# Define a 2x2 matrix (you can change these values)
A = np.array([[2, 1],
              [1, 2]])

print("Matrix A:")
print(A)
print()

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("\nEigenvectors (as columns):")
print(eigenvectors)
print()

# Extract individual eigenvectors
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 1]

print(f"Eigenvector 1: {v1} with eigenvalue {eigenvalues[0]:.3f}")
print(f"Eigenvector 2: {v2} with eigenvalue {eigenvalues[1]:.3f}")

# --- Plot 1: Original Eigenvectors ---
plt.subplot(1, 2, 1)
ax = plt.gca()
ax.set_aspect('equal')

# Plot origin
plt.plot(0, 0, 'ko', markersize=8, label='Origin')

# Plot eigenvectors
plt.arrow(0, 0, v1[0], v1[1], head_width=0.1, head_length=0.1, 
          fc='red', ec='red', linewidth=2, label=f'v1 (λ={eigenvalues[0]:.2f})')
plt.arrow(0, 0, v2[0], v2[1], head_width=0.1, head_length=0.1, 
          fc='blue', ec='blue', linewidth=2, label=f'v2 (λ={eigenvalues[1]:.2f})')

# Add grid and labels
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Eigenvectors of Matrix A')
plt.legend()
plt.xlim(-2, 2)
plt.ylim(-2, 2)

# --- Plot 2: Transformation Effect ---
plt.subplot(1, 2, 2)
ax = plt.gca()
ax.set_aspect('equal')

# Apply transformation to eigenvectors
Av1 = A @ v1
Av2 = A @ v2

# Plot original eigenvectors (lighter)
plt.arrow(0, 0, v1[0], v1[1], head_width=0.15, head_length=0.15, 
          fc='pink', ec='pink', linewidth=2, alpha=0.5, label='Original v1')
plt.arrow(0, 0, v2[0], v2[1], head_width=0.15, head_length=0.15, 
          fc='lightblue', ec='lightblue', linewidth=2, alpha=0.5, label='Original v2')

# Plot transformed eigenvectors
plt.arrow(0, 0, Av1[0], Av1[1], head_width=0.15, head_length=0.15, 
          fc='red', ec='red', linewidth=2.5, label=f'A·v1 = {eigenvalues[0]:.2f}·v1')
plt.arrow(0, 0, Av2[0], Av2[1], head_width=0.15, head_length=0.15, 
          fc='blue', ec='blue', linewidth=2.5, label=f'A·v2 = {eigenvalues[1]:.2f}·v2')

# Add grid and labels
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Effect of Transformation A')
plt.legend(fontsize=9)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.tight_layout()
plt.show()

# --- Additional Analysis ---
print("\n" + "="*50)
print("INTERPRETATION:")
print("="*50)

for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T), 1):
    print(f"\nEigenvector {i}: {vec}")
    print(f"Eigenvalue {i}: {val:.3f}")
    
    if val > 1:
        print(f"  → Stretched by factor of {val:.3f}")
    elif val < -1:
        print(f"  → Stretched by factor of {abs(val):.3f} and flipped")
    elif 0 < val < 1:
        print(f"  → Compressed by factor of {val:.3f}")
    elif -1 < val < 0:
        print(f"  → Compressed by factor of {abs(val):.3f} and flipped")
    elif val == 1:
        print(f"  → No change (invariant)")
    elif val == -1:
        print(f"  → Flipped (reversed direction)")
    
    # Verify: A·v = λ·v
    verification = np.allclose(A @ vec, val * vec)
    print(f"  Verification (A·v = λ·v): {verification}")
