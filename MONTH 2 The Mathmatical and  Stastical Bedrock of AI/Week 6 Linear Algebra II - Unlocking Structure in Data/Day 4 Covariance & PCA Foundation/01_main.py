#Day 4: Covariance & PCA Foundation
#Goal: Connect linear algebra to statistics and prepare for dimensionality reduction.

#Mathematics for Machine Learning, Chapter 4: Section on covariance matrices
#BITS Pilani Course: PCA introduction
#Review how eigendecomposition applies to covariance matrices


#Covariance Matrix: Calculate and interpret the covariance matrix for a dataset (e.g., iris dataset). Visualize the correlation between features.
#Principal Component Analysis (PCA) from Scratch: Implement PCA using NumPy's eigenvalue decomposition. Apply it to a dataset and visualize the first two principal components.

#Key Concepts to Master
#Covariance and correlation
#Eigendecomposition of covariance matrices
#Variance explained by principal components
#Dimensionality reduction intuition
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. DATA GENERATION ---
np.random.seed(42)
height = np.random.normal(170, 10, 100)
weight = height * 0.5 + np.random.normal(0, 5, 100)
data = np.stack([height, weight], axis=1)

# --- 2. COVARIANCE MATRIX ---
# Logic: Ensure the matrix captures how Height & Weight move together
cov_matrix = np.cov(data, rowvar=False)
print("Covariance Matrix:\n", cov_matrix)

# Heatmap Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(cov_matrix, annot=True, fmt='.2f', 
            xticklabels=['Height', 'Weight'], 
            yticklabels=['Height', 'Weight'], cmap='Blues')
plt.title("Step 1: Covariance Heatmap")
plt.show()

# --- 3. EIGENDECOMPOSITION ---
# Logic: We MUST center the data so the origin (0,0) is the mean.
data_centered = data - np.mean(data, axis=0)
cov_centered = np.cov(data_centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_centered)

print("\nEigenvalues (Variance magnitude):", eigenvalues)
print("Eigenvectors (Directions):\n", eigenvectors)

# Plotting Eigenvectors over Centered Data
plt.figure(figsize=(6, 6))
plt.scatter(data_centered[:, 0], data_centered[:, 1], alpha=0.4, label="Centered Data")

# Logic for Plotting Vectors
colors = ['red', 'green']
for i in range(len(eigenvalues)):
    # Scale vector for visibility: direction * sqrt(variance) * constant
    v = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 3
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
               color=colors[i], label=f'PC{i+1} Direction')

plt.xlabel("Height (centered)")
plt.ylabel("Weight (centered)")
plt.legend()
plt.title("Step 2: Eigenvectors showing Max Variance")
plt.grid(True)
plt.show()

# --- 4. PCA PROJECTION (Dimensionality Reduction) ---
# Logic: Sort by Eigenvalues (highest variance first)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project 2D data onto the 1D line of the 1st Principal Component
# Dot product logic: Data (100,2) @ Top Eigenvector (2,1) = (100,1)
top_eigenvector = eigenvectors[:, 0]
data_reduced = np.dot(data_centered, top_eigenvector)

print("\nOriginal shape:", data.shape)
print("Reduced shape (1D):", data_reduced.shape)

# Final Visualization: The 1D "Shadow" of our data
plt.figure(figsize=(8, 2))
plt.scatter(data_reduced, np.zeros_like(data_reduced), alpha=0.6, color='purple')
plt.title("Step 3: 1D Projection (PCA Result)")
plt.xlabel("Principal Component 1 Value")
plt.yticks([]) # Hide Y axis as it's now 1D
plt.show()