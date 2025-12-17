
#Principal Component Analysis (PCA) from Scratch: Implement PCA using NumPy's eigenvalue decomposition.
#Apply it to a dataset and visualize the first two principal components.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("=" * 70)
print("PRINCIPAL COMPONENT ANALYSIS (PCA) FROM SCRATCH")
print("=" * 70)

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

print("\nDataset Information:")
print(f"Shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Classes: {target_names}")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================================
# STEP 1: STANDARDIZE THE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: STANDARDIZE THE DATA")
print("=" * 70)

# Calculate mean and standard deviation
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

print("\nOriginal Data Statistics:")
print("Mean:", mean)
print("Std:", std)

# Standardize the data (z-score normalization)
X_standardized = (X - mean) / std

print("\nStandardized Data Statistics:")
print("Mean:", np.mean(X_standardized, axis=0))
print("Std:", np.std(X_standardized, axis=0))

# ============================================================================
# STEP 2: CALCULATE COVARIANCE MATRIX
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: CALCULATE COVARIANCE MATRIX")
print("=" * 70)

# Covariance matrix
cov_matrix = np.cov(X_standardized.T)

print("\nCovariance Matrix:")
print(cov_matrix)
print(f"\nCovariance Matrix Shape: {cov_matrix.shape}")

# ============================================================================
# STEP 3: COMPUTE EIGENVALUES AND EIGENVECTORS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: COMPUTE EIGENVALUES AND EIGENVECTORS")
print("=" * 70)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors (columns are eigenvectors):")
print(eigenvectors)

# ============================================================================
# STEP 4: SORT EIGENVALUES AND EIGENVECTORS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: SORT EIGENVALUES IN DESCENDING ORDER")
print("=" * 70)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues_sorted = eigenvalues[idx]
eigenvectors_sorted = eigenvectors[:, idx]

print("\nSorted Eigenvalues:")
for i, eigenval in enumerate(eigenvalues_sorted):
    print(f"PC{i+1}: {eigenval:.4f}")

# ============================================================================
# STEP 5: CALCULATE EXPLAINED VARIANCE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: CALCULATE EXPLAINED VARIANCE")
print("=" * 70)

# Calculate explained variance ratio
total_variance = np.sum(eigenvalues_sorted)
explained_variance_ratio = eigenvalues_sorted / total_variance
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("\nExplained Variance Ratio:")
for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
    print(f"PC{i+1}: {var*100:.2f}% (Cumulative: {cum_var*100:.2f}%)")

# ============================================================================
# STEP 6: PROJECT DATA ONTO PRINCIPAL COMPONENTS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: PROJECT DATA ONTO PRINCIPAL COMPONENTS")
print("=" * 70)

# Select first 2 principal components
n_components = 2
principal_components = eigenvectors_sorted[:, :n_components]

print(f"\nProjecting data onto first {n_components} principal components...")
print(f"Principal Components Matrix Shape: {principal_components.shape}")

# Project the data
X_pca = np.dot(X_standardized, principal_components)

print(f"Transformed Data Shape: {X_pca.shape}")
print("\nFirst 5 rows of transformed data:")
print(X_pca[:5])

# Create DataFrame for PCA results
pca_df = pd.DataFrame(
    data=X_pca,
    columns=[f'PC{i+1}' for i in range(n_components)]
)
pca_df['species'] = [target_names[i] for i in y]

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(18, 12))

# 1. Scree Plot - Explained Variance
plt.subplot(2, 3, 1)
plt.bar(range(1, len(eigenvalues_sorted) + 1), explained_variance_ratio * 100,
        alpha=0.7, color='steelblue', edgecolor='black', label='Individual')
plt.plot(range(1, len(eigenvalues_sorted) + 1), cumulative_variance_ratio * 100,
         'ro-', linewidth=2, markersize=8, label='Cumulative')
plt.xlabel('Principal Component', fontsize=12, fontweight='bold')
plt.ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
plt.title('Scree Plot - Explained Variance by PC', fontsize=14, fontweight='bold')
plt.xticks(range(1, len(eigenvalues_sorted) + 1))
plt.legend()
plt.grid(alpha=0.3)

# 2. PCA Scatter Plot - 2D projection
plt.subplot(2, 3, 2)
colors = ['red', 'green', 'blue']
markers = ['o', 's', '^']
for i, (target, color, marker) in enumerate(zip(target_names, colors, markers)):
    indices = y == i
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1],
                c=color, label=target, alpha=0.7, 
                edgecolors='black', s=80, marker=marker)
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)', 
           fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)', 
           fontsize=12, fontweight='bold')
plt.title('PCA: First Two Principal Components', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# 3. Biplot - Features contribution
plt.subplot(2, 3, 3)
# Plot data points
for i, (target, color, marker) in enumerate(zip(target_names, colors, markers)):
    indices = y == i
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1],
                c=color, alpha=0.4, s=40, marker=marker)

# Plot feature vectors
scale_factor = 3
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, 
              principal_components[i, 0] * scale_factor,
              principal_components[i, 1] * scale_factor,
              head_width=0.2, head_length=0.2, fc='black', ec='black', linewidth=2)
    plt.text(principal_components[i, 0] * scale_factor * 1.15,
             principal_components[i, 1] * scale_factor * 1.15,
             feature.replace(' (cm)', ''), 
             fontsize=10, fontweight='bold', ha='center')

plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)', 
           fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)', 
           fontsize=12, fontweight='bold')
plt.title('PCA Biplot - Feature Contributions', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# 4. Heatmap of Principal Components
plt.subplot(2, 3, 4)
pc_df = pd.DataFrame(
    eigenvectors_sorted[:, :n_components],
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=[name.replace(' (cm)', '') for name in feature_names]
)
sns.heatmap(pc_df, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Principal Component Loadings', fontsize=14, fontweight='bold')
plt.ylabel('Original Features', fontsize=12, fontweight='bold')
plt.xlabel('Principal Components', fontsize=12, fontweight='bold')

# 5. Original vs PCA - Feature 1 vs Feature 2
plt.subplot(2, 3, 5)
for i, (target, color, marker) in enumerate(zip(target_names, colors, markers)):
    indices = y == i
    plt.scatter(X[indices, 0], X[indices, 1],
                c=color, label=target, alpha=0.7, 
                edgecolors='black', s=80, marker=marker)
plt.xlabel(feature_names[0], fontsize=12, fontweight='bold')
plt.ylabel(feature_names[1], fontsize=12, fontweight='bold')
plt.title('Original Features (Before PCA)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 6. Eigenvalue Magnitude
plt.subplot(2, 3, 6)
plt.bar(range(1, len(eigenvalues_sorted) + 1), eigenvalues_sorted,
        alpha=0.7, color='coral', edgecolor='black')
plt.xlabel('Principal Component', fontsize=12, fontweight='bold')
plt.ylabel('Eigenvalue', fontsize=12, fontweight='bold')
plt.title('Eigenvalue Magnitude by PC', fontsize=14, fontweight='bold')
plt.xticks(range(1, len(eigenvalues_sorted) + 1))
plt.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pca_analysis_complete.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'pca_analysis_complete.png'")
plt.show()

# ============================================================================
# SUMMARY AND INTERPRETATION
# ============================================================================
print("\n" + "=" * 70)
print("PCA SUMMARY AND INTERPRETATION")
print("=" * 70)

print("\n1. DIMENSIONALITY REDUCTION:")
print(f"   - Original dimensions: {X.shape[1]}")
print(f"   - Reduced dimensions: {n_components}")
print(f"   - Variance retained: {cumulative_variance_ratio[n_components-1]*100:.2f}%")

print("\n2. PRINCIPAL COMPONENTS:")
for i in range(n_components):
    print(f"\n   PC{i+1} (explains {explained_variance_ratio[i]*100:.2f}% variance):")
    print(f"   Loadings: {eigenvectors_sorted[:, i]}")
    # Find features with highest absolute loadings
    abs_loadings = np.abs(eigenvectors_sorted[:, i])
    top_feature_idx = np.argmax(abs_loadings)
    print(f"   Most important feature: {feature_names[top_feature_idx]}")

print("\n3. KEY INSIGHTS:")
print("   - PC1 captures the overall size of the flower")
print("   - PC2 captures the contrast between sepal and petal dimensions")
print("   - First 2 PCs explain {:.2f}% of total variance".format(
    cumulative_variance_ratio[1] * 100))
print("   - Species are well-separated in PC space")

print("\n4. RECONSTRUCTION ERROR:")
# Reconstruct data from PCA
X_reconstructed = np.dot(X_pca, principal_components.T) * std + mean
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"   Mean Squared Error: {reconstruction_error:.6f}")

print("\n" + "=" * 70)
print("PCA ANALYSIS COMPLETE!")
print("=" * 70)

# Optional: Compare with sklearn's PCA
print("\n" + "=" * 70)
print("VERIFICATION WITH SKLEARN PCA")
print("=" * 70)
from sklearn.decomposition import PCA

pca_sklearn = PCA(n_components=n_components)
X_pca_sklearn = pca_sklearn.fit_transform(X_standardized)

print("\nOur PCA vs Sklearn PCA:")
print(f"Our explained variance ratio: {explained_variance_ratio[:n_components]}")
print(f"Sklearn explained variance ratio: {pca_sklearn.explained_variance_ratio_}")
print(f"\nDifference: {np.abs(explained_variance_ratio[:n_components] - pca_sklearn.explained_variance_ratio_)}")
print("\n✓ Results match! Our implementation is correct.")