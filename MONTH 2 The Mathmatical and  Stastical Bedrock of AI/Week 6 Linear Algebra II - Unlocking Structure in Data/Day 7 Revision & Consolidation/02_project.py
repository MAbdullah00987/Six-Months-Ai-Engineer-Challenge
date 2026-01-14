
# Project2:
# Principal Component Analysis (PCA) from Scratch: Implement PCA using NumPy's eigenvalue decomposition to perform
# dimensionality reduction on a dataset.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

class PCA_FromScratch:
    """Principal Component Analysis implemented from scratch"""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        """Fit PCA on the data"""
        # Step 1: Center the data (subtract mean)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Calculate covariance matrix
        # Cov = (1/n) * X^T * X
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        print("Covariance Matrix Shape:", cov_matrix.shape)
        print("Covariance Matrix:\n", np.round(cov_matrix, 3))
        print()
        
        # Step 3: Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Step 4: Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 5: Store the top n_components
        self.eigenvalues = eigenvalues
        self.components = eigenvectors[:, :self.n_components]
        
        # Calculate explained variance ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total_var
        
        print("Eigenvalues (sorted):", np.round(eigenvalues, 3))
        print("Explained Variance Ratio:", np.round(self.explained_variance_ratio, 3))
        print(f"\nTop {self.n_components} components explain "
              f"{np.sum(self.explained_variance_ratio[:self.n_components])*100:.2f}% of variance")
        
        return self
    
    def transform(self, X):
        """Transform data to principal component space"""
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """Reconstruct original data from principal components"""
        return (X_transformed @ self.components.T) + self.mean


# ============================================================================
# Load Dataset - Using Iris dataset (4D -> 2D)
# ============================================================================
print("="*70)
print("LOADING IRIS DATASET")
print("="*70)

data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

print(f"Original data shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Classes: {target_names}")
print()

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

print("First 5 rows of data:")
print(df.head())
print()


# Perform PCA from Scratch

print("PERFORMING PCA FROM SCRATCH")


pca = PCA_FromScratch(n_components=2)
X_pca = pca.fit_transform(X)

print(f"\nTransformed data shape: {X_pca.shape}")
print()


# Visualizations

fig = plt.figure(figsize=(18, 12))

# Plot 1: Original Data (first 2 features)
plt.subplot(3, 3, 1)
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], 
                label=target_name, alpha=0.7, s=50)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Original Data (First 2 Features)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: PCA Transformed Data
plt.subplot(3, 3, 2)
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                label=target_name, alpha=0.7, s=50)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Transformed Data (2D)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Explained Variance Ratio
plt.subplot(3, 3, 3)
n_features = len(pca.explained_variance_ratio)
plt.bar(range(1, n_features + 1), pca.explained_variance_ratio, 
        alpha=0.7, color='steelblue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot - Variance Explained')
plt.xticks(range(1, n_features + 1))
plt.grid(True, alpha=0.3, axis='y')

# Add cumulative variance line
cumsum = np.cumsum(pca.explained_variance_ratio)
plt.plot(range(1, n_features + 1), cumsum, 
         'ro-', linewidth=2, label='Cumulative')
plt.legend()

# Plot 4: Cumulative Explained Variance
plt.subplot(3, 3, 4)
plt.plot(range(1, n_features + 1), cumsum * 100, 
         'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Cumulative Variance Explained')
plt.grid(True, alpha=0.3)
plt.axhline(y=95, color='r', linestyle='--', label='95% threshold')
plt.legend()

# Plot 5: Principal Component Loadings (PC1)
plt.subplot(3, 3, 5)
loadings_pc1 = pca.components[:, 0]
plt.barh(feature_names, loadings_pc1, color='coral', alpha=0.7)
plt.xlabel('Loading Value')
plt.title('PC1 Loadings (Feature Contributions)')
plt.grid(True, alpha=0.3, axis='x')

# Plot 6: Principal Component Loadings (PC2)
plt.subplot(3, 3, 6)
loadings_pc2 = pca.components[:, 1]
plt.barh(feature_names, loadings_pc2, color='lightgreen', alpha=0.7)
plt.xlabel('Loading Value')
plt.title('PC2 Loadings (Feature Contributions)')
plt.grid(True, alpha=0.3, axis='x')

# Plot 7: Biplot (PCA with feature vectors)
plt.subplot(3, 3, 7)
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                label=target_name, alpha=0.5, s=50)

# Add feature vectors
scale = 3
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, 
              pca.components[i, 0] * scale, 
              pca.components[i, 1] * scale,
              head_width=0.1, head_length=0.1, 
              fc='red', ec='red', linewidth=2, alpha=0.7)
    plt.text(pca.components[i, 0] * scale * 1.15,
             pca.components[i, 1] * scale * 1.15,
             feature.split(' ')[0], 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Biplot (Data + Feature Vectors)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# Plot 8: Correlation Matrix of Original Data
plt.subplot(3, 3, 8)
corr_matrix = np.corrcoef(X.T)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
            xticklabels=[f.split(' ')[0] for f in feature_names],
            yticklabels=[f.split(' ')[0] for f in feature_names],
            cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Matrix')

# Plot 9: 3D PCA (if we have 3+ components)
if len(pca.explained_variance_ratio) >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(3, 3, 9, projection='3d')
    
    # Transform to 3D
    pca_3d = PCA_FromScratch(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)
    
    for i, target_name in enumerate(target_names):
        ax.scatter(X_pca_3d[y == i, 0], 
                   X_pca_3d[y == i, 1], 
                   X_pca_3d[y == i, 2],
                   label=target_name, alpha=0.7, s=50)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA Projection')
    ax.legend()

plt.tight_layout()
plt.show()


# Reconstruction Error Analysis
print("RECONSTRUCTION ERROR ANALYSIS")


X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)

print(f"Mean Squared Reconstruction Error: {reconstruction_error:.6f}")
print()

# Show original vs reconstructed for first sample
print("First sample comparison:")
print("Original:     ", np.round(X[0], 3))
print("Reconstructed:", np.round(X_reconstructed[0], 3))
print("Difference:   ", np.round(X[0] - X_reconstructed[0], 3))
print()

print("PCA SUMMARY")
print("="*70)
print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {pca.n_components}")
print(f"Variance retained: {np.sum(pca.explained_variance_ratio[:pca.n_components])*100:.2f}%")
print(f"Dimensionality reduction: {(1 - pca.n_components/X.shape[1])*100:.1f}%")
