#Project 1 (Complete): Data Representation
#Load full dataset (100+ samples, 4-5 features)
#Represent as feature matrix X and target vector y
#Compute X^T X (Gram matrix)
#alculate column-wise means using matrix operations
#Standardize features using broadcasting
#Compute correlation matrix
#Visualize data matrix as heatmap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

# Set random seed for reproducibility
np.random.seed(42)

print("PROJECT 9: DATA REPRESENTATION")



# STEP 1: Load Full Dataset (100+ samples, 4-5 features)

print("\n[STEP 1] Loading Dataset...")

# Using Iris dataset (150 samples, 4 features)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f"Dataset Shape: {df.shape}")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(data.feature_names)}")
print(f"\nFeature names: {data.feature_names}")
print(f"\nFirst 5 rows:")
print(df.head())

# STEP 2: Represent as Feature Matrix X and Target Vector y

print("Creating Feature Matrix X and Target Vector y")


X = data.data  # Feature matrix (n_samples x n_features)
y = data.target  # Target vector (n_samples,)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nX (first 3 samples):\n{X[:3]}")
print(f"\ny (first 10 targets): {y[:10]}")

#Compute X^T X (Gram Matrix)

print("\n" + "=" * 60)
print("[STEP 3] Computing Gram Matrix (X^T X)")
print("=" * 60)

# Gram matrix represents relationships between features
XTX = X.T @ X  # Using @ operator for matrix multiplication
# Alternative: XTX = np.dot(X.T, X)

print(f"Gram Matrix shape: {XTX.shape}")
print(f"\nGram Matrix (X^T X):\n{XTX}")
print(f"\nGram Matrix represents feature covariances (unnormalized)")


# STEP 4: Calculate Column-wise Means Using Matrix Operations

print("Computing Column-wise Means")


# Method 1: Using NumPy's mean function
means_np = np.mean(X, axis=0)

# Method 2: Using matrix operations (manual calculation)
n_samples = X.shape[0]
ones_vector = np.ones((n_samples, 1))
means_manual = (ones_vector.T @ X) / n_samples
means_manual = means_manual.flatten()

print(f"Column-wise means (numpy): {means_np}")
print(f"Column-wise means (matrix ops): {means_manual}")
print(f"Are they equal? {np.allclose(means_np, means_manual)}")


# STEP 5: Standardize Features Using Broadcasting

print("Standardizing Features")


# Compute standard deviations
stds = np.std(X, axis=0)

print(f"Standard deviations: {stds}")

# Standardize: Z = (X - mean) / std using broadcasting
X_standardized = (X - means_np) / stds

print(f"\nOriginal X (first 3 samples):\n{X[:3]}")
print(f"\nStandardized X (first 3 samples):\n{X_standardized[:3]}")

# Verify standardization
new_means = np.mean(X_standardized, axis=0)
new_stds = np.std(X_standardized, axis=0)
print(f"\nMeans after standardization (should be ~0): {new_means}")
print(f"Stds after standardization (should be ~1): {new_stds}")


# STEP 6: Compute Correlation Matrix

print("Computing Correlation Matrix")


# Method 1: Using NumPy
corr_matrix_np = np.corrcoef(X.T)

# Method 2: Manual calculation using standardized data
corr_matrix_manual = (X_standardized.T @ X_standardized) / (n_samples - 1)

print(f"Correlation Matrix (NumPy):\n{corr_matrix_np}")
print(f"\nCorrelation Matrix (Manual):\n{corr_matrix_manual}")

# Create DataFrame for better visualization
corr_df = pd.DataFrame(corr_matrix_np, 
                       columns=data.feature_names,
                       index=data.feature_names)
print(f"\nCorrelation Matrix (as DataFrame):")
print(corr_df)


# STEP 7: Visualize Data Matrix as Heatmap


print("Creating Visualizations")


# Create a figure with multiple subplots
fig = plt.figure(figsize=(18, 12))

# 1. Original Data Heatmap
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(X[:50], cmap='viridis', cbar_kws={'label': 'Value'}, ax=ax1)
ax1.set_title('Original Data Matrix (First 50 Samples)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Features')
ax1.set_ylabel('Samples')

# 2. Standardized Data Heatmap
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(X_standardized[:50], cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Standardized Value'}, ax=ax2)
ax2.set_title('Standardized Data Matrix (First 50 Samples)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Features')
ax2.set_ylabel('Samples')

# 3. Correlation Matrix Heatmap
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, cbar_kws={'label': 'Correlation'}, ax=ax3)
ax3.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

# 4. Gram Matrix Heatmap
ax4 = plt.subplot(2, 3, 4)
gram_df = pd.DataFrame(XTX, columns=data.feature_names, index=data.feature_names)
sns.heatmap(gram_df, annot=True, fmt='.1f', cmap='YlOrRd',
            square=True, linewidths=1, cbar_kws={'label': 'Value'}, ax=ax4)
ax4.set_title('Gram Matrix (X^T X)', fontsize=12, fontweight='bold')

# 5. Feature Distributions (Before Standardization)
ax5 = plt.subplot(2, 3, 5)
for i, feature in enumerate(data.feature_names):
    ax5.hist(X[:, i], alpha=0.5, label=feature, bins=20)
ax5.set_title('Feature Distributions (Original)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Value')
ax5.set_ylabel('Frequency')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# 6. Feature Distributions (After Standardization)
ax6 = plt.subplot(2, 3, 6)
for i, feature in enumerate(data.feature_names):
    ax6.hist(X_standardized[:, i], alpha=0.5, label=feature, bins=20)
ax6.set_title('Feature Distributions (Standardized)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Standardized Value')
ax6.set_ylabel('Frequency')
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data_representation_project.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'data_representation_project.png'")
plt.show()





#Important Points use in the project
#1. Loaded dataset with", X.shape[0], "samples and", X.shape[1], "features")
#2. Computed Gram matrix showing feature relationships")
#3. Calculated means using matrix operations")
#4. Standardized features to zero mean and unit variance")
#5. Computed correlation matrix showing feature correlations")
#6. Visualized all matrices as heatmaps")
