
#Covariance Matrix: Calculate and interpret the covariance matrix for a dataset (e.g., iris dataset).
#Visualize the correlation between features. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

print("=" * 60)
print("IRIS DATASET - COVARIANCE MATRIX ANALYSIS")
print("=" * 60)

# Display basic information about the dataset
print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

# Extract only numeric features for covariance calculation
numeric_features = df.iloc[:, :4]

# 1. Calculate Covariance Matrix using NumPy
print("\n" + "=" * 60)
print("COVARIANCE MATRIX (using NumPy)")
print("=" * 60)
cov_matrix_np = np.cov(numeric_features.T)
print(cov_matrix_np)

# 2. Calculate Covariance Matrix using Pandas
print("\n" + "=" * 60)
print("COVARIANCE MATRIX (using Pandas)")
print("=" * 60)
cov_matrix_pd = numeric_features.cov()
print(cov_matrix_pd)

# 3. Calculate Correlation Matrix
print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)
corr_matrix = numeric_features.corr()
print(corr_matrix)

# Interpretation
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("\n1. COVARIANCE INTERPRETATION:")
print("   - Positive covariance: Features increase together")
print("   - Negative covariance: One increases as other decreases")
print("   - Magnitude: Strength of linear relationship")

print("\n2. KEY FINDINGS:")
for i in range(len(iris.feature_names)):
    for j in range(i+1, len(iris.feature_names)):
        cov_val = cov_matrix_pd.iloc[i, j]
        corr_val = corr_matrix.iloc[i, j]
        print(f"\n   {iris.feature_names[i]} vs {iris.feature_names[j]}:")
        print(f"   - Covariance: {cov_val:.4f}")
        print(f"   - Correlation: {corr_val:.4f}")
        if abs(corr_val) > 0.8:
            print(f"   - Strong {'positive' if corr_val > 0 else 'negative'} relationship")
        elif abs(corr_val) > 0.5:
            print(f"   - Moderate {'positive' if corr_val > 0 else 'negative'} relationship")
        else:
            print(f"   - Weak relationship")

# Create visualizations
fig = plt.figure(figsize=(18, 12))

# 1. Heatmap of Covariance Matrix
plt.subplot(2, 3, 1)
sns.heatmap(cov_matrix_pd, annot=True, fmt='.3f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Covariance Matrix Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Features')

# 2. Heatmap of Correlation Matrix
plt.subplot(2, 3, 2)
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Features')

# 3. Pairplot showing relationships
plt.subplot(2, 3, 3)
plt.axis('off')
plt.text(0.5, 0.5, 'See separate pairplot below', 
         ha='center', va='center', fontsize=12)

# 4. Scatter plots for highly correlated features
plt.subplot(2, 3, 4)
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], 
           c=iris.target, cmap='viridis', alpha=0.6, edgecolors='black')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title(f'Petal Length vs Width\n(Correlation: {corr_matrix.iloc[2, 3]:.3f})', 
          fontweight='bold')
plt.colorbar(label='Species')
plt.grid(alpha=0.3)

# 5. Another scatter plot
plt.subplot(2, 3, 5)
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], 
           c=iris.target, cmap='viridis', alpha=0.6, edgecolors='black')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title(f'Sepal Length vs Petal Length\n(Correlation: {corr_matrix.iloc[0, 2]:.3f})', 
          fontweight='bold')
plt.colorbar(label='Species')
plt.grid(alpha=0.3)

# 6. Variance of each feature
plt.subplot(2, 3, 6)
variances = np.diag(cov_matrix_pd)
plt.bar(range(len(iris.feature_names)), variances, color='steelblue', edgecolor='black')
plt.xticks(range(len(iris.feature_names)), 
           [name.replace(' (cm)', '') for name in iris.feature_names], 
           rotation=45, ha='right')
plt.ylabel('Variance')
plt.title('Variance of Each Feature', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('covariance_analysis.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print("Figure saved as 'covariance_analysis.png'")
print("=" * 60)
plt.show()

# Create a detailed pairplot
print("\nGenerating detailed pairplot...")
pairplot_fig = sns.pairplot(df, hue='species', diag_kind='hist', 
                            plot_kws={'alpha': 0.6, 'edgecolor': 'black'},
                            height=2.5)
pairplot_fig.fig.suptitle('Iris Dataset: Pairwise Relationships', 
                          y=1.02, fontsize=16, fontweight='bold')
plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
print("Pairplot saved as 'iris_pairplot.png'")
plt.show()

# Calculate eigenvalues and eigenvectors
print("\n" + "=" * 60)
print("EIGENVALUES AND EIGENVECTORS (for PCA insights)")
print("=" * 60)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_np)
print("\nEigenvalues:")
for i, val in enumerate(eigenvalues):
    print(f"  λ{i+1}: {val:.4f} ({val/sum(eigenvalues)*100:.2f}% of variance)")

print("\nEigenvectors:")
print(eigenvectors)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nKey Takeaways:")
print("1. Petal measurements show strong positive correlation")
print("2. Sepal width has weaker correlation with other features")
print("3. Petal length has highest variance among features")
print("4. All petal and sepal length features are positively correlated")