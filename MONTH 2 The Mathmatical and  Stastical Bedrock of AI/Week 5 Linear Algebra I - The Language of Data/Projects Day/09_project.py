
#9: Data Representation: Take a small tabular dataset and represent it as a matrix (features) and a vector (target).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a small tabular dataset (house prices example)
data = {
    'Size_sqft': [1500, 1800, 2400, 2000, 1600],
    'Bedrooms': [3, 3, 4, 3, 2],
    'Age_years': [10, 15, 5, 8, 20],
    'Distance_km': [2.5, 3.0, 1.5, 2.0, 4.5],
    'Price_1000s': [300, 340, 450, 380, 280]  # Target variable
}

# Create DataFrame
df = pd.DataFrame(data)

print("Original Dataset:")
print(df)
print("\n" + "="*60 + "\n")

# Separate features (X) and target (y)
# Features matrix: all columns except the target
X = df.drop('Price_1000s', axis=1).values
# Target vector: the price column
y = df['Price_1000s'].values

print("Feature Matrix (X):")
print(X)
print(f"\nShape: {X.shape}")
print(f"Type: {type(X)}")
print("\n" + "="*60 + "\n")

print("Target Vector (y):")
print(y)
print(f"\nShape: {y.shape}")
print(f"Type: {type(y)}")
print("\n" + "="*60 + "\n")

# Display feature names
feature_names = df.drop('Price_1000s', axis=1).columns.tolist()
print(f"Feature names: {feature_names}")
print(f"Target name: Price_1000s")
print("\n" + "="*60 + "\n")

# Show some matrix operations
print("Matrix Operations Examples:")
print(f"Mean of each feature: {X.mean(axis=0)}")
print(f"Std of each feature: {X.std(axis=0)}")
print(f"Mean of target: {y.mean()}")
print(f"Std of target: {y.std()}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Feature Matrix and Target Vector Visualization', fontsize=16, fontweight='bold')

# 1. Heatmap of feature matrix
im = axes[0, 0].imshow(X, cmap='viridis', aspect='auto')
axes[0, 0].set_title('Feature Matrix (X) Heatmap')
axes[0, 0].set_xlabel('Feature Index')
axes[0, 0].set_ylabel('Sample Index')
axes[0, 0].set_xticks(range(len(feature_names)))
axes[0, 0].set_xticklabels(feature_names, rotation=45, ha='right')
plt.colorbar(im, ax=axes[0, 0])

# 2. Target vector bar plot
axes[0, 1].bar(range(len(y)), y, color='coral', edgecolor='black')
axes[0, 1].set_title('Target Vector (y)')
axes[0, 1].set_xlabel('Sample Index')
axes[0, 1].set_ylabel('Price (1000s)')
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Feature correlation with target
correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
axes[1, 0].barh(feature_names, correlations, color='steelblue', edgecolor='black')
axes[1, 0].set_title('Feature Correlation with Target')
axes[1, 0].set_xlabel('Correlation Coefficient')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=1)
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Matrix dimensions visualization
axes[1, 1].axis('off')
info_text = f"""
Dataset Representation:

Feature Matrix (X):
  Shape: {X.shape}
  Rows (samples): {X.shape[0]}
  Columns (features): {X.shape[1]}
  
Target Vector (y):
  Shape: {y.shape}
  Length: {len(y)}

Features: {', '.join(feature_names)}
Target: Price_1000s

Matrix-Vector Relationship:
  Each row in X corresponds to
  one element in y
"""
axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Additional: Show how to access elements
print("\nAccessing Elements:")
print(f"First sample features: {X[0]}")
print(f"First sample target: {y[0]}")
print(f"Second feature (all samples): {X[:, 1]}")
print(f"\nMatrix multiplication example (X^T @ X):")
print(X.T @ X)