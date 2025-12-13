# ============================================================================
# PROJECT 2: DATA REPRESENTATION
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


print("\n" + "="*60)
print("PROJECT 2: DATA REPRESENTATION WITH IRIS DATASET")
print("="*60)

# Create a small Iris dataset sample
iris_data = {
    'sepal_length': [5.1, 4.9, 6.2, 5.9],
    'sepal_width': [3.5, 3.0, 2.9, 3.0],
    'petal_length': [1.4, 1.4, 4.3, 4.2],
    'petal_width': [0.2, 0.2, 1.3, 1.5],
    'species': ['setosa', 'setosa', 'versicolor', 'versicolor']
}

df = pd.DataFrame(iris_data)
print("\n1. Iris Dataset Sample:")
print(df)

print("\n2. Each Data Point as a Vector:")
print("-" * 60)
for idx, row in df.iterrows():
    vector = row[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    print(f"Sample {idx} ({row['species']}): {vector}")
    print(f"   As vector: [{vector[0]:.1f}, {vector[1]:.1f}, {vector[2]:.1f}, {vector[3]:.1f}]")
    print(f"   Magnitude: {vector_magnitude(vector):.4f}")
    print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n3. Creating Visualizations...")

# Visualization 1: 2D Vector Space (using first 2 features)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: 2D representation (Sepal)
ax1 = axes[0, 0]
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    ax1.scatter(species_data['sepal_length'], 
                species_data['sepal_width'],
                label=species, s=200, alpha=0.6)
    
    # Draw vectors from origin
    for idx, row in species_data.iterrows():
        ax1.arrow(0, 0, row['sepal_length'], row['sepal_width'],
                  head_width=0.15, head_length=0.15, fc='gray', 
                  ec='gray', alpha=0.3, width=0.02)

ax1.set_xlabel('Sepal Length', fontsize=12)
ax1.set_ylabel('Sepal Width', fontsize=12)
ax1.set_title('2D Vector Representation (Sepal Features)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Plot 2: 2D representation (Petal)
ax2 = axes[0, 1]
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    ax2.scatter(species_data['petal_length'], 
                species_data['petal_width'],
                label=species, s=200, alpha=0.6)
    
    for idx, row in species_data.iterrows():
        ax2.arrow(0, 0, row['petal_length'], row['petal_width'],
                  head_width=0.15, head_length=0.15, fc='gray', 
                  ec='gray', alpha=0.3, width=0.02)

ax2.set_xlabel('Petal Length', fontsize=12)
ax2.set_ylabel('Petal Width', fontsize=12)
ax2.set_title('2D Vector Representation (Petal Features)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# Plot 3: Vector magnitudes
ax3 = axes[1, 0]
feature_vectors = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
magnitudes = [vector_magnitude(v) for v in feature_vectors]
colors = ['blue' if s == 'setosa' else 'orange' for s in df['species']]
ax3.bar(range(len(magnitudes)), magnitudes, color=colors, alpha=0.6)
ax3.set_xlabel('Sample Index', fontsize=12)
ax3.set_ylabel('Vector Magnitude', fontsize=12)
ax3.set_title('Vector Magnitudes for Each Data Point', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(magnitudes)))
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Feature correlation heatmap
ax4 = axes[1, 1]
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
correlation = df[feature_cols].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=ax4, cbar_kws={'label': 'Correlation'})
ax4.set_title('Feature Correlation Matrix\n(Related to Dot Products)', 
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('vector_operations_day1.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: vector_operations_day1.png")

# 3D Visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for species in df['species'].unique():
    species_data = df[df['species'] == species]
    ax.scatter(species_data['sepal_length'], 
               species_data['petal_length'],
               species_data['petal_width'],
               label=species, s=200, alpha=0.6)
    
    # Draw vectors from origin
    for idx, row in species_data.iterrows():
        ax.plot([0, row['sepal_length']], 
                [0, row['petal_length']], 
                [0, row['petal_width']],
                'gray', alpha=0.3, linewidth=1)

ax.set_xlabel('Sepal Length', fontsize=12)
ax.set_ylabel('Petal Length', fontsize=12)
ax.set_zlabel('Petal Width', fontsize=12)
ax.set_title('3D Vector Representation of Iris Data', fontsize=14, fontweight='bold')
ax.legend()
plt.savefig('vector_3d_representation.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: vector_3d_representation.png")

plt.show()
