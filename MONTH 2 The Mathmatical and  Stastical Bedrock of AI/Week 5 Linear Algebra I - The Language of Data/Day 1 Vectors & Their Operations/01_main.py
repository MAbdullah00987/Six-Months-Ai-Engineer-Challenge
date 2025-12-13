#Day 1: Vectors & Their Operations
#Read: Mathematics for ML Chapter 2 (Sections 2.1-2.3) - Vectors, norms, inner products
#Watch: Coursera Imperial College LA Course - Week 1 Videos (Vectors introduction)
#Focus: Understanding vectors as data points, geometric vs algebraic view, dot products

#Project 1: Vector Operations from Scratch
#Implement: vector_add(), scalar_multiply(), dot_product()
#Test with various vector sizes
#Compare results with NumPy for verification

#Project 2: Data Representation
#Load a simple dataset (e.g., Iris with 3-4 samples)
#Represent each data point as a vector
#Visualize vectors in 2D/3D space

#Exercise Set:
#Calculate magnitudes and unit vectors for 5 different vectors
#Compute angles between vector pairs using dot product
#Practice: 10 problems from Chapter 2 exercises

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*60)
print("DAY 1: VECTORS & THEIR OPERATIONS")
print("="*60)

# ============================================================================
# PROJECT 1: VECTOR OPERATIONS FROM SCRATCH
# ============================================================================

print("\n" + "="*60)
print("PROJECT 1: VECTOR OPERATIONS FROM SCRATCH")
print("="*60)

def vector_add(v1, v2):
    """
    Add two vectors element-wise
    
    Parameters:
    v1, v2: lists or arrays representing vectors
    
    Returns:
    list: sum of vectors
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    result = [v1[i] + v2[i] for i in range(len(v1))]
    return result

def scalar_multiply(scalar, vector):
    """
    Multiply a vector by a scalar
    
    Parameters:
    scalar: number to multiply by
    vector: list or array
    
    Returns:
    list: scaled vector
    """
    result = [scalar * v for v in vector]
    return result

def dot_product(v1, v2):
    """
    Calculate dot product of two vectors
    
    Parameters:
    v1, v2: lists or arrays
    
    Returns:
    float: dot product
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    result = sum(v1[i] * v2[i] for i in range(len(v1)))
    return result

# Test with various vector sizes
print("\n1. Testing Vector Addition:")
v1 = [1, 2, 3]
v2 = [4, 5, 6]
custom_add = vector_add(v1, v2)
numpy_add = np.add(v1, v2)
print(f"   Vectors: {v1} + {v2}")
print(f"   Custom implementation: {custom_add}")
print(f"   NumPy result: {numpy_add.tolist()}")
print(f"   ✓ Match: {custom_add == numpy_add.tolist()}")

print("\n2. Testing Scalar Multiplication:")
scalar = 3
v = [1, 2, 3, 4]
custom_scalar = scalar_multiply(scalar, v)
numpy_scalar = (scalar * np.array(v)).tolist()
print(f"   Scalar: {scalar}, Vector: {v}")
print(f"   Custom implementation: {custom_scalar}")
print(f"   NumPy result: {numpy_scalar}")
print(f"   ✓ Match: {custom_scalar == numpy_scalar}")

print("\n3. Testing Dot Product:")
v1 = [1, 2, 3]
v2 = [4, 5, 6]
custom_dot = dot_product(v1, v2)
numpy_dot = np.dot(v1, v2)
print(f"   Vectors: {v1} · {v2}")
print(f"   Custom implementation: {custom_dot}")
print(f"   NumPy result: {numpy_dot}")
print(f"   ✓ Match: {custom_dot == numpy_dot}")

# ============================================================================
# EXERCISE SET: MAGNITUDES, UNIT VECTORS, AND ANGLES
# ============================================================================

print("\n" + "="*60)
print("EXERCISE SET: MAGNITUDES, UNIT VECTORS & ANGLES")
print("="*60)

def vector_magnitude(v):
    """Calculate the magnitude (length) of a vector"""
    return np.sqrt(np.sum(np.array(v)**2))

def unit_vector(v):
    """Calculate the unit vector (normalized vector)"""
    mag = vector_magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize zero vector")
    return np.array(v) / mag

def angle_between_vectors(v1, v2, degrees=True):
    """
    Calculate angle between two vectors using dot product
    Formula: cos(θ) = (v1·v2) / (|v1| * |v2|)
    """
    v1_array = np.array(v1)
    v2_array = np.array(v2)
    
    dot = np.dot(v1_array, v2_array)
    mag1 = vector_magnitude(v1_array)
    mag2 = vector_magnitude(v2_array)
    
    cos_angle = dot / (mag1 * mag2)
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1, 1)
    
    angle_rad = np.arccos(cos_angle)
    
    if degrees:
        return np.degrees(angle_rad)
    return angle_rad

# Define 5 different vectors for practice
vectors = {
    'v1': [1, 0, 0],
    'v2': [3, 4, 0],
    'v3': [1, 1, 1],
    'v4': [2, -1, 2],
    'v5': [0, 0, 5]
}

print("\n1. Calculating Magnitudes and Unit Vectors:")
print("-" * 60)

results = []
for name, vec in vectors.items():
    mag = vector_magnitude(vec)
    unit = unit_vector(vec)
    results.append({
        'Vector': name,
        'Original': str(vec),
        'Magnitude': f"{mag:.4f}",
        'Unit Vector': f"[{unit[0]:.4f}, {unit[1]:.4f}, {unit[2]:.4f}]"
    })
    print(f"{name} = {vec}")
    print(f"   Magnitude: {mag:.4f}")
    print(f"   Unit vector: [{unit[0]:.4f}, {unit[1]:.4f}, {unit[2]:.4f}]")
    print()

print("\n2. Computing Angles Between Vector Pairs:")
print("-" * 60)

# Calculate angles between all pairs
vector_pairs = [
    ('v1', 'v2'),
    ('v1', 'v3'),
    ('v2', 'v3'),
    ('v3', 'v4'),
    ('v4', 'v5')
]

angle_results = []
for v1_name, v2_name in vector_pairs:
    v1 = vectors[v1_name]
    v2 = vectors[v2_name]
    angle = angle_between_vectors(v1, v2)
    
    print(f"{v1_name} {v1} and {v2_name} {v2}")
    print(f"   Angle: {angle:.2f}°")
    print(f"   Dot product: {np.dot(v1, v2):.4f}")
    print()
    
    angle_results.append({
        'Vector 1': v1_name,
        'Vector 2': v2_name,
        'Angle (degrees)': f"{angle:.2f}°",
        'Dot Product': f"{np.dot(v1, v2):.4f}"
    })

# ============================================================================
# PROJECT 2: DATA REPRESENTATION
# ============================================================================

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

# ============================================================================
# 10 PRACTICE PROBLEMS FROM CHAPTER 2
# ============================================================================

print("\n" + "="*60)
print("10 PRACTICE PROBLEMS - CHAPTER 2 EXERCISES")
print("="*60)

print("\nProblem 1: Vector Addition")
print("-" * 60)
a = np.array([2, 3, 1])
b = np.array([1, -1, 4])
result = a + b
print(f"a = {a}, b = {b}")
print(f"a + b = {result}")

print("\nProblem 2: Scalar Multiplication")
print("-" * 60)
v = np.array([1, 2, 3])
k = 5
result = k * v
print(f"v = {v}, k = {k}")
print(f"{k}v = {result}")

print("\nProblem 3: Dot Product")
print("-" * 60)
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
dot = np.dot(x, y)
print(f"x = {x}, y = {y}")
print(f"x · y = {dot}")

print("\nProblem 4: Vector Magnitude")
print("-" * 60)
v = np.array([3, 4])
mag = np.linalg.norm(v)
print(f"v = {v}")
print(f"|v| = {mag}")

print("\nProblem 5: Unit Vector")
print("-" * 60)
v = np.array([3, 4, 0])
unit = v / np.linalg.norm(v)
print(f"v = {v}")
print(f"û = {unit}")
print(f"Verification |û| = {np.linalg.norm(unit):.10f}")

print("\nProblem 6: Angle Between Vectors")
print("-" * 60)
a = np.array([1, 0])
b = np.array([1, 1])
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.degrees(np.arccos(cos_theta))
print(f"a = {a}, b = {b}")
print(f"Angle = {theta:.2f}°")

print("\nProblem 7: Orthogonal Vectors")
print("-" * 60)
v1 = np.array([1, 2])
v2 = np.array([2, -1])
dot = np.dot(v1, v2)
print(f"v1 = {v1}, v2 = {v2}")
print(f"v1 · v2 = {dot}")
print(f"Are they orthogonal? {dot == 0}")

print("\nProblem 8: Linear Combination")
print("-" * 60)
v1 = np.array([1, 0, 1])
v2 = np.array([0, 1, 1])
c1, c2 = 2, 3
result = c1*v1 + c2*v2
print(f"v1 = {v1}, v2 = {v2}")
print(f"{c1}v1 + {c2}v2 = {result}")

print("\nProblem 9: Distance Between Points")
print("-" * 60)
p1 = np.array([1, 2, 3])
p2 = np.array([4, 6, 8])
distance = np.linalg.norm(p2 - p1)
print(f"p1 = {p1}, p2 = {p2}")
print(f"Distance = {distance:.4f}")

print("\nProblem 10: Projection of v onto u")
print("-" * 60)
v = np.array([3, 4])
u = np.array([1, 0])
proj = (np.dot(v, u) / np.dot(u, u)) * u
print(f"v = {v}, u = {u}")
print(f"proj_u(v) = {proj}")
print(f"Magnitude of projection = {np.linalg.norm(proj):.4f}")

