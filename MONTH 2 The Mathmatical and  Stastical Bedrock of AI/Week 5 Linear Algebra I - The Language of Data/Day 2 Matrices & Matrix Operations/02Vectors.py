import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
# Creating vectors
vector_1d = np.array([1, 2, 3, 4])
vector_2d = np.array([[1], [2], [3], [4]])  # Column vector
vector_row = np.array([[1, 2, 3, 4]])  # Row vector
print(f"1D Vector: {vector_1d}")
print(f"Shape: {vector_1d.shape}")
'''

'''
#Norms (Vector Magnitude)
# Norms measure the "size" or "length" of vectors.

# Different types of norms
vector = np.array([3, 4])

# L1 Norm (Manhattan distance)
l1_norm = np.linalg.norm(vector, ord=1)
print(f"L1 Norm: {l1_norm}")  # |3| + |4| = 7

# L2 Norm (Euclidean distance)
l2_norm = np.linalg.norm(vector, ord=2)
print(f"L2 Norm: {l2_norm}")  # sqrt(3² + 4²) = 5.0

# Infinity Norm (Maximum absolute value)
inf_norm = np.linalg.norm(vector, ord=np.inf)
print(f"Infinity Norm: {inf_norm}")  # max(|3|, |4|) = 4
'''

'''
#Inner Products (Dot Products)
# Measures similarity and projection between vectors.

# Inner product / Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Multiple ways to compute
dot_product_1 = np.dot(a, b)
dot_product_2 = a @ b  # Matrix multiplication operator
dot_product_3 = np.inner(a, b)

print(f"Dot Product: {dot_product_1}")  # 1*4 + 2*5 + 3*6 = 32
'''

'''
#Practical Applications with NumPyA.
#  Vector Operations for Data Analysis

# Simulating real-world data
prices = np.array([100, 150, 200, 175, 225])
quantities = np.array([10, 8, 12, 15, 9])

# Total revenue using dot product
total_revenue = np.dot(prices, quantities)
print(f"Total Revenue: ${total_revenue}")

# Normalize prices (using L2 norm)
normalized_prices = prices / np.linalg.norm(prices)
print(f"Normalized Prices: {normalized_prices}")

# Distance between price vectors (comparing two time periods)
prices_week1 = np.array([100, 150, 200])
prices_week2 = np.array([105, 148, 210])

price_change = np.linalg.norm(prices_week2 - prices_week1)
print(f"Price Change Magnitude: {price_change:.2f}")
'''


#B.Similarity and Distance Calculations
#Cosine similarity (used in recommendation systems)
def cosine_similarity(v1, v2):
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_prod / (norm_v1 * norm_v2)

# User preferences
user1_ratings = np.array([5, 4, 0, 0, 3])  # Movie ratings
user2_ratings = np.array([4, 5, 0, 0, 4])
user3_ratings = np.array([0, 0, 5, 4, 0])

similarity_1_2 = cosine_similarity(user1_ratings, user2_ratings)
similarity_1_3 = cosine_similarity(user1_ratings, user3_ratings)

print(f"User 1 & 2 Similarity: {similarity_1_2:.3f}")
print(f"User 1 & 3 Similarity: {similarity_1_3:.3f}")

# Create a similarity matrix
products = ['Product A', 'Product B', 'Product C', 'Product D']
features = np.array([
    [1, 2, 3, 4, 5],  # Product A features
    [1, 2, 3, 3, 4],  # Product B features
    [5, 4, 3, 2, 1],  # Product C features
    [1, 1, 3, 4, 5]   # Product D features
])

# Compute similarity matrix
n = len(products)
similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = cosine_similarity(features[i], features[j])

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=products, yticklabels=products, 
            cbar_kws={'label': 'Cosine Similarity'})
plt.title('Product Similarity Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


'''

'''
#C. Matrix Operations and Transformations

# Creating transformation matrices
rotation_matrix = np.array([
    [np.cos(np.pi/4), -np.sin(np.pi/4)],
    [np.sin(np.pi/4), np.cos(np.pi/4)]
])

# Original vector
point = np.array([1, 0])

# Rotate the point
rotated_point = rotation_matrix @ point
print(f"Original: {point}")
print(f"Rotated 45°: {rotated_point}")



sns.set_style("whitegrid")

# Visualizing vectors in 2D space
fig, ax = plt.subplots(figsize=(10, 8))

# Define vectors
vectors = {
    'v1': np.array([3, 2]),
    'v2': np.array([1, 4]),
    'v1+v2': np.array([4, 6])
}

colors = ['red', 'blue', 'green']

for (name, vec), color in zip(vectors.items(), colors):
    ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', 
              scale=1, color=color, width=0.008, label=name)
    
    # Show L2 norm
    norm = np.linalg.norm(vec)
    ax.text(vec[0]/2, vec[1]/2, f'||{name}|| = {norm:.2f}', 
            fontsize=10, color=color)

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 7)
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.set_title('Vector Addition and Norms', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()
'''

'''
#B. Visualizing Inner Products and Angles
# Angle between vectors using dot product
def angle_between_vectors(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle_rad)

# Create multiple vectors
angles_data = []
for i in range(0, 181, 20):
    rad = np.radians(i)
    v2 = np.array([np.cos(rad), np.sin(rad)])
    v1 = np.array([1, 0])
    
    dot_prod = np.dot(v1, v2)
    angles_data.append({'Angle': i, 'Dot Product': dot_prod})

import pandas as pd
df = pd.DataFrame(angles_data)

# Visualize relationship
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df, x='Angle', y='Dot Product', marker='o', linewidth=2.5)
ax.set_title('Dot Product vs Angle Between Unit Vectors', fontsize=14, fontweight='bold')
ax.set_xlabel('Angle (degrees)', fontsize=12)
ax.set_ylabel('Dot Product Value', fontsize=12)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Orthogonal (90°)')
ax.legend()
plt.tight_layout()
plt.show()
'''
'''
#Advanced ApplicationsA. Principal Component Analysis (PCA) Intuition
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
data = np.random.randn(100, 5)  # 100 samples, 5 features

# Apply PCA
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(data)

# Explained variance (using norms of principal components)
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(10, 5))

# Plot 1: Original data projection
plt.subplot(1, 2, 1)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.6, c='blue')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Projection')
plt.grid(True, alpha=0.3)

# Plot 2: Explained variance
plt.subplot(1, 2, 2)
plt.bar(range(1, len(explained_var) + 1), explained_var, color='green', alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance Explained by Components')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''

'''
#B. Distance Metrics Comparison

# Compare different distance metrics
point1 = np.array([1, 2, 3])
point2 = np.array([4, 6, 8])

distances = {
    'Euclidean (L2)': np.linalg.norm(point1 - point2, ord=2),
    'Manhattan (L1)': np.linalg.norm(point1 - point2, ord=1),
    'Chebyshev (L∞)': np.linalg.norm(point1 - point2, ord=np.inf),
}

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(distances.keys(), distances.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_ylabel('Distance Value', fontsize=12)
ax.set_title('Comparison of Distance Metrics', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()


#5. Best Practices for Stronger Logic
#1. Use element-wise operations instead of nested loops
#2. Utilize vectorized operations for performance
#3. Use NumPy's built-in functions for common operations
#4. Use broadcasting for element-wise operations
#5. Use vectorized operations for performance

'''
#Vectorization Over Loops

# Slow - Using loops
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = 0
for i in range(len(a)):
    result += a[i] * b[i]

#Fast - Vectorized
result = np.dot(a, b)
'''

'''
#Broadcasting for Efficiency
# Normalize multiple vectors at once
vectors = np.random.rand(1000, 50)  # 1000 vectors of dimension 50
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
normalized = vectors / norms  # Broadcasting!
'''

'''
#Memory-Efficient Operations
# Use in-place operations when possible
large_array = np.random.rand(10000, 100)
large_array /= np.linalg.norm(large_array, axis=1, keepdims=True)  # In-place
'''


#Exercise 1: Build a Recommendation Systempython
def recommend_items(user_vector, item_matrix, top_n=3):
    """
    Recommend items based on cosine similarity
    """
    similarities = np.array([
        cosine_similarity(user_vector, item) 
        for item in item_matrix
    ])
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return top_indices, similarities[top_indices]

# Test
user_pref = np.array([5, 3, 0, 1])
items = np.array([
    [5, 4, 0, 1],
    [0, 0, 5, 4],
    [4, 3, 1, 2],
    [5, 5, 0, 0]
])

recommendations, scores = recommend_items(user_pref, items)
print(f"Recommended items: {recommendations}")
print(f"Similarity scores: {scores}")

#Key Takeaways
#Vectors → Data representation, feature engineering
#Norms → Measuring magnitude, normalization, regularization
#Inner Products → Similarity, projections, transformations