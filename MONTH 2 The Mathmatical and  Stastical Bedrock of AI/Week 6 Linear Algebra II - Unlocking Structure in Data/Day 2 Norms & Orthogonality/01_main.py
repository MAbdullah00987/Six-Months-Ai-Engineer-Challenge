
#Day 2: Norms & Orthogonality
#Goal: Learn different ways to measure vectors and understand perpendicularity.

#Mathematics for Machine Learning, Chapter 4: Sections on norms and inner products
#BITS Pilani Course: Lectures on vector norms and orthogonality

#Check for Orthogonality: Write a function that checks if the columns of a matrix are orthogonal (using dot products). Visualize orthogonal vs non-orthogonal vectors.
#Linear Independence Check: Use matrix rank to determine if a set of vectors is linearly independent. Create test cases with dependent and independent vectors.

#Key Concepts to Master
#L1, L2, and infinity norms
#Orthogonal and orthonormal vectors
#Linear independence vs dependence
#Matrix rank

#Vector Norms (Measuring Vectors)
#A vector norm is a function that assigns a non-negative length or size to a vector. 
#In data science, norms are primarily used as distance metrics to quantify how "far" apart 
#two data points (vectors) are, which is fundamental to algorithms like clustering, regression,
# and regularization.

#Key Norms and their Use in NumPy

#Name	                     Formula (for vector x)   	NumPy Implementation	Logic Strengthening Use Case
#L2 Norm (Euclidean Norm)	 $\sqrt{\sum_{i=1}^{n}	    x_i	                        ^2}$ 
#L1 Norm (Manhattan Norm)	 $\sum_{i=1}^{n}            x_i	                          $
#L-infinity Norm (Max Norm)	 $\max_{i}	                x_i	                          $  

#Python/NumPy Logic Example (Feature Scaling)
#When you scale data, you often want to normalize it so all features have a unit norm (a magnitude of 1), which prevents features with naturally larger values from dominating a model.


import numpy as np
'''
# A sample data point (e.g., age, income, spending)
vector_a = np.array([25, 50000, 1500])

# 1. Calculate the L2 Norm (magnitude)
l2_norm = np.linalg.norm(vector_a, 2)
print(f"L2 Norm: {l2_norm:.2f}")

# 2. Normalize the vector (Unit Vector)
# Dividing the vector by its norm scales it to length 1.
unit_vector = vector_a / l2_norm
print(f"Unit Vector: {unit_vector}")
print(f"New L2 Norm (should be 1): {np.linalg.norm(unit_vector, 2):.4f}")
'''
'''
#2. Inner Products and Orthogonality
#Inner Product
#The inner product (or dot product) of two vectors, 
#In data science, the inner product is directly related to similarity and correlation:
#A large positive inner product means the vectors point in the same general direction (highly similar/correlated).
#A large negative inner product means they point in opposite directions (highly dissimilar/negatively correlated).
#Orthogonality
#Two vectors, 
#Orthogonality is critical because it means the two vectors convey completely independent information—changing one does not affect the projection of the other.
#Python/NumPy Logic Example (Checking Independence)
#In Principal Component Analysis (PCA), the goal is to find a new set of dimensions (principal components) that are orthogonal to each other, thus capturing the maximum variance with minimal redundant information.


# Two vectors (representing two features in a dataset)
v1 = np.array([3, 4])
v2 = np.array([-4, 3])
v3 = np.array([1, 1])

# Calculate the inner product (dot product)
dot_product_1_2 = np.dot(v1, v2)
dot_product_1_3 = np.dot(v1, v3)

print(f"Dot Product v1 and v2: {dot_product_1_2}")
print(f"Are v1 and v2 orthogonal? {dot_product_1_2 == 0}")

print(f"\nDot Product v1 and v3: {dot_product_1_3}")
print(f"Are v1 and v3 orthogonal? {dot_product_1_3 == 0}")
'''

