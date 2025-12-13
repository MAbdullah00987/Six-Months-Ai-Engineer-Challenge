# ============================================================================
# PROJECT 1: VECTOR OPERATIONS FROM SCRATCH
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



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