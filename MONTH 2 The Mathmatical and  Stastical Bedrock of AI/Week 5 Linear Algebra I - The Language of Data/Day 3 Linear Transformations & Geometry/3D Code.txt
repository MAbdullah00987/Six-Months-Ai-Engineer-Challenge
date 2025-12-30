#Day 3: Linear Transformations & Geometry

#Read: Mathematics for ML Chapter 2 (Section 2.7) - Linear mappings
#Watch: Coursera Week 2-3 - Linear transformations
#Focus: How matrices transform space, determinants, special matrices

#Project 3: Geometric Transformations
#Create rotation matrix: R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]
#Create scaling matrix: S = [[sx, 0], [0, sy]]
#Create shearing matrix
#Apply to a square/triangle and plot before/after

#Project 6: Linear Transformation Visualizer
#Build interactive visualization showing grid transformation
#Display eigenvectors if time permits
#Show effect of different 2×2 matrices

#Exercise Set:
#Apply 5 different transformations to unit square
#Calculate determinants and interpret geometric meaning



#Part 1: Understanding Linear TransformationsLinear transformations are functions that map vectors from one space to another while preserving:
#Addition: T(u + v) = T(u) + T(v)
#Scalar multiplication: T(cv) = cT(v)
#Every linear transformation can be represented by a matrix, and matrix multiplication IS applying that transformation.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns

# PART 1: 3D TRANSFORMATIONS

# Create a 3D cube
def create_cube():
    """Create vertices of a unit cube"""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
    ]).T
    return vertices

# Define faces for visualization
faces = [
    [0, 1, 2, 3],  # bottom
    [4, 5, 6, 7],  # top
    [0, 1, 5, 4],  # front
    [2, 3, 7, 6],  # back
    [0, 3, 7, 4],  # left
    [1, 2, 6, 5]   # right
]

def plot_3d_transformation(ax, matrix, title, vertices):
    """Plot 3D transformation"""
    # Transform vertices
    transformed = matrix @ vertices
    
    # Create faces
    cube_faces = []
    for face in faces:
        cube_faces.append([transformed.T[face[i]] for i in range(4)])
    
    # Plot
    poly = Poly3DCollection(cube_faces, alpha=0.3, facecolor='cyan', 
                            edgecolor='black', linewidth=2)
    ax.add_collection3d(poly)
    
    # Plot vertices
    ax.scatter(transformed[0], transformed[1], transformed[2], 
               c='red', s=50, marker='o')
    
    # Plot basis vectors
    origin = np.zeros((3, 1))
    colors = ['red', 'green', 'blue']
    labels = ['x', 'y', 'z']
    
    for i in range(3):
        direction = matrix[:, i]
        ax.quiver(0, 0, 0, direction[0], direction[1], direction[2],
                 color=colors[i], arrow_length_ratio=0.1, linewidth=2,
                 label=f'{labels[i]}-axis')
    
    # Calculate determinant (volume scaling)
    det = np.linalg.det(matrix)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_title(f'{title}\ndet = {det:.3f} (volume × {abs(det):.2f})', 
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

# Create different 3D transformations
fig = plt.figure(figsize=(18, 12))

# Original cube
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
plot_3d_transformation(ax1, np.eye(3), 'Original Cube', create_cube())

# 3D Rotation around z-axis
angle = np.pi / 4
rotation_z = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
])
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
plot_3d_transformation(ax2, rotation_z, 'Rotation (Z-axis, 45°)', create_cube())

# 3D Scaling
scaling_3d = np.array([
    [2, 0, 0],
    [0, 1, 0],
    [0, 0, 0.5]
])
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
plot_3d_transformation(ax3, scaling_3d, 'Scaling (2x, 1x, 0.5x)', create_cube())

# 3D Shear
shear_3d = np.array([
    [1, 0.5, 0],
    [0, 1, 0],
    [0, 0, 1]
])
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
plot_3d_transformation(ax4, shear_3d, 'Shear (XY)', create_cube())

# Projection onto plane
projection_3d = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
plot_3d_transformation(ax5, projection_3d, 'Projection (onto XY)', create_cube())

# Complex transformation
complex_3d = rotation_z @ scaling_3d
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
plot_3d_transformation(ax6, complex_3d, 'Composite: Scale→Rotate', create_cube())

plt.tight_layout()
plt.show()

#3D DETERMINANT = VOLUME SCALING FACTOR
#Just like 2D det = area scaling, 3D det = volume scaling!"