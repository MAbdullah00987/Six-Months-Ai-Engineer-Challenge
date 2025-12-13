
#3: Geometric Transformations: Use matrices in NumPy to apply 2D transformations (rotation, scaling, shearing) to a set of points and plot the results.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a set of points forming a shape (square)
def create_square():
    """Create points for a square"""
    points = np.array([
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2],
        [0, 0]  # Close the shape
    ])
    return points

# Rotation matrix
def rotation_matrix(angle_degrees):
    """Create a 2D rotation matrix"""
    theta = np.radians(angle_degrees)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

# Scaling matrix
def scaling_matrix(sx, sy):
    """Create a 2D scaling matrix"""
    return np.array([
        [sx, 0],
        [0, sy]
    ])

# Shearing matrix
def shearing_matrix(shx, shy):
    """Create a 2D shearing matrix"""
    return np.array([
        [1, shx],
        [shy, 1]
    ])

# Apply transformation
def apply_transformation(points, transformation_matrix):
    """Apply transformation matrix to points"""
    # Transpose for matrix multiplication, then transpose back
    transformed = (transformation_matrix @ points.T).T
    return transformed

# Create original points
original_points = create_square()

# Apply transformations
rotated_45 = apply_transformation(original_points, rotation_matrix(45))
scaled = apply_transformation(original_points, scaling_matrix(1.5, 0.5))
sheared = apply_transformation(original_points, shearing_matrix(0.5, 0.3))

# Combined transformation (rotate then scale)
combined_matrix = scaling_matrix(1.2, 1.2) @ rotation_matrix(30)
combined = apply_transformation(original_points, combined_matrix)

# Create DataFrame to display transformation matrices
print("=" * 60)
print("TRANSFORMATION MATRICES")
print("=" * 60)

print("\n1. Rotation Matrix (45 degrees):")
print(pd.DataFrame(rotation_matrix(45), 
                   columns=['x', 'y'], 
                   index=['x\'', 'y\'']))

print("\n2. Scaling Matrix (sx=1.5, sy=0.5):")
print(pd.DataFrame(scaling_matrix(1.5, 0.5), 
                   columns=['x', 'y'], 
                   index=['x\'', 'y\'']))

print("\n3. Shearing Matrix (shx=0.5, shy=0.3):")
print(pd.DataFrame(shearing_matrix(0.5, 0.3), 
                   columns=['x', 'y'], 
                   index=['x\'', 'y\'']))

# Display points in DataFrame
print("\n" + "=" * 60)
print("ORIGINAL AND TRANSFORMED POINTS")
print("=" * 60)

points_df = pd.DataFrame({
    'Original_X': original_points[:, 0],
    'Original_Y': original_points[:, 1],
    'Rotated_X': rotated_45[:, 0],
    'Rotated_Y': rotated_45[:, 1],
    'Scaled_X': scaled[:, 0],
    'Scaled_Y': scaled[:, 1],
    'Sheared_X': sheared[:, 0],
    'Sheared_Y': sheared[:, 1]
})

print("\n", points_df.round(3))

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('2D Geometric Transformations', fontsize=16, fontweight='bold')

# Function to plot shape
def plot_shape(ax, original, transformed, title, color='blue'):
    ax.plot(original[:, 0], original[:, 1], 'ko-', 
            linewidth=2, markersize=8, label='Original', alpha=0.5)
    ax.plot(transformed[:, 0], transformed[:, 1], 
            color=color, marker='o', linewidth=2, 
            markersize=8, label='Transformed')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Plot transformations
plot_shape(axes[0, 0], original_points, original_points, 
           'Original Shape', 'black')

plot_shape(axes[0, 1], original_points, rotated_45, 
           'Rotation (45°)', 'red')

plot_shape(axes[0, 2], original_points, scaled, 
           'Scaling (sx=1.5, sy=0.5)', 'green')

plot_shape(axes[1, 0], original_points, sheared, 
           'Shearing (shx=0.5, shy=0.3)', 'purple')

plot_shape(axes[1, 1], original_points, combined, 
           'Combined (Rotate 30° + Scale 1.2)', 'orange')

# All transformations together
axes[1, 2].plot(original_points[:, 0], original_points[:, 1], 
                'ko-', linewidth=2, label='Original', alpha=0.5)
axes[1, 2].plot(rotated_45[:, 0], rotated_45[:, 1], 
                'r-', linewidth=2, label='Rotated', alpha=0.7)
axes[1, 2].plot(scaled[:, 0], scaled[:, 1], 
                'g-', linewidth=2, label='Scaled', alpha=0.7)
axes[1, 2].plot(sheared[:, 0], sheared[:, 1], 
                'purple', linewidth=2, label='Sheared', alpha=0.7)
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_aspect('equal')
axes[1, 2].legend()
axes[1, 2].set_title('All Transformations', fontweight='bold')
axes[1, 2].set_xlabel('X')
axes[1, 2].set_ylabel('Y')

plt.tight_layout()
plt.show()

# Additional example: Transform a more complex shape
print("\n" + "=" * 60)
print("BONUS: Transforming a House Shape")
print("=" * 60)

# Create a house shape
house = np.array([
    [0, 0], [4, 0], [4, 3], [0, 3], [0, 0],  # Base
    [0, 3], [2, 5], [4, 3]  # Roof
])

# Apply rotation to house
rotated_house = apply_transformation(house, rotation_matrix(30))

# Plot house transformation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('House Transformation Example', fontsize=14, fontweight='bold')

ax1.plot(house[:, 0], house[:, 1], 'bo-', linewidth=2, markersize=8)
ax1.set_title('Original House')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

ax2.plot(house[:, 0], house[:, 1], 'bo-', 
         linewidth=2, markersize=8, label='Original', alpha=0.3)
ax2.plot(rotated_house[:, 0], rotated_house[:, 1], 
         'ro-', linewidth=2, markersize=8, label='Rotated 30°')
ax2.set_title('Rotated House (30°)')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.legend()
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.tight_layout()
plt.show()
