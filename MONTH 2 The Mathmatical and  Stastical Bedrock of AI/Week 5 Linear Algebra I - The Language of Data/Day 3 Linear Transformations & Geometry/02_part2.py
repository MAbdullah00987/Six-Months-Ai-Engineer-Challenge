
#Part 2: The Unit Square - Our Canvas
#The unit square has corners at (0,0), (1,0), (1,1), (0,1).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Define the unit square
unit_square = np.array([
    [0, 1, 1, 0, 0],  # x coordinates
    [0, 0, 1, 1, 0]   # y coordinates
])

def plot_transformation(ax, matrix, title, color='blue'):
    """Plot original and transformed unit square"""
    
    # Apply transformation
    transformed = matrix @ unit_square
    
    # Calculate determinant
    det = np.linalg.det(matrix)
    
    # Plot original square
    ax.plot(unit_square[0], unit_square[1], 'k--', linewidth=2, label='Original', alpha=0.5)
    ax.fill(unit_square[0], unit_square[1], alpha=0.2, color='gray')
    
    # Plot transformed square
    ax.plot(transformed[0], transformed[1], color=color, linewidth=2.5, label='Transformed')
    ax.fill(transformed[0], transformed[1], alpha=0.3, color=color)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Add basis vectors
    origin = np.array([[0, 0], [0, 0]])
    
    # Original basis vectors
    ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, 
              color='red', alpha=0.5, width=0.006, label='i (original)')
    ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, 
              color='green', alpha=0.5, width=0.006, label='j (original)')
    
    # Transformed basis vectors
    new_i = matrix @ np.array([1, 0])
    new_j = matrix @ np.array([0, 1])
    ax.quiver(0, 0, new_i[0], new_i[1], angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.008, label='i (transformed)', linewidth=2)
    ax.quiver(0, 0, new_j[0], new_j[1], angles='xy', scale_units='xy', scale=1, 
              color='green', width=0.008, label='j (transformed)', linewidth=2)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{title}\nMatrix: {matrix.tolist()}\nDeterminant: {det:.3f}', 
                 fontsize=10, fontweight='bold')
    
    # Add interpretation
    interpretation = ""
    if abs(det) < 0.01:
        interpretation = "COLLAPSED to line/point!"
    elif det < 0:
        interpretation = f"Area scaled by {abs(det):.2f}x + FLIPPED"
    else:
        interpretation = f"Area scaled by {det:.2f}x"
    
    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Create figure with 5 transformations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()


# TRANSFORMATION 1: SCALING

scaling_matrix = np.array([
    [2, 0],
    [0, 1.5]
])
plot_transformation(axes[0], scaling_matrix, 'SCALING (Stretch)', 'blue')
axes[0].text(0.5, -2.5, 'Scales x by 2, y by 1.5\nPreserves shape orientation', 
             ha='center', fontsize=9, style='italic')


# TRANSFORMATION 2: ROTATION

angle = np.pi / 4  # 45 degrees
rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
plot_transformation(axes[1], rotation_matrix, 'ROTATION (45°)', 'green')
axes[1].text(0.5, -2.5, 'Rotates 45° counterclockwise\nPreserves area (det=1)', 
             ha='center', fontsize=9, style='italic')


# TRANSFORMATION 3: SHEAR

shear_matrix = np.array([
    [1, 0.5],
    [0, 1]
])
plot_transformation(axes[2], shear_matrix, 'SHEAR (Horizontal)', 'orange')
axes[2].text(0.5, -2.5, 'Skews horizontally\nPreserves area (det=1)', 
             ha='center', fontsize=9, style='italic')


# TRANSFORMATION 4: REFLECTION

reflection_matrix = np.array([
    [1, 0],
    [0, -1]
])
plot_transformation(axes[3], reflection_matrix, 'REFLECTION (over x-axis)', 'red')
axes[3].text(0.5, -2.5, 'Flips over x-axis\nNegative det → orientation reversed', 
             ha='center', fontsize=9, style='italic')


# TRANSFORMATION 5: PROJECTION (Singular)
projection_matrix = np.array([
    [1, 0],
    [0, 0]
])
plot_transformation(axes[4], projection_matrix, 'PROJECTION (onto x-axis)', 'purple')
axes[4].text(0.5, -2.5, 'Collapses to line\ndet=0 → loses dimension!', 
             ha='center', fontsize=9, style='italic')

# Remove extra subplot
fig.delaxes(axes[5])

plt.suptitle('Linear Transformations: How Matrices Transform Space', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

#Another Example 
#DETERMINANT GEOMETRIC INTERPRETATION


transformations = {
    'Scaling': scaling_matrix,
    'Rotation': rotation_matrix,
    'Shear': shear_matrix,
    'Reflection': reflection_matrix,
    'Projection': projection_matrix
}

for name, matrix in transformations.items():
    det = np.linalg.det(matrix)
    print(f"\n{name}:")
    print(f"  Matrix:\n{matrix}")
    print(f"  Determinant: {det:.4f}")
    
    if abs(det) < 1e-10:
        print(f"  → SINGULAR: Space collapses, no inverse exists")
        print(f"  → Information is LOST")
    elif det < 0:
        print(f"  → Area scaled by {abs(det):.4f}x")
        print(f"  → Orientation REVERSED (inside-out)")
    elif abs(det - 1) < 1e-10:
        print(f"  → Area PRESERVED (rigid motion)")
    else:
        print(f"  → Area scaled by {det:.4f}x")
        print(f"  → Orientation preserved")
    
    # Check if invertible
    if abs(det) > 1e-10:
        inv = np.linalg.inv(matrix)
        print(f"  → INVERTIBLE (transformation reversible)")
    else:
        print(f"  → NOT invertible (one-way transformation)")

"""
1. DETERMINANT = Area Scaling Factor
   - |det| tells you how much areas are multiplied
   - det = 0 means space collapses (dimension reduced)
   - det < 0 means orientation flips (mirroring)

2. BASIS VECTORS = Columns of Matrix
   - Column 1 = where î (1,0) goes
   - Column 2 = where ĵ (0,1) goes
   - These define the entire transformation!

3. INVERTIBILITY
   - det ≠ 0 → Matrix is invertible → Transformation reversible
   - det = 0 → Matrix is singular → Information lost forever

4. SPECIAL CASES
   - Rotation: det = 1 (preserves area and orientation)
   - Reflection: det = -1 (preserves area, flips orientation)
   - Projection: det = 0 (collapses dimension)
"""
