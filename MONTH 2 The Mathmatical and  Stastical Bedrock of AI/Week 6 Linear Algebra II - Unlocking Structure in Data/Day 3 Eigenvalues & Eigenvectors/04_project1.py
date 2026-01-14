#Project 1: Geometric Transformations

#Create rotation matrix: R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]
#Create scaling matrix: S = [[sx, 0], [0, sy]]
#Create shearing matrix
#Apply to a square/triangle and plot before/after


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("="*80)
print("PROJECT 3: GEOMETRIC TRANSFORMATIONS")
print("="*80)


# PART 1: DEFINE TRANSFORMATION MATRICES

print("\n" + "="*80)
print("PART 1: TRANSFORMATION MATRIX DEFINITIONS")
print("="*80)

def rotation_matrix(theta):
    """
    Create rotation matrix for angle theta (in radians)
    R(θ) = [[cos θ, -sin θ],
            [sin θ,  cos θ]]
    
    Positive theta -> counterclockwise rotation
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def scaling_matrix(sx, sy):
    """
    Create scaling matrix
    S = [[sx,  0],
         [0,  sy]]
    
    sx: scale factor in x direction
    sy: scale factor in y direction
    """
    return np.array([
        [sx, 0],
        [0, sy]
    ])

def shearing_matrix(shx=0, shy=0):
    """
    Create shearing matrix
    
    Horizontal shear (x direction):
    Sh_x = [[1, shx],
            [0,  1 ]]
    
    Vertical shear (y direction):
    Sh_y = [[1,   0],
            [shy, 1]]
    
    Combined:
    Sh = [[1,   shx],
          [shy,  1 ]]
    """
    return np.array([
        [1, shx],
        [shy, 1]
    ])

def reflection_matrix(axis='x'):
    """
    Create reflection matrix
    - 'x': reflect over x-axis
    - 'y': reflect over y-axis
    - 'origin': reflect through origin
    """
    if axis == 'x':
        return np.array([[1, 0], [0, -1]])
    elif axis == 'y':
        return np.array([[-1, 0], [0, 1]])
    elif axis == 'origin':
        return np.array([[-1, 0], [0, -1]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'origin'")

def translation_vector(tx, ty):
    """
    Translation is NOT a linear transformation, but we include it
    for completeness. For homogeneous coordinates:
    T = [[1, 0, tx],
         [0, 1, ty],
         [0, 0, 1 ]]
    """
    return np.array([tx, ty])

# Display the matrices
print("\n1. ROTATION MATRIX (45° counterclockwise):")
theta = np.pi / 4  # 45 degrees
R = rotation_matrix(theta)
print(R)
print(f"   Angle: {np.degrees(theta)}°")
print(f"   Determinant: {np.linalg.det(R):.4f} (preserves area)")

print("\n2. SCALING MATRIX (2x in x, 0.5x in y):")
S = scaling_matrix(2, 0.5)
print(S)
print(f"   Determinant: {np.linalg.det(S):.4f} (area scaled by this factor)")

print("\n3. SHEARING MATRIX (horizontal shear by 0.5):")
Sh = shearing_matrix(shx=0.5, shy=0)
print(Sh)
print(f"   Determinant: {np.linalg.det(Sh):.4f} (preserves area)")

print("\n4. REFLECTION MATRIX (over x-axis):")
Ref = reflection_matrix('x')
print(Ref)
print(f"   Determinant: {np.linalg.det(Ref):.4f} (negative = orientation flip)")


# PART 2: DEFINE SHAPES

print("\n" + "="*80)
print("PART 2: DEFINE GEOMETRIC SHAPES")
print("="*80)

def create_square(size=1, center=(0, 0)):
    """Create a square centered at given point"""
    cx, cy = center
    half = size / 2
    return np.array([
        [cx - half, cx + half, cx + half, cx - half, cx - half],
        [cy - half, cy - half, cy + half, cy + half, cy - half]
    ])

def create_triangle(size=1, center=(0, 0)):
    """Create an equilateral triangle"""
    cx, cy = center
    height = size * np.sqrt(3) / 2
    return np.array([
        [cx, cx + size/2, cx - size/2, cx],
        [cy + height/2, cy - height/2, cy - height/2, cy + height/2]
    ])

def create_rectangle(width=2, height=1, center=(0, 0)):
    """Create a rectangle"""
    cx, cy = center
    hw, hh = width/2, height/2
    return np.array([
        [cx - hw, cx + hw, cx + hw, cx - hw, cx - hw],
        [cy - hh, cy - hh, cy + hh, cy + hh, cy - hh]
    ])

def create_pentagon(size=1, center=(0, 0)):
    """Create a regular pentagon"""
    cx, cy = center
    angles = np.linspace(0, 2*np.pi, 6)
    x = cx + size * np.cos(angles)
    y = cy + size * np.sin(angles)
    return np.array([x, y])

def create_house(size=1, center=(0, 0)):
    """Create a house shape"""
    cx, cy = center
    s = size
    return np.array([
        [cx - s, cx + s, cx + s, cx + 1.5*s, cx, cx - 1.5*s, cx - s, cx - s],
        [cy - s, cy - s, cy, cy, cy + 1.5*s, cy, cy, cy - s]
    ])

# Create shapes
square = create_square(size=1)
triangle = create_triangle(size=1)
rectangle = create_rectangle(width=2, height=1)
pentagon = create_pentagon(size=1)
house = create_house(size=0.5)

print("\nCreated shapes:")
print(f"  - Square: {square.shape}")
print(f"  - Triangle: {triangle.shape}")
print(f"  - Rectangle: {rectangle.shape}")
print(f"  - Pentagon: {pentagon.shape}")
print(f"  - House: {house.shape}")


# PART 3: APPLY TRANSFORMATIONS AND VISUALIZE

print("\n" + "="*80)
print("PART 3: APPLY TRANSFORMATIONS TO SHAPES")
print("="*80)

def apply_transformation(shape, matrix):
    """Apply transformation matrix to shape"""
    return matrix @ shape

def plot_before_after(original, transformed, title, ax=None, 
                      original_color='blue', transformed_color='red'):
    """Plot original and transformed shapes side by side"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot original shape
    ax.plot(original[0], original[1], 'o-', color=original_color, 
            linewidth=2, markersize=6, label='Original', alpha=0.7)
    ax.fill(original[0], original[1], color=original_color, alpha=0.2)
    
    # Plot transformed shape
    ax.plot(transformed[0], transformed[1], 's-', color=transformed_color, 
            linewidth=2, markersize=6, label='Transformed', alpha=0.7)
    ax.fill(transformed[0], transformed[1], color=transformed_color, alpha=0.2)
    
    # Plot origin and axes
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    all_x = np.concatenate([original[0], transformed[0]])
    all_y = np.concatenate([original[1], transformed[1]])
    margin = 0.5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    return ax


# VISUALIZATION 1: SINGLE TRANSFORMATIONS ON SQUARE

print("\n>>> Applying transformations to SQUARE...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

transformations = [
    (rotation_matrix(np.pi/4), "Rotation (45°)", square),
    (rotation_matrix(np.pi/2), "Rotation (90°)", square),
    (scaling_matrix(2, 0.5), "Scaling (2x, 0.5x)", square),
    (shearing_matrix(shx=0.5), "Horizontal Shear (0.5)", square),
    (shearing_matrix(shy=0.5), "Vertical Shear (0.5)", square),
    (reflection_matrix('x'), "Reflection (x-axis)", square),
]

for idx, (matrix, title, shape) in enumerate(transformations):
    transformed = apply_transformation(shape, matrix)
    plot_before_after(shape, transformed, title, ax=axes[idx])
    
    # Add matrix info
    det = np.linalg.det(matrix)
    matrix_str = f"det = {det:.2f}"
    axes[idx].text(0.02, 0.98, matrix_str, transform=axes[idx].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Single Transformations on Square', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# VISUALIZATION 2: DIFFERENT SHAPES, SAME TRANSFORMATION

print("\n>>> Applying ROTATION (60°) to different shapes...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

theta_60 = np.pi / 3  # 60 degrees
R_60 = rotation_matrix(theta_60)

shapes_dict = {
    'Square': square,
    'Triangle': triangle,
    'Rectangle': rectangle,
    'Pentagon': pentagon,
    'House': house,
}

for idx, (name, shape) in enumerate(shapes_dict.items()):
    transformed = apply_transformation(shape, R_60)
    plot_before_after(shape, transformed, 
                     f"Rotation 60° on {name}", ax=axes[idx])

# Remove extra subplot
fig.delaxes(axes[5])

plt.suptitle('Rotation (60°) Applied to Different Shapes', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# VISUALIZATION 3: COMPOSITION OF TRANSFORMATIONS

print("\n" + "="*80)
print("PART 4: COMPOSITION OF TRANSFORMATIONS")
print("="*80)

print("\n>>> Demonstrating transformation composition...")

# Define multiple transformations
R_45 = rotation_matrix(np.pi/4)
S_scale = scaling_matrix(1.5, 0.7)
Sh_shear = shearing_matrix(shx=0.3)

# Apply in different orders
shape = square

# Order 1: Scale -> Rotate -> Shear
comp1 = Sh_shear @ R_45 @ S_scale
transformed1 = comp1 @ shape

# Order 2: Rotate -> Shear -> Scale
comp2 = S_scale @ Sh_shear @ R_45
transformed2 = comp2 @ shape

# Order 3: Shear -> Scale -> Rotate
comp3 = R_45 @ S_scale @ Sh_shear
transformed3 = comp3 @ shape

fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# Original
plot_before_after(shape, shape, "Original Square", ax=axes[0, 0],
                 transformed_color='blue')

# Composition 1
plot_before_after(shape, transformed1, 
                 "Scale → Rotate → Shear", ax=axes[0, 1])
det1 = np.linalg.det(comp1)
axes[0, 1].text(0.02, 0.98, f"det = {det1:.3f}", 
                transform=axes[0, 1].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Composition 2
plot_before_after(shape, transformed2, 
                 "Rotate → Shear → Scale", ax=axes[1, 0])
det2 = np.linalg.det(comp2)
axes[1, 0].text(0.02, 0.98, f"det = {det2:.3f}", 
                transform=axes[1, 0].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Composition 3
plot_before_after(shape, transformed3, 
                 "Shear → Scale → Rotate", ax=axes[1, 1])
det3 = np.linalg.det(comp3)
axes[1, 1].text(0.02, 0.98, f"det = {det3:.3f}", 
                transform=axes[1, 1].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.suptitle('Composition: Order Matters!', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n ORDER MATTERS in matrix multiplication!")
print(f"   Composition 1 det: {det1:.4f}")
print(f"   Composition 2 det: {det2:.4f}")
print(f"   Composition 3 det: {det3:.4f}")
print("   Notice: All have same determinant (product of individual dets)")
print("   But the shapes look different!")


# VISUALIZATION 4: ROTATION ANIMATION (Multiple Angles)

print("\n" + "="*80)
print("PART 5: ROTATION AT DIFFERENT ANGLES")
print("="*80)

print("\n>>> Creating rotation sequence...")

angles = np.linspace(0, 2*np.pi, 13)  # 0° to 360° in 30° steps

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for idx, angle in enumerate(angles):
    if idx >= 15:  # Only 15 subplots
        break
    
    R = rotation_matrix(angle)
    transformed = apply_transformation(triangle, R)
    
    plot_before_after(triangle, transformed, 
                     f"Rotation: {np.degrees(angle):.0f}°",
                     ax=axes[idx])

plt.suptitle('Triangle Rotation Sequence (0° to 360°)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


#VISUALIZATION 5: TRANSFORMATION EFFECTS COMPARISON

print("\n" + "="*80)
print("PART 6: COMPARING TRANSFORMATION EFFECTS")
print("="*80)

# Create a comprehensive comparison
fig = plt.figure(figsize=(20, 14))

# Define transformations
transforms_comparison = [
    ("Original", np.eye(2)),
    ("Rotate 30°", rotation_matrix(np.pi/6)),
    ("Rotate 90°", rotation_matrix(np.pi/2)),
    ("Rotate 180°", rotation_matrix(np.pi)),
    ("Scale (2, 2)", scaling_matrix(2, 2)),
    ("Scale (0.5, 0.5)", scaling_matrix(0.5, 0.5)),
    ("Scale (2, 0.5)", scaling_matrix(2, 0.5)),
    ("Stretch X", scaling_matrix(2, 1)),
    ("Stretch Y", scaling_matrix(1, 2)),
    ("Shear X", shearing_matrix(shx=0.5)),
    ("Shear Y", shearing_matrix(shy=0.5)),
    ("Reflect X", reflection_matrix('x')),
    ("Reflect Y", reflection_matrix('y')),
    ("Reflect Origin", reflection_matrix('origin')),
    ("Combined", rotation_matrix(np.pi/6) @ scaling_matrix(1.5, 0.7)),
]

for idx, (name, matrix) in enumerate(transforms_comparison):
    ax = plt.subplot(3, 5, idx + 1)
    transformed = apply_transformation(square, matrix)
    plot_before_after(square, transformed, name, ax=ax)
    
    # Add determinant
    det = np.linalg.det(matrix)
    ax.text(0.98, 0.02, f"det={det:.2f}", transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.suptitle('Comprehensive Transformation Comparison', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


#PART 7: CREATE SUMMARY TABLE WITH PANDAS

print("\n" + "="*80)
print("PART 7: TRANSFORMATION SUMMARY TABLE")
print("="*80)

# Create summary data
summary_data = []

for name, matrix in transforms_comparison:
    det = np.linalg.det(matrix)
    trace = np.trace(matrix)
    
    # Analyze properties
    preserves_area = abs(abs(det) - 1) < 0.01
    preserves_orientation = det > 0
    is_singular = abs(det) < 0.01
    
    summary_data.append({
        'Transformation': name,
        'Determinant': f"{det:.3f}",
        'Trace': f"{trace:.3f}",
        'Preserves Area': '✓' if preserves_area else '✗',
        'Preserves Orientation': '✓' if preserves_orientation else '✗',
        'Invertible': '✗' if is_singular else '✓',
    })

df_summary = pd.DataFrame(summary_data)

print("\nTRANSFORMATION PROPERTIES SUMMARY:")
print("="*80)
print(df_summary.to_string(index=False))


#PART 8: ADVANCED - MULTIPLE SHAPES WITH TRANSFORMATION

print("\n" + "="*80)
print("PART 8: TRANSFORMING MULTIPLE SHAPES SIMULTANEOUSLY")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Create multiple shapes at different positions
shapes_positioned = {
    'Square': create_square(size=0.5, center=(-1, 0)),
    'Triangle': create_triangle(size=0.6, center=(0, 0)),
    'Pentagon': create_pentagon(size=0.5, center=(1, 0)),
}

# Define transformation
complex_transform = rotation_matrix(np.pi/4) @ scaling_matrix(1.5, 0.8)

# Plot 1: Original
for name, shape in shapes_positioned.items():
    axes[0].plot(shape[0], shape[1], 'o-', linewidth=2, 
                markersize=4, label=name, alpha=0.7)
    axes[0].fill(shape[0], shape[1], alpha=0.2)

axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].axvline(x=0, color='k', linewidth=0.5)
axes[0].set_title('Original Shapes', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-2, 2)

# Plot 2: Transformed
for name, shape in shapes_positioned.items():
    transformed = apply_transformation(shape, complex_transform)
    axes[1].plot(transformed[0], transformed[1], 's-', linewidth=2, 
                markersize=4, label=name, alpha=0.7)
    axes[1].fill(transformed[0], transformed[1], alpha=0.2)

axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linewidth=0.5)
axes[1].axvline(x=0, color='k', linewidth=0.5)
axes[1].set_title('Transformed (Rotate 45° + Scale)', 
                 fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-2, 2)

# Plot 3: Overlay
for name, shape in shapes_positioned.items():
    transformed = apply_transformation(shape, complex_transform)
    axes[2].plot(shape[0], shape[1], '--', linewidth=1.5, 
                alpha=0.5, color='gray')
    axes[2].fill(shape[0], shape[1], alpha=0.1, color='gray')
    axes[2].plot(transformed[0], transformed[1], '-', linewidth=2, 
                label=name)
    axes[2].fill(transformed[0], transformed[1], alpha=0.3)

axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='k', linewidth=0.5)
axes[2].axvline(x=0, color='k', linewidth=0.5)
axes[2].set_title('Before (dashed) & After (solid)', 
                 fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].set_xlim(-3, 3)
axes[2].set_ylim(-2, 2)

plt.tight_layout()
plt.show()


#Project Summary 
"""
 COMPLETED TASKS:

1.  Created rotation matrix R(θ)
2.  Created scaling matrix S(sx, sy)
3.  Created shearing matrix Sh(shx, shy)
4.  Created additional transformations (reflection)
5.  Applied transformations to square
6.  Applied transformations to triangle
7.  Applied transformations to multiple shapes
8.  Visualized before/after comparisons
9.  Demonstrated composition of transformations
10. Created comprehensive summary table

KEY INSIGHTS:

 DETERMINANT tells you:
  - Area scaling factor (|det|)
  - Orientation preservation (det > 0) or flip (det < 0)
  - Invertibility (det ≠ 0)

 ROTATION:
  - Preserves shape, size, and area (det = 1)
  - Changes orientation in space

 SCALING:
  - Changes size
  - det = sx × sy (area scaling factor)

 SHEARING:
  - Preserves area (det = 1)
  - Distorts angles but keeps parallel lines parallel

 COMPOSITION:
  - Order matters! A @ B ≠ B @ A
  - Final determinant = product of individual determinants

NEXT STEPS:
- Experiment with different angles and scale factors
- Try combining 3+ transformations
- Apply to more complex shapes
- Implement 3D transformations
- Use in image processing applications
"""

