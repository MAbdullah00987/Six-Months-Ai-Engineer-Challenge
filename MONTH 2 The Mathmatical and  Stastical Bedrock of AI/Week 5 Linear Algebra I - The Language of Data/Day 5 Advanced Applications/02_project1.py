
#Project 1: Affine Transformations
#Implement translation + rotation + scaling
#Use homogeneous coordinates (3×3 matrices for 2D)
#Create animation of shape moving and rotating
#Apply to letter shapes or simple graphics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


print("PROJECT 1: AFFINE TRANSFORMATIONS WITH HOMOGENEOUS COORDINATES")



# PART 1: UNDERSTANDING HOMOGENEOUS COORDINATES
print("PART 1: HOMOGENEOUS COORDINATES FOR 2D")

print("""
HOMOGENEOUS COORDINATES:
- Regular 2D point: (x, y)
- Homogeneous form: (x, y, 1)
- Allows translation with matrix multiplication!

3×3 Transformation Matrix:
┌              ┐
│ a  b  tx │  → (a,b) = rotation/scale, (tx,ty) = translation
│ c  d  ty │
│ 0  0  1  │
└              ┘
""")


# PART 2: BASIC TRANSFORMATION MATRICES


print("PART 2: CREATING TRANSFORMATION MATRICES")

def translation_matrix(tx, ty):
    """Create translation matrix"""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def rotation_matrix(theta):
    """Create rotation matrix (angle in radians)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def scaling_matrix(sx, sy):
    """Create scaling matrix"""
    return np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ])

# Example matrices
print("\n1. TRANSLATION by (3, 2):")
T = translation_matrix(3, 2)
print(T)

print("\n2. ROTATION by 45°:")
R = rotation_matrix(np.pi/4)
print(R)

print("\n3. SCALING by (2, 1.5):")
S = scaling_matrix(2, 1.5)
print(S)


# PART 3: APPLYING TRANSFORMATIONS


print("PART 3: APPLYING TRANSFORMATIONS TO POINTS")


def to_homogeneous(points):
    """Convert points to homogeneous coordinates"""
    if points.ndim == 1:
        return np.append(points, 1)
    else:
        ones = np.ones((points.shape[0], 1))
        return np.hstack([points, ones])

def from_homogeneous(points):
    """Convert from homogeneous to Cartesian"""
    if points.ndim == 1:
        return points[:2] / points[2]
    else:
        return points[:, :2] / points[:, 2:3]

def apply_transformation(points, matrix):
    """Apply transformation matrix to points"""
    homo_points = to_homogeneous(points)
    transformed = (matrix @ homo_points.T).T
    return from_homogeneous(transformed)

# Test with a simple square
square = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]  # Close the shape
])

print("\nOriginal square:")
print(square)

# Apply transformations
translated = apply_transformation(square, translation_matrix(2, 1))
rotated = apply_transformation(square, rotation_matrix(np.pi/4))
scaled = apply_transformation(square, scaling_matrix(2, 0.5))

print("\nAfter translation (2, 1):")
print(translated[:3])

# Visualize basic transformations
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

transformations = [
    ("Original", square, 'blue'),
    ("Translated (+2, +1)", translated, 'green'),
    ("Rotated (45°)", rotated, 'red'),
    ("Scaled (2x, 0.5y)", scaled, 'orange')
]

for ax, (title, shape, color) in zip(axes.flat, transformations):
    ax.plot(shape[:, 0], shape[:, 1], 'o-', linewidth=2, 
            markersize=8, color=color, label=title)
    ax.fill(shape[:, 0], shape[:, 1], alpha=0.3, color=color)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 4)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig('basic_transformations.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: basic_transformations.png")
plt.show()


# PART 4: COMPOSITE TRANSFORMATIONS


print("PART 4: COMPOSITE TRANSFORMATIONS")

print("""
COMBINING TRANSFORMATIONS:
- Matrix multiplication: M_combined = M3 × M2 × M1
- Order matters! (not commutative)
- Apply from RIGHT to LEFT
""")

# Example: Rotate around point (1, 1) instead of origin
print("\nExample: Rotate square around point (1, 1)")

# Method: Translate to origin → Rotate → Translate back
center = np.array([1, 1])
angle = np.pi/3  # 60 degrees

# Step by step
T1 = translation_matrix(-center[0], -center[1])  # Move to origin
R = rotation_matrix(angle)                        # Rotate
T2 = translation_matrix(center[0], center[1])    # Move back

# Composite matrix
M_composite = T2 @ R @ T1

print(f"\n1. Translate to origin: T(-1, -1)")
print(f"2. Rotate by {np.degrees(angle):.1f}°")
print(f"3. Translate back: T(1, 1)")
print(f"\nComposite matrix:")
print(M_composite)

# Apply composite transformation
rotated_around_center = apply_transformation(square, M_composite)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before
ax = axes[0]
ax.plot(square[:, 0], square[:, 1], 'bo-', linewidth=2, 
        markersize=8, label='Original')
ax.fill(square[:, 0], square[:, 1], alpha=0.3, color='blue')
ax.plot(center[0], center[1], 'r*', markersize=20, 
        label='Rotation center')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.set_title('Before Rotation', fontsize=12, fontweight='bold')
ax.legend()

# After
ax = axes[1]
ax.plot(square[:, 0], square[:, 1], 'b--', linewidth=1, 
        alpha=0.5, label='Original')
ax.plot(rotated_around_center[:, 0], rotated_around_center[:, 1], 
        'ro-', linewidth=2, markersize=8, label='Rotated')
ax.fill(rotated_around_center[:, 0], rotated_around_center[:, 1], 
        alpha=0.3, color='red')
ax.plot(center[0], center[1], 'r*', markersize=20, 
        label='Rotation center')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.set_title(f'After {np.degrees(angle):.1f}° Rotation', 
             fontsize=12, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('composite_rotation.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: composite_rotation.png")
plt.show()


# PART 5: LETTER SHAPES



print("PART 5: CREATING AND TRANSFORMING LETTER SHAPES")

def create_letter_A():
    """Create letter 'A' shape"""
    return np.array([
        [0, 0], [0.5, 2], [1, 0],     # Outer triangle
        [1, 0], [0.65, 0.8], [0.35, 0.8], [0, 0]  # With crossbar
    ])

def create_letter_F():
    """Create letter 'F' shape"""
    return np.array([
        [0, 0], [0, 2], [1.2, 2],     # Vertical and top
        [1.2, 2], [0.2, 2], [0.2, 1.2],  # Top bar
        [0.2, 1.2], [0.8, 1.2], [0.8, 1.0],  # Middle bar
        [0.8, 1.0], [0.2, 1.0], [0.2, 0], [0, 0]  # Back to start
    ])

def create_letter_T():
    """Create letter 'T' shape"""
    return np.array([
        [0, 2], [1, 2], [1, 1.8],     # Top bar
        [1, 1.8], [0.6, 1.8], [0.6, 0],  # Right side down
        [0.6, 0], [0.4, 0], [0.4, 1.8],  # Vertical stem
        [0.4, 1.8], [0, 1.8], [0, 2]     # Back to start
    ])

# Create letters
letter_A = create_letter_A()
letter_F = create_letter_F()
letter_T = create_letter_T()

print("\nCreated letters: A, F, T")

# Apply various transformations to letters
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Letter A transformations
ax = axes[0, 0]
ax.plot(letter_A[:, 0], letter_A[:, 1], 'bo-', linewidth=2)
ax.fill(letter_A[:, 0], letter_A[:, 1], alpha=0.3, color='blue')
ax.set_title('Letter A - Original', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Scaled A
scaled_A = apply_transformation(letter_A, scaling_matrix(1.5, 1.5))
ax = axes[0, 1]
ax.plot(scaled_A[:, 0], scaled_A[:, 1], 'go-', linewidth=2)
ax.fill(scaled_A[:, 0], scaled_A[:, 1], alpha=0.3, color='green')
ax.set_title('Letter A - Scaled (1.5x)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Rotated A
rotated_A = apply_transformation(letter_A, rotation_matrix(np.pi/6))
ax = axes[0, 2]
ax.plot(rotated_A[:, 0], rotated_A[:, 1], 'ro-', linewidth=2)
ax.fill(rotated_A[:, 0], rotated_A[:, 1], alpha=0.3, color='red')
ax.set_title('Letter A - Rotated (30°)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Letter F transformations
ax = axes[1, 0]
ax.plot(letter_F[:, 0], letter_F[:, 1], 'mo-', linewidth=2)
ax.fill(letter_F[:, 0], letter_F[:, 1], alpha=0.3, color='magenta')
ax.set_title('Letter F - Original', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Sheared F (shear transformation)
shear_matrix = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
sheared_F = apply_transformation(letter_F, shear_matrix)
ax = axes[1, 1]
ax.plot(sheared_F[:, 0], sheared_F[:, 1], 'co-', linewidth=2)
ax.fill(sheared_F[:, 0], sheared_F[:, 1], alpha=0.3, color='cyan')
ax.set_title('Letter F - Sheared', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Letter T - Composite
M_combo = (translation_matrix(1, 0.5) @ 
           rotation_matrix(-np.pi/8) @ 
           scaling_matrix(1.2, 0.8))
transformed_T = apply_transformation(letter_T, M_combo)
ax = axes[1, 2]
ax.plot(letter_T[:, 0], letter_T[:, 1], 'b--', alpha=0.3, label='Original')
ax.plot(transformed_T[:, 0], transformed_T[:, 1], 'yo-', linewidth=2, 
        label='Transformed')
ax.fill(transformed_T[:, 0], transformed_T[:, 1], alpha=0.3, color='yellow')
ax.set_title('Letter T - Composite Transform', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.legend()

for ax in axes.flat:
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('letter_transformations.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: letter_transformations.png")
plt.show()


# PART 6: ANIMATED TRANSFORMATIONS


print("PART 6: CREATING ANIMATIONS")

print("\nCreating animated transformation sequence...")

# Create a simple star shape
def create_star(points=5, outer_radius=1, inner_radius=0.4):
    """Create a star shape"""
    angles = np.linspace(0, 2*np.pi, 2*points + 1)
    vertices = []
    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = r * np.cos(angle - np.pi/2)
        y = r * np.sin(angle - np.pi/2)
        vertices.append([x, y])
    return np.array(vertices)

star = create_star()

# Animation 1: Rotation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Animation: Rotating Star', fontsize=14, fontweight='bold')

line, = ax.plot([], [], 'ro-', linewidth=2, markersize=8)
fill = ax.fill([], [], alpha=0.3, color='red')[0]

def init():
    line.set_data([], [])
    return line, fill

def animate(frame):
    angle = frame * np.pi / 30  # Rotation angle
    M = rotation_matrix(angle)
    rotated = apply_transformation(star, M)
    line.set_data(rotated[:, 0], rotated[:, 1])
    fill.set_xy(rotated)
    return line, fill

anim = FuncAnimation(fig, animate, init_func=init, frames=60, 
                     interval=50, blit=True)
anim.save('rotation_animation.gif', writer='pillow', fps=20)
print("✓ Saved: rotation_animation.gif")
plt.close()

# Animation 2: Translation + Rotation + Scaling
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Animation: Moving, Rotating, and Scaling Star', 
             fontsize=14, fontweight='bold')

line, = ax.plot([], [], 'bo-', linewidth=2, markersize=8)
fill = ax.fill([], [], alpha=0.3, color='blue')[0]
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def animate_complex(frame):
    t = frame / 60.0
    
    # Varying transformations over time
    tx = 4 * np.cos(2 * np.pi * t) - 2  # Circular path
    ty = 2 * np.sin(2 * np.pi * t)
    angle = 4 * np.pi * t  # Multiple rotations
    scale = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * t))  # Pulsing
    
    # Composite transformation
    M = (translation_matrix(tx, ty) @ 
         rotation_matrix(angle) @ 
         scaling_matrix(scale, scale))
    
    transformed = apply_transformation(star, M)
    line.set_data(transformed[:, 0], transformed[:, 1])
    fill.set_xy(transformed)
    time_text.set_text(f'Frame: {frame}\nAngle: {np.degrees(angle):.1f}°\n'
                       f'Scale: {scale:.2f}')
    return line, fill, time_text

anim2 = FuncAnimation(fig, animate_complex, frames=120, 
                      interval=50, blit=True)
anim2.save('complex_animation.gif', writer='pillow', fps=20)
print("✓ Saved: complex_animation.gif")
plt.close()

# Animation 3: Letter transformation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-2, 4)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Animation: Transforming Letter A', 
             fontsize=14, fontweight='bold')

line, = ax.plot([], [], 'mo-', linewidth=2, markersize=8)
fill = ax.fill([], [], alpha=0.3, color='magenta')[0]

def animate_letter(frame):
    t = frame / 100.0
    
    # Smooth transition through transformations
    if t < 0.33:  # First third: translate
        progress = t / 0.33
        tx = 2 * progress
        M = translation_matrix(tx, 0)
    elif t < 0.66:  # Second third: rotate
        progress = (t - 0.33) / 0.33
        angle = np.pi * progress
        M = translation_matrix(2, 0) @ rotation_matrix(angle)
    else:  # Final third: scale
        progress = (t - 0.66) / 0.34
        scale = 1 + 0.5 * progress
        M = (translation_matrix(2, 0) @ 
             rotation_matrix(np.pi) @ 
             scaling_matrix(scale, scale))
    
    transformed = apply_transformation(letter_A, M)
    line.set_data(transformed[:, 0], transformed[:, 1])
    fill.set_xy(transformed)
    return line, fill

anim3 = FuncAnimation(fig, animate_letter, frames=100, 
                      interval=50, blit=True)
anim3.save('letter_animation.gif', writer='pillow', fps=20)
print(" Saved: letter_animation.gif")
plt.close()


# PART 7: TRANSFORMATION ANALYSIS WITH PANDAS


print("PART 7: ANALYZING TRANSFORMATIONS WITH PANDAS")

# Create dataset of transformation properties
transformations_data = []

test_point = np.array([[1, 1]])

for angle_deg in range(0, 361, 30):
    angle_rad = np.radians(angle_deg)
    
    # Different transformations
    transforms = {
        'Rotation': rotation_matrix(angle_rad),
        'Scale_2x': scaling_matrix(2, 2),
        'Translate': translation_matrix(1, 1),
        'Composite': (translation_matrix(1, 1) @ 
                     rotation_matrix(angle_rad) @ 
                     scaling_matrix(1.5, 1.5))
    }
    
    for name, M in transforms.items():
        result = apply_transformation(test_point, M)[0]
        
        # Calculate properties
        det = np.linalg.det(M[:2, :2])  # 2x2 submatrix
        
        transformations_data.append({
            'Angle (deg)': angle_deg,
            'Transform': name,
            'Result_X': result[0],
            'Result_Y': result[1],
            'Distance': np.linalg.norm(result),
            'Determinant': det
        })

df = pd.DataFrame(transformations_data)

print("\nTransformation Data Sample:")
print(df.head(10))

print("\n\nStatistics by Transformation Type:")
print(df.groupby('Transform')[['Distance', 'Determinant']].describe())

# Visualize with Seaborn
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distance from origin over angles
ax = axes[0, 0]
for transform_type in df['Transform'].unique():
    data = df[df['Transform'] == transform_type]
    ax.plot(data['Angle (deg)'], data['Distance'], 
            marker='o', label=transform_type, linewidth=2)
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Distance from Origin')
ax.set_title('Distance vs Angle for Different Transforms')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Trajectory in 2D space
ax = axes[0, 1]
for transform_type in df['Transform'].unique():
    data = df[df['Transform'] == transform_type]
    ax.plot(data['Result_X'], data['Result_Y'], 
            marker='o', label=transform_type, linewidth=2, markersize=4)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Transformation Trajectories')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Heatmap of determinants
pivot_det = df.pivot_table(values='Determinant', 
                            index='Transform', 
                            columns='Angle (deg)')
ax = axes[1, 0]
sns.heatmap(pivot_det, annot=False, cmap='coolwarm', 
            center=1, ax=ax, cbar_kws={'label': 'Determinant'})
ax.set_title('Determinant Heatmap\n(1 = area-preserving)')

# Plot 4: Box plot of distances
ax = axes[1, 1]
df.boxplot(column='Distance', by='Transform', ax=ax)
ax.set_title('Distance Distribution by Transform Type')
ax.set_xlabel('Transform Type')
ax.set_ylabel('Distance from Origin')
plt.suptitle('')  # Remove automatic title

plt.tight_layout()
plt.savefig('transformation_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: transformation_analysis.png")
plt.show()


# PART 8: PRACTICAL APPLICATION - IMAGE-LIKE GRID

print("\n" + "="*80)
print("PART 8: TRANSFORMING A GRID (IMAGE-LIKE)")
print("="*80)

# Create a grid of points
x = np.linspace(0, 4, 20)
y = np.linspace(0, 3, 15)
X, Y = np.meshgrid(x, y)
grid_points = np.column_stack([X.ravel(), Y.ravel()])

print(f"Created grid: {grid_points.shape[0]} points")

# Apply various transformations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

transforms_grid = [
    ("Original Grid", np.eye(3)),
    ("Rotated 30°", rotation_matrix(np.pi/6)),
    ("Scaled (0.8, 1.2)", scaling_matrix(0.8, 1.2)),
    ("Sheared", np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])),
    ("Translated", translation_matrix(-1, -0.5)),
    ("Composite", (rotation_matrix(np.pi/8) @ 
                   scaling_matrix(0.7, 0.7) @ 
                   translation_matrix(1, 0.5)))
]

for ax, (title, M) in zip(axes.flat, transforms_grid):
    transformed = apply_transformation(grid_points, M)
    X_new = transformed[:, 0].reshape(15, 20)
    Y_new = transformed[:, 1].reshape(15, 20)
    
    ax.scatter(transformed[:, 0], transformed[:, 1], 
               c=grid_points[:, 0], cmap='viridis', 
               s=20, alpha=0.6)
    
    # Draw grid lines
    for i in range(15):
        ax.plot(X_new[i, :], Y_new[i, :], 'b-', alpha=0.3, linewidth=0.5)
    for j in range(20):
        ax.plot(X_new[:, j], Y_new[:, j], 'b-', alpha=0.3, linewidth=0.5)
    
    ax.set_title(title, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 6)
    ax.set_ylim(-3, 6)

plt.tight_layout()
plt.savefig('grid_transformations.png', dpi=150, bbox_inches='tight')
print("\nSaved: grid_transformations.png")
plt.show()


#  Important Notes

print("PROJECT Notes")

"""
 COMPLETED ALL OBJECTIVES:
1. HOMOGENEOUS COORDINATES (3×3 for 2D)
   - Implemented translation, rotation, scaling matrices
   - Used homogeneous form: (x, y, 1)

2. BASIC TRANSFORMATIONS
   - Translation: Move objects
   - Rotation: Rotate around origin
   - Scaling: Resize objects
   - Shearing: Skew objects

3. COMPOSITE TRANSFORMATIONS
   - Combined multiple transformations
   - Matrix multiplication order matters
   - Rotation around arbitrary points

4. LETTER SHAPES
   - Created letters: A, F, T
   - Applied various transformations
   - Visualized results

5. ANIMATIONS (3 GIF files created)
   - rotation_animation.gif: Spinning star
   - complex_animation.gif: Moving + rotating + scaling
   - letter_animation.gif: Letter A transformation sequence

6. DATA ANALYSIS WITH PANDAS
   - Tracked transformation properties
   - Statistical analysis
   - Visualized with Seaborn

7. GRID TRANSFORMATIONS
   - Applied to 300-point grid
   - Image-like transformation demo

FILES GENERATED:
- basic_transformations.png
- composite_rotation.png
- letter_transformations.png
- rotation_animation.gif
- complex_animation.gif
- letter_animation.gif
- transformation_analysis.png
- grid_transformations.png

TOTAL: 8 visualization files
"""

print(summary)

# Create summary DataFrame
summary_df = pd.DataFrame({
    'Transformation': ['Translation', 'Rotation', 'Scaling', 'Shearing', 'Composite'],
    'Matrix Size': ['3×3'] * 5,
    'Preserves Area': ['Yes', 'Yes', 'No', 'Yes', 'Depends'],
    'Preserves Angles': ['Yes', 'Yes', 'Yes', 'No', 'Depends'],
    'Use Case': [
        'Moving objects',
        'Rotating objects',
        'Resizing objects',
        'Skewing objects',
        'Complex transforms'
    ]
})

print("\nTransformation Summary Table:")
print(summary_df.to_string(index=False))
