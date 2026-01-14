
# Project 5:
# Determinant and Area: Show visually how the determinant of a 2x2 matrix relates to the change in area
# of a transformed shape.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection

# Set style
sns.set_style("white")
plt.rcParams['figure.figsize'] = (18, 14)

def create_shape(shape_type='square', scale=1.0):
    """Create different shapes for transformation"""
    if shape_type == 'square':
        # Unit square
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * scale
    elif shape_type == 'triangle':
        # Triangle
        return np.array([[0, 0], [1, 0], [0.5, 1]]) * scale
    elif shape_type == 'circle':
        # Circle approximated by polygon
        theta = np.linspace(0, 2*np.pi, 30)
        x = np.cos(theta) * scale
        y = np.sin(theta) * scale
        return np.column_stack([x, y])
    elif shape_type == 'rectangle':
        # Rectangle
        return np.array([[0, 0], [1.5, 0], [1.5, 1], [0, 1]]) * scale
    elif shape_type == 'L_shape':
        # L-shaped polygon
        return np.array([[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], 
                        [0.5, 1], [0, 1]]) * scale

def calculate_polygon_area(vertices):
    """Calculate area of polygon using shoelace formula"""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0

def transform_shape(shape, matrix):
    """Apply matrix transformation to shape"""
    # Shape is Nx2, matrix is 2x2
    # Transform each point
    transformed = shape @ matrix.T
    return transformed

def plot_transformation(ax, original, transformed, matrix, title):
    """Plot original and transformed shapes"""
    
    # Calculate areas
    original_area = calculate_polygon_area(original)
    transformed_area = calculate_polygon_area(transformed)
    det = np.linalg.det(matrix)
    
    # Plot original shape
    poly_original = Polygon(original, alpha=0.4, facecolor='blue', 
                           edgecolor='darkblue', linewidth=2, label='Original')
    ax.add_patch(poly_original)
    
    # Plot transformed shape
    poly_transformed = Polygon(transformed, alpha=0.4, facecolor='red', 
                              edgecolor='darkred', linewidth=2, label='Transformed')
    ax.add_patch(poly_transformed)
    
    # Plot grid lines to show transformation
    for i in range(len(original)):
        ax.plot([original[i, 0], transformed[i, 0]], 
               [original[i, 1], transformed[i, 1]], 
               'k--', alpha=0.3, linewidth=0.5)
    
    # Set equal aspect and limits
    ax.set_aspect('equal')
    
    # Calculate bounds
    all_points = np.vstack([original, transformed])
    margin = 0.5
    ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Labels
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Add text with calculations
    area_ratio = transformed_area / original_area if original_area > 0 else 0
    
    text_str = f'det(A) = {det:.2f}\n'
    text_str += f'Original Area = {original_area:.2f}\n'
    text_str += f'New Area = {transformed_area:.2f}\n'
    text_str += f'Area Ratio = {area_ratio:.2f}\n'
    text_str += f'|det(A)| = {abs(det):.2f}'
    
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=9)
    
    return det, original_area, transformed_area


# Create comprehensive visualizations


fig = plt.figure(figsize=(20, 16))

# Define test matrices
matrices = {
    'Identity\n(No change)': np.array([[1, 0], [0, 1]]),
    'Scale by 2\n(Stretch)': np.array([[2, 0], [0, 2]]),
    'Scale by 3 (x only)': np.array([[3, 0], [0, 1]]),
    'Scale by 0.5\n(Shrink)': np.array([[0.5, 0], [0, 0.5]]),
    'Shear\n(Horizontal)': np.array([[1, 1], [0, 1]]),
    'Rotation 45°\n(Preserves area)': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                                 [np.sin(np.pi/4), np.cos(np.pi/4)]]),
    'Reflection\n(Flips, det < 0)': np.array([[-1, 0], [0, 1]]),
    'Compression\n(det < 1)': np.array([[0.5, 0], [0, 2]]),
    'Complex\nTransform': np.array([[2, 1], [1, 2]]),
}

# Create unit square for all transformations
square = create_shape('square', scale=1.0)

# Plot each transformation
results = []
for idx, (name, matrix) in enumerate(matrices.items(), 1):
    ax = plt.subplot(3, 4, idx)
    transformed = transform_shape(square, matrix)
    det, orig_area, new_area = plot_transformation(ax, square, transformed, matrix, name)
    results.append({
        'Name': name.replace('\n', ' '),
        'Determinant': det,
        'Original Area': orig_area,
        'New Area': new_area,
        'Area Ratio': new_area / orig_area if orig_area > 0 else 0
    })


# Additional analysis plots

# Plot 10: Determinant vs Area Ratio
ax10 = plt.subplot(3, 4, 10)
dets = [r['Determinant'] for r in results]
ratios = [r['Area Ratio'] for r in results]
abs_dets = [abs(d) for d in dets]

ax10.scatter(dets, ratios, s=200, c=range(len(dets)), 
            cmap='viridis', edgecolors='black', linewidth=2, alpha=0.7)
ax10.plot([-3, 5], [-3, 5], 'r--', linewidth=2, alpha=0.5, label='y = x (theory)')
ax10.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax10.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
ax10.set_xlabel('Determinant', fontsize=11, fontweight='bold')
ax10.set_ylabel('Area Ratio (New/Original)', fontsize=11, fontweight='bold')
ax10.set_title('Determinant = Area Scaling Factor', fontsize=12, fontweight='bold')
ax10.grid(True, alpha=0.3)
ax10.legend()

# Add annotations
for i, r in enumerate(results):
    if abs(r['Determinant']) > 0.1:  # Skip near-zero determinants
        ax10.annotate(f"{i+1}", (r['Determinant'], r['Area Ratio']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

# Plot 11: Different shapes with same transformation
ax11 = plt.subplot(3, 4, 11)
transform_matrix = np.array([[2, 0.5], [0.5, 1.5]])
det = np.linalg.det(transform_matrix)

shapes_to_test = ['square', 'triangle', 'circle']
colors = ['blue', 'green', 'orange']
shape_results = []

for i, (shape_type, color) in enumerate(zip(shapes_to_test, colors)):
    shape = create_shape(shape_type, scale=0.8)
    transformed = transform_shape(shape, transform_matrix)
    
    orig_area = calculate_polygon_area(shape)
    new_area = calculate_polygon_area(transformed)
    
    # Plot with offset for visibility
    offset = np.array([i * 3, 0])
    poly_orig = Polygon(shape + offset, alpha=0.3, facecolor=color, 
                       edgecolor=color, linewidth=2)
    poly_trans = Polygon(transformed + offset, alpha=0.6, facecolor=color, 
                        edgecolor='black', linewidth=2, linestyle='--')
    ax11.add_patch(poly_orig)
    ax11.add_patch(poly_trans)
    
    shape_results.append({
        'shape': shape_type,
        'orig_area': orig_area,
        'new_area': new_area,
        'ratio': new_area / orig_area
    })

ax11.set_xlim(-1, 10)
ax11.set_ylim(-1, 4)
ax11.set_aspect('equal')
ax11.grid(True, alpha=0.3)
ax11.set_title(f'Same Transform on Different Shapes\ndet = {det:.2f}', 
              fontsize=12, fontweight='bold')

# Add text
text = 'All shapes scale by |det|:\n'
for sr in shape_results:
    text += f"{sr['shape']}: {sr['ratio']:.2f}x\n"
ax11.text(0.02, 0.98, text, transform=ax11.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 12: Visual proof with grid
ax12 = plt.subplot(3, 4, 12)

# Create a grid of points
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)

# Transform the grid
transform_matrix = np.array([[1.5, 0.5], [0.5, 1.5]])
points = np.column_stack([X.ravel(), Y.ravel()])
transformed_points = transform_shape(points, transform_matrix)

X_trans = transformed_points[:, 0].reshape(X.shape)
Y_trans = transformed_points[:, 1].reshape(Y.shape)

# Plot original grid
ax12.plot(X, Y, 'b-', alpha=0.3, linewidth=0.5)
ax12.plot(X.T, Y.T, 'b-', alpha=0.3, linewidth=0.5)

# Plot transformed grid
ax12.plot(X_trans, Y_trans, 'r-', alpha=0.6, linewidth=1)
ax12.plot(X_trans.T, Y_trans.T, 'r-', alpha=0.6, linewidth=1)

# Highlight one unit square
unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
trans_square = transform_shape(unit_square, transform_matrix)

poly1 = Polygon(unit_square, alpha=0.3, facecolor='blue', 
               edgecolor='darkblue', linewidth=3)
poly2 = Polygon(trans_square, alpha=0.3, facecolor='red', 
               edgecolor='darkred', linewidth=3)
ax12.add_patch(poly1)
ax12.add_patch(poly2)

ax12.set_xlim(-3, 5)
ax12.set_ylim(-3, 5)
ax12.set_aspect('equal')
ax12.grid(True, alpha=0.3)
ax12.axhline(y=0, color='k', linewidth=0.5)
ax12.axvline(x=0, color='k', linewidth=0.5)
ax12.set_title('Grid Transformation Visualization', fontsize=12, fontweight='bold')

det = np.linalg.det(transform_matrix)
ax12.text(0.02, 0.98, f'det = {det:.2f}\nEach square\nscales by {det:.2f}', 
         transform=ax12.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()


# Summary table
print("DETERMINANT AND AREA TRANSFORMATION")

import pandas as pd
df = pd.DataFrame(results)
df['|det|'] = df['Determinant'].abs()
df['Matches?'] = np.isclose(df['|det|'], df['Area Ratio'], atol=0.01)

print("\n", df.to_string(index=False))

print("KEY INSIGHTS")
print("""
1. FUNDAMENTAL THEOREM:
   |det(A)| = Area Scaling Factor
   
   • If |det| = 2, area doubles
   • If |det| = 0.5, area halves
   • If |det| = 1, area preserved (rotations, reflections)
   • If |det| = 0, shape collapses to line/point

2. SIGN OF DETERMINANT:
   • det > 0: Orientation preserved
   • det < 0: Orientation reversed (reflection)
   • The absolute value |det| gives the area scaling

3. SPECIAL CASES:
   • Identity (det=1): No change
   • Rotation (det=1): Area preserved, shape rotated
   • Reflection (det=-1): Area preserved, shape flipped
   • Shear: Area preserved (det=1), shape skewed
   • Singular (det=0): Shape collapses, no inverse

4. UNIVERSAL PROPERTY:
   The determinant scales area the SAME way
   for ANY shape (square, circle, triangle, etc.)

5. GEOMETRIC INTERPRETATION:
   det(A) = (signed) area of parallelogram
   formed by column vectors of A
""")

# Interactive example with step-by-step visualization

print("STEP-BY-STEP EXAMPLE")


matrix_example = np.array([[2, 1], [0, 2]])
det_example = np.linalg.det(matrix_example)

print(f"\nMatrix A = {matrix_example}")
print(f"det(A) = {det_example:.2f}")
print(f"\nThis means ANY shape will have its area scaled by {det_example:.2f}")

# Test with different shapes
test_shapes = ['square', 'triangle', 'circle']
print("\nVerification:")
for shape_type in test_shapes:
    shape = create_shape(shape_type, scale=1.0)
    transformed = transform_shape(shape, matrix_example)
    
    orig_area = calculate_polygon_area(shape)
    new_area = calculate_polygon_area(transformed)
    ratio = new_area / orig_area
    
    print(f"  {shape_type.capitalize():10s}: {orig_area:.3f} → {new_area:.3f} "
          f"(ratio = {ratio:.3f}, matches det = {np.isclose(ratio, det_example)})")
          