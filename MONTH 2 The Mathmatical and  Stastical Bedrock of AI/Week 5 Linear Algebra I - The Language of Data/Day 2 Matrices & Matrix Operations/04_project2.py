
#Project 2: Image as a Matrix 
#* Create a 10×10 black & white image matrix
#* Perform transpose, flip (using slicing)
#* Apply brightness adjustment (scalar multiplication)
#* Visualize with matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

print("=" * 60)
print("PROJECT 5: IMAGE AS A MATRIX")
print("=" * 60)

# ============================================
# PART 1: CREATE 10x10 BLACK & WHITE IMAGE
# ============================================
print("\n[PART 1] Creating 10x10 Black & White Image Matrix\n")

# Method 1: Random binary image (0s and 1s)
np.random.seed(42)  # For reproducibility
image_matrix = np.random.randint(0, 2, size=(10, 10))

print("Original Image Matrix (Random Binary):")
print(image_matrix)
print(f"\nShape: {image_matrix.shape}")
print(f"Data type: {image_matrix.dtype}")
print(f"Min value: {image_matrix.min()}, Max value: {image_matrix.max()}")

# Method 2: Create a pattern (Checkerboard)
checkerboard = np.zeros((10, 10), dtype=int)
checkerboard[::2, ::2] = 1  # Every even row, even column
checkerboard[1::2, 1::2] = 1  # Every odd row, odd column

print("\n\nCheckerboard Pattern:")
print(checkerboard)

# Method 3: Create a gradient pattern
gradient = np.linspace(0, 1, 10).reshape(10, 1) * np.ones((10, 10))
print("\n\nGradient Pattern (0 to 1):")
print(np.round(gradient, 2))

# ============================================
# PART 2: TRANSPOSE OPERATION
# ============================================
print("\n" + "=" * 60)
print("[PART 2] Transpose Operation")
print("=" * 60)

# Transpose swaps rows with columns
transposed = image_matrix.T  # or np.transpose(image_matrix)

print("\nTransposed Image Matrix:")
print(transposed)

# Verify transpose properties
print(f"\nOriginal shape: {image_matrix.shape}")
print(f"Transposed shape: {transposed.shape}")
print(f"Are they equal? {np.array_equal(image_matrix, transposed.T)}")

# ============================================
# PART 3: FLIP OPERATIONS (Using Slicing)
# ============================================
print("\n" + "=" * 60)
print("[PART 3] Flip Operations Using Slicing")
print("=" * 60)

# Vertical flip (flip upside down) - reverse rows
vertical_flip = image_matrix[::-1, :]
print("\nVertical Flip (Upside Down):")
print(vertical_flip)

# Horizontal flip (mirror left-right) - reverse columns
horizontal_flip = image_matrix[:, ::-1]
print("\n\nHorizontal Flip (Left-Right Mirror):")
print(horizontal_flip)

# Both flips combined (180° rotation)
both_flips = image_matrix[::-1, ::-1]
print("\n\nBoth Flips (180° Rotation):")
print(both_flips)

# Diagonal flip (combination of transpose and flip)
diagonal_flip = image_matrix.T[::-1, :]
print("\n\nDiagonal Flip:")
print(diagonal_flip)

# ============================================
# PART 4: BRIGHTNESS ADJUSTMENT
# ============================================
print("\n" + "=" * 60)
print("[PART 4] Brightness Adjustment (Scalar Multiplication)")
print("=" * 60)

# Convert to float for brightness operations
image_float = image_matrix.astype(float)

# Increase brightness (multiply by scalar > 1)
brightness_increase = np.clip(image_float * 1.5, 0, 1)
print("\nBrightness Increased (1.5x):")
print(brightness_increase)

# Decrease brightness (multiply by scalar < 1)
brightness_decrease = image_float * 0.5
print("\n\nBrightness Decreased (0.5x):")
print(brightness_decrease)

# Invert colors (1 - image)
inverted = 1 - image_float
print("\n\nInverted Colors:")
print(inverted)

# ============================================
# PART 5: PANDAS DATAFRAME ANALYSIS
# ============================================
print("\n" + "=" * 60)
print("[PART 5] Pandas DataFrame Analysis")
print("=" * 60)

# Convert to DataFrame
df_image = pd.DataFrame(image_matrix, 
                        columns=[f'Col{i}' for i in range(10)],
                        index=[f'Row{i}' for i in range(10)])

print("\nImage as DataFrame:")
print(df_image)

# Statistical analysis
print("\n\nStatistical Summary:")
print(df_image.describe())

# Row-wise and column-wise statistics
print("\n\nRow sums (horizontal intensity):")
print(df_image.sum(axis=1))

print("\n\nColumn sums (vertical intensity):")
print(df_image.sum(axis=0))

# Find brightest and darkest regions
print(f"\n\nBrightest row: Row{df_image.sum(axis=1).idxmax()}")
print(f"Darkest row: Row{df_image.sum(axis=1).idxmin()}")

# ============================================
# PART 6: ADVANCED NUMPY OPERATIONS
# ============================================
print("\n" + "=" * 60)
print("[PART 6] Advanced NumPy Operations")
print("=" * 60)

# Rotation using numpy (90°, 180°, 270°)
rotate_90 = np.rot90(image_matrix, k=1)  # k=1 means 90° counter-clockwise
rotate_180 = np.rot90(image_matrix, k=2)
rotate_270 = np.rot90(image_matrix, k=3)

print("\n90° Rotation:")
print(rotate_90)

# Convolution (simple edge detection)
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Manual convolution on a gradient image for better effect
from scipy.ndimage import convolve
edges = convolve(gradient, kernel, mode='constant')

print("\n\nEdge Detection (using convolution):")
print(np.round(edges, 2))

# Element-wise operations
squared = image_matrix ** 2
sqrt_img = np.sqrt(image_float)

print(f"\n\nElement-wise operations:")
print(f"Original sum: {image_matrix.sum()}")
print(f"Squared sum: {squared.sum()}")
print(f"Square root sum: {sqrt_img.sum():.2f}")

# ============================================
# PART 7: VISUALIZATION WITH MATPLOTLIB
# ============================================
print("\n" + "=" * 60)
print("[PART 7] Creating Visualizations...")
print("=" * 60)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Image Matrix Manipulation Project', fontsize=20, fontweight='bold', y=0.995)

# Define all images to display
images_to_plot = [
    (image_matrix, "Original Image"),
    (checkerboard, "Checkerboard Pattern"),
    (gradient, "Gradient Pattern"),
    (transposed, "Transposed"),
    (vertical_flip, "Vertical Flip"),
    (horizontal_flip, "Horizontal Flip"),
    (both_flips, "Both Flips (180°)"),
    (rotate_90, "90° Rotation"),
    (brightness_increase, "Brightness +50%"),
    (brightness_decrease, "Brightness -50%"),
    (inverted, "Inverted Colors"),
    (np.clip(edges, 0, 1), "Edge Detection")
]

# Plot all images
for idx, (img, title) in enumerate(images_to_plot, 1):
    ax = plt.subplot(3, 4, idx)
    im = ax.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar for each subplot
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('image_matrix_transformations.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: image_matrix_transformations.png")

# ============================================
# PART 8: SEABORN HEATMAPS
# ============================================
print("\n" + "=" * 60)
print("[PART 8] Creating Seaborn Heatmaps...")
print("=" * 60)

fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle('Seaborn Heatmap Analysis', fontsize=20, fontweight='bold')

heatmap_data = [
    (image_matrix, "Original Matrix", "binary"),
    (transposed, "Transposed", "binary"),
    (gradient, "Gradient", "viridis"),
    (brightness_increase, "Brightness Increased", "YlOrRd"),
    (inverted, "Inverted", "gray_r"),
    (df_image, "Pandas DataFrame", "coolwarm")
]

for ax, (data, title, cmap) in zip(axes.flatten(), heatmap_data):
    sns.heatmap(data, annot=True, fmt='.1f', cmap=cmap, 
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
                ax=ax, vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('seaborn_heatmaps.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: seaborn_heatmaps.png")

# ============================================
# PART 9: STATISTICAL VISUALIZATION
# ============================================
print("\n" + "=" * 60)
print("[PART 9] Creating Statistical Plots...")
print("=" * 60)

fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle('Statistical Analysis of Image Matrix', fontsize=18, fontweight='bold')

# 1. Histogram of pixel intensities
axes[0, 0].hist(image_matrix.flatten(), bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Histogram of Pixel Values', fontweight='bold')
axes[0, 0].set_xlabel('Pixel Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# 2. Row-wise intensity plot
row_sums = image_matrix.sum(axis=1)
axes[0, 1].bar(range(10), row_sums, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Row-wise Intensity Sum', fontweight='bold')
axes[0, 1].set_xlabel('Row Index')
axes[0, 1].set_ylabel('Sum of Pixels')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Column-wise intensity plot
col_sums = image_matrix.sum(axis=0)
axes[1, 0].plot(range(10), col_sums, marker='o', linewidth=2, 
                markersize=8, color='green', label='Column Sums')
axes[1, 0].fill_between(range(10), col_sums, alpha=0.3, color='green')
axes[1, 0].set_title('Column-wise Intensity Sum', fontweight='bold')
axes[1, 0].set_xlabel('Column Index')
axes[1, 0].set_ylabel('Sum of Pixels')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Comparison of transformations
transformations = ['Original', 'Transposed', 'V-Flip', 'H-Flip', 'Inverted']
sums = [image_matrix.sum(), transposed.sum(), vertical_flip.sum(), 
        horizontal_flip.sum(), inverted.sum()]
axes[1, 1].barh(transformations, sums, color=['blue', 'orange', 'green', 'red', 'purple'], 
                edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Total Intensity Comparison', fontweight='bold')
axes[1, 1].set_xlabel('Total Pixel Sum')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('statistical_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: statistical_analysis.png")

# Show all plots
plt.show()



# Main Points
#Created 10x10 image matrices (random, checkerboard, gradient)
#Performed transpose operations
#Applied flip operations using NumPy slicing
#djusted brightness using scalar multiplication
# Analyzed data using Pandas DataFrame
# Visualized with Matplotlib (12 transformations)
#Created Seaborn heatmaps (6 variations)
# Generated statistical analysis plots
#Total images generated: 3 PNG files
# 1. image_matrix_transformations.png
# 2. seaborn_heatmaps.png
# 3. statistical_analysis.png

