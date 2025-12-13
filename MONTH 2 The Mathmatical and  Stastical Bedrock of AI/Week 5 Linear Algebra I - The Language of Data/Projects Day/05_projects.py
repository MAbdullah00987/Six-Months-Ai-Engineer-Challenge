
#5: Image as a Matrix: Represent a simple black and white image as a matrix and perform operations like transposition.

import numpy as np
import matplotlib.pyplot as plt

# Create a simple black and white image as a matrix
# 0 represents black, 1 represents white
print("Creating a simple black and white image as a matrix...")
print("=" * 50)

# Example 1: Create a simple pattern (8x8 image)
image_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

print("\nOriginal Image Matrix:")
print(image_matrix)
print(f"Shape: {image_matrix.shape}")

# Perform matrix operations
print("\n" + "=" * 50)
print("Matrix Operations:")
print("=" * 50)

# 1. Transpose
transposed_image = np.transpose(image_matrix)
print("\n1. Transposed Image Matrix:")
print(transposed_image)
print(f"Shape: {transposed_image.shape}")

# 2. Horizontal flip (flip left-right)
horizontal_flip = np.fliplr(image_matrix)
print("\n2. Horizontal Flip:")
print(horizontal_flip)

# 3. Vertical flip (flip up-down)
vertical_flip = np.flipud(image_matrix)
print("\n3. Vertical Flip:")
print(vertical_flip)

# 4. Rotate 90 degrees
rotate_90 = np.rot90(image_matrix)
print("\n4. Rotate 90° Counter-clockwise:")
print(rotate_90)

# 5. Invert colors (negative)
inverted_image = 1 - image_matrix
print("\n5. Inverted Image (Negative):")
print(inverted_image)

# Visualize all operations
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Image Matrix Operations', fontsize=16, fontweight='bold')

# Original
axes[0, 0].imshow(image_matrix, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Transposed
axes[0, 1].imshow(transposed_image, cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title('Transposed')
axes[0, 1].axis('off')

# Horizontal Flip
axes[0, 2].imshow(horizontal_flip, cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title('Horizontal Flip')
axes[0, 2].axis('off')

# Vertical Flip
axes[1, 0].imshow(vertical_flip, cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title('Vertical Flip')
axes[1, 0].axis('off')

# Rotated
axes[1, 1].imshow(rotate_90, cmap='gray', vmin=0, vmax=1)
axes[1, 1].set_title('Rotated 90°')
axes[1, 1].axis('off')

# Inverted
axes[1, 2].imshow(inverted_image, cmap='gray', vmin=0, vmax=1)
axes[1, 2].set_title('Inverted (Negative)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Additional matrix operations
print("\n" + "=" * 50)
print("Additional Matrix Statistics:")
print("=" * 50)
print(f"Mean value: {np.mean(image_matrix):.2f}")
print(f"Sum of all pixels: {np.sum(image_matrix)}")
print(f"Number of white pixels: {np.sum(image_matrix == 1)}")
print(f"Number of black pixels: {np.sum(image_matrix == 0)}")
print(f"Percentage of white pixels: {(np.sum(image_matrix == 1) / image_matrix.size) * 100:.2f}%")