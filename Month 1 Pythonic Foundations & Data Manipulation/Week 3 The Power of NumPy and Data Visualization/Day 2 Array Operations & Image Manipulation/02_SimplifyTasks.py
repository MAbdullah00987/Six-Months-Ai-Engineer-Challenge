

#Day 2: Array Operations & Image Manipulation - Complete Guide
'''
#1. 2D Arrays (Matrices)
#Definition: Arrays with rows and columns, like a spreadsheet or table.
#Simple Explanation: Think of a 2D array as a grid. The first number tells you which row, the second tells you which column.

import numpy as np

arr_2d = np.array([1,2,3],
                  [4,5,6],
                  [4,6,7])

print(arr_2d)

# Accessing elements
print(arr_2d[0, 0])    # 1 (row 0, col 0)
print(arr_2d[1, 2])    # 6 (row 1, col 2)
print(arr_2d[2, 1])    # 8 (row 2, col 1)

# Get entire row
print(arr_2d[1])       # [4 5 6] (all of row 1)

# Get entire column
print(arr_2d[:, 1])    # [2 5 8] (all of column 1)

# Array properties
print(arr_2d.shape)    # (3, 3) - 3 rows, 3 columns
print(arr_2d.size)     # 9 - total elements
print(arr_2d.ndim)     # 2 - number of dimensions

#2. Array Reshaping - .reshape()
##Definition: Changes the shape of an array without changing its data.
#Simple Explanation: Like rearranging building blocks - same blocks, different arrangement. The total number of elements must stay the same.

# 1D to 2D
arr_1d = np.arange(12)
print(arr_1d)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

arr_2d = arr_1d.reshape(3, 4)  # 3 rows, 4 columns
print(arr_2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

arr_2d = arr_1d.reshape(4, 3)  # 4 rows, 3 columns
print(arr_2d)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

# 2D to 1D (flatten)
flattened = arr_2d.reshape(-1)  # -1 means "figure it out"
print(flattened)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# 2D to 3D (like stacking images)
arr_3d = np.arange(24).reshape(2, 3, 4)  # 2 layers, 3 rows, 4 cols
print(arr_3d.shape)  # (2, 3, 4)

# Using -1 (auto-calculate dimension)
arr = np.arange(20).reshape(4, -1)  # 4 rows, auto columns
print(arr.shape)  # (4, 5)

# Common mistake - must have same total elements
# np.arange(10).reshape(3, 4)  # ERROR! 10 ≠ 12

#3. Transpose - .transpose() or .T
#Definition: Flips an array over its diagonal (swaps rows and columns).
#Simple Explanation: Like rotating a table 90 degrees - rows become columns, columns become rows.

# Original array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print("Original shape:", arr.shape)  # (2, 3)
print(arr)
# [[1 2 3]
#  [4 5 6]]

# Transpose method 1
transposed = arr.T
print("Transposed shape:", transposed.shape)  # (3, 2)
print(transposed)
# [[1 4]
#  [2 5]
#  [3 6]]

# Transpose method 2 (same result)
transposed = arr.transpose()

# Visual explanation
print("Row 0 became column 0:", arr[0], "→", transposed[:, 0])
print("Row 1 became column 1:", arr[1], "→", transposed[:, 1])

# Practical use - matrix multiplication needs matching dimensions
A = np.array([[1, 2]])      # shape (1, 2)
B = np.array([[3, 4]])      # shape (1, 2)
# A @ B won't work, but A @ B.T works:
result = A @ B.T            # (1, 2) @ (2, 1) = (1, 1)
print(result)               # [[11]] (1*3 + 2*4)

#4. Boolean Indexing (Boolean Masks)
#Definition: Using True/False conditions to filter arrays.
#Simple Explanation: Like a filter - keep only elements that meet your condition. NumPy checks each element and keeps the "True" ones.

arr = np.array([10, 25, 30, 15, 40, 35, 5])

# Create boolean mask (True/False array)
mask = arr > 20
print(mask)  # [False  True  True False  True  True False]

# Use mask to filter
filtered = arr[mask]
print(filtered)  # [25 30 40 35]

# One-line version (most common)
result = arr[arr > 20]
print(result)  # [25 30 40 35]

# Multiple conditions (use & for AND, | for OR)
result = arr[(arr > 15) & (arr < 35)]
print(result)  # [25 30]

result = arr[(arr < 15) | (arr > 35)]
print(result)  # [10  5 40]

# Modify values using boolean indexing
arr[arr < 20] = 0  # Set all values < 20 to 0
print(arr)  # [ 0 25 30  0 40 35  0]

# 2D arrays
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Get all elements > 5
print(arr_2d[arr_2d > 5])  # [6 7 8 9]

# Replace values
arr_2d[arr_2d % 2 == 0] = -1  # Make even numbers -1
print(arr_2d)
# [[ 1 -1  3]
#  [-1  5 -1]
#  [ 7 -1  9]]

# Practical example: remove outliers
data = np.array([10, 12, 11, 200, 13, 9, 15])
mean = np.mean(data)
std = np.std(data)
# Keep only values within 2 standard deviations
clean_data = data[np.abs(data - mean) < 2 * std]
print(clean_data)  # [10 12 11 13  9 15] (removed 200)

#5. Fancy Indexing
#Definition: Using arrays of indices to select multiple elements at once.
#Simple Explanation: Instead of picking one element, give NumPy a list of positions and it grabs all of them at once.

arr = np.array([10, 20, 30, 40, 50, 60, 70])

# Select specific indices
indices = [0, 2, 5]  # Want elements at positions 0, 2, 5
result = arr[indices]
print(result)  # [10 30 60]

# Can use NumPy array as indices too
indices = np.array([1, 3, 4])
result = arr[indices]
print(result)  # [20 40 50]

# 2D fancy indexing
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Select specific rows
rows = [0, 2]  # Want rows 0 and 2
result = arr_2d[rows]
print(result)
# [[1 2 3]
#  [7 8 9]]

# Select specific elements by (row, col) pairs
rows = [0, 1, 2]
cols = [0, 1, 2]  # Get diagonal: (0,0), (1,1), (2,2)
result = arr_2d[rows, cols]
print(result)  # [1 5 9]

# Random sampling
arr = np.arange(100)
random_indices = np.random.randint(0, 100, size=10)
random_sample = arr[random_indices]
print(random_sample)  # 10 random values from arr

# Reorder elements
arr = np.array([10, 20, 30, 40])
new_order = [3, 1, 0, 2]  # Reverse and shuffle
result = arr[new_order]
print(result)  # [40 20 10 30]

#6. Stacking Arrays - np.vstack() and np.hstack()
#Vertical Stack - np.vstack()
#Definition: Stack arrays vertically (on top of each other, adds rows).
#Simple Explanation: Like stacking pancakes - put one array on top of another.

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Stack vertically
result = np.vstack([arr1, arr2])
print(result)
# [[1 2 3]
#  [4 5 6]]

# Multiple arrays
arr3 = np.array([7, 8, 9])
result = np.vstack([arr1, arr2, arr3])
print(result)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# 2D arrays
top = np.array([[1, 2], [3, 4]])
bottom = np.array([[5, 6], [7, 8]])
result = np.vstack([top, bottom])
print(result)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ============================================
# STEP 1: CREATE A SAMPLE IMAGE
# ============================================
print("=" * 60)
print("CREATING SAMPLE IMAGE")
print("=" * 60)

# Create a simple colored image (100x100 pixels)
# Red gradient on top, blue gradient on bottom
width, height = 200, 200
sample_image = np.zeros((height, width, 3), dtype=np.uint8)

# Top half: Red gradient
for i in range(height // 2):
    sample_image[i, :, 0] = int(255 * i / (height // 2))  # Red channel

# Bottom half: Blue gradient
for i in range(height // 2, height):
    sample_image[i, :, 2] = int(255 * (i - height // 2) / (height // 2))  # Blue channel

# Add green in the middle
sample_image[height // 2 - 20:height // 2 + 20, :, 1] = 255

print(f"Image shape: {sample_image.shape}")
print(f"Image dtype: {sample_image.dtype}")
print(f"Min pixel value: {sample_image.min()}")
print(f"Max pixel value: {sample_image.max()}")

