
#Vectorization vs Loops
#Purpose: Vectorization allows you to perform operations on entire arrays at once using optimized C code,
#rather than writing explicit Python loops that process elements one-by-one.


#Examples Comparing Loops vs Vectorization
#Calculating Squares of Numbers
#Using Loops (Slow)
import numpy as np
import time

# Create array of 1 million numbers
numbers = np.arange(1000000)

# Loop approach
start = time.time()
squares_loop = []
for num in numbers:
    squares_loop.append(num ** 2)
squares_loop = np.array(squares_loop)
loop_time = time.time() - start
print(f"Loop time: {loop_time:.4f} seconds")
#Using Vectorization (Fast)
# Vectorized approach
start = time.time()
squares_vectorized = numbers ** 2
vectorized_time = time.time() - start
print(f"Vectorized time: {vectorized_time:.4f} seconds")
print(f"Speedup: {loop_time / vectorized_time:.1f}x faster")

# Typical result: Vectorization is 50-100x faster

#Why this matters: When processing large datasets (sensor data, financial records, image pixels), the time difference becomes massive—seconds vs minutes or even hours.

#Applying Conditional Logic
Using Loops (Slow)
python# Temperature data in Celsius
temps_celsius = np.array([15, 22, 30, 18, 25, 32, 28, 20])

# Convert to Fahrenheit, but cap at 85°F using a loop
temps_fahrenheit_loop = []
for temp in temps_celsius:
    fahrenheit = (temp * 9/5) + 32
    if fahrenheit > 85:
        temps_fahrenheit_loop.append(85)
    else:
        temps_fahrenheit_loop.append(fahrenheit)

print("Loop result:", temps_fahrenheit_loop)
Using Vectorization (Fast)
python# Vectorized approach with np.where
temps_fahrenheit = (temps_celsius * 9/5) + 32
temps_capped = np.where(temps_fahrenheit > 85, 85, temps_fahrenheit)

print("Vectorized result:", temps_capped)

# Both give same result: [59. 71.6 85. 64.4 77. 85. 82.4 68.]

# Why: np.where applies conditional logic across entire array instantly
Why this matters: Data cleaning often requires conditional transformations (capping outliers, replacing values). Vectorization makes this trivial on millions of records.

Example 3: Calculating Distance Between Points
Using Loops (Slow)
python# Calculate Euclidean distance between corresponding points
x1 = np.array([1, 2, 3, 4, 5])
y1 = np.array([2, 4, 6, 8, 10])
x2 = np.array([5, 6, 7, 8, 9])
y2 = np.array([10, 12, 14, 16, 18])

# Loop approach
distances_loop = []
for i in range(len(x1)):
    dist = np.sqrt((x2[i] - x1[i])**2 + (y2[i] - y1[i])**2)
    distances_loop.append(dist)

print("Loop distances:", distances_loop)
Using Vectorization (Fast)
python# Vectorized approach - operations on entire arrays
distances_vectorized = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

print("Vectorized distances:", distances_vectorized)

# Both give: [8.944 8.944 8.944 8.944 8.944]

# Why: All subtractions, squares, and square roots happen
# simultaneously across the entire arrays
Why this matters: Calculating distances is common in machine learning (K-nearest neighbors, clustering). With thousands of data points, vectorization is the only practical approach.
