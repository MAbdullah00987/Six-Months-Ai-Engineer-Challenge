

#Performance Comparison - Compare loop vs vectorized operations

import numpy as np
import time

#Example One : Simple Addition

print("="*90)
print("Example 1: Adding Arrays")
print("="*90)

# Create large arrays for testing
size = 1000000
array1 = np.random.randint(1,100,size)
array2 = np.random.randint(1,100,size)

# Convert to lists for loop comparison
list1 = array1.tolist()
list2 = array2.tolist()

# METHOD 1: Using Loop

start_time = time.time()
result_loop = []
for i in range(len(list1)):
    result_loop.append(list1[i] + list2[i])
loop_time = time.time() - start_time

# METHOD 2: Using Vectorized Operation
start_time = time.time()
result_vectorized = array1 + array2
vectorized_time = time.time() - start_time


# Display Results
print(f"\nArray size: {size:,} elements")
print(f"Loop time: {loop_time:.6f} seconds")
print(f"Vectorized time: {vectorized_time:.6f} seconds")
print(f"Speedup: {loop_time/vectorized_time:.2f}x faster")


#EXAMPLE Two: Multiplication and Square
print("\n" + "=" * 50)
print("EXAMPLE Two: Multiply by 5 and Square")
print("=" * 50)

numbers = np.random.randint(1, 50, size)
numbers_list = numbers.tolist()

# METHOD 1: Using Loop
start_time = time.time()
result_loop = []
for num in numbers_list:
    result_loop.append((num * 5) ** 2)
loop_time = time.time() - start_time

# METHOD 2: Using Vectorized Operation
start_time = time.time()
result_vectorized = (numbers * 5) ** 2
vectorized_time = time.time() - start_time

print(f"\nArray size: {size:,} elements")
print(f"Loop time: {loop_time:.6f} seconds")
print(f"Vectorized time: {vectorized_time:.6f} seconds")
print(f"Speedup: {loop_time/vectorized_time:.2f}x faster")

#EXAMPLE Three: Finding Sum and Mean

print("\n" + "=" * 50)
print("EXAMPLE 3: Calculate Sum and Mean")
print("=" * 50)


print("\n" + "=" * 50)
print("EXAMPLE 3: Calculate Sum and Mean")
print("=" * 50)

data = np.random.randint(1, 1000, size)
data_list = data.tolist()

# METHOD 1: Using Loop
start_time = time.time()
total = 0
for num in data_list:
    total += num
mean_loop = total / len(data_list)
loop_time = time.time() - start_time

# METHOD 2: Using Vectorized Operation
start_time = time.time()
total_vectorized = np.sum(data)
mean_vectorized = np.mean(data)
vectorized_time = time.time() - start_time

print(f"\nArray size: {size:,} elements")
print(f"Loop time: {loop_time:.6f} seconds")
print(f"Vectorized time: {vectorized_time:.6f} seconds")
print(f"Speedup: {loop_time/vectorized_time:.2f}x faster")