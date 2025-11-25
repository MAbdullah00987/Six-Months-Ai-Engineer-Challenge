

#Day 1: NumPy Fundamentals & Performance - Complete Guide

#Day 1: NumPy Fundamentals & Performance - Complete Guide
#1. What is NumPy?
#Definition: NumPy (Numerical Python) is a Python library for working with arrays and performing mathematical operations efficiently.
#Simple Explanation: Think of NumPy as a supercharged version of Python lists. Instead of working with one number at a time, NumPy lets you
# work with entire collections of numbers at once, making calculations much faster.

'''
import numpy as np

#python - Slow way of creating list 
my_list = [1,2,3,4]

#python - fast way of creating list
my_array = np.array([1,4,5,7])

print(my_list)
print(my_array)

#Question: How do I create an array of numbers from 0 to 9?

import numpy as np 

array = np.arange(10)
print (array)

array = np.arange(-10)
print(array)
# but if i need to create an of numbers from 9 to 0
# there are three methods i can do that 

#method one 
array2 = np.arange(9,-1,-1)
print(array2)

#np.arange(9, -1, -1) means:
##Start at 9
#Stop before -1 (so it includes 0)
#Step by -1 (go backwards)

#method two 
array3 = np.flip(np.arange(10))
print(array3)


# Create 0 to 9, then reverse with slicing
array4 = np.arange(10)[::-1]
print(array4)



#2. Question: How can I find the sum and average of an array?
import numpy as np

numbers = np.array([10,20,30,40])
total = np.sum(numbers)
average = np.mean(numbers)
print(f"sum: {total}")
print(f"average: {average}")

#Calculate percentage increase/decrease between elements

numbers1 = np.array([1,2,3,4,56,7,8])
# Percentage change from one element to the next

percentage_change = ((numbers1[1:]) - numbers1[:-1] / numbers1[:-1]) * 100
print(percentage_change)

#Using division method np.divide()

numbers2 = np.array([10,20,30,50,60])
total = np.sum(numbers)

percentages = np.divide(numbers,total)*100
print(percentages)


#3. Question: How do I create a 3x3 matrix filled with zeros?

import numpy as np

matrix = np.zeros((3,3))
matrixx = np.ones((3,3))

print(matrix)
print(matrixx)

#Matrix filled only five values using np.full()
import numpy as np

matrix1 = np.full((3,3),5)
print (matrix1)

#matrix filled with any specific value
matrix = np.full((3, 3), 7)
print(matrix)

matrix = np.full((3, 3), 3.14)
print(matrix)

#Identity matrix (diagonal ones, rest zeros)

matrix = np.eye(3)
print(matrix)

#Matrix with custom values
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(matrix)


#4. Question: How can I multiply each element in an array by 2?

import numpy as np 
array = np.array(([12,56,78,78]))

doubled = array * 5
print(doubled)


#5. Question: How do I find the maximum and minimum values in an array?
import numpy as np 
values = np.array([1,4,7,9,5])

maximum = np.max(values)
minimum = np.min(values)

print(f"maximum: {maximum}")
print(f"minimum: {minimum}")

'''

'''
#2. Creating NumPy Arrays
#np.array() - Convert lists to arrays
#Definition: Converts Python lists or tuples into NumPy arrays.
#Simple Explanation: Takes your regular Python list and transforms it into a NumPy array that can do math operations faster.

import numpy as np

array = np.array([1,2,5,6])
print(array)

array_1 = np.array([[3,4,6,7],[3,7,9,0]])
print(array_1)


#np.arange() - Create sequences
#Definition: Creates arrays with evenly spaced values within a given range (like Python's range() but returns an array).
#Simple Explanation: Generates a sequence of numbers automatically, like counting from start to stop.

arr1 = np.arange(12)
print(arr1)

arr2 = np.arange(5,12,3)
print(arr2)

arr3 = np.arange(0,1,0.2)


#np.linspace() - Create evenly spaced points
#Definition: Creates an array with a specified number of evenly spaced values between a start and stop value.
#Simple Explanation: Instead of saying "count by 2s", you say "give me exactly 5 numbers between 0 and 10".


import numpy as np

arr3 = np.linspace(0,10,15)
print(arr3)

arr4 = np.linspace(0,3,6)
print(arr4)


#Key Difference from arange:
#arange(0, 10, 2) → "count from 0 to 10 by 2s" → [0, 2, 4, 6, 8]
#linspace(0, 10, 5) → "give me 5 numbers from 0 to 10" → [0, 2.5, 5, 7.5, 10]


#3. Array Slicing - arr[start:stop:step]
#Definition: Extracting portions of an array using index notation.
#Simple Explanation: Like cutting a piece of cake - you specify where to start cutting, where to stop, and how big each slice should be.
#Syntax: array[start:stop:step]

#start: where to begin (inclusive)
#stop: where to end (exclusive)
#step: jump size (default is 1)

import numpy as np

arr = np.array([0,1,2,3,4,5,6,7,8,9])

print(arr[2:6]) # [2 3 4 5]

print(arr[::2]) # [0 2 4 6 8]

print(arr[3:])  # [3 4 5 6 7 8 9]

print(arr[:5])  # [0 1 2 3 4]

print(arr[::-1])  # [9 8 7 6 5 4 3 2 1 0]

# 2D array slicing
arr2d = np.array([[1,2,4,5],[2,3,5,6],[3,6,9,6],[2,8,4,0]])
print(arr2d[0:2, 1:3])



#4. Vectorization vs Loops
#Definition: Vectorization means applying operations to entire arrays at once instead of looping through elements one by one.
#Simple Explanation: Instead of adding numbers one at a time (slow), NumPy adds all numbers simultaneously (fast) using optimized C code under the hood.

import numpy as np

#python example 
numbers = [1,2,3,4,5]
result = []
for num in numbers:
    result.apend(num * 2)
print(result)

#vectorized example
numbers = np.array([1,4,8,6])
result = numbers * 2
print(result)

#Why Faster?
#Loops: Python checks each number's type, does the math, stores result (slow)
#Vectorization: NumPy assumes all numbers are same type, does math in optimized C code (fast)

import time
arr_list = list(range(1000000))
start = time.time()
result_loop =[x*2 for x in arr_list]
print(f"Loop time: {time.time() - start:.4f} seconds")

arr_num = np.arange(1000000)
start = time.time()
result_vec = arr_num * 2
print(f"Vectorized time: {time.time() - start:.4f} seconds")



#5. Basic Array Operations
#Definition: Mathematical operations applied to arrays.
#Simple Explanation: Do math on entire arrays just like you would with single numbers.

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Arithmetic operations
print(arr + 10)      # [11 12 13 14 15]
print(arr * 2)       # [ 2  4  6  8 10]
print(arr ** 2)      # [ 1  4  9 16 25] (squared)
print(arr / 2)       # [0.5 1.  1.5 2.  2.5]

# Operations between arrays
arr2 = np.array([10, 20, 30, 40, 50])
print(arr + arr2)    # [11 22 33 44 55]
print(arr * arr2)    # [ 10  40  90 160 250]

# Boolean operations
print(arr > 3)      
print(arr[arr > 3]) 


#6. Statistical Functions
#Mean (Average)
#Definition: Sum of all values divided by count.
#Simple Explanation: The "middle" value if you add everything up and divide equally.

import numpy as np

arr = np.array([1,3,6,8])
mean = np.mean(arr)
print(mean)

#Median
#Definition: The middle value when data is sorted.
#Simple Explanation: Line up all numbers smallest to largest, pick the middle one.

arr = np.array([10, 20, 30, 40, 50])
median = np.median(arr)
print(median)  # 30.0 (middle value)

arr2 = np.array([10, 20, 25, 30, 40, 50])
print(np.median(arr2))  

#Standard Deviation (std)
#Definition: Measures how spread out numbers are from the mean.
#Simple Explanation: Tells you if numbers are close together or far apart. Small std = numbers are similar, large std = numbers vary a lot.

arr1 = np.array([10, 11, 12, 13, 14])
print(np.std(arr1))  # ~1.41 (small spread)

# Numbers far apart
arr2 = np.array([1, 50, 100, 150, 200])
print(np.std(arr2))  # ~73.9 (large spread)

# All same numbers
arr3 = np.array([5, 5, 5, 5, 5])
print(np.std(arr3))  # 0.0 (no spread)

'''
#7. Random Data Generation
#Definition: Creating arrays filled with random numbers.
#Simple Explanation: Let NumPy pick random numbers for you - useful for testing and simulations.

import numpy as np

random_arr = np.random(6)
print(random_arr)

# Random integers
random_ints = np.random.randint(1, 100, size=10)
print(random_ints)  # [45 23 67 89 12 34 78 56 90 21]

# Normal distribution (bell curve)
normal_data = np.random.randn(1000)  # Mean=0, Std=1
print(np.mean(normal_data))  # ~0
print(np.std(normal_data))   # ~1

