
#Day 1: NumPy Fundamentals & Performance

#NumPy Arrays

#Purpose: NumPy arrays allow you to perform fast mathematical operations on large datasets, store data efficiently, and work with
#multi-dimensional data structures that are essential for scientific computing, data analysis, and machine learning.

#Examples of Creating NumPy Arrays
#Creating arrays from scratch for a dataset
import numpy as np

# Create temperature readings for a week
temperatures = np.array([72, 75, 78, 73, 71, 74, 76])
print(temperatures)  # [72 75 78 73 71 74 76]

# Why: You need to convert raw data into a NumPy array to perform calculations
avg_temp = np.mean(temperatures)
print(f"Average temperature: {avg_temp}°F")

#Example: Creating arrays with specific patterns
# Create a dataset of 100 evenly spaced points for plotting
x = np.linspace(0, 10, 100)  # 100 points from 0 to 10
y = np.sin(x)  # Calculate sine values

# Why: Essential for creating smooth curves in data visualization
# or generating training data for models

#Example: Creating multi-dimensional arrays (matrices)
# Create a 3x3 identity matrix for linear algebra
identity = np.eye(3)
print(identity)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Create a grid of zeros to represent an image
image = np.zeros((256, 256, 3))  # 256x256 RGB image

# Why: Matrices are used in linear algebra, image processing,
# and neural network operations


#Example: Filtering data with boolean indexing
# Sales data for different products
sales = np.array([120, 450, 89, 320, 510, 95, 380])

# Find all products with sales above 300
high_performers = sales[sales > 300]
print(high_performers)  # [450 320 510 380]
   
# Why: Quickly filter large datasets without writing loops
# Essential for data cleaning and analysis
#Reshaping data for machine learning
# You have 12 data points that need to be organized into batches
data = np.arange(12)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Reshape into 3 batches of 4 samples each
batches = data.reshape(3, 4)
print(batches)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Why: Machine learning models require specific input shapes
# (batch_size, features) - reshaping is constant in ML workflows
#Combining arrays (stacking data)
# You have features from different sources
ages = np.array([25, 30, 35, 40])
incomes = np.array([50000, 60000, 75000, 80000])
experience = np.array([2, 5, 8, 12])

# Stack them into a feature matrix for a model
features = np.column_stack([ages, incomes, experience])
print(features)
# [[   25 50000     2]
#  [   30 60000     5]
#  [   35 75000     8]
#  [   40 80000    12]]

# Why: Combining separate arrays into structured datasets
# is crucial for preparing data for analysis or modeling

