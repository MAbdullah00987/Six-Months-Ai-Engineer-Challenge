

#Day 1: The Scikit-Learn API & Simple Linear Regression

#Objective: Master the "Estimator" API. Every model in sklearn follows the same structure: fit() to train, predict() to test.

#Concept: Supervised Learning, Features ($X$) vs. Targets ($y$), Train/Test Split.

#Task: Project - Salary Prediction.
#Use a simple dataset (YearsExperience vs. Salary).
#Split data using train_test_split.
#Train a LinearRegression model.
#Visualize the "Best Fit Line" using Matplotlib.


#ML Fundamentals & Scikit-Learn Basics
#Learning Objectives:

#Understand the ML pipeline
#Differentiate supervised vs unsupervised learning
#Learn Scikit-Learn's consistent API

#Study Materials:

#Géron Chapter 1: "The Machine Learning Landscape" (pages 1-35)
#Andrew Ng Course 1, Week 1: "Introduction to Machine Learning"
#Scikit-Learn documentation: Getting Started tutorial

#Part 1: NumPy & Pandas Foundations for ML

import numpy as np
import pandas as pd

print("="*70)
print("SECTION 1: NumPy Arrays - The Foundation of ML")
print("="*70)

# 1.1 Creating Feature Matrix (X) and Target Vector (y)
# This is how data is structured in ML: rows=samples, columns=features

# Feature matrix: house size (sqft), bedrooms, age
X = np.array([
    [1200, 3, 10],  # house 1
    [1500, 4, 5],   # house 2
    [1800, 3, 15],  # house 3
    [2000, 5, 2],   # house 4
    [2200, 4, 8]    # house 5
])

# Target vector: house prices ($1000s)
y = np.array([250, 310, 340, 410, 450])

print("\n1.1 Feature Matrix X (samples × features):")
print(X)
print(f"Shape: {X.shape} - means {X.shape[0]} samples, {X.shape[1]} features")

print("\n1.2 Target Vector y:")
print(y)
print(f"Shape: {y.shape} - means {y.shape[0]} samples")

# 1.3 Array Operations - Essential for ML math
print("\n" + "="*70)
print("SECTION 2: Array Operations for ML Computations")
print("="*70)

# Mean normalization (feature scaling technique)
X_mean = np.mean(X, axis=0)  # mean of each column
X_std = np.std(X, axis=0)    # std dev of each column
X_normalized = (X - X_mean) / X_std

print("\n2.1 Feature Means:", X_mean)
print("Feature Std Devs:", X_std)
print("\n2.2 Normalized Features (mean=0, std=1):")
print(X_normalized)

# Matrix multiplication - core of linear regression
weights = np.array([0.2, 30, -5])  # hypothetical model weights
predictions = X @ weights  # matrix-vector multiplication

print("\n2.3 Matrix Multiplication (X @ weights):")
print(f"Weights: {weights}")
print(f"Predictions: {predictions}")

# 1.4 Boolean indexing - filtering data
print("\n" + "="*70)
print("SECTION 3: Data Filtering (Train/Test Split Logic)")
print("="*70)

# Create train/test split manually
np.random.seed(42)
n_samples = len(y)
indices = np.arange(n_samples)
np.random.shuffle(indices)

split_point = int(0.8 * n_samples)  # 80% train, 20% test
train_idx = indices[:split_point]
test_idx = indices[split_point:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\n3.1 Train set: {len(train_idx)} samples")
print(f"Indices: {train_idx}")
print(f"Test set: {len(test_idx)} samples")
print(f"Indices: {test_idx}")

# 1.5 Advanced NumPy - Broadcasting
print("\n" + "="*70)
print("SECTION 4: Broadcasting - Efficient Computations")
print("="*70)

# Add bias term (column of ones) for linear regression
bias = np.ones((X.shape[0], 1))
X_with_bias = np.concatenate([bias, X], axis=1)

print("\n4.1 Adding bias term:")
print(X_with_bias)
print(f"New shape: {X_with_bias.shape}")

# Element-wise operations (used in activation functions)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.array([-2, -1, 0, 1, 2])
sigmoid_values = sigmoid(z_values)

print("\n4.2 Sigmoid activation (element-wise):")
print(f"z: {z_values}")
print(f"sigmoid(z): {sigmoid_values}")

print("\n" + "="*70)
print("SECTION 5: Pandas for Real-World Data")
print("="*70)

# 5.1 Create DataFrame (typical ML data structure)
df = pd.DataFrame(X, columns=['sqft', 'bedrooms', 'age'])
df['price'] = y

print("\n5.1 DataFrame structure:")
print(df)
print("\n5.2 DataFrame info:")
print(df.info())

# 5.2 Descriptive statistics
print("\n5.3 Statistical summary:")
print(df.describe())

# 5.3 Feature engineering with Pandas
df['price_per_sqft'] = df['price'] / df['sqft'] * 1000  # new feature
df['is_old'] = (df['age'] > 10).astype(int)  # binary feature

print("\n5.4 Feature engineering:")
print(df[['sqft', 'price', 'price_per_sqft', 'age', 'is_old']])

# 5.4 Correlation analysis
print("\n5.5 Feature correlations with price:")
correlations = df.corr()['price'].sort_values(ascending=False)
print(correlations)

# 5.5 Groupby operations (useful for categorical features)
df['size_category'] = pd.cut(df['sqft'], bins=[0, 1500, 2000, 3000], 
                              labels=['small', 'medium', 'large'])
print("\n5.6 Grouped statistics:")
print(df.groupby('size_category', observed=True)['price'].agg(['mean', 'std', 'count']))

# 5.6 Handling missing data (critical for real datasets)
df_with_missing = df.copy()
df_with_missing.loc[1, 'bedrooms'] = np.nan
df_with_missing.loc[3, 'age'] = np.nan

print("\n5.7 Missing data handling:")
print(f"Missing values:\n{df_with_missing.isnull().sum()}")

# Imputation strategies - FIXED: Only calculate mean for numeric columns
df_filled = df_with_missing.fillna(df_with_missing.mean(numeric_only=True))
print(f"\nAfter mean imputation:\n{df_filled[['sqft', 'bedrooms', 'age', 'price']]}")

print("\n" + "="*70)
print("SECTION 6: Advanced Indexing for ML Workflows")
print("="*70)

# 6.1 Fancy indexing
selected_features = ['sqft', 'bedrooms']
X_selected = df[selected_features].values

print("\n6.1 Feature selection:")
print(f"Selected features: {selected_features}")
print(X_selected)

# 6.2 Boolean masking for data filtering
expensive_houses = df[df['price'] > 350]
print("\n6.2 Filtering expensive houses (>$350k):")
print(expensive_houses)

# 6.3 Multi-condition filtering
modern_large = df[(df['age'] < 10) & (df['sqft'] > 1800)]
print("\n6.3 Modern & large houses:")
print(modern_large)


print("KEY TAKEAWAYS")
"""
1. NumPy arrays = foundation for ML (matrices for X, vectors for y)
2. Shape matters: (n_samples, n_features) is standard
3. Array operations: mean, std, @ (matmul) are core ML operations
4. Broadcasting enables efficient computations
5. Pandas DataFrames organize data with column names
6. Feature engineering creates new predictive variables
7. Correlation analysis identifies important features
8. Missing data handling is crucial for real datasets
9. Boolean indexing enables train/test splits and filtering
10. These skills are prerequisites for scikit-learn workflows
"""