import pandas as pd
import numpy as np
import time

np.random.seed(42)
large_df = pd.DataFrame({
    'Group': np.random.choice(['A', 'B', 'C', 'D'], 100000),
    'Value': np.random.randn(100000)
})

print("Large Dataset:")
print(large_df.head(10))
print(f"Shape: {large_df.shape}")

# Method 1: Using transform (FAST PATH)
print("\n=== Method 1: Fast Path (Optimized) ===")
start_time = time.time()
for _ in range(10):
    large_df['Value'] - large_df.groupby('Group')['Value'].transform('mean')
end_time = time.time()
print(f"Time taken (10 runs): {end_time - start_time:.4f} seconds")

# Method 2: Using transform with lambda (SLOWER - no fast path)
print("\n=== Method 2: Lambda (No Fast Path) ===")
start_time = time.time()
for _ in range(10):
    large_df['Value'] - large_df.groupby('Group')['Value'].transform(lambda x: x.mean())
end_time = time.time()
print(f"Time taken (10 runs): {end_time - start_time:.4f} seconds")

#Common Fast Path Operations
# Reset to original sales data
sales = pd.DataFrame({
    'Region': ['North', 'North', 'North', 'South', 'South', 'South', 'East', 'East'],
    'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
    'Sales': [100, 150, 200, 120, 180, 220, 90, 160],
    'Quantity': [10, 15, 20, 12, 18, 22, 9, 16]
})

grouped = sales.groupby('Region')

# Fast path operations (use string names)
print("=== Fast Path Operations ===")

# 1. Deviation from group mean
sales['Sales_vs_Mean'] = sales['Sales'] - grouped['Sales'].transform('mean')
print("\n1. Deviation from Mean:")
print(sales[['Region', 'Sales', 'Sales_vs_Mean']])

# 2. Percentage of group total
sales['Pct_of_Total'] = sales['Sales'] / grouped['Sales'].transform('sum') * 100
print("\n2. Percentage of Group Total:")
print(sales[['Region', 'Sales', 'Pct_of_Total']])

# 3. Ratio to group max
sales['Ratio_to_Max'] = sales['Sales'] / grouped['Sales'].transform('max')
print("\n3. Ratio to Group Maximum:")
print(sales[['Region', 'Sales', 'Ratio_to_Max']])

# 4. Difference from group min
sales['Above_Min'] = sales['Sales'] - grouped['Sales'].transform('min')
print("\n4. Above Group Minimum:")
print(sales[['Region', 'Sales', 'Above_Min']])

# 5. Cumulative sum within groups
sales['Cumulative_Sales'] = grouped['Sales'].transform('cumsum')
print("\n5. Cumulative Sum:")
print(sales[['Region', 'Product', 'Sales', 'Cumulative_Sales']])
