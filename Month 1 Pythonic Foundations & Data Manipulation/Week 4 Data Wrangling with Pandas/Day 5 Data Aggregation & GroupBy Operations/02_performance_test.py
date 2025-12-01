import pandas as pd
import numpy as np
import time

# Why Column Selection Matters
# Large dataset example
large_data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 1000000),
    'Value1': np.random.randn(1000000),
    'Value2': np.random.randn(1000000),
    'Value3': np.random.randn(1000000),
    'Value4': np.random.randn(1000000),
    'Value5': np.random.randn(1000000)
})

print("Benchmarking...")

# Efficient: Only process needed column
start_time = time.time()
# Run 100 times to get a measurable difference
for _ in range(10):
    large_data.groupby('Category')['Value1'].mean()
end_time = time.time()
print(f"Efficient method (Select then Group): {end_time - start_time:.4f} seconds")

# Less efficient: Process all columns then select
start_time = time.time()
for _ in range(10):
    large_data.groupby('Category').mean()['Value1']
end_time = time.time()
print(f"Less efficient method (Group then Select): {end_time - start_time:.4f} seconds")

print("\nColumn selection improves performance on large datasets!")
