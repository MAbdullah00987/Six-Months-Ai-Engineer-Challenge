
#Chapter 8: Data Wrangling: Join, Combine, and Reshape
#This chapter covers tools for combining multiple datasets and rearranging them into different structural forms.

#1. Hierarchical Indexing (MultiIndex)

#Concept: Enables storing multiple index levels on an axis, allowing higher-dimensional data to be represented in a lower-dimensional form (like a 2D DataFrame).
#Selection: Supports partial indexing (selecting subsets of data) and selecting from "inner" levels.
#Reordering: swaplevel() interchanges index levels, and sort_index() sorts data using specific levels.
#Columns as Index: set_index() converts one or more columns into row indexes, while reset_index() moves the index levels back into columns.

#2. Combining and Merging Datasets

#Database-Style Joins (merge):
#pandas.merge() connects rows based on one or more keys, similar to SQL joins.
#Supports inner (default), left, right, and outer joins.
#Can merge on specific columns (on, left_on, right_on) or indexes (left_index=True).
#Concatenation (concat):
#pandas.concat() glues objects together along an axis (default is axis 0/rows).
#The keys argument creates a hierarchical index to identify the source of the concatenated parts.
#ignore_index=True discards existing indexes and creates a new range index.
#Combine with Overlap: combine_first() "patches" missing data in one object with values from another object, aligning them by index.

#3. Reshaping and Pivoting

#Stacking:
#stack() pivots columns into the rows (creating a taller, longer format).
#nstack() pivots rows into columns (creating a wider format).
#Pivoting to Wide Format:
#pivot() transforms long-format data (one row per observation) into wide format (multiple time series in columns).
#Pivoting to Long Format:
#melt() is the inverse of pivot(). It merges multiple columns into one, producing a longer DataFrame, often used for reshaping data into a format suitable for analysis.


import pandas as pd
import numpy as np


#1. Hierarchical Indexing (MultiIndex)

#Concept: Enables storing multiple index levels on an axis, allowing higher-dimensional data to be represented in a lower-dimensional form (like a 2D DataFrame).
#Selection: Supports partial indexing (selecting subsets of data) and selecting from "inner" levels.
#Reordering: swaplevel() interchanges index levels, and sort_index() sorts data using specific levels.
#Columns as Index: set_index() converts one or more columns into row indexes, while reset_index() moves the index levels back into columns.

#Basic Concept: Creating a MultiIndex
#A MultiIndex allows you to have multiple levels of row or column labels. Think of it like organizing data in nested categories.

# Creating a simple MultiIndex DataFrame
arrays = [
    ['California', 'California', 'Texas', 'Texas', 'New York', 'New York'],
    ['Los Angeles', 'San Francisco', 'Houston', 'Dallas', 'New York City', 'Buffalo']
]

index = pd.MultiIndex.from_arrays(arrays, names=['State', 'City'])
data = np.random.randn(6, 2)
df = pd.DataFrame(data, index=index, columns=['Temperature', 'Humidity'])

print(df)


#Explanation: The DataFrame has two index levels: State (outer) and City (inner). This is more efficient than storing state names repeatedly in a column.

#2. Selection with MultiIndex
#You can select data at different levels of the hierarchy.

# Selecting all data for California
print(df.loc['California'])

# Selecting specific city
print(df.loc[('Texas', 'Houston')])

# Selecting multiple states
print(df.loc[['California', 'Texas']])

# Using xs() for cross-section selection
print(df.xs('Houston', level='City'))

# Selecting from inner level across all outer levels
print(df.xs('Houston', level=1))  # level=1 is the City level


#3. Reordering: swaplevel() and sort_index()
#These functions help reorganize your MultiIndex structure.

# Using our previous df
print("Original:")
print(df)

# Swap the levels (City becomes outer, State becomes inner)
df_swapped = df.swaplevel()
print("\nAfter swaplevel():")
print(df_swapped)

# Sort by the new outer level
df_sorted = df_swapped.sort_index()
print("\nAfter sort_index():")
print(df_sorted)

# Sort by specific level
df_sorted_by_city = df.sort_index(level='City')
print("\nSorted by City (original structure):")
print(df_sorted_by_city)

#Converting Columns to Index: set_index() and reset_index()
#These are crucial for transforming data structure.

# Starting with a regular DataFrame
data = {
    'Country': ['USA', 'USA', 'UK', 'UK', 'Japan', 'Japan'],
    'City': ['NYC', 'LA', 'London', 'Manchester', 'Tokyo', 'Osaka'],
    'Population': [8.3, 3.9, 8.9, 2.8, 13.9, 2.7],
    'GDP': [1700, 1000, 800, 120, 1600, 400]
}

df_regular = pd.DataFrame(data)
print("Regular DataFrame:")
print(df_regular)

# Convert columns to MultiIndex
df_multi = df_regular.set_index(['Country', 'City'])
print("\nWith MultiIndex:")
print(df_multi)

# Reset index (move index back to columns)
df_reset = df_multi.reset_index()
print("\nAfter reset_index():")
print(df_reset)

# Reset only one level
df_partial_reset = df_multi.reset_index(level='City')
print("\nReset only City level:")
print(df_partial_reset)


#Comprehensive Practical Example: Sales Analysis
#Let me create a complete example that uses all these concepts:

# Create sales data for multiple stores, products, and time periods
sales = pd.DataFrame({
    'Store': ['Store A', 'Store A', 'Store A', 'Store A', 
              'Store B', 'Store B', 'Store B', 'Store B'],
    'Product': ['Laptop', 'Laptop', 'Phone', 'Phone',
                'Laptop', 'Laptop', 'Phone', 'Phone'],
    'Quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q2'],
    'Units_Sold': [100, 120, 200, 180, 80, 95, 150, 170],
    'Revenue': [100000, 120000, 150000, 135000, 80000, 95000, 112500, 127500]
})

# Convert to MultiIndex
sales_multi = sales.set_index(['Store', 'Product', 'Quarter'])
print("MultiIndex Sales Data:")
print(sales_multi)

# Selection: Get all Store A data
print("\n--- Store A Performance ---")
print(sales_multi.loc['Store A'])

# Selection: Get all Laptop sales across stores
print("\n--- All Laptop Sales ---")
print(sales_multi.xs('Laptop', level='Product'))

# Reordering: Group by Product first
print("\n--- Grouped by Product ---")
sales_by_product = sales_multi.swaplevel(0, 1).sort_index()
print(sales_by_product)

# Reset to analyze differently
sales_reset = sales_multi.reset_index()
print("\n--- Back to Regular DataFrame ---")
print(sales_reset)

# Create different index structure
sales_time_first = sales_reset.set_index(['Quarter', 'Store', 'Product'])
print("\n--- Time-based View ---")
print(sales_time_first)


#2. Combining and Merging Datasets

#Database-Style Joins (merge):
#pandas.merge() connects rows based on one or more keys, similar to SQL joins.
#Supports inner (default), left, right, and outer joins.
#Can merge on specific columns (on, left_on, right_on) or indexes (left_index=True).
#Concatenation (concat):
#pandas.concat() glues objects together along an axis (default is axis 0/rows).
#The keys argument creates a hierarchical index to identify the source of the concatenated parts.
#ignore_index=True discards existing indexes and creates a new range index.
#Combine with Overlap: combine_first() "patches" missing data in one object with values from another object, aligning them by index.


#1. Database-Style Joins with merge()
#Think of merge() like SQL JOIN operations - it combines DataFrames based on common columns or indexes.
#Basic Merge Types

# Create sample DataFrames
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'city': ['NYC', 'LA', 'Chicago', 'Boston']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [1, 2, 2, 3, 5],  # Note: customer 5 doesn't exist in customers
    'amount': [250, 150, 300, 400, 200]
})

print("Customers:")
print(customers)
print("\nOrders:")
print(orders)

#a) Inner Join (default) - Only matching records

# Inner join - only customers who have orders
inner_merge = pd.merge(customers, orders, on='customer_id', how='inner')
print("\nInner Join:")
print(inner_merge)

#Explanation: Only customers 1, 2, and 3 appear because they have matching orders. Customer 4 (David) has no orders, and order from customer 5 has no matching customer.

#Merging on Different Column Names
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'dept_id': [10, 20, 10]
})

departments = pd.DataFrame({
    'department_id': [10, 20, 30],
    'dept_name': ['Sales', 'Marketing', 'IT']
})

# Merge using left_on and right_on
emp_dept = pd.merge(employees, departments, 
                    left_on='dept_id', 
                    right_on='department_id',
                    how='left')
print("\nEmployees with Departments:")
print(emp_dept)


#Concatenation with concat()
#concat() stacks DataFrames together - either vertically (rows) or horizontally (columns).
jan_sales = pd.DataFrame({
    'product': ['Laptop', 'Phone'],
    'sales': [10, 25]
})

feb_sales = pd.DataFrame({
    'product': ['Laptop', 'Phone'],
    'sales': [15, 30]
})

mar_sales = pd.DataFrame({
    'product': ['Laptop', 'Phone'],
    'sales': [12, 28]
})

# Concatenate vertically
all_sales = pd.concat([jan_sales, feb_sales, mar_sales])
print("Concatenated Sales:")
print(all_sales)

#Combine with Overlap: combine_first()
#This method "patches" missing values from one DataFrame with values from another.

df1 = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [np.nan, 5, 6],
    'C': [7, 8, 9]
}, index=[0, 1, 2])

# Another dataset with different missing values
df2 = pd.DataFrame({
    'A': [10, 20, np.nan],
    'B': [40, np.nan, 60],
    'C': [np.nan, 80, 90]
}, index=[0, 1, 2])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Combine: df1's values take priority, df2 fills gaps
combined = df1.combine_first(df2)
print("\nCombined (df1.combine_first(df2)):")
print(combined)

#Complete Real-World Example: Sales Analysis

products = pd.DataFrame({
    'product_id': [1, 2, 3, 4],
    'product_name': ['Laptop', 'Phone', 'Tablet', 'Monitor'],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Accessories']
})

# 2. Sales from Q1
q1_sales = pd.DataFrame({
    'product_id': [1, 2, 3],
    'units_sold': [50, 100, 30],
    'revenue': [50000, 50000, 9000]
})

# 3. Sales from Q2
q2_sales = pd.DataFrame({
    'product_id': [1, 2, 4],
    'units_sold': [60, 120, 25],
    'revenue': [60000, 60000, 5000]
})

# Merge products with Q1 sales
q1_complete = pd.merge(products, q1_sales, on='product_id', how='left')
print("Q1 Sales (with product info):")
print(q1_complete)

# Merge products with Q2 sales
q2_complete = pd.merge(products, q2_sales, on='product_id', how='left')

# Concatenate Q1 and Q2 with keys
all_quarters = pd.concat([q1_complete, q2_complete],
                         keys=['Q1', 'Q2'],
                         names=['Quarter', 'Record'])
print("\nAll Quarters:")
print(all_quarters)

# Analysis by quarter
print("\nTotal Revenue by Quarter:")
print(all_quarters.groupby('Quarter')['revenue'].sum())

