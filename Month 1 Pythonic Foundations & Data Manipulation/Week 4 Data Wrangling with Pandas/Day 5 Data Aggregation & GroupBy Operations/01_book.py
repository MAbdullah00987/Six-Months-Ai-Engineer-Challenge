
#Python for data analysis by Wes McKinney
# Chapter 10: Data Aggregation and Group Operations 

#The Split-Apply-Combine Paradigm
#The core logic of group operations in pandas follows the "split-apply-combine" process:
#Split: Data is divided into groups based on one or more keys.
#Apply: A function is applied to each group independently, producing a new value.
#Combine: The results of all function applications are merged into a single result object.

#1. The groupby Mechanism
#Creating Groups: You can group data using list/arrays, DataFrame column names, dictionaries, Series, or functions.
#Iterating: The GroupBy object supports iteration, generating a sequence of 2-tuples containing the group name and the chunk of data.
#Column Selection: You can index a GroupBy object to select specific columns for aggregation, which is useful for large datasets.
#Grouping by Index: You can aggregate using levels of a hierarchical index by passing the level number or name.

#2. Data Aggregation
#Aggregation refers to data transformations that produce scalar values from arrays.
#Optimized Methods: pandas provides optimized implementations for common statistical methods like count, sum, mean, median, std, var, min, and max.
#Custom Aggregation: The agg (or aggregate) method allows you to pass your own functions to aggregate data.
#Multiple Functions: You can pass a list of functions to agg to compute multiple statistics at once, resulting in a DataFrame with hierarchical columns.
#Column-Specific Functions: You can pass a dictionary to agg to apply different functions to different columns.

#3. General Group Operations (apply)
#The apply Method: This is the most general-purpose GroupBy method. It splits the object, invokes a passed function on each piece, and concatenates the pieces.
#Flexibility: Unlike aggregation methods that must return a scalar, apply can return a pandas object or a scalar, allowing for more complex operations like top-N selection or group-wise linear regression.
#Quantile Analysis: Combining groupby with cut or qcut allows for bucket or quantile analysis.

#4. Group Transforms
#The transform Method: This method is specialized for operations where the result must have the same shape as the input group. It is useful for operations like standardization (subtracting group mean and dividing by group standard deviation).
#Unwrapped GroupBys: Built-in aggregate functions used with transform often have a "fast path" that allows for efficient "unwrapped" group operations (e.g., df['value'] - g.transform('mean')).

#5. Pivot Tables and Cross-Tabulation
#Pivot Tables: The pivot_table method allows you to aggregate data by one or more keys, arranging the data in a rectangle with rows and columns. It supports partial totals via the margins keyword.
#Cross-Tabulation: The crosstab function is a specialized version of a pivot table designed to compute group frequencies (counts).

import pandas as pd
import numpy as np

#1. The groupby Mechanism
#Creating Groups: You can group data using list/arrays, DataFrame column names, dictionaries, Series, or functions.
#Iterating: The GroupBy object supports iteration, generating a sequence of 2-tuples containing the group name and the chunk of data.
#Column Selection: You can index a GroupBy object to select specific columns for aggregation, which is useful for large datasets.
#Grouping by Index: You can aggregate using levels of a hierarchical index by passing the level number or name.

#Creating Groups - Different Methods
#The groupby() method splits data into groups based on various criteria.
#Grouping by Single Column

# Sample sales data
sales = pd.DataFrame({
    'Region': ['North', 'South', 'North', 'South', 'East', 'East', 'West'],
    'Product': ['Laptop', 'Laptop', 'Phone', 'Phone', 'Tablet', 'Laptop', 'Phone'],
    'Sales': [1000, 1500, 800, 900, 600, 1200, 750],
    'Quantity': [2, 3, 4, 5, 3, 2, 4]
})

print("Original Data:")
print(sales)

# Group by Region
grouped = sales.groupby('Region')
print("\nGroupBy Object:")
print(grouped)
print("Type:", type(grouped))

# Apply aggregation
print("\nTotal Sales by Region:")
print(grouped['Sales'].sum())


