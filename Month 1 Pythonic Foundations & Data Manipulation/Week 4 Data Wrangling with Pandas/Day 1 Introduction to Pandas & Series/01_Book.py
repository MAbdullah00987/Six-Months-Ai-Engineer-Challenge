# Python for Data Analysis by Wes McKinney (Chapters 5-10)
# Chapter 5: Getting Started with pandas

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#This chapter introduces the core data structures and essential operations of the pandas library, which serves as the foundation for data analysis in Python.

#1. Core Data Structures
#Series: A one-dimensional array-like object containing a sequence of values and an associated
#array of data labels called the index. It behaves like a fixed-length, ordered dictionary.

'''
# Learning about Series in pandas 
#A pandas Series is:
#A one-dimensional structure (single column of data)
##Labeled with an index for each value
#Ordered (maintains sequence)
#Dictionary-like (access by label)
#Array-like (access by position, supports vectorized operations)
#Fixed-length (requires explicit operations to change size)
##temperatures = pd.Series([23,56,78,89])
#print(temperatures)

data = pd.Series([23,67,89.9,98,0.6])
print(data)
#Explanation: By default, pandas assigns integer indices (0, 1, 2, 3, 4) to each value.

#Series with Custom Index (Labels)
temperatures1 = pd.Series(
        [23,67,87,54,87],
        index = ['Mon','Tue','Wed','Thu','Fri']
        )

print(temperatures1)
#Explanation: Now each temperature value has a meaningful label (day of the week) instead of just numbers.

#Fixed-Length Ordered Dictionary
# Creating a Series from a dictionary

student_grades = pd.Series({
        'Alice': 56,
        "Bob" : 89,
        "Charlie" : 78,
        "David" : 92,
        "Eve" : 65
})

print(student_grades)
print("\n Accessing list of dictionary:")
print(f"Alice,s grades : {student_grades['Alice']}")
print(f"Bob,s grades : {student_grades['Bob']}")
print(f"Charlie,s grades : {student_grades['Charlie']}")
print(f"David,s grades : {student_grades['David']}")
print(f"Eve,s grades : {student_grades['Eve']}")

#Explanation:
#Like a dictionary: You can access values using keys (index labels)
#Ordered: Unlike regular Python dictionaries (pre-3.7), Series always maintains order
#Fixed-length: The Series has 4 elements; you can't just add a 5th without explicitly doing so

#Accessing Data
sales = pd.Series([23,56,78,90],
        index = ['Mon','Tue','Wed','Thu']
        )

print(f"March Sales: {sales['Mon']}")
print(f"Tuesday Sales: {sales['Tue']}")
print(f"Wednesday Sales: {sales['Wed']}")
print(f"Thursday Sales: {sales['Thu']}")

print(f"First Month Sales: {sales[0]}")
print(f"Second Month Sales: {sales[1]}")
print(f"Third Month Sales: {sales[2]}")
print(f"Fourth Month Sales: {sales[3]}")

print("\nQ1 Sales: ")
print(sales['Mon':'Thu'])

#Series Operations
prices = pd.Series([10, 15, 20, 25], index=['apple', 'banana', 'cherry', 'date'])

# Mathematical operations
print("Original prices:")
print(prices)

print("\nAfter 20% discount:")
print(prices * 0.8)

print("\nPrices above $15:")
print(prices[prices > 15])

#Series Operations
prices = pd.Series([10, 15, 20, 25,67], 
index = ['apple','banana','Mango','Orange','Pineapple'])

print(Series)

print("\nAfter 20% discount:")
print(prices * 0.8)

print("\nPrices above $15:")
print(prices[prices > 15])

#Why "Fixed-Length"?
colors = pd.Series(['red','blue','green','yellow','white'])
print(f"Original length: {len(colors)}")
print(colors)
# To add an element, you need to explicitly do it

colors['Secondary2'] = 'yellow'
print(f"New length: {len(colors)}")
print(colors)

#Explanation: Unlike a regular list where you can easily .append(), a Series requires explicit assignment to modify its length.
'''

'''
#DataFrame:
#  A rectangular table of data containing an ordered collection of columns, which can 
#be of different value types (numeric, string, boolean, etc.). It has both row and column indexes.

#What is a DataFrame?
#A DataFrame is the primary data structure in pandas, representing a two-dimensional labeled data structure. Think of it as:
#A spreadsheet or Excel table
#A SQL database table
#A collection of Series objects sharing the same index
#Key Characteristics:
#Two-dimensional: Has both rows and columns
#Rectangular: All columns have the same length
#Heterogeneous: Different columns can have different data types
#Labeled: Both rows and columns have labels (indexes)
#Size-mutable: You can add or remove columns and rows

#Creating a Simple DataFrame
data = {
        'Name ' : ['John','james','jill','jane','joe'],
        'Age ' : [23,24,35,56,67],
        'City ' : ['New York','Los Angeles','Chicago','Houston','Phoenix']
}

df = pd.DataFrame(data)
print(df)

#Explanation:
#Columns: Name, Age, City (different data types: string, integer, string)
#Row index: 0, 1, 2, 3 (automatically assigned)
#Rectangular: All columns have 4 values

#DataFrame with Custom Row Index
employees = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Department': ['HR', 'IT', 'Sales', 'IT'],
    'Salary': [60000, 75000, 65000, 72000]
} , index=['E001', 'E002', 'E003', 'E004'])

print(employees)

#Explanation: Now rows have meaningful labels (employee IDs) instead of just numbers.
#Understanding the Structure
#Accessing Row and Column Indexes

sales_data = pd.DataFrame({
        'Product': ['Laptop', 'Phone', 'Tablet', 'Monitor'],
        'Price': [1200, 800, 400, 300],
        'Stock': [15, 30, 25, 10]
}, index = ['P001','P002','P003','P004'])

print("DataFrame:")
print(sales_data)

print("\n Column Index (Column Names) ")
print(sales_data.columns)

print("\n Row Index (Row Names)")
print(sales_data.index)

print("\n Data Types of Each Column")
print(sales_data.shape)

#Different Data Types in Columns
#Heterogeneous Data Types

students_records = pd.DataFrame({
        'Name': ['John', 'Emma', 'Michael', 'Sophia'],      # string
    'Age': [20, 22, 21, 23],                            # integer
    'GPA': [3.5, 3.8, 3.2, 3.9],                        # float
    'Graduated': [False, False, False, True],            # boolean
    'Enrollment_Date': pd.to_datetime(['2020-09-01', '2019-09-01', 
                                        '2020-09-01', '2018-09-01']) 
})

print(students_records)
print("\n Data Types of each column")
print(students_records.dtypes)
#Explanation: Each column can have its own data type - this is what makes DataFrames "heterogeneous."

#Accessing Data
#Selecting Columns and Rows

student_records = pd.DataFrame({
    'Product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'],
    'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories'],
    'Price': [1200, 800, 400, 300, 50],
    'Stock': [15, 30, 25, 10, 100]
})

print("Full DataFrame:")
print(student_records.dtypes)

# Selecting a single column (returns a Series)
print("\n--- Single Column (Series) ---")
print(student_records['Product'])
print(f"Type: {type(student_records['Product'])}")

# Selecting multiple columns (returns a DataFrame)
print("\n--- Multiple Columns (DataFrame) ---")
print(student_records[['Product', 'Price']])
print(f"Type: {type(student_records[['Product', 'Price']])}")

# Selecting rows by position using iloc
print("\n--- First Two Rows ---")
print(student_records.iloc[0:2])

# Selecting rows by label using loc
student_records_indexed = student_records.set_index('Product')
print("\n--- Select by Label ---")
print(student_records_indexed.loc['Laptop'])

#Creating DataFrames Different Ways
#Various Creation Methods
df1 = pd.DataFrame({
        'A': [1,2,3],
        'B': [4,5,6]
})

print("Method 1 - Dictionary:")
print(df1)

df2 = pd.DataFrame([
    {'Name': 'Alice', 'Score': 85},
    {'Name': 'Bob', 'Score': 92},
    {'Name': 'Charlie', 'Score': 78}
])

print("\nMethod 2 - List of Dictionaries:")
print(df2)

df3 = pd.DataFrame(
    [[1, 'A', True], [2, 'B', False], [3, 'C', True]],
    columns=['Number', 'Letter', 'Flag']
)
print("\nMethod 3 - 2D List:")
print(df3)

series1 = pd.Series([10, 20, 30], name='Values1')
series2 = pd.Series([40, 50, 60], name='Values2')
df4 = pd.DataFrame({'Col1': series1, 'Col2': series2})
print("\nMethod 4 - From Series:")
print(df4)

#DataFrame Operations
#Adding and Removing Columns

store = pd.DataFrame({
         'Product': ['Apple', 'Banana', 'Orange'],
        'Price': [1.5, 0.5, 2.0],
        'Quantity': [100, 150, 80]
})

print("Original DataFrame:")
print(store)
# Add a new column (calculated)
store['Total_Value'] = store['Price'] * store['Quantity']
print("\nAfter adding Total_Value column:")
print(store)

# Add another column (constant value)
store['In_Stock'] = True
print("\nAfter adding In_Stock column:")
print(store)

# Remove a column
store_reduced = store.drop('In_Stock', axis=1)
print("\nAfter removing In_Stock column:")
print(store_reduced)

#Filtering and Sorting
sales = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West', 'North', 'South'],
    'Product': ['A', 'B', 'A', 'C', 'B', 'A'],
    'Sales': [250, 150, 300, 200, 180, 220],
    'Profit': [50, 30, 60, 40, 35, 45]
})
print("Original DataFrame:")
print(sales)

# Filter rows where Sales > 200
print("\nSales greater than 200:")
print(sales[sales['Sales'] > 200])

# Filter with multiple conditions
print("\nNorth region with Sales > 200:")
print(sales[(sales['Region'] == 'North') & (sales['Sales'] > 200)])

# Sort by Sales (descending)
print("\nSorted by Sales (highest first):")
print(sales.sort_values('Sales', ascending=False))

# Sort by multiple columns
print("\nSorted by Region, then Sales:")
print(sales.sort_values(['Region', 'Sales']))

#Summary Statistics
weather = pd.DataFrame({
    'City': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'],
    'Temperature': [75, 85, 68, 90, 95],
    'Humidity': [65, 45, 70, 80, 30],
    'Wind_Speed': [12, 8, 15, 10, 5]
})
print("Weather DataFrame:")
print(weather)

print("\n--- Basic Statistics ---")
print(weather.describe())

print("\n--- Specific Statistics ---")
print(f"Average Temperature: {weather['Temperature'].mean():.1f}°F")
print(f"Maximum Humidity: {weather['Humidity'].max()}%")
print(f"Minimum Wind Speed: {weather['Wind_Speed'].min()} mph")

print("\n--- Info About DataFrame ---")
weather.info()

'''
'''
#Index Objects: Responsible for holding axis labels and metadata. They are immutable, making it 
# safer to share them among data structures.

#What is an Index Object?
#An Index is a fundamental component of pandas that holds the axis labels (row or column labels) for Series and DataFrames. It's responsible for data alignment and provides fast lookup capabilities.
#Key Characteristics:
#Immutable: Once created, you cannot change individual elements (makes it safe to share)
#Array-like: Behaves like an array but with special properties
#Can be shared: Multiple data structures can safely reference the same Index
#Enables alignment: Automatically aligns data during operations
#Fast lookup: Optimized for finding labels quickly

#Understanding Index in Series

cities = pd.Series([8000000, 4000000, 2700000, 1500000],
                   index=['New York', 'Los Angeles', 'Chicago', 'Houston'])

print("Series with Index")
print(cities)

print("\nIndex Of Series cities")
print("Series with Index:")
print(cities)
print("\n The Index Object")
print(cities.index)
print(f"Type: {type(cities.index)}")

#Explanation: You cannot change individual elements of an Index, but you can replace the entire Index object. This immutability prevents accidental modifications.

#Index in DataFrames
#Row and Column Indexes

sales_data = pd.DataFrame({
    'Q1': [100, 150, 200],
    'Q2': [110, 160, 210],
    'Q3': [120, 170, 220],
    'Q4': [130, 180, 230]
}, index=['Product A', 'Product B', 'Product C'])

print("DataFrame:")
print(sales_data)

print("\n--- Row Index (index) ---")
print(sales_data.index)
print(f"Type: {type(sales_data.index)}")

print("\n--- Column Index (columns) ---")
print(sales_data.columns)
print(f"Type: {type(sales_data.columns)}")

print("\n--- Index Properties ---")
print(f"Row index name: {sales_data.index.name}")
print(f"Number of rows: {len(sales_data.index)}")
print(f"Number of columns: {len(sales_data.columns)}")

#Different Index Types
df1 = pd.DataFrame({'A': [1, 2, 3]})
print("1. RangeIndex (default):")
print(df1.index)
print(f"Type: {type(df1.index)}\n")

# Int64Index
df2 = pd.DataFrame({'A': [1, 2, 3]}, index=[10, 20, 30])
print("2. Index with integers:")
print(df2.index)
print(f"Type: {type(df2.index)}\n")

# DatetimeIndex
dates = pd.date_range('2024-01-01', periods=3)
df3 = pd.DataFrame({'Sales': [100, 150, 200]}, index=dates)
print("3. DatetimeIndex:")
print(df3.index)
print(f"Type: {type(df3.index)}\n")

# MultiIndex (hierarchical)
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
multi_idx = pd.MultiIndex.from_arrays(arrays, names=['Letter', 'Number'])
df4 = pd.Series([10, 20, 30, 40], index=multi_idx)
print("4. MultiIndex:")
print(df4)
print(f"Type: {type(df4.index)}\n")

#Resetting and Setting Indexes

df = pd.DataFrame({
    'City': ['NYC', 'LA', 'Chicago'],
    'Population': [8000000, 4000000, 2700000],
    'Area': [302, 469, 227]
})

print("Original DataFrame:")
print(df)
print(f"Current index: {df.index.tolist()}\n")

# Set 'City' as the index
df_indexed = df.set_index('City')
print("After setting 'City' as index:")
print(df_indexed)
print(f"Current index: {df_indexed.index.tolist()}\n")

# Reset index back to default
df_reset = df_indexed.reset_index()
print("After resetting index:")
print(df_reset)
print(f"Current index: {df_reset.index.tolist()}")

#Index Set Operations
idx1 = pd.Index(['a', 'b', 'c', 'd'])
idx2 = pd.Index(['c', 'd', 'e', 'f'])

print("Index 1:", idx1.tolist())
print("Index 2:", idx2.tolist())

# Union (all unique elements from both)
print(f"\nUnion: {idx1.union(idx2).tolist()}")

# Intersection (common elements)
print(f"Intersection: {idx1.intersection(idx2).tolist()}")

# Difference (elements in idx1 but not in idx2)
print(f"Difference (idx1 - idx2): {idx1.difference(idx2).tolist()}")

# Symmetric difference (elements in either but not both)
print(f"Symmetric Difference: {idx1.symmetric_difference(idx2).tolist()}")

#Explanation: Index objects support set operations, which are useful for finding common labels, unique labels, etc.

#Key Takeaways
#Index Objects are:
#Immutable: Cannot change individual elements (safe to share)
#Axis labels: Hold row/column labels for Series and DataFrames
#Enable alignment: Automatically align data during operations
#Fast lookup: Optimized for finding and accessing data by label
#Various types: RangeIndex, DatetimeIndex, MultiIndex, etc.
#Support set operations: Union, intersection, difference
#Metadata holders: Store information about the structure of your data
'''
'''
#2. Essential Functionality
#Reindexing: The reindex method allows you to create a new object with data conformed to a new 
# index, filling in missing values (NaN) where indices do not match.
#What is Reindexing?Reindexing is the process of conforming a DataFrame or Series to a new index. It creates a new object with data rearranged to match the new index labels, introducing missing values (NaN) where labels don't exist in the original data, and dropping data for labels not in the new index.Key Characteristics:
#Creates new objects: Doesn't modify the original data
#Conforms to new index: Rearranges data to match new labels
#Handles missing labels: Fills with NaN for labels that don't exist
#Can drop data: Removes data for labels not in new index
#Works on rows and columns: Can reindex both axes in DataFrames

#Simple Reindexing with Series

original = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print("Original Series:")
print(original)

new_index = ['a', 'c', 'e', 'f']
reindexed = original.reindex(new_index)

print("\nReindexed Series (new index: ['a', 'c', 'e', 'f']):")
print(reindexed)

print("\n--- What happened? ---")
print("'a' and 'c': Kept their values (10 and 30)")
print("'b' and 'd': Dropped (not in new index)")
print("'e' and 'f': Added with NaN (didn't exist in original)")

#Explanation:
#Labels in both indexes keep their values
#New labels get NaN
#Old labels not in the new index are dropped
#Note: dtype changed from int64 to float64 (NaN requires float type)

#Reindexing with Custom Fill Values

sales = pd.Series([100, 200, 300], index=['Jan', 'Feb', 'Mar'])
print("Original Sales:")
print(sales)

# Reindex with fill_value instead of NaN
new_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
reindexed = sales.reindex(new_months, fill_value=0)

print("\nReindexed with fill_value=0:")
print(reindexed)

#Explanation: Instead of NaN, missing labels are filled with 0 (or any value you specify).

#Reindexing DataFrame Rows
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['row1', 'row2', 'row3'])

print("Original DataFrame:")
print(df)

new_row_index = ['row1', 'row3', 'row4', 'row5']
df_reindexed = df.reindex(new_row_index)

print("\nReindexed Rows:")
print(df_reindexed)

#Explanation: New rows are added with NaN, and old rows not in the new index are dropped.
#Reindexing DataFrame Columns

#Aligning Data from Different Sources
# Sales data from two different months (different products tracked)
january_sales = pd.Series({
    'Laptop': 50,
    'Phone': 100,
    'Tablet': 30
})

february_sales = pd.Series({
    'Laptop': 55,
    'Phone': 95,
    'Monitor': 20  # New product in February
})

print("January Sales:")
print(january_sales)
print("\nFebruary Sales:")
print(february_sales)

# Align February to January's index (to compare same products)
feb_aligned = february_sales.reindex(january_sales.index, fill_value=0)

print("\nFebruary Sales (aligned to January products):")
print(feb_aligned)

# Now we can safely compare
print("\nSales Change (Feb - Jan):")
print(feb_aligned - january_sales)

#Explanation: Data is aligned based on labels, and missing values are filled with 0.

#Preparing Data for Time Series Analysis
irregular_data = pd.Series(
    [10, 15, 25, 30],
    index=pd.to_datetime(['2024-01-01 08:00', '2024-01-01 10:00', 
                          '2024-01-01 13:00', '2024-01-01 15:00'])
)

print("Irregular Time Series:")
print(irregular_data)

# Create complete hourly index
complete_hours = pd.date_range('2024-01-01 08:00', '2024-01-01 15:00', freq='h')
regular_data = irregular_data.reindex(complete_hours)

print("\nRegular Hourly Time Series:")
print(regular_data)

# Fill missing values with interpolation
regular_filled = irregular_data.reindex(complete_hours).interpolate()

print("\nWith Interpolated Values:")
print(regular_filled)

#Explanation: Data is aligned based on labels, and missing values are filled with 0.

#Combining Reindex with Other Operations
# Student test scores (not all students took all tests)
test1 = pd.Series({'Alice': 85, 'Bob': 90, 'Charlie': 78})
test2 = pd.Series({'Alice': 88, 'Bob': 92, 'Diana': 95})
test3 = pd.Series({'Bob': 87, 'Charlie': 82, 'Diana': 91, 'Eve': 89})

# Get all unique students
all_students = sorted(set(test1.index) | set(test2.index) | set(test3.index))
print(f"All students: {all_students}\n")

# Reindex all tests to include all students
test1_aligned = test1.reindex(all_students)
test2_aligned = test2.reindex(all_students)
test3_aligned = test3.reindex(all_students)

# Create DataFrame
scores_df = pd.DataFrame({
    'Test 1': test1_aligned,
    'Test 2': test2_aligned,
    'Test 3': test3_aligned
})

print("All Test Scores (aligned):")
print(scores_df)

# Calculate average (ignoring NaN)
scores_df['Average'] = scores_df.mean(axis=1)
print("\nWith Averages:")
print(scores_df.round(2))

#Explanation: Data is aligned based on labels, and missing values are filled with 0.

#Key Takeaways
#Reindexing allows you to:
#Conform data to a new index: Rearrange data to match new labels
#Add missing labels: Introduce new labels with NaN or fill values
#Remove unwanted labels: Drop labels not in the new index
#Align data sources: Make different datasets compatible for operations
#Fill missing values: Use methods like ffill, bfill, or nearest
#Work with time series: Create regular intervals from irregular data

#Common Parameters:
#index: New row index
#columns: New column index
#fill_value: Value to use for missing labels (default: NaN)
#method: Fill method ('ffill', 'bfill', 'nearest')
#limit: Maximum number of consecutive fills
#tolerance: Maximum distance for nearest matching
'''
'''
#Selection and Filtering:
#loc: Used for label-based indexing (selecting rows/columns by name).
#iloc: Used for integer-based indexing (selecting rows/columns by position).
#Note: The book advises using loc and iloc over chained indexing (e.g., df[][]) to avoid ambiguity 
# and performance warnings.

#What are loc and iloc?
#loc and iloc are the primary methods for selecting and filtering data in pandas:

#loc: Label-based indexing (uses index/column names)
#iloc: Integer-based indexing (uses numerical positions)
#Key Characteristics:

#Explicit and clear: Avoid ambiguity in data selection
#Efficient: Better performance than chained indexing
#Safe: Prevent SettingWithCopyWarning
#Consistent: Predictable behavior across different scenarios
#Powerful: Support complex selection patterns

#loc (Label-Based Indexing)
#Basic loc Usage - Selecting Rows

employees = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Department': ['HR', 'IT', 'Sales', 'IT', 'HR'],
    'Salary': [60000, 75000, 65000, 72000, 68000]
}, index=['E001', 'E002', 'E003', 'E004', 'E005'])

print("Original DataFrame:")
print(employees)

# Select single row by label
print("\n--- Select row 'E001' ---")
print(employees.loc['E001'])

# Select multiple rows by labels
print("\n--- Select rows 'E001' and 'E003' ---")
print(employees.loc[['E001', 'E003']])

# Select range of rows (inclusive on both ends!)
print("\n--- Select rows from 'E002' to 'E004' (inclusive) ---")
print(employees.loc['E002':'E004'])

#loc - Selecting Columns
print("--- Select 'Name' column ---")
print(employees.loc[:, 'Name'])  # : means "all rows"

# Select multiple columns
print("\n--- Select 'Name' and 'Salary' columns ---")
print(employees.loc[:, ['Name', 'Salary']])

# Select range of columns
print("\n--- Select columns from 'Age' to 'Department' ---")
print(employees.loc[:, 'Age':'Department'])

#loc - Selecting Rows AND Columns Together
print("--- Rows 'E001' & 'E003', Columns 'Name' & 'Salary' ---")
print(employees.loc[['E001', 'E003'], ['Name', 'Salary']])

# Select row range and column range
print("\n--- Rows 'E002':'E004', Columns 'Name':'Department' ---")
print(employees.loc['E002':'E004', 'Name':'Department'])

# Select all rows, specific columns
print("\n--- All rows, only 'Name' and 'Age' ---")
print(employees.loc[:, ['Name', 'Age']])

# Select specific row, all columns
print("\n--- Row 'E003', all columns ---")
print(employees.loc['E003', :])

#iloc (Integer-Based Indexing)
#Basic iloc Usage - Selecting by Position

print("Original DataFrame:")
print(employees)

# Select single row by position (0-indexed)
print("\n--- Select first row (position 0) ---")
print(employees.iloc[0])

# Select multiple rows by positions
print("\n--- Select rows at positions 0 and 2 ---")
print(employees.iloc[[0, 2]])

# Select range of rows (end is exclusive!)
print("\n--- Select rows from position 1 to 3 (exclusive) ---")
print(employees.iloc[1:3])  # Gets positions 1 and 2, NOT 3

#Advanced Boolean Filtering
# Student records
students = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Math': [85, 92, 78, 95, 88, 70],
    'Science': [90, 85, 82, 98, 91, 75],
    'English': [88, 90, 85, 92, 87, 80]
})

print("Student Records:")
print(students)

# Filter using .isin() method
print("\n--- Students named Alice, Bob, or Diana ---")
print(students.loc[students['Name'].isin(['Alice', 'Bob', 'Diana'])])

# Filter using string methods
print("\n--- Students whose names start with 'E' or 'F' ---")
print(students.loc[students['Name'].str.startswith(('E', 'F'))])

# Filter using .between() method
print("\n--- Students with Math scores between 80 and 90 ---")
print(students.loc[students['Math'].between(80, 90)])

# Complex condition: average score > 85
students['Average'] = students[['Math', 'Science', 'English']].mean(axis=1)
print("\n--- Students with average > 85 ---")
print(students.loc[students['Average'] > 85])

#Comparison: loc vs iloc
#Side-by-Side Comparison
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Score': [85, 92, 78, 95]
}, index=[10, 20, 30, 40])

print("DataFrame with custom index:")
print(df)

print("\n=== Using loc (label-based) ===")
print("loc[20] - Get row with index label 20:")
print(df.loc[20])

print("\nloc[20:40] - Slice from label 20 to 40 (INCLUSIVE):")
print(df.loc[20:40])

print("\n=== Using iloc (position-based) ===")
print("iloc[1] - Get row at position 1:")
print(df.iloc[1])

print("\niloc[1:3] - Slice from position 1 to 3 (EXCLUSIVE):")
print(df.iloc[1:3])

print("\n=== Key Difference ===")
print("loc uses LABEL 20 → gets Bob")
print("iloc uses POSITION 1 → gets Bob (same result, different approach)")

#Quick Reference Table
#Operation           loc (Label-based)                   iloc (Position-based)
#Single row          df.loc['label']                     df.iloc[0]
#Multiple rows       df.loc[['A', 'B']]                  df.iloc[[0, 2]]
#Row slice           df.loc['A':'C'] (inclusive)         df.iloc[0:3] (exclusive)
#Single column       df.loc[:, 'col']                    df.iloc[:, 0]
#Multiple columns    df.loc[:, ['col1', 'col3']]         df.iloc[:, [0, 2]]
#Rows & columns      df.loc['A':'C', ['col1', 'col2']]   df.iloc[0:3, [0, 1]]
#Boolean filter      df.loc[df['col'] > 5]               df.iloc[mask.values]
#Assignment          df.loc[mask, 'col'] = value         df.iloc[0:3, 1] = value
'''

'''
#Data Alignment: When performing arithmetic between objects with different indexes, pandas
#  automatically aligns the data based on labels. Non-overlapping labels result in NaN 
#(missing data).

#What is Data Alignment?Data Alignment is pandas' automatic process of matching data based on index labels when performing operations between Series or DataFrames. This ensures operations happen on the correct corresponding elements, even when the indexes are different or in different orders.Key Characteristics:
#Automatic: Happens without explicit commands
#Label-based: Aligns by index labels, not positions
#Union of indexes: Result contains all labels from both objects
#NaN for missing: Non-overlapping labels produce NaN
#Order-independent: Labels are matched regardless of order

#Simple Alignment - Different Order
# Two Series with same labels but different order

series1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
series2 = pd.Series([1, 2, 3], index=['c', 'a', 'b'])

print("Series 1:")
print(series1)
print("\nSeries 2:")
print(series2)

# Add them together - pandas aligns by label!
result = series1 + series2

print("\nSeries 1 + Series 2:")
print(result)
print("\n--- How alignment worked ---")
print("a: 10 + 2 = 12")
print("b: 20 + 3 = 23")
print("c: 30 + 1 = 31")

#Explanation: Even though series2 has labels in a different order (c, a, b), pandas automatically matched them correctly by label, not by position!

#Partial Overlap - Introducing NaN
series1 = pd.Series([100, 200, 300], index=['A', 'B', 'C'])
series2 = pd.Series([10, 20, 30], index=['B', 'C', 'D'])

print("Series 1:")
print(series1)
print("\nSeries 2:")
print(series2)

# Add them together
result = series1 + series2

print("\nSeries 1 + Series 2:")
print(result)
print("\n--- Explanation ---")
print("A: Only in series1 → 100 + NaN = NaN")
print("B: In both → 200 + 10 = 210")
print("C: In both → 300 + 20 = 320")
print("D: Only in series2 → NaN + 30 = NaN")

#No Overlap - All NaN
prices_store1 = pd.Series([10, 15, 20], index=['apple', 'banana', 'orange'])
prices_store2 = pd.Series([5, 8, 12], index=['grape', 'melon', 'pear'])

print("Store 1 Prices:")
print(prices_store1)
print("\nStore 2 Prices:")
print(prices_store2)

# Try to add them
result = prices_store1 + prices_store2

print("\nStore 1 + Store 2:")
print(result)
print("\n--- Why all NaN? ---")
print("No overlapping labels between the two Series!")

#Data Alignment with DataFrames
#DataFrame Alignment - Rows and Columns

df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}, index=['row1', 'row2', 'row3'])

df2 = pd.DataFrame({
    'B': [10, 20],
    'C': [30, 40]
}, index=['row2', 'row3'])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Add DataFrames
result = df1 + df2

print("\nDataFrame 1 + DataFrame 2:")
print(result)

print("\n--- Explanation ---")
print("Result has union of rows: row1, row2, row3")
print("Result has union of columns: A, B, C")
print("Alignment happens on BOTH axes simultaneously!")
'''


#Function Application:
#apply: Applies a function across an axis (rows or columns) of a DataFrame.
#applymap: Applies a function element-wise to a DataFrame.
#map: Applies a function element-wise to a Series.

#What are apply, applymap, and map?These methods allow you to apply custom functions to your data:
#apply: Applies a function along an axis (rows or columns) of a DataFrame or to a Series
#applymap: Applies a function element-wise to every cell in a DataFrame (deprecated in newer pandas, use .map() instead)
#map: Applies a function element-wise to a Series, or maps values using a dictionary
#Key Characteristics:
#Flexible: Work with custom functions, lambda functions, or built-in functions
#Powerful: Enable complex transformations not available in built-in methods
#Vectorization alternative: When vectorized operations aren't possible
#Element-wise or row/column-wise: Different granularity levels

numbers = pd.Series([1, 2, 3, 4, 5])

print("Original Series:")
print(numbers)

# Apply a function to each element
squared = numbers.apply(lambda x: x ** 2)
print("\nSquared (using apply):")
print(squared)

# Apply with a named function
def categorize(x):
    if x < 3:
        return 'Low'
    elif x < 5:
        return 'Medium'
    else:
        return 'High'

categories = numbers.apply(categorize)
print("\nCategories:")
print(categories)

#map() for Series Transformations
fruits = pd.Series(['apple', 'banana', 'cherry', 'apple', 'banana'])

print("Original Series:")
print(fruits)

# Map with a dictionary (value replacement)
fruit_colors = {
    'apple': 'red',
    'banana': 'yellow',
    'cherry': 'red'
}
colors = fruits.map(fruit_colors)
print("\nMapped to colors (using dictionary):")
print(colors)

# Map with a function
prices = pd.Series([1.5, 2.0, 3.5, 1.5, 2.0])
discounted = prices.map(lambda x: x * 0.8)
print("\nOriginal Prices:")
print(prices)
print("\nDiscounted Prices (20% off):")
print(discounted)

# Map with string methods
print("\nUppercase fruits:")
print(fruits.map(str.upper))

#Explanation:
#map() with dictionary: Replaces values based on key-value pairs
#map() with function: Applies function to each element


#map() vs apply() on Series
numbers = pd.Series([10, 20, 30, 40, 50])

print("Original Series:")
print(numbers)

# Both achieve the same result for simple operations
result_map = numbers.map(lambda x: x * 2)
result_apply = numbers.apply(lambda x: x * 2)

print("\nUsing map:")
print(result_map)
print("\nUsing apply:")
print(result_apply)

print("\n--- Key Difference ---")
print("For Series: map() and apply() are similar")
print("map() is specifically for Series")
print("apply() works on both Series and DataFrames")


#Understanding apply() with DataFrames - Along Axis
#apply() on DataFrame Columns (axis=0)

df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': [100, 200, 300, 400]
})

print("Original DataFrame:")
print(df)

# Apply function to each column (axis=0, default)
column_sums = df.apply(sum, axis=0)
print("\nSum of each column (axis=0):")
print(column_sums)

column_means = df.apply(np.mean, axis=0)
print("\nMean of each column:")
print(column_means)

# Custom function on columns
def column_range(col):
    return col.max() - col.min()

ranges = df.apply(column_range, axis=0)
print("\nRange of each column:")
print(ranges)



#Sorting: Data can be sorted by index (sort_index) or by value (sort_values).

#What is Sorting in Pandas?Sorting allows you to arrange your data in a specific order. Pandas provides two main sorting methods:
#sort_index(): Sorts by row or column labels (index)
#sort_values(): Sorts by actual data values
#Key Characteristics:
#Returns new object: Original data unchanged by default (use inplace=True to modify)
#Flexible: Sort ascending or descending
#Multi-level: Sort by multiple columns/indexes
#NaN handling: Control where missing values appear
#Stable sorting: Maintains relative order of equal elements

#Basic Index Sorting - Series
series = pd.Series([30, 10, 40, 20], index=['d', 'b', 'a', 'c'])

print("Original Series (unsorted index):")
print(series)

# Sort by index (ascending - default)
sorted_asc = series.sort_index()
print("\nSorted by index (ascending):")
print(sorted_asc)

# Sort by index (descending)
sorted_desc = series.sort_index(ascending=False)
print("\nSorted by index (descending):")
print(sorted_desc)


#3. Summarizing and Computing Statistics
#Reductions: Methods like sum, mean, and std compute statistics. By default, these methods exclude 
# missing data (NaN).
#Correlation and Covariance: Methods like corr and cov compute relationships between Series or 
# DataFrame columns.
#Value Counts: The value_counts method computes a histogram of unique values, useful for 
#understanding data distribution.

