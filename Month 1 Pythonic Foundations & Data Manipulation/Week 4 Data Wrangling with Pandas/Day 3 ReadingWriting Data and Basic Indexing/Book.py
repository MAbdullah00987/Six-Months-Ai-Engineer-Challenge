
#Chapter 7: Data Cleaning and Preparation
#This chapter focuses on the tools and techniques for handling missing data, manipulating strings, and transforming datasets into clean, usable formats.
#1. Handling Missing Data

#Representation: pandas uses NaN (Not a Number) for missing floating-point values and the built-in Python None for missing object data.
#Identification: The isna() and notna() methods identify missing values.
#Filtering:

#dropna() removes missing data. It can drop rows (default) or columns (axis="columns") containing any nulls, or only those that are all null (how="all").
#thresh argument allows keeping rows with a certain number of observations.
#Filling:
#fillna() fills missing data with a constant value or a dictionary of values for different columns.
#Interpolation methods like ffill (forward fill) and bfill (backward fill) propagate values to fill gaps


#1. Missing Data Representation
#Pandas uses different types:
#NaN - for numeric data
#None - for object/string data
#NaT - for datetime data
#All are treated as "missing" by pandas!


import pandas as pd
import numpy as np
import re

# 1. REPRESENTATION OF MISSING DATA


print("\nTypes of Missing Values")
print("""
Pandas uses different representations for missing data:

1. NaN (Not a Number) - for numeric and float data
2. None - for object/string data (Python's built-in)
3. pd.NA - newer, experimental missing value indicator
4. NaT (Not a Time) - for datetime data

Both NaN and None are treated as "missing" by pandas.
""")

# Create DataFrame with various missing value types
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [5.5, np.nan, 7.5, 8.5, 9.5],
    'C': ['apple', None, 'cherry', 'date', 'elderberry'],
    'D': pd.date_range('2024-01-01', periods=5),
    'E': [10, 20, 30, 40, 50]
})

# Introduce a NaT (Not a Time)
data.loc[2, 'D'] = pd.NaT

print("\nDataFrame with missing values:")
print(data)
print("\nData types:")
print(data.dtypes)

# Show how missing values appear
print("\nHow Missing Values Appear")
print(f"NaN in numeric column: {data.loc[2, 'A']}")
print(f"Type: {type(data.loc[2, 'A'])}")
print(f"\nNone in string column: {data.loc[1, 'C']}")
print(f"Type: {type(data.loc[1, 'C'])}")
print(f"\nNaT in datetime column: {data.loc[2, 'D']}")
print(f"Type: {type(data.loc[2, 'D'])}")

#2. Identifying Missing Data 
#Key Methods:
#isna() / isnull() - Returns True where data is missing
#notna() / notnull() - Returns True where data exists
#.isna().sum() - Count missing values per column

# 2. IDENTIFYING MISSING DATA

print("2. IDENTIFYING MISSING DATA")
print("=" * 70)

# Create a more comprehensive example
sales_data = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=10),
    'Product': ['Laptop', 'Phone', 'Tablet', None, 'Monitor', 
                'Keyboard', 'Mouse', 'Laptop', None, 'Phone'],
    'Quantity': [5, np.nan, 3, 4, np.nan, 10, 15, 7, 8, np.nan],
    'Price': [1200, 800, np.nan, 300, 250, 50, 25, 1200, 400, 800],
    'Revenue': [6000, np.nan, np.nan, 1200, np.nan, 500, 375, 8400, np.nan, np.nan]
})

print("\nSales Data with Missing Values:")
print(sales_data)

# Method 1: isna() / isnull() - Find missing values
print("\nUsing isna() to identify missing values ")
print("\nBoolean mask showing where values are missing:")
print(sales_data.isna())

print("\nCount of missing values per column ")
print(sales_data.isna().sum())

print("\nPercentage of missing values per column ")
missing_percent = (sales_data.isna().sum() / len(sales_data) * 100).round(2)
print(missing_percent)

# Method 2: notna() / notnull() - Find non-missing values
print("\nUsing notna() to identify non-missing values ")
print("\nCount of non-missing values per column:")
print(sales_data.notna().sum())

# Check specific rows with missing data
print("\nRows with ANY missing values ")
rows_with_missing = sales_data[sales_data.isna().any(axis=1)]
print(rows_with_missing)

print("\nRows with NO missing values ")
complete_rows = sales_data[sales_data.notna().all(axis=1)]
print(complete_rows)


#3. Removing Missing Data 
#dropna() options:
#df.dropna()              # Drop rows with ANY missing
#df.dropna(how='all')     # Drop only if ALL missing
#df.dropna(axis='columns') # Drop columns with missing
#df.dropna(thresh=4)      # Keep rows with ≥4 non-null values
#df.dropna(subset=['col']) # Drop based on specific columns

# 3. FILTERING OUT MISSING DATA (dropna)

print("\nOriginal DataFrame:")
print(sales_data)
print(f"Shape: {sales_data.shape}")

# Method 1: Drop rows with ANY missing values (default)
print("\nDrop rows with ANY missing values")
df_drop_any = sales_data.dropna()
print(df_drop_any)
print(f"Shape: {df_drop_any.shape} (removed {len(sales_data) - len(df_drop_any)} rows)")

# Method 2: Drop rows where ALL values are missing
print("\nDrop rows where ALL values are missing")
# Create a row with all missing values for demonstration
sales_data_with_empty = sales_data.copy()
sales_data_with_empty.loc[10] = [pd.NaT, None, np.nan, np.nan, np.nan]
print("\nDataFrame with a completely empty row:")
print(sales_data_with_empty)

df_drop_all = sales_data_with_empty.dropna(how='all')
print("\nAfter dropna(how='all'):")
print(df_drop_all)
print(f"Shape: {df_drop_all.shape}")

# Method 3: Drop columns with missing values
print("\nDrop columns with ANY missing values")
df_drop_cols = sales_data.dropna(axis='columns')
print(df_drop_cols)
print(f"Shape: {df_drop_cols.shape}")

# Method 4: Drop columns where ALL values are missing
print("\nDrop columns where ALL values are missing")
all_missing_data = sales_data.copy()
all_missing_data['Empty_Column'] = np.nan
print("\nDataFrame with completely empty column:")
print(all_missing_data)

df_drop_cols_all = all_missing_data.dropna(axis='columns', how='all')
df_thresh = sales_data.dropna(thresh=4)
print("\nAfter dropna(thresh=4) - keep rows with at least 4 non-null values:")
print(df_thresh)
print(f"Shape: {df_thresh.shape}")

# Method 6: Drop based on specific columns
print("\nDrop rows with missing values in specific columns ")
df_subset = sales_data.dropna(subset=['Product', 'Quantity'])
print("\nDrop if 'Product' OR 'Quantity' is missing:")
print(df_subset)

#4. Filling Missing Data 
#Multiple strategies:
#python# Constant value
#df.fillna(0)

# Different values per column
#df.fillna({'col1': 0, 'col2': 'Unknown'})

# Statistical measures
#df.fillna(df.mean())    # Use mean
#df.fillna(df.median())  # Use median

# 4. FILLING MISSING DATA (fillna)
print("\n" + "=" * 70)
print("4. FILLING MISSING DATA (fillna)")
print("=" * 70)

print("\nOriginal Data:")
print(sales_data)

# Method 1: Fill with constant value
print("\nFill with constant value")
df_fill_zero = sales_data.fillna(0)
print("\nFill all missing values with 0:")
print(df_fill_zero)

# Method 2: Fill with different values per column (dictionary)
print("\nFill with different values per column")
df_fill_dict = sales_data.fillna({
    'Product': 'Unknown',
    'Quantity': 0,
    'Price': sales_data['Price'].mean(),
    'Revenue': 0
})
print("\nFill with custom values:")
print(df_fill_dict)

# Method 3: Fill with statistical measures
print("\nFill with statistical measures ")
df_fill_stats = sales_data.copy()
df_fill_stats['Quantity'] = df_fill_stats['Quantity'].fillna(
    df_fill_stats['Quantity'].mean()
)
df_fill_stats['Price'] = df_fill_stats['Price'].fillna(
    df_fill_stats['Price'].median()
)
df_fill_stats['Revenue'] = df_fill_stats['Revenue'].fillna(
    df_fill_stats['Revenue'].mean()
)
print("\nFill Quantity with mean, Price with median:")
print(df_fill_stats)
print(f"\nQuantity mean: {sales_data['Quantity'].mean():.2f}")
print(f"Price median: {sales_data['Price'].median():.2f}")


#5. Forward/Backward Fill 
#For time series:
#pythondf.fillna(method='ffill')  # Forward fill (copy last value)
#df.fillna(method='bfill')  # Backward fill (copy next value)
#df.fillna(method='ffill', limit=1)  # Limit consecutive fills

# 5. FORWARD FILL AND BACKWARD FILL
print("\n" + "=" * 70)
print("5. FORWARD FILL (ffill) AND BACKWARD FILL (bfill)")
print("=" * 70)

# Create time series data for better demonstration
time_series_data = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=10),
    'Temperature': [32, 34, np.nan, np.nan, 38, 40, np.nan, 42, 43, np.nan],
    'Humidity': [65, np.nan, np.nan, 70, 72, np.nan, 75, np.nan, 78, 80],
    'Status': ['Good', 'Good', None, None, 'Fair', 'Fair', None, 'Good', 'Good', None]
})

print("\nTime Series Data with Missing Values:")
print(time_series_data)

# Forward Fill (ffill) - propagate last valid value forward
print("\nForward Fill (ffill) ")
print("Propagates the last valid value forward to fill gaps")
df_ffill = time_series_data.ffill()
print(df_ffill)

# Backward Fill (bfill) - propagate next valid value backward
print("\nBackward Fill (bfill)")
print("Propagates the next valid value backward to fill gaps")
df_bfill = time_series_data.bfill()
print(df_bfill)

# Limit the number of consecutive fills
print("\nForward Fill with limit")
print("Only fill the first 1 consecutive missing value")
df_ffill_limit = time_series_data.ffill(limit=1)
print(df_ffill_limit)


#6. Interpolation 
#Smart filling for continuous data:
#pythondf.interpolate()  # Linear interpolation
#df.interpolate(method='polynomial', order=2)  # Polynomial
#df.interpolate(method='time')  # Time-based

# 6. INTERPOLATION


print("""
Interpolation estimates missing values based on surrounding values.
Great for numeric data where values should be continuous.
""")

numeric_data = pd.DataFrame({
    'Index': range(1, 11),
    'Value': [10, 15, np.nan, np.nan, 30, 35, np.nan, 45, 50, np.nan]
})

print("\nOriginal Data:")
print(numeric_data)

# Linear interpolation (default)
print("\nLinear Interpolation")
df_interp = numeric_data.copy()
df_interp['Value'] = df_interp['Value'].interpolate()
print(df_interp)
print("\nHow it works: Missing values filled based on linear progression")
print("20 → 25 → 30 (linear steps between 15 and 30)")

# Polynomial interpolation
print("\nPolynomial Interpolation ")
df_poly = numeric_data.copy()
try:
    df_poly['Value'] = df_poly['Value'].interpolate(method='polynomial', order=2)
    print(df_poly)
except ImportError:
    print("Error: 'scipy' library is required for polynomial interpolation.")
    print("Please install it using: pip install scipy")
except Exception as e:
    print(f"An error occurred during polynomial interpolation: {e}")

# Time-based interpolation
print("\n Time-based Interpolation ")
time_data = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=7),
    'Value': [100, 110, np.nan, np.nan, 150, 160, np.nan]
})
time_data = time_data.set_index('Date')
print("\nOriginal:")
print(time_data)

time_data['Value'] = time_data['Value'].interpolate(method='time')
print("\nAfter time interpolation:")
print(time_data)


#Data Transformation
#Duplicates: duplicated() returns a boolean Series indicating duplicate rows, and drop_duplicates() removes them.
#Mapping: The map() method on a Series transforms values based on a function or dictionary mapping (e.g., mapping meat types to animals).
#Replacement: replace() provides a flexible way to substitute specific values (e.g., sentinel values like -999) with others.
#Renaming Axes: The rename() method transforms row indexes and column names using functions or dictionaries without modifying the original object.
#Discretization: cut() divides continuous data into bins (either defined edges or equal length), while qcut() bins data based on sample quantiles.
#Outliers: Outliers can be detected or filtered using boolean indexing and array operations (e.g., capping values outside a specific range).
#Permutation/Sampling: numpy.random.permutation reorders a Series or DataFrame rows. sample() allows for random selection with or without replacement.
#Dummy Variables: get_dummies() converts categorical variables into a "one-hot" encoded matrix (columns of 1s and 0s) for machine learning.


#1. Duplicates 
#pythondf.duplicated()           # Find duplicates
#df.drop_duplicates()      # Remove duplicates
#df.drop_duplicates(subset=['col'], keep='first')  # Control which to keep


# 1. HANDLING DUPLICATES

print("\n" + "=" * 70)
print("1. HANDLING DUPLICATES")
print("=" * 70)

# Create data with duplicates
customer_data = pd.DataFrame({
    'customer_id': [1, 2, 3, 2, 4, 5, 3, 6],
    'name': ['John', 'Emma', 'Michael', 'Emma', 'Sophia', 'David', 'Michael', 'Lisa'],
    'city': ['New York', 'Boston', 'Chicago', 'Boston', 'Seattle', 'Miami', 'Chicago', 'Denver'],
    'purchase': [100, 200, 150, 200, 300, 250, 150, 180]
})

print("\nOriginal Data:")
print(customer_data)

# Identify duplicates
print("\nIdentifying Duplicates")
print("\nBoolean mask showing duplicate rows:")
print(customer_data.duplicated())

print("\nRows that are duplicates:")
print(customer_data[customer_data.duplicated()])

# Check duplicates based on specific columns
print("\nDuplicates based on specific column (customer_id)")
print(customer_data.duplicated(subset=['customer_id']))

print("\nDuplicate customers:")
print(customer_data[customer_data.duplicated(subset=['customer_id'], keep=False)])

# Remove duplicates
print("\nRemoving Duplicates")
df_no_dup = customer_data.drop_duplicates()
print("\nAfter drop_duplicates():")
print(df_no_dup)
print(f"Shape changed from {customer_data.shape} to {df_no_dup.shape}")

# Keep different occurrences
print("\ndrop_duplicates() with keep parameter")
print("\nkeep='first' (default) - keep first occurrence:")
print(customer_data.drop_duplicates(subset=['customer_id'], keep='first'))

print("\nkeep='last' - keep last occurrence:")
print(customer_data.drop_duplicates(subset=['customer_id'], keep='last'))

print("\nkeep=False - remove all duplicates:")
print(customer_data.drop_duplicates(subset=['customer_id'], keep=False))

#2. Mapping 
#Transform values using dictionaries or functions:
#pythondf['col'].map({'old': 'new'})      # Dictionary mapping
#df['col'].map(lambda x: x * 2)     # Function mapping

# 2. MAPPING VALUES

# Create food data
food_data = pd.DataFrame({
    'food': ['bacon', 'chicken', 'beef', 'pork', 'salmon', 'tuna', 'turkey'],
    'price': [8.99, 5.99, 12.99, 7.99, 15.99, 13.99, 6.99]
})

print("\nOriginal Food Data:")
print(food_data)

# Map foods to their animal source
print("\nMapping with Dictionary ")
meat_to_animal = {
    'bacon': 'pig',
    'chicken': 'chicken',
    'beef': 'cow',
    'pork': 'pig',
    'salmon': 'fish',
    'tuna': 'fish',
    'turkey': 'turkey'
}

food_data['animal'] = food_data['food'].map(meat_to_animal)
print("\nAfter mapping food to animal:")
print(food_data)

# Map with function
print("\nMapping with Function")
def price_category(price):
    if price < 7:
        return 'Budget'
    elif price < 10:
        return 'Standard'
    else:
        return 'Premium'

food_data['category'] = food_data['price'].map(price_category)
print("\nAfter mapping prices to categories:")
print(food_data)

# Map with lambda function
print("\nMapping with Lambda Function ")
food_data['expensive'] = food_data['price'].map(lambda x: 'Yes' if x > 10 else 'No')
print("\nMarked expensive items:")
print(food_data)

#3. Replacing 
#pythondf.replace(-999, np.nan)           # Replace single value
#df.replace({col: {old: new}})      # Column-specific
#df.replace(r'\d+', '', regex=True) # Regex patterns

# 3. REPLACING VALUES
print("\n" + "=" * 70)
print("3. REPLACING VALUES")
print("=" * 70)

# Create data with sentinel values
sensor_data = pd.DataFrame({
    'sensor_id': [1, 2, 3, 4, 5],
    'temperature': [23.5, -999, 24.1, 25.3, -999],
    'humidity': [65, 70, -999, 75, 80],
    'status': ['active', 'faulty', 'active', 'maintenance', 'active']
})

print("\nData with sentinel values (-999 means missing):")
print(sensor_data)

# Replace single value
print("\nReplace single value ")
df_replaced = sensor_data.replace(-999, np.nan)
print("\nReplaced -999 with NaN:")
print(df_replaced)

# Replace multiple values
print("\nReplace multiple values")
df_multi = sensor_data.replace([-999, 'faulty'], [np.nan, 'inactive'])
print("\nReplaced -999 with NaN and 'faulty' with 'inactive':")
print(df_multi)

# Replace with dictionary (different values per column)
print("\n--- Replace with dictionary ---")
df_dict = sensor_data.replace({
    'temperature': {-999: np.nan},
    'humidity': {-999: np.nan},
    'status': {'faulty': 'inactive', 'maintenance': 'servicing'}
})
print("\nColumn-specific replacements:")
print(df_dict)

# Replace using regex
print("\nReplace using regex")
text_data = pd.DataFrame({
    'text': ['Hello123', 'World456', 'Data789', 'Science000']
})
print("\nOriginal text:")
print(text_data)

text_data['cleaned'] = text_data['text'].replace(r'\d+', '', regex=True)
print("\nRemoved all digits using regex:")
print(text_data)

#4. Renaming 
#pythondf.rename(columns={'old': 'new'})  # Rename columns
#df.rename(columns=str.upper)       # Use function

# 4. RENAMING AXES (Index and Columns)
print("\n" + "=" * 70)
print("4. RENAMING AXES (Index and Columns)")
print("=" * 70)

# Create sample data
sales_data = pd.DataFrame({
    'prod': ['A', 'B', 'C'],
    'qty': [100, 150, 200],
    'rev': [1000, 1500, 2000]
})

print("\nOriginal DataFrame:")
print(sales_data)

# Rename columns with dictionary
print("\nRename columns with dictionary ")
df_renamed = sales_data.rename(columns={
    'prod': 'product',
    'qty': 'quantity',
    'rev': 'revenue'
})
print(df_renamed)

# Rename with function
print("\nRename columns with function ")
df_upper = sales_data.rename(columns=str.upper)
print("\nUppercase column names:")
print(df_upper)

# Rename index
print("\nRename index ")
df_idx = sales_data.rename(index={0: 'first', 1: 'second', 2: 'third'})
print(df_idx)

# Rename in place
print("\nRename in place (modifies original)")
sales_copy = sales_data.copy()
sales_copy.rename(columns={'prod': 'product'}, inplace=True)
print(sales_copy)

#3. String Manipulation
#Built-in Methods: Standard Python string methods like split, strip, join, and find are essential for basic text cleanup.
#Regular Expressions: The re module handles complex pattern matching. Functions like findall, search, and sub are used to locate and modify patterns within strings.
#Vectorized String Functions: pandas Series have a .str attribute that enables string operations (like .str.contains() or .str.findall()) to be applied array-wise, handling missing values automatically

#1. Built-in Python String Methods 
#Basic operations for single strings:
#pythontext.strip()          # Remove whitespace
#text.lower()          # Convert to lowercase
#text.split(',')       # Split into list
#text.replace('old', 'new')  # Replace text
#text.find('word')     # Find position
#text.startswith('H')  # Check start

#1. BUILT-IN PYTHON STRING METHODS

print("\nBasic String Operations")

# Single string examples
text = "  Hello, World! Welcome to Python.  "
print(f"Original: '{text}'")
print(f"strip(): '{text.strip()}'")           # Remove whitespace
print(f"lower(): '{text.lower()}'")           # Convert to lowercase
print(f"upper(): '{text.upper()}'")           # Convert to uppercase
print(f"replace(): '{text.replace('Python', 'pandas')}'")

# Split and join
print("\nSplit and Join")
sentence = "apple,banana,cherry,date"
print(f"Original: '{sentence}'")
fruits = sentence.split(',')
print(f"split(','): {fruits}")
print(f"join(): '{' | '.join(fruits)}'")

# Find and index
print("\nFind and Index")
text = "Python is great. Python is powerful."
print(f"Original: '{text}'")
print(f"find('Python'): {text.find('Python')}")           # First occurrence
print(f"rfind('Python'): {text.rfind('Python')}")         # Last occurrence
print(f"count('Python'): {text.count('Python')}")         # Count occurrences
print(f"startswith('Python'): {text.startswith('Python')}")
print(f"endswith('.'): {text.endswith('.')}")

# String formatting
print("\nString Formattin")
name = "Alice"
age = 30
print(f"format(): 'Name: {name}, Age: {age}'")
print(f"capitalize(): '{name.lower().capitalize()}'")
print(f"title(): '{'hello world'.title()}'")

#2. Regular Expressions (Regex) 🔍
#Pattern matching for complex text:
#Common Patterns:

#\d - digit (0-9)
#\w - word character
#\s - whitespace
#+ - one or more
#* - zero or more
#. - any character

#Key Functions:
#pythonre.findall(pattern, text)    # Find all matches
#re.search(pattern, text)     # Find first match
#re.sub(pattern, new, text)   # Replace pattern



# 2. REGULAR EXPRESSIONS (re module)

print("\n" + "=" * 70)
print("2. REGULAR EXPRESSIONS (Pattern Matching)")
print("=" * 70)

print("\n--- What are Regular Expressions? ---")
print("""
Regular expressions (regex) are patterns used to match text.

COMMON PATTERNS:
\d  - any digit (0-9)
\w  - any word character (letter, digit, underscore)
\s  - any whitespace (space, tab, newline)
.   - any character except newline
*   - 0 or more repetitions
+   - 1 or more repetitions
?   - 0 or 1 repetition
[abc] - match a, b, or c
[0-9] - match any digit
^   - start of string
$   - end of string
""")

# Example 1: Finding patterns
print("\n--- Example 1: Finding Patterns ---")
text = "Contact us at info@example.com or support@example.org"
print(f"Text: '{text}'")

# Find email addresses
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
emails = re.findall(email_pattern, text)
print(f"Emails found: {emails}")

# Example 2: Search for pattern
print("\n--- Example 2: Search for Pattern ---")
text = "My phone number is 555-123-4567"
print(f"Text: '{text}'")

phone_pattern = r'\d{3}-\d{3}-\d{4}'
match = re.search(phone_pattern, text)
if match:
    print(f"Phone found: {match.group()}")
    print(f"Position: {match.start()} to {match.end()}")

# Example 3: Substitution (replace)
print("\n--- Example 3: Substitution ---")
text = "Price: $100, Sale: $50, Total: $150"
print(f"Original: '{text}'")

# Remove dollar signs
cleaned = re.sub(r'\$', '', text)
print(f"After removing $: '{cleaned}'")

# Replace digits
masked = re.sub(r'\d+', 'XXX', text)
print(f"Masked numbers: '{masked}'")

# Example 4: Extract specific patterns
print("\n--- Example 4: Extract URLs ---")
text = "Visit https://example.com or http://test.org for more info"
print(f"Text: '{text}'")

url_pattern = r'https?://[^\s]+'
urls = re.findall(url_pattern, text)
print(f"URLs found: {urls}")

# Example 5: Complex pattern matching
print("\n--- Example 5: Date Extraction ---")
text = "Important dates: 2024-01-15, 2024-12-31, and 2025-06-20"
print(f"Text: '{text}'")

date_pattern = r'\d{4}-\d{2}-\d{2}'
dates = re.findall(date_pattern, text)
print(f"Dates found: {dates}")

# Split with regex
print("\n--- Split with Regex ---")
text = "apple, banana; cherry| date"
print(f"Original: '{text}'")
parts = re.split(r'[,;|]', text)
print(f"Split on multiple delimiters: {[p.strip() for p in parts]}")

#4. Categorical Data
#Extension Type: pandas uses the Categorical type for data with repeated values (like "apple", "orange") to improve memory usage and performance.
#Structure: It separates the distinct values (categories) from the integer codes that reference them