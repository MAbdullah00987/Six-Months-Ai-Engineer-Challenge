
#Chapter 6: Data Loading, Storage, and File Formats
#This chapter focuses on Input/Output (I/O)—reading data into pandas and exporting it.

#1. Reading Text Files
#read_csv and read_table: These are the most common functions for loading delimited text data.
#Handling Messy Data: The chapter details arguments for handling headers, skipping rows, using custom delimiters (including regex), and handling missing values.
#Sentinels: You can specify which strings (e.g., "NULL", "NA") should be treated as missing data.
#Reading in Chunks: For large files, you can process data in pieces using the chunksize argument, which returns an iterator object.
#JSON: The read_json function converts JSON strings into Series or DataFrames.

#2. Binary Data Formats
#Pickle: The to_pickle and read_pickle methods allow for serializing pandas objects in Python’s native binary format. This is efficient for temporary storage but not recommended for long-term archival due to version compatibility issues.
#HDF5: Using HDFStore allows for efficient storage of large amounts of scientific array data, supporting compression and efficient read/write of small sections of large arrays.
#Excel: pandas supports reading and writing Excel files (.xls, .xlsx) using read_excel and to_excel.

#3. Web APIs and Databases
#Web APIs: The chapter demonstrates using the requests library to fetch data from websites and converting JSON responses into pandas DataFrames.
#Databases: pandas can interact with SQL databases (like SQLite, PostgreSQL, MySQL). The read_sql function (often used with SQLAlchemy) allows you to execute a query and load the results directly into a DataFrame.


import pandas as pd
import numpy as np
from io import StringIO
import json
import os
import tempfile

'''
1. Basic CSV Reading
#pd.read_csv() is your go-to function
#Automatically detects columns and data types
#Works with files or strings
'''
# 1. BASIC CSV READING
print("\n" + "=" * 70)
print("1. BASIC CSV READING")
print("=" * 70)

# Create sample CSV data in memory
csv_data = """Name,Age,City,Salary
John,28,New York,75000
Emma,34,Los Angeles,82000
Michael,45,Chicago,95000
Sophia,29,Houston,68000"""

print("\nSample CSV Data:")
print(csv_data)

df = pd.read_csv(StringIO(csv_data))
print("\nDataFrame from CSV:")
print(df)
print(f"\nData Types:\n{df.dtypes}")

'''
2. Handling Messy Data
No Header?
Use header=None and provide names=['col1', 'col2']
Skip Rows?
skiprows=3 skips first 3 rows
comment='#' skips lines starting with #
Custom Delimiters?
sep='\t' for tabs
sep='|' for pipes
sep=r'\s+' for multiple spaces (regex)

#Example 2a: No Header
csv_no_header = """John,28,New York,75000
Emma,34,Los Angeles,82000
Michael,45,Chicago,95000"""

print("\n--- CSV Without Header ---")
print(csv_no_header)

df_no_header = pd.read_csv(StringIO(csv_no_header), 
                           header=None,  # No header row
                           names=['Name', 'Age', 'City', 'Salary'])  # Provide column names
print("\nDataFrame:")
print(df_no_header)

# Example 2b: Skipping Rows 
csv_with_comments = """# Sales Data Report
# Generated on 2024-01-15
# Department: Electronics
Name,Age,City,Salary
John,28,New York,75000
Emma,34,Los Angeles,82000"""

print("\nSkipping Comment Rows")
print(csv_with_comments)

df_skip = pd.read_csv(StringIO(csv_with_comments), 
                      skiprows=3)  # Skip first 3 rows
print("\nDataFrame (skipped first 3 rows):")
print(df_skip)

# Alternative: Skip rows by function
df_skip_comments = pd.read_csv(StringIO(csv_with_comments), 
                               comment='#')  # Skip lines starting with #
print("\nDataFrame (skipped comment lines):")
print(df_skip_comments)

# Example 2c: Custom Delimiters 
print("\nCustom Delimiters ")

# Tab-delimited data
tsv_data = """Name\tAge\tCity\tSalary
John\t28\tNew York\t75000
Emma\t34\tLos Angeles\t82000"""

print("\nTab-delimited data:")
print(tsv_data)

df_tsv = pd.read_csv(StringIO(tsv_data), sep='\t')  # or use read_table()
print("\nDataFrame:")
print(df_tsv)

# Example 2d: Regex Delimiter 
print("\nRegex Delimiter (Multiple Spaces)")

# Data with varying whitespace
regex_data = """Name    Age  City            Salary
John    28   New York        75000
Emma    34   Los Angeles     82000"""

print(regex_data)

df_regex = pd.read_csv(StringIO(regex_data), sep=r'\s+', engine='python')
print("\nDataFrame:")
print(df_regex)

'''

'''
3. Missing Values (Sentinels)
Specify what represents missing data:

na_values=['NA', 'NULL', '?', '-999']
Can be different per column!
Common sentinels: "NA", "NULL", "N/A", "-999", "?"

# Data with various missing value representations
messy_data = """Name,Age,City,Salary,Status
John,28,New York,75000,Active
Emma,NA,Los Angeles,82000,Active
Michael,45,NULL,N/A,Inactive
Sophia,29,Houston,-999,Active
David,?,Miami,68000,NULL"""

print("Messy data with various missing value indicators:")
print(messy_data)

# Read with default missing value handling
print("\n--- Default Missing Value Handling ---")
df_default = pd.read_csv(StringIO(messy_data))
print(df_default)
print("\nNull values detected:")
print(df_default.isna().sum())

# Read with default missing value handling
print("\n--- Default Missing Value Handling ---")
df_default = pd.read_csv(StringIO(messy_data))
print(df_default)
print("\nNull values detected:")
print(df_default.isna().sum())

# Read with custom missing value sentinels
print("\n--- Custom Missing Value Sentinels ---")
df_custom = pd.read_csv(StringIO(messy_data), 
                        na_values=['NA', 'NULL', 'N/A', '-999', '?'])
print(df_custom)
print("\nNull values detected:")
print(df_custom.isna().sum())

# Different sentinels for different columns
print("\n--- Column-Specific Sentinels ---")
df_column_specific = pd.read_csv(StringIO(messy_data),
                                 na_values={'Age': ['NA', '?'],
                                           'City': ['NULL'],
                                           'Salary': ['N/A', '-999'],
                                           'Status': ['NULL']})
print(df_column_specific)

'''

'''
4. Reading in Chunks
For huge files that don't fit in memory:
for chunk in pd.read_csv('huge_file.csv', chunksize=1000):
    # Process each chunk
    total += chunk['amount'].sum()
Processes 1000 rows at a time
Saves memory!
Perfect for big data
'''
# Create a larger dataset
large_csv = """ID,Name,Product,Amount,Date
1,John,Laptop,1200,2024-01-01
2,Emma,Phone,800,2024-01-02
3,Michael,Tablet,400,2024-01-03
4,Sophia,Monitor,300,2024-01-04
5,David,Keyboard,50,2024-01-05
6,Lisa,Mouse,25,2024-01-06
7,James,Laptop,1200,2024-01-07
8,Emily,Phone,800,2024-01-08
9,Robert,Tablet,400,2024-01-09
10,Maria,Monitor,300,2024-01-10"""

print("Large dataset example:")
print(large_csv[:200] + "...")

# Read in chunks
print("\n--- Processing in Chunks of 3 rows ---")
chunk_iterator = pd.read_csv(StringIO(large_csv), chunksize=3)

chunk_num = 1
total_amount = 0

for chunk in chunk_iterator:
    print(f"\n--- Chunk {chunk_num} ---")
    print(chunk)
    chunk_total = chunk['Amount'].sum()
    print(f"Chunk Total: ${chunk_total:,.2f}")
    total_amount += chunk_total
    chunk_num += 1

print(f"\n--- Grand Total from all chunks: ${total_amount:,.2f} ---")

# Practical example: Process large file and keep only summary
print("\n--- Practical: Aggregate Statistics from Chunks ---")
chunk_iterator = pd.read_csv(StringIO(large_csv), chunksize=4)

aggregated_stats = {
    'total_amount': 0,
    'row_count': 0,
    'max_amount': 0
}

for chunk in chunk_iterator:
    aggregated_stats['total_amount'] += chunk['Amount'].sum()
    aggregated_stats['row_count'] += len(chunk)
    aggregated_stats['max_amount'] = max(aggregated_stats['max_amount'], 
                                         chunk['Amount'].max())

print("Aggregated Statistics:")
print(f"Total Rows Processed: {aggregated_stats['row_count']}")
print(f"Total Amount: ${aggregated_stats['total_amount']:,.2f}")
print(f"Average Amount: ${aggregated_stats['total_amount']/aggregated_stats['row_count']:,.2f}")
print(f"Max Amount: ${aggregated_stats['max_amount']:,.2f}")


'''
5. JSON Data

pd.read_json() converts JSON to DataFrame
lines=True for JSON Lines format (one object per line)
json_normalize() flattens nested JSON

When to use each:

CSV: Spreadsheet-like data, reports
Chunks: Files > 1GB, streaming data
JSON: API responses, web data, nested structures

'''

print("\n--- JSON Format: List of Records ---")
json_records = '''[
    {"name": "John", "age": 28, "city": "New York", "salary": 75000},
    {"name": "Emma", "age": 34, "city": "Los Angeles", "salary": 82000},
    {"name": "Michael", "age": 45, "city": "Chicago", "salary": 95000}
]'''

print(json_records)

df_json = pd.read_json(StringIO(json_records))
print("\nDataFrame from JSON:")
print(df_json)

# Example 5b: JSON Lines Format 
print("\n--- JSON Lines Format (one object per line) ---")
json_lines = '''{"name": "John", "age": 28, "city": "New York"}
{"name": "Emma", "age": 34, "city": "Los Angeles"}
{"name": "Michael", "age": 45, "city": "Chicago"}'''

print(json_lines)

df_json_lines = pd.read_json(StringIO(json_lines), lines=True)
print("\nDataFrame from JSON Lines:")
print(df_json_lines)

# Example 5c: Nested JSON 
print("\n--- Nested JSON ---")
nested_json = '''[
    {
        "name": "John",
        "age": 28,
        "address": {"city": "New York", "zip": "10001"}
    },
    {
        "name": "Emma",
        "age": 34,
        "address": {"city": "Los Angeles", "zip": "90001"}
    }
]'''

print(nested_json)

df_nested = pd.read_json(StringIO(nested_json))
print("\nDataFrame with nested data:")
print(df_nested)

# Normalize nested JSON
from pandas import json_normalize
data = json.loads(nested_json)
df_normalized = json_normalize(data)
print("\nNormalized (flattened) DataFrame:")
print(df_normalized)



#2. Binary Data Formats
#Pickle: The to_pickle and read_pickle methods allow for serializing pandas objects in Python’s native
#  binary format. This is efficient for temporary storage but not recommended for long-term archival due to version compatibility issues.
#HDF5: Using HDFStore allows for efficient storage of large amounts of scientific array data, supporting 
# compression and efficient read/write of small sections of large arrays.
#Excel: pandas supports reading and writing Excel files (.xls, .xlsx) using read_excel and to_excel.


#1. Pickle Format (.pkl) 
#Best for: Temporary storage, caching, intermediate results
#Pros:#
# Fastest format for read/write
#Preserves exact pandas data types
#Perfect for caching between analysis steps
#Cons:
# Python-only (can't open in Excel or other tools)
# Security risk - never open untrusted pickle files!
# Version compatibility issues
#Use when: You need fast temporary storage during data processing

temp_dir = tempfile.gettempdir()
sales_data = df

# Save to pickle
pickle_file = os.path.join(temp_dir, 'sales_data.pkl')
print(f"\nSaving to Pickle: {pickle_file} ")
sales_data.to_pickle(pickle_file)
print(f"Saved successfully!")

# Check file size
file_size = os.path.getsize(pickle_file)
print(f"File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

# Read from pickle
print("\nReading from Pickle")
df_from_pickle = pd.read_pickle(pickle_file)
print("Data loaded successfully!")
print(df_from_pickle.head())

# Verify data integrity
print("\nData Integrity Check ")
print(f"Original shape: {sales_data.shape}")
print(f"Loaded shape: {df_from_pickle.shape}")
print(f"Data identical: {sales_data.equals(df_from_pickle)}")
print(f"Data types preserved: {all(sales_data.dtypes == df_from_pickle.dtypes)}")

#2. HDF5 Format (.h5) 🗄️
#Best for: Large datasets (GB to TB), scientific data
#Pros:

#Excellent for huge datasets
#Supports compression (saves space)
#Can read specific sections without loading entire file
#Store multiple DataFrames in one file
#Fast querying capabilities
#Cons:
#Requires PyTables library (pip install tables)
#More complex than CSV or Pickle
#Use when: Working with large time series, scientific datasets, or need compression

# Note: HDF5 requires the 'tables' package
# Create a dummy employee dataframe for the HDF5 example
employee_data = pd.DataFrame({
    'EmployeeID': [101, 102, 103],
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Department': ['HR', 'IT', 'Finance'],
    'Salary': [60000, 75000, 80000]
})

try:
    # Save to HDF5
    hdf5_file = os.path.join(temp_dir, 'company_data.h5')
    print(f"\nSaving to HDF5: {hdf5_file}")
    
    # Method 1: Simple save (fixed format)
    sales_data.to_hdf(hdf5_file, key='sales', mode='w')
    print("Sales data saved to key 'sales'")
    
    # Add another DataFrame to the same file
    employee_data.to_hdf(hdf5_file, key='employees', mode='a')
    print("Employee data saved to key 'employees'")
    
    # Check file size
    file_size = os.path.getsize(hdf5_file)
    print(f"File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    # Read from HDF5
    print("\nReading from HDF5")
    df_sales = pd.read_hdf(hdf5_file, key='sales')
    df_employees = pd.read_hdf(hdf5_file, key='employees')
    print("Sales data loaded from key 'sales'")
    print(df_sales.head(3))
    print("\nEmployee data loaded from key 'employees'")
    print(df_employees.head(3))
    
    # Using HDFStore for more control
    print("\nUsing HDFStore for Advanced Operations")
    with pd.HDFStore(hdf5_file, mode='r') as store:
        print("Keys in HDF5 file:", list(store.keys()))
        print(f"\nSales data info:")
        print(store.info())
    
    # Querying HDF5 (requires table format)
    print("\nSaving with Table Format (allows querying)")
    hdf5_query_file = os.path.join(temp_dir, 'queryable_data.h5')
    sales_data.to_hdf(hdf5_query_file, key='sales', mode='w', format='table', data_columns=True)
    
    # Query specific data
    queried_data = pd.read_hdf(hdf5_query_file, key='sales', 
                               where='Salary > 80000')
    print(f"✓ Queried {len(queried_data)} rows where Salary > 80000")
    print(queried_data.head())
    
    # Compression
    print("\nHDF5 with Compression")
    hdf5_compressed = os.path.join(temp_dir, 'compressed_data.h5')
    sales_data.to_hdf(hdf5_compressed, key='sales', mode='w', 
                      complevel=9, complib='blosc')
    
    uncompressed_size = os.path.getsize(hdf5_file)
    compressed_size = os.path.getsize(hdf5_compressed)
    print(f"Uncompressed: {uncompressed_size:,} bytes")
    print(f"Compressed: {compressed_size:,} bytes")
    print(f"Compression ratio: {uncompressed_size/compressed_size:.2f}x")

except ImportError:
    print("\nError: The 'tables' library is required for HDF5 operations.")
    print("Please install it using: pip install tables")
except Exception as e:
    print(f"\nAn error occurred during HDF5 operations: {e}")

#3. Excel Format (.xlsx)
#Best for: Business reports, sharing with non-technical users
#Pros:
#Universal - everyone has Excel
#Multiple sheets in one file
#Perfect for business stakeholders
#Easy to open and edit

#Cons:
#Slow for large datasets
#Limited to ~1 million rows per sheet
#Requires openpyxl library (pip install openpyxl)
#Use when: Creating reports for business users or sharing with non-programmers

# Note: Excel requires 'openpyxl' package
try:
    # Save to Excel
    excel_file = os.path.join(temp_dir, 'company_report.xlsx')
    print(f"\nSaving to Excel: {excel_file}")
    
    # Method 1: Save single DataFrame
    sales_data.head(20).to_excel(excel_file, sheet_name='Sales', index=False)
    print("Sales data saved to sheet 'Sales'")
    
    # Method 2: Save multiple sheets
    excel_multi_file = os.path.join(temp_dir, 'complete_report.xlsx')
    print(f"\nSaving Multiple Sheets: {excel_multi_file}")
    
    with pd.ExcelWriter(excel_multi_file, engine='openpyxl') as writer:
        sales_data.head(20).to_excel(writer, sheet_name='Sales', index=False)
        employee_data.to_excel(writer, sheet_name='Employees', index=False)
        
        # Create a summary sheet
        summary = pd.DataFrame({
            'Metric': ['Total Sales Records', 'Total Employees', 
                      'Average Revenue', 'Average Salary'],
            'Value': [len(sales_data), len(employee_data),
                     sales_data['Amount'].mean(), employee_data['Salary'].mean()]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print("Saved 3 sheets: Sales, Employees, Summary")
    
    # Check file size
    file_size = os.path.getsize(excel_multi_file)
    print(f"File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    # Read from Excel
    print("\nReading from Excel")
    
    # Read specific sheet
    df_excel_sales = pd.read_excel(excel_multi_file, sheet_name='Sales')
    print("Loaded 'Sales' sheet:")
    print(df_excel_sales.head())
    
    # Read all sheets
    print("\nReading All Sheets")
    all_sheets = pd.read_excel(excel_multi_file, sheet_name=None)
    print(f"Available sheets: {list(all_sheets.keys())}")
    
    for sheet_name, df in all_sheets.items():
        print(f"\n{sheet_name}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Read with specific options
    print("\n--- Reading with Options ---")
    df_custom = pd.read_excel(excel_multi_file, 
                             sheet_name='Employees',
                             usecols=['Name', 'Department', 'Salary'],
                             nrows=10)
    print("✓ Loaded only 3 columns and 10 rows:")
    print(df_custom)

except ImportError:
    print("\nError: The 'openpyxl' library is required for Excel operations.")
    print("Please install it using: pip install openpyxl")
except Exception as e:
    print(f"\nAn error occurred during Excel operations: {e}")

# 4. FORMAT COMPARISON
print("\nSpeed Comparison ")
import time

# Test pickle
start = time.time()
sales_data.to_pickle(os.path.join(temp_dir, 'speed_test.pkl'))
pickle_write_time = time.time() - start

start = time.time()
pd.read_pickle(os.path.join(temp_dir, 'speed_test.pkl'))
pickle_read_time = time.time() - start

# Test CSV for comparison
start = time.time()
sales_data.to_csv(os.path.join(temp_dir, 'speed_test.csv'), index=False)
csv_write_time = time.time() - start

start = time.time()
pd.read_csv(os.path.join(temp_dir, 'speed_test.csv'))
csv_read_time = time.time() - start

print("\nPerformance Comparison (100 rows):")
print(f"{'Format':<15} {'Write Time':<15} {'Read Time':<15}")
print("-" * 45)
print(f"{'Pickle':<15} {pickle_write_time:.4f}s{'':<8} {pickle_read_time:.4f}s")
print(f"{'CSV':<15} {csv_write_time:.4f}s{'':<8} {csv_read_time:.4f}s")

print("\n--- File Size Comparison ---")
pickle_size = os.path.getsize(os.path.join(temp_dir, 'speed_test.pkl'))
csv_size = os.path.getsize(os.path.join(temp_dir, 'speed_test.csv'))

print(f"{'Format':<15} {'File Size':<20}")
print("-" * 35)
print(f"{'Pickle':<15} {pickle_size:,} bytes ({pickle_size/1024:.2f} KB)")
print(f"{'CSV':<15} {csv_size:,} bytes ({csv_size/1024:.2f} KB)")

#3. Web APIs and Databases
#Web APIs: The chapter demonstrates using the requests library to fetch data from websites and converting JSON responses into pandas DataFrames.
#Databases: pandas can interact with SQL databases (like SQLite, PostgreSQL, MySQL). The read_sql 
# function (often used with SQLAlchemy) allows you to execute a query and load the results directly into a DataFrame.

#PART 1: Web APIs 
#What You'll Learn:
#Basic Concepts:
#APIs return data (usually JSON format)
#Convert JSON → pandas DataFrame
#Handle different JSON structures

#5 Real-World Examples:
#Weather API - Simple JSON with nested forecast data
#Stock Price API - Time series data with OHLC prices
#User API - Simple list of user records
#Nested JSON - Complex product data with reviews
#Paginated API - Combining multiple pages of results

#Key Functions:
#json.loads() - Parse JSON string
#pd.DataFrame() - Convert to DataFrame
#pd.json_normalize() - Flatten nested JSON


# 1. SIMULATED API EXAMPLES (No internet required)

# Simulate API response for weather data
print("\nExample 1: Weather API Response ")
weather_api_response = '''
{
    "city": "New York",
    "forecast": [
        {
            "date": "2024-01-01",
            "temperature": 45,
            "humidity": 65,
            "condition": "Cloudy"
        },
        {
            "date": "2024-01-02",
            "temperature": 48,
            "humidity": 70,
            "condition": "Rainy"
        },
        {
            "date": "2024-01-03",
            "temperature": 42,
            "humidity": 60,
            "condition": "Sunny"
        },
        {
            "date": "2024-01-04",
            "temperature": 50,
            "humidity": 55,
            "condition": "Partly Cloudy"
        }
    ]
}
'''

print("Raw API Response (JSON):")
print(weather_api_response[:200] + "...")

# Parse JSON and convert to DataFrame
weather_data = json.loads(weather_api_response)
df_weather = pd.DataFrame(weather_data['forecast'])
print("\nConverted to DataFrame:")
print(df_weather)
print(f"\nData types:\n{df_weather.dtypes}")

# Convert date column to datetime
df_weather['date'] = pd.to_datetime(df_weather['date'])
print("\nAfter converting date column:")
print(df_weather.dtypes)


# Example 2: Stock Price API

print("\nExample 2: Stock Price API Response")
stock_api_response = '''
{
    "symbol": "AAPL",
    "prices": [
        {"timestamp": "2024-01-01T09:30:00", "open": 185.50, "high": 187.20, "low": 184.80, "close": 186.90, "volume": 50000000},
        {"timestamp": "2024-01-02T09:30:00", "open": 186.90, "high": 189.50, "low": 186.00, "close": 188.75, "volume": 55000000},
        {"timestamp": "2024-01-03T09:30:00", "open": 188.75, "high": 190.00, "low": 187.50, "close": 189.25, "volume": 48000000},
        {"timestamp": "2024-01-04T09:30:00", "open": 189.25, "high": 191.50, "low": 188.90, "close": 190.80, "volume": 52000000}
    ]
}
'''

stock_data = json.loads(stock_api_response)
df_stock = pd.DataFrame(stock_data['prices'])
print(f"\nStock: {stock_data['symbol']}")
print(df_stock)

# Calculate daily returns
df_stock['daily_return'] = df_stock['close'].pct_change() * 100
print("\nWith calculated daily returns:")
print(df_stock[['timestamp', 'close', 'daily_return']])


# Example 3: User API (List of Dictionaries)

print("\nExample 3: User API Response (List Format) ")
users_api_response = '''
[
    {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com",
        "age": 28,
        "country": "USA"
    },
    {
        "id": 2,
        "name": "Emma Smith",
        "email": "emma@example.com",
        "age": 34,
        "country": "UK"
    },
    {
        "id": 3,
        "name": "Michael Chen",
        "email": "michael@example.com",
        "age": 29,
        "country": "Canada"
    }
]
'''

users_data = json.loads(users_api_response)
df_users = pd.DataFrame(users_data)
print("\nUsers DataFrame:")
print(df_users)


# Example 4: Nested JSON (Complex API Response)

print("\nExample 4: Nested JSON API Response ")
nested_api_response = '''
{
    "results": [
        {
            "product_id": 1,
            "name": "Laptop",
            "price": 1200,
            "details": {
                "brand": "Dell",
                "ram": "16GB",
                "storage": "512GB SSD"
            },
            "reviews": [
                {"rating": 5, "comment": "Excellent"},
                {"rating": 4, "comment": "Good value"}
            ]
        },
        {
            "product_id": 2,
            "name": "Phone",
            "price": 800,
            "details": {
                "brand": "Apple",
                "ram": "8GB",
                "storage": "256GB"
            },
            "reviews": [
                {"rating": 5, "comment": "Amazing"},
                {"rating": 5, "comment": "Best phone"}
            ]
        }
    ]
}
'''

nested_data = json.loads(nested_api_response)

# Method 1: Basic conversion (nested data stays as dict)
df_basic = pd.DataFrame(nested_data['results'])
print("\nBasic conversion (nested data intact):")
print(df_basic)

# Method 2: Normalize nested data (flatten structure)
df_normalized = pd.json_normalize(nested_data['results'])
print("\nNormalized (flattened) DataFrame:")
print(df_normalized)


# Example 5: Handling Paginated API Responses

print("\nExample 5: Paginated API Response")
print("""
Many APIs return data in pages. You need to fetch multiple pages and 
combine them into a single DataFrame.
""")

# Simulate multiple pages of data
page1 = '''{"page": 1, "data": [{"id": 1, "value": 100}, {"id": 2, "value": 200}]}'''
page2 = '''{"page": 2, "data": [{"id": 3, "value": 300}, {"id": 4, "value": 400}]}'''
page3 = '''{"page": 3, "data": [{"id": 5, "value": 500}, {"id": 6, "value": 600}]}'''

all_data = []
for page_response in [page1, page2, page3]:
    page_data = json.loads(page_response)
    all_data.extend(page_data['data'])
    print(f"Fetched page {page_data['page']}: {len(page_data['data'])} records")

df_paginated = pd.DataFrame(all_data)
print(f"\nCombined DataFrame ({len(df_paginated)} total records):")
print(df_paginated)

#PART 2: Databases 
#What You'll Learn:
#Database Operations:
#Writing Data - df.to_sql() saves DataFrames to database
#Reading Data - pd.read_sql() executes queries
#Complex Queries - JOINs, GROUP BY, aggregations
#Safe Queries - Parameterized queries (prevent SQL injection!)
#Analysis Pipeline - Complete workflow example
#SQL Examples Include:
#Simple SELECT queries
#WHERE filters
#GROUP BY aggregations
#JOIN multiple tables
#Statistical calculations
# PART 2: DATABASES


print("\n--- What is a Database? ---")
print("""
A database is an organized collection of structured data. SQL databases 
use structured query language (SQL) to manage and query data.

POPULAR SQL DATABASES:
- SQLite: Lightweight, file-based, no server needed
- PostgreSQL: Powerful, open-source, production-ready
- MySQL: Popular, widely used, good for web applications
- SQL Server: Microsoft's enterprise database

PANDAS + SQL WORKFLOW:
1. Connect to database
2. Execute SQL query
3. Load results into DataFrame
4. Analyze data with pandas
5. (Optional) Write results back to database
""")


# 1.SQLite DATABASE EXAMPLES

print("\n" + "=" * 70)
print("1. WORKING WITH SQLite DATABASE")
print("=" * 70)

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')
print("\n✓ Created in-memory SQLite database")

# Create sample data
employees = pd.DataFrame({
    'employee_id': range(1, 11),
    'name': ['John Doe', 'Emma Smith', 'Michael Chen', 'Sophia Lee', 
             'David Brown', 'Lisa Wang', 'James Wilson', 'Emily Davis',
             'Robert Taylor', 'Maria Garcia'],
    'department': ['Sales', 'IT', 'HR', 'Sales', 'IT', 
                   'Finance', 'Sales', 'HR', 'IT', 'Finance'],
    'salary': [75000, 85000, 65000, 72000, 88000, 
               78000, 70000, 67000, 90000, 80000],
    'hire_date': pd.date_range('2020-01-01', periods=10, freq='2M')
})

sales = pd.DataFrame({
    'sale_id': range(1, 16),
    'employee_id': [1, 1, 4, 4, 1, 7, 7, 4, 1, 7, 4, 1, 7, 4, 1],
    'amount': [15000, 22000, 18000, 25000, 19000, 
               21000, 17000, 23000, 20000, 16000,
               24000, 18500, 22500, 19500, 21500],
    'sale_date': pd.date_range('2024-01-01', periods=15, freq='2D')
})

print("\nSample Data Created:")
print(f"Employees: {len(employees)} records")
print(employees.head())
print(f"\nSales: {len(sales)} records")
print(sales.head())


# 2. WRITING DATA TO DATABASE

print("\n--- Writing DataFrames to Database ---")

# Write DataFrames to database tables
employees.to_sql('employees', conn, if_exists='replace', index=False)
print("✓ Created table 'employees'")

sales.to_sql('sales', conn, if_exists='replace', index=False)
print("✓ Created table 'sales'")


# 3. READING DATA FROM DATABASE

print("\n--- Reading Data from Database ---")

# Method 1: Read entire table
df_emp = pd.read_sql('SELECT * FROM employees', conn)
print("\nQuery: SELECT * FROM employees")
print(df_emp.head())

# Method 2: Read with WHERE clause
df_it = pd.read_sql('SELECT * FROM employees WHERE department = "IT"', conn)
print("\nQuery: SELECT * FROM employees WHERE department = 'IT'")
print(df_it)

# Method 3: Aggregate queries
df_dept_avg = pd.read_sql('''
    SELECT department, 
           COUNT(*) as employee_count,
           AVG(salary) as avg_salary,
           MAX(salary) as max_salary
    FROM employees
    GROUP BY department
    ORDER BY avg_salary DESC
''', conn)
print("\nQuery: Average salary by department")
print(df_dept_avg)


# 4. COMPLEX SQL QUERIES (JOINS)

print("\n--- Complex Queries with JOINs ---")

# Join employees and sales tables
df_sales_detail = pd.read_sql('''
    SELECT 
        e.name,
        e.department,
        s.sale_id,
        s.amount,
        s.sale_date
    FROM sales s
    JOIN employees e ON s.employee_id = e.employee_id
    ORDER BY s.sale_date
''', conn)

print("\nQuery: Sales with employee details (JOIN)")
print(df_sales_detail.head(10))

# Calculate total sales per employee
df_employee_sales = pd.read_sql('''
    SELECT 
        e.name,
        e.department,
        COUNT(s.sale_id) as total_sales,
        SUM(s.amount) as total_revenue,
        AVG(s.amount) as avg_sale_amount
    FROM employees e
    LEFT JOIN sales s ON e.employee_id = s.employee_id
    GROUP BY e.employee_id, e.name, e.department
    ORDER BY total_revenue DESC
''', conn)

print("\nQuery: Total sales per employee")
print(df_employee_sales)