
#Accumulator Pattern in Strings
#What is the Accumulator Pattern?
#The accumulator pattern is a programming technique where you:

#Start with an empty or initial value (the accumulator)
#Loop through data
#Gradually build up (accumulate) the result by adding to it in each iteration
#Return the final accumulated value

#Why Do We Need It?
#Key Benefits:

#Build complex strings incrementally - Construct strings piece by piece
#Process data sequentially - Transform data one element at a time
#Filter and collect - Select certain items while building a string
##Avoid memory issues - More efficient than creating many temporary strings
#Clear logic flow - Easy to understand and modify

#Common Use Cases:

#Creating formatted reports
#Building HTML/XML content
#Processing text files line by line
#Creating CSV data
#Generating dynamic messages


#Type 1: String Accumulator with Concatenation
#Building strings by repeatedly adding to them using + or +=
#Example 1: Building a simple sentence
# Start with empty accumulator
sentence = ""

# Words to add
words = ["Python", "is", "a", "powerful", "language"]

# Accumulate words into sentence
for word in words:
    sentence += word + " "

# Remove trailing space
sentence = sentence.strip()
print(sentence)
# Output: Python is a powerful language

# Step-by-step visualization:
print("\n=== Step by Step ===")
result = ""
for i, word in enumerate(words, 1):
    result += word + " "
    print(f"Step {i}: '{result.strip()}'")
# Output:
# Step 1: 'Python'
# Step 2: 'Python is'
# Step 3: 'Python is a'
# Step 4: 'Python is a powerful'
# Step 5: 'Python is a powerful language'

#Example 2: Building formatted output
# Creating a formatted student report
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78},
    {"name": "David", "grade": 95}
]

# Start with empty report (accumulator)
report = ""

# Add header
report += "=" * 40 + "\n"
report += "STUDENT GRADE REPORT\n"
report += "=" * 40 + "\n"

# Accumulate student data
for student in students:
    line = f"{student['name']:<15} Grade: {student['grade']}\n"
    report += line

# Add footer
report += "=" * 40 + "\n"
total = sum(s['grade'] for s in students)
average = total / len(students)
report += f"Average Grade: {average:.1f}\n"
report += "=" * 40

print(report)

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Accumulate only even numbers
even_string = ""
for num in numbers:
    if num % 2 == 0:
        even_string += str(num) + ", "

# Remove trailing comma and space
even_string = even_string.rstrip(", ")
print(f"Even numbers: {even_string}")
# Output: Even numbers: 2, 4, 6, 8, 10

# Accumulate with conditions and formatting
text = "Hello World! Python is AMAZING. Let's CODE!"
uppercase_letters = ""
lowercase_letters = ""
digits = ""
special = ""

for char in text:
    if char.isupper():
        uppercase_letters += char
    elif char.islower():
        lowercase_letters += char
    elif char.isdigit():
        digits += char
    elif not char.isspace():
        special += char

print(f"Uppercase: {uppercase_letters}")
print(f"Lowercase: {lowercase_letters}")
print(f"Digits: {digits}")
print(f"Special: {special}")
# Output:
# Uppercase: HWPACL
# Lowercase: elloorldythonissomeode
# Digits: 
# Special: !'

#Type 2: List Accumulator with join()
#Accumulating strings in a list, then joining them (MORE EFFICIENT)

#Example 1: Building strings efficiently
# BETTER WAY: Accumulate in list, then join
words = ["Python", "is", "awesome", "for", "programming"]

# Accumulate in list (more efficient than string concatenation)
word_list = []
for word in words:
    word_list.append(word)

# Join at the end
sentence = " ".join(word_list)
print(sentence)
# Output: Python is awesome for programming

# Why is this better?
# Strings are immutable - each += creates a NEW string
# Lists are mutable - appending is fast

# Performance comparison:
import time

# Method 1: String concatenation (SLOWER)
start = time.time()
result1 = ""
for i in range(10000):
    result1 += str(i) + " "
time1 = time.time() - start
print(f"String concat time: {time1:.4f}s")

# Method 2: List + join (FASTER)
start = time.time()
result2_list = []
for i in range(10000):
    result2_list.append(str(i))
result2 = " ".join(result2_list)
time2 = time.time() - start
print(f"List + join time: {time2:.4f}s")
print(f"List method is {time1/time2:.1f}x faster!")

#Example 2: Building HTML content
# Creating HTML table using list accumulator
data = [
    ["Name", "Age", "City"],
    ["Alice", "25", "New York"],
    ["Bob", "30", "London"],
    ["Charlie", "28", "Paris"]
]

# Accumulate HTML parts in list
html_parts = []

# Add opening tags
html_parts.append("<table border='1'>")

# Process each row
for i, row in enumerate(data):
    if i == 0:
        # Header row
        html_parts.append("  <thead>")
        html_parts.append("    <tr>")
        for cell in row:
            html_parts.append(f"      <th>{cell}</th>")
        html_parts.append("    </tr>")
        html_parts.append("  </thead>")
        html_parts.append("  <tbody>")
    else:
        # Data rows
        html_parts.append("    <tr>")
        for cell in row:
            html_parts.append(f"      <td>{cell}</td>")
        html_parts.append("    </tr>")

# Add closing tags
html_parts.append("  </tbody>")
html_parts.append("</table>")

# Join all parts
html = "\n".join(html_parts)
print(html)



#Example 3: Processing log files
# Simulate log entries
log_entries = [
    "2024-11-14 10:00:00 INFO User logged in",
    "2024-11-14 10:05:00 ERROR Database connection failed",
    "2024-11-14 10:06:00 INFO Retrying connection",
    "2024-11-14 10:07:00 ERROR Connection timeout",
    "2024-11-14 10:10:00 WARNING Low memory",
    "2024-11-14 10:15:00 INFO User logged out",
    "2024-11-14 10:20:00 ERROR Disk space low"
]

# Accumulate only ERROR messages
error_log = []
warning_log = []
info_log = []

for entry in log_entries:
    if "ERROR" in entry:
        error_log.append(entry)
    elif "WARNING" in entry:
        warning_log.append(entry)
    elif "INFO" in entry:
        info_log.append(entry)

# Create summary report
report_lines = []
report_lines.append("=" * 60)
report_lines.append("LOG SUMMARY REPORT")
report_lines.append("=" * 60)
report_lines.append("")
report_lines.append(f"Total Entries: {len(log_entries)}")
report_lines.append(f"Errors: {len(error_log)}")
report_lines.append(f"Warnings: {len(warning_log)}")
report_lines.append(f"Info: {len(info_log)}")
report_lines.append("")
report_lines.append("ERROR MESSAGES:")
report_lines.append("-" * 60)
report_lines.extend(error_log)
report_lines.append("=" * 60)

# Join and print
report = "\n".join(report_lines)
print(report)




#Type 3: Accumulator with String Methods
#Using string methods like .join() directly with comprehensions or generators
#Example 1: Creating CSV data
# Data to convert to CSV
employees = [
    {"name": "Alice Johnson", "department": "Engineering", "salary": 75000},
    {"name": "Bob Smith", "department": "Marketing", "salary": 65000},
    {"name": "Charlie Brown", "department": "Sales", "salary": 70000},
    {"name": "Diana Prince", "department": "HR", "salary": 68000}
]

# Accumulate CSV rows
csv_lines = []

# Add header
header = ",".join(employees[0].keys())
csv_lines.append(header)

# Add data rows
for emp in employees:
    row = ",".join(str(value) for value in emp.values())
    csv_lines.append(row)

# Join all lines
csv_content = "\n".join(csv_lines)
print(csv_content)

# Output:
# name,department,salary
# Alice Johnson,Engineering,75000
# Bob Smith,Marketing,65000
# Charlie Brown,Sales,70000
# Diana Prince,HR,68000

# Alternative: Using list comprehension (more Pythonic)
csv_lines2 = [",".join(employees[0].keys())]  # Header
csv_lines2.extend([",".join(str(v) for v in emp.values()) for emp in employees])
csv_content2 = "\n".join(csv_lines2)
print("\n" + csv_content2)

#Example 2: Creating formatted menu
# Menu items with prices
menu_items = [
    ("Burger", 8.99),
    ("Pizza", 12.99),
    ("Salad", 6.99),
    ("Pasta", 10.99),
    ("Soda", 2.99),
    ("Coffee", 3.99)
]

# Accumulate menu lines with formatting
menu_lines = []

# Add header
menu_lines.append("=" * 40)
menu_lines.append("RESTAURANT MENU".center(40))
menu_lines.append("=" * 40)
menu_lines.append("")

# Group items (let's say first 4 are food, last 2 are drinks)
menu_lines.append("FOOD:")
menu_lines.append("-" * 40)
for item, price in menu_items[:4]:
    line = f"{item:<30} ${price:>6.2f}"
    menu_lines.append(line)

menu_lines.append("")
menu_lines.append("BEVERAGES:")
menu_lines.append("-" * 40)
for item, price in menu_items[4:]:
    line = f"{item:<30} ${price:>6.2f}"
    menu_lines.append(line)

menu_lines.append("=" * 40)

# Join with newlines
menu = "\n".join(menu_lines)
print(menu)



#Example 3: Text transformation
# Transform text with accumulator pattern
text = "The quick brown fox jumps over the lazy dog"

# Example 1: Reverse each word
words = text.split()
reversed_words = []
for word in words:
    reversed_words.append(word[::-1])
result1 = " ".join(reversed_words)
print(f"Reversed words: {result1}")
# Output: ehT kciuq nworb xof spmuj revo eht yzal god



#Example 3: Creating markdown documentation
# Generate markdown documentation
functions = [
    {
        "name": "calculate_total",
        "params": ["price", "quantity", "tax_rate"],
        "returns": "float",
        "description": "Calculates the total price including tax"
    },
    {
        "name": "format_currency",
        "params": ["amount", "currency"],
        "returns": "str",
        "description": "Formats a number as currency string"
    },
    {
        "name": "validate_email",
        "params": ["email"],
        "returns": "bool",
        "description": "Validates email format"
    }
]

# Accumulate markdown lines
md_lines = []

# Document header
md_lines.append("# API Documentation")
md_lines.append("")
md_lines.append("This document describes the available functions.")
md_lines.append("")

# Function documentation
for func in functions:
    md_lines.append(f"## `{func['name']}()`")
    md_lines.append("")
    md_lines.append(func['description'])
    md_lines.append("")
    md_lines.append("**Parameters:**")
    for param in func['params']:
        md_lines.append(f"- `{param}`")
    md_lines.append("")
    md_lines.append(f"**Returns:** `{func['returns']}`")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

# Join all lines
markdown = "\n".join(md_lines)
print(markdown)

# Output:
# # API Documentation
# 
# This document describes the available functions.
# 
# ## `calculate_total()`
# 
# Calculates the total price including tax
# 
# **Parameters:**
# - `price`
# - `quantity`
# - `tax_rate`
# 
# **Returns:** `float`
# 
# ---
# 
# ## `format_currency()`
# ...