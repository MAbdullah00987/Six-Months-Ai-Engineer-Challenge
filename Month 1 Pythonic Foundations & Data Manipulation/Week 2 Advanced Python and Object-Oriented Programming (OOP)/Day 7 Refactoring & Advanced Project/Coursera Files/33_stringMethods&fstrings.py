

#String Formatting Methods and F-Strings in PythonWhy Do We Need String Formatting?String formatting allows you to:

#Insert variables into strings - Combine text with dynamic values
#Control output appearance - Align text, set width, decimal places
#Build readable messages - Create clear, formatted output
#Simplify string construction - Avoid messy concatenation with +
#Make code maintainable - Easier to read and modify
#1. .format() Method - Three ExamplesThe .format() method uses {} placeholders that get replaced with values.Example 1: Basic .format() usage
# Simple placeholder replacement
name = "Alice"
age = 25
city = "New York"

# Using .format() with positional arguments
message = "My name is {}, I am {} years old, and I live in {}.".format(name, age, city)
print(message)
# Output: My name is Alice, I am 25 years old, and I live in New York.

# Using indexed placeholders (can reorder or reuse)
message2 = "Hello {0}! {0}, you are {1} years old.".format(name, age)
print(message2)
# Output: Hello Alice! Alice, you are 25 years old.

# Using named placeholders
message3 = "Name: {n}, Age: {a}, City: {c}".format(n=name, a=age, c=city)
print(message3)
# Output: Name: Alice, Age: 25, City: New YorkExample 2: Number formatting with .format()
# Formatting numbers
price = 19.99
quantity = 3
tax_rate = 0.08

# Format with decimal places
total = price * quantity
print("Price: ${:.2f}".format(price))
# Output: Price: $19.99

# Format with thousands separator
large_number = 1234567.89
print("Amount: ${:,.2f}".format(large_number))
# Output: Amount: $1,234,567.89

# Format percentages
print("Tax rate: {:.1%}".format(tax_rate))
# Output: Tax rate: 8.0%

# Padding and alignment
items = [("Apple", 1.50), ("Banana", 0.75), ("Orange", 2.00)]
print("\n=== Receipt ===")
for item, cost in items:
    print("{:<10} ${:>6.2f}".format(item, cost))
# Output:
# Apple      $  1.50
# Banana     $  0.75
# Orange     $  2.00Example 3: Advanced .format() with alignment and width
# Creating formatted tables
students = [
    {"name": "Alice", "math": 85, "english": 92, "science": 88},
    {"name": "Bob", "math": 78, "english": 85, "science": 90},
    {"name": "Charlie", "math": 92, "english": 88, "science": 85}
]

# Table header
print("{:<10} {:>6} {:>8} {:>8} {:>8}".format("Name", "Math", "English", "Science", "Average"))
print("-" * 50)

# Table rows
for student in students:
    avg = (student["math"] + student["english"] + student["science"]) / 3
    print("{:<10} {:>6} {:>8} {:>8} {:>8.1f}".format(
        student["name"],
        student["math"],
        student["english"],
        student["science"],
        avg
    ))

name = "Bob"
age = 30
profession = "Engineer"

# Simple variable insertion
greeting = f"Hello, my name is {name}."
print(greeting)
# Output: Hello, my name is Bob.

# Multiple variables
intro = f"I'm {name}, I'm {age} years old, and I work as a {profession}."
print(intro)
# Output: I'm Bob, I'm 30 years old, and I work as a Engineer.

# Expressions inside f-strings
print(f"{name} will be {age + 5} years old in 5 years.")
# Output: Bob will be 35 years old in 5 years.

# Method calls inside f-strings
print(f"My name in uppercase: {name.upper()}")
# Output: My name in uppercase: BOB

# Calculations
x = 10
y = 20
print(f"The sum of {x} and {y} is {x + y}")
# Output: The sum of 10 and 20 is 30Example 2: F-strings with formatting options
# Number formatting with f-strings
pi = 3.14159265359
price = 49.99
large_num = 1_000_000

# Decimal places
print(f"Pi to 2 decimals: {pi:.2f}")
# Output: Pi to 2 decimals: 3.14

print(f"Pi to 4 decimals: {pi:.4f}")
# Output: Pi to 4 decimals: 3.1416

# Currency formatting
print(f"Price: ${price:.2f}")
# Output: Price: $49.99

# Thousands separator
print(f"Large number: {large_num:,}")
# Output: Large number: 1,000,000

# Percentage
success_rate = 0.856
print(f"Success rate: {success_rate:.1%}")
# Output: Success rate: 85.6%

# Scientific notation
big = 123456789
print(f"Scientific: {big:.2e}")
# Output: Scientific: 1.23e+08

# Binary, octal, hex
number = 42
print(f"Decimal: {number}, Binary: {number:b}, Hex: {number:x}")
# Output: Decimal: 42, Binary: 101010, Hex: 2aExample 3: F-strings with alignment and width
# Alignment and padding
products = [
    {"name": "Laptop", "price": 999.99, "stock": 5},
    {"name": "Mouse", "price": 25.50, "stock": 50},
    {"name": "Keyboard", "price": 75.00, "stock": 30},
    {"name": "Monitor", "price": 299.99, "stock": 15}
]

# Create inventory report
print(f"{'Product':<15} {'Price':>10} {'Stock':>8} {'Value':>12}")
print("=" * 50)

for item in products:
    total_value = item['price'] * item['stock']
    print(f"{item['name']:<15} ${item['price']:>9.2f} {item['stock']:>8} ${total_value:>11.2f}")



# Center alignment
title = "Sales Report"
print(f"\n{title:=^50}")  # Centers with = padding
# Output: ===================Sales Report===================

# Date formatting
from datetime import datetime
now = datetime.now()
print(f"\nReport generated: {now:%Y-%m-%d %H:%M:%S}")
# Output: Report generated: 2024-11-14 10:30:45