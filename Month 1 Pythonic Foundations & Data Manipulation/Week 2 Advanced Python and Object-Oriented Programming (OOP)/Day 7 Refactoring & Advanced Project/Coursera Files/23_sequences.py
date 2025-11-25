

#Sequences in Python
#Definition of Sequences
#Sequences are ordered collections of items where each element has a specific position (index). They are data structures that store multiple values in a single variable, maintaining the order in which elements are added.
#Simple explanation: A sequence is like a numbered list where each item has a position (0, 1, 2, 3...), and you can access, modify, or iterate through items in order.

#Types of Sequences in Python
#1. List - Mutable, ordered collection
#pythonmy_list = [1, 2, 3, 4, 5]
#2. Tuple - Immutable, ordered collection
#pythonmy_tuple = (1, 2, 3, 4, 5)
#3. String - Immutable sequence of characters
#pythonmy_string = "Hello"
#4. Range - Immutable sequence of numbers
#pythonmy_range = range(1, 10)
#5. Bytes - Immutable sequence of bytes
#pythonmy_bytes = b"Hello"
#6. Bytearray - Mutable sequence of bytes
#pythonmy_bytearray = bytearray(b"Hello")

#Why We Need to Use Sequences
#1. Store Multiple Values
#Instead of creating separate variables, store related data together.
#python#  Without sequences - messy and limited
#student1 = "Ali"
#student2 = "Sara"
#student3 = "Ahmed"
#student4 = "Fatima"
#student5 = "Hassan"

# With sequences - clean and scalable
#students = ["Ali", "Sara", "Ahmed", "Fatima", "Hassan"]

#2. Maintain Order
#Sequences preserve the order of elements, which is crucial for many applications.
#python# Order matters in these cases
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
steps_in_recipe = ["Mix flour", "Add eggs", "Bake for 30 minutes"]
leaderboard = ["Player1", "Player2", "Player3"]

print(days_of_week[0])  # Monday (first day)
print(days_of_week[6])  # Sunday (last day)

#3. Efficient Data Management
#Easy to add, remove, search, and manipulate multiple items.
# Shopping cart management
cart = ["Apple", "Banana", "Milk"]

# Add items
cart.append("Bread")
print(cart)  # ['Apple', 'Banana', 'Milk', 'Bread']

# Remove items
cart.remove("Banana")
print(cart)  # ['Apple', 'Milk', 'Bread']

# Check if item exists
if "Milk" in cart:
    print("Milk is in cart")

# Count items
print(f"Total items: {len(cart)}")  # Total items: 3

#4. Iteration and Processing
#Process multiple items with loops efficiently.
# Calculate total of prices
prices = [100, 250, 50, 300, 150]

total = 0
for price in prices:
    total += price

print(f"Total: {total}")  # Total: 850

# Or using built-in function
print(f"Total: {sum(prices)}")  # Total: 850

# Apply discount to all items
discounted_prices = []
for price in prices:
    discounted_prices.append(price * 0.9)  # 10% discount

print(discounted_prices)  # [90.0, 225.0, 45.0, 270.0, 135.0]

#5. Indexing and Slicing
#Access specific elements or ranges of elements easily.
numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Indexing - access single element
print(numbers[0])      # 10 (first element)
print(numbers[4])      # 50 (fifth element)
print(numbers[-1])     # 90 (last element)
print(numbers[-2])     # 80 (second to last)

# Slicing - access multiple elements
print(numbers[2:5])    # [30, 40, 50] (index 2 to 4)
print(numbers[:3])     # [10, 20, 30] (first 3)
print(numbers[5:])     # [60, 70, 80, 90] (from index 5 to end)
print(numbers[::2])    # [10, 30, 50, 70, 90] (every 2nd element)
print(numbers[::-1])   # [90, 80, 70, 60, 50, 40, 30, 20, 10] (reversed)

#6. Data Organization
#Group related information together logically.
# Student information
student_info = ["Ahmed", 20, "Computer Science", 3.8]
name = student_info[0]
age = student_info[1]
major = student_info[2]
gpa = student_info[3]

print(f"{name} is {age} years old, studying {major} with GPA {gpa}")
# Ahmed is 20 years old, studying Computer Science with GPA 3.8

# Employee records
employees = [
    ["Ali", "Manager", 80000],
    ["Sara", "Developer", 60000],
    ["Hassan", "Designer", 55000]
]

for employee in employees:
    print(f"{employee[0]}: {employee[1]} - ${employee[2]}")

# Output:
# Ali: Manager - $80000
# Sara: Developer - $60000
# Hassan: Designer - $55000

#7. Mathematical and Statistical Operations
#Perform calculations on collections of numbers.
test_scores = [85, 92, 78, 95, 88, 76, 90]

# Statistical operations
average = sum(test_scores) / len(test_scores)
highest = max(test_scores)
lowest = min(test_scores)
total_students = len(test_scores)

print(f"Average Score: {average:.2f}")      # Average Score: 86.29
print(f"Highest Score: {highest}")          # Highest Score: 95
print(f"Lowest Score: {lowest}")            # Lowest Score: 76
print(f"Total Students: {total_students}")  # Total Students: 7

# Count passing students (60+)
passing = 0
for score in test_scores:
    if score >= 60:
        passing += 1

print(f"Passing Students: {passing}")  # Passing Students: 7

#8. Pattern Matching and Searching
#Find specific elements or patterns in data.
# Find specific items
products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Webcam"]

search_term = "Keyboard"
if search_term in products:
    position = products.index(search_term)
    print(f"{search_term} found at position {position}")
    # Keyboard found at position 2

# Find all items starting with 'M'
m_products = []
for product in products:
    if product.startswith('M'):
        m_products.append(product)

print(m_products)  # ['Mouse', 'Monitor']

# Count occurrences
votes = ["Ali", "Sara", "Ali", "Hassan", "Ali", "Sara"]
ali_votes = votes.count("Ali")
print(f"Ali got {ali_votes} votes")  # Ali got 3 votes

#9. String Manipulation
#Strings are sequences of characters, allowing powerful text processing.
message = "Hello World"

# Access characters
print(message[0])      # H
print(message[6])      # W
print(message[-1])     # d

# Slicing strings
print(message[0:5])    # Hello
print(message[6:])     # World

# String operations
print(message.upper())      # HELLO WORLD
print(message.lower())      # hello world
print(message.split())      # ['Hello', 'World']

# Check substring
if "World" in message:
    print("Found World!")

# Iteration through characters
for char in message:
    print(char, end="-")
# H-e-l-l-o- -W-o-r-l-d-

#10. Memory Efficiency
#Store and manage large amounts of data efficiently.
# Generate large sequence efficiently
numbers = list(range(1, 1001))  # 1 to 1000

print(f"First 10: {numbers[:10]}")
print(f"Last 10: {numbers[-10:]}")
print(f"Total: {len(numbers)}")

# Count even numbers
even_count = 0
for num in numbers:
    if num % 2 == 0:
        even_count += 1

print(f"Even numbers: {even_count}")  # Even numbers: 500

#Common Sequence Operations
#All sequences support these operations:
sequence = [10, 20, 30, 40, 50]

# 1. Length
print(len(sequence))           # 5

# 2. Membership test
print(30 in sequence)          # True
print(60 in sequence)          # False

# 3. Minimum and Maximum
print(min(sequence))           # 10
print(max(sequence))           # 50

# 4. Index (find position)
print(sequence.index(30))      # 2

# 5. Count occurrences
numbers = [1, 2, 2, 3, 2, 4]
print(numbers.count(2))        # 3

# 6. Concatenation
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2
print(combined)                # [1, 2, 3, 4, 5, 6]

# 7. Repetition
repeated = [1, 2] * 3
print(repeated)                # [1, 2, 1, 2, 1, 2]

# 8. Iteration
for item in sequence:
    print(item, end=" ")       # 10 20 30 40 50

#Real-World Examples
#Example 1: Todo List Application
# Task management using list
todo_list = []

# Add tasks
todo_list.append("Buy groceries")
todo_list.append("Call dentist")
todo_list.append("Finish homework")
todo_list.append("Pay bills")

print("TODO LIST:")
for i, task in enumerate(todo_list, 1):
    print(f"{i}. {task}")

# Mark task as done (remove)
completed_task = "Call dentist"
if completed_task in todo_list:
    todo_list.remove(completed_task)
    print(f"\nCompleted: {completed_task}")

print("\nREMAINING TASKS:")
for i, task in enumerate(todo_list, 1):
    print(f"{i}. {task}")



#Example 2: Grade Management System
# Student grades tracking
student_grades = {
    "Math": [85, 90, 78, 92, 88],
    "English": [75, 82, 88, 79, 85],
    "Science": [92, 95, 89, 91, 93]
}

print("GRADE REPORT\n")

for subject, grades in student_grades.items():
    average = sum(grades) / len(grades)
    highest = max(grades)
    lowest = min(grades)
    
    print(f"{subject}:")
    print(f"  Grades: {grades}")
    print(f"  Average: {average:.2f}")
    print(f"  Highest: {highest}")
    print(f"  Lowest: {lowest}")
    print()


#Example 3: Temperature Monitoring
# Weekly temperature tracking
weekly_temps = [28, 32, 30, 29, 31, 27, 26]  # Celsius
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

print("🌡️  WEEKLY TEMPERATURE REPORT\n")

for day, temp in zip(days, weekly_temps):
    if temp > 30:
        status = "Hot"
    elif temp > 25:
        status = "Warm"
    else:
        status = "Cool"
    
    print(f"{day}: {temp}°C {status}")

# Statistics
avg_temp = sum(weekly_temps) / len(weekly_temps)
max_temp = max(weekly_temps)
min_temp = min(weekly_temps)
max_day = days[weekly_temps.index(max_temp)]
min_day = days[weekly_temps.index(min_temp)]

print(f"\nSTATISTICS:")
print(f"Average Temperature: {avg_temp:.1f}°C")
print(f"Hottest Day: {max_day} ({max_temp}°C)")
print(f"Coolest Day: {min_day} ({min_temp}°C)")



#Key Advantages of Sequences
# Advantage         Description                              Example Use Case 
# Ordered           Elements maintain insertion order        Task lists, steps in process
# Indexed           Access elements by position              Get first/last item
# Iterable          Loop through elements easily             Process all students
# Flexible Size     Can grow/shrink (for lists)              Shopping cart
# Versatile         Store any data type                      Mixed data structures
# Memory Efficient  Store multiple values in one variable    Large datasets