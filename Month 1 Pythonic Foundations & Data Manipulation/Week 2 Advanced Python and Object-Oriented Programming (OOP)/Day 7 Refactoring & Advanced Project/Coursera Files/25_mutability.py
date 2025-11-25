
#Mutability in Python
#Definition of Mutability
#Mutability refers to whether an object's value can be changed after it is created.

#Mutable objects can be modified after creation (their content can change)
#Immutable objects cannot be modified after creation (their content is fixed)

#Simple explanation:
#Mutable = Can be edited/changed (like a whiteboard - you can erase and rewrite)
#Immutable = Cannot be changed (like ink on paper - once written, it's permanent)

#
#Mutable vs Immutable Types in Python
#Mutable Types (Can be changed):

#Lists - [1, 2, 3]
#Dictionaries - {"name": "Ali", "age": 20}
#Sets - {1, 2, 3}
#Bytearrays - bytearray(b"hello")
#User-defined classes (by default)

#Immutable Types (Cannot be changed):

#Integers - 42
#Floats - 3.14
#Strings - "Hello"
#Tuples - (1, 2, 3)
#Frozen Sets - frozenset({1, 2, 3})
#Booleans - True, False
#Bytes - b"hello"
#None - None


##Why We Need Mutability
#Benefits of Mutable Objects:
#1. Efficiency - Modify in Place
#No need to create new objects when making changes.
# Mutable list - efficient modification
shopping_cart = ["Apple", "Banana"]
print(f"Before: {shopping_cart}")
print(f"Memory ID: {id(shopping_cart)}")

shopping_cart.append("Orange")  # Modifies same object
print(f"After: {shopping_cart}")
print(f"Memory ID: {id(shopping_cart)}")  # Same ID - same object

# Output:
# Before: ['Apple', 'Banana']
# Memory ID: 140234567891234
# After: ['Apple', 'Banana', 'Orange']
# Memory ID: 140234567891234  (Same object modified)

#2. Dynamic Data Structures
#Build data structures that grow and shrink.
# Building a dynamic list of tasks
tasks = []
tasks.append("Task 1")
tasks.append("Task 2")
tasks.append("Task 3")
tasks.remove("Task 2")
print(tasks)  # ['Task 1', 'Task 3']

#3. Shared State
#Multiple references can see the same changes.
# Shared shopping cart
cart = ["Milk", "Bread"]
customer_view = cart
admin_view = cart

customer_view.append("Eggs")
print(f"Customer sees: {customer_view}")  # ['Milk', 'Bread', 'Eggs']
print(f"Admin sees: {admin_view}")        # ['Milk', 'Bread', 'Eggs']
# Both see the same changes!

#Benefits of Immutable Objects:
#1. Safety - Prevent Accidental Changes
#Data can't be modified accidentally.
# Immutable tuple - safe configuration
database_config = ("localhost", 5432, "mydb")
# database_config[0] = "newhost"  # ❌ TypeError: Cannot modify tuple

print(database_config)  # ('localhost', 5432, 'mydb')

#2. Hashable - Can be Dictionary Keys
#Immutable objects can be used as dictionary keys or in sets.
# Using tuples as dictionary keys (immutable)
locations = {
    (40.7128, -74.0060): "New York",
    (51.5074, -0.1278): "London",
    (35.6762, 139.6503): "Tokyo"
}

print(locations[(40.7128, -74.0060)])  # New York

# Lists cannot be keys (mutable)
# locations[[40.7128, -74.0060]] = "NYC"  # ❌ TypeError: unhashable type
#3. Thread Safety
#Immutable objects are inherently thread-safe.
# Immutable string - safe to share between threads
username = "JohnDoe"
# No thread can modify it, they can only create new strings

#4. Predictability
#alues won't change unexpectedly.
# Immutable - predictable behavior
name = "Ahmed"
original_name = name
name = name.upper()  # Creates NEW string

print(f"Original: {original_name}")  # Ahmed
print(f"Modified: {name}")           # AHMED
# Original remains unchanged

#Understanding Mutability Better
#Concept 1: Memory and Identity
# MUTABLE - Modification keeps same identity
mutable_list = [1, 2, 3]
print(f"Original ID: {id(mutable_list)}")
mutable_list.append(4)
print(f"After append ID: {id(mutable_list)}")  # SAME ID
print(mutable_list)  # [1, 2, 3, 4]

print("\n" + "="*50 + "\n")

# IMMUTABLE - "Modification" creates new object
immutable_string = "hello"
print(f"Original ID: {id(immutable_string)}")
immutable_string = immutable_string + " world"
print(f"After concat ID: {id(immutable_string)}")  # DIFFERENT ID
print(immutable_string)  # hello world

#Concept 2: Reference vs Copy
# MUTABLE - References share the same object
original_list = [1, 2, 3]
reference = original_list  # Same object
copy_list = original_list.copy()  # New object

original_list.append(4)

print(f"Original: {original_list}")    # [1, 2, 3, 4]
print(f"Reference: {reference}")       # [1, 2, 3, 4] - Changed!
print(f"Copy: {copy_list}")            # [1, 2, 3] - Unchanged

print("\n" + "="*50 + "\n")

# IMMUTABLE - Assignment always creates new binding
original_string = "Hello"
reference_string = original_string
original_string = original_string + " World"

print(f"Original: {original_string}")      # Hello World
print(f"Reference: {reference_string}")    # Hello - Unchanged
# Strings are immutable, so "modification" creates new object

#5 Examples of Mutable Objects
#Example 1: Student Grade Management (List)
print("="*70)
print("EXAMPLE 1: MUTABLE LIST - STUDENT GRADES")
print("="*70)

# Mutable list allows dynamic grade management
student_grades = [85, 90, 78]
print(f"Initial grades: {student_grades}")
print(f"Memory address: {id(student_grades)}")

# Add new grade
student_grades.append(92)
print(f"\nAfter adding grade: {student_grades}")
print(f"Memory address: {id(student_grades)}")  # SAME address

# Update existing grade
student_grades[1] = 95  # Change second grade
print(f"\nAfter updating: {student_grades}")
print(f"Memory address: {id(student_grades)}")  # STILL same address

# Remove lowest grade
lowest = min(student_grades)
student_grades.remove(lowest)
print(f"\nAfter removing lowest: {student_grades}")
print(f"Memory address: {id(student_grades)}")  # STILL same address

# Sort grades
student_grades.sort(reverse=True)
print(f"\nSorted (high to low): {student_grades}")
print(f"Memory address: {id(student_grades)}")  # STILL same address

# Calculate statistics
average = sum(student_grades) / len(student_grades)
print(f"\nAverage Grade: {average:.2f}")


#Example 2: User Profile Management (Dictionary)
print("="*70)
print("EXAMPLE 2: MUTABLE DICTIONARY - USER PROFILE")
print("="*70)

# Mutable dictionary for dynamic profile updates
user_profile = {
    "username": "john_doe",
    "email": "john@example.com",
    "age": 25
}

print(f"Initial profile: {user_profile}")
print(f"Memory address: {id(user_profile)}")

# Add new fields
user_profile["city"] = "Jhelum"
user_profile["country"] = "Pakistan"
print(f"\nAfter adding location: {user_profile}")
print(f"Memory address: {id(user_profile)}")  # SAME

# Update existing fields
user_profile["age"] = 26
user_profile["email"] = "john.doe@newmail.com"
print(f"\nAfter updating: {user_profile}")
print(f"Memory address: {id(user_profile)}")  # SAME

# Add preferences
user_profile["preferences"] = {
    "theme": "dark",
    "notifications": True
}
print(f"\nWith preferences: {user_profile}")
print(f"Memory address: {id(user_profile)}")  # SAME

# Remove a field
del user_profile["age"]
print(f"\nAfter removing age: {user_profile}")
print(f"Memory address: {id(user_profile)}")  # SAME

print("\nSame object throughout all modifications!")


#Example 3: Unique Items Collection (Set)
print("="*70)
print("EXAMPLE 3: MUTABLE SET - UNIQUE VISITORS")
print("="*70)

# Mutable set for tracking unique visitors
website_visitors = {"user1", "user2", "user3"}
print(f"Initial visitors: {website_visitors}")
print(f"Count: {len(website_visitors)}")
print(f"Memory address: {id(website_visitors)}")

# Add new visitors (duplicates automatically ignored)
website_visitors.add("user4")
website_visitors.add("user2")  # Duplicate - won't be added
print(f"\nAfter adding visitors: {website_visitors}")
print(f"Count: {len(website_visitors)}")
print(f"Memory address: {id(website_visitors)}")  # SAME

# Add multiple visitors
new_visitors = {"user5", "user6", "user1"}  # user1 is duplicate
website_visitors.update(new_visitors)
print(f"\nAfter bulk add: {website_visitors}")
print(f"Count: {len(website_visitors)}")
print(f"Memory address: {id(website_visitors)}")  # SAME

# Remove visitor
website_visitors.remove("user3")
print(f"\nAfter removing user3: {website_visitors}")
print(f"Memory address: {id(website_visitors)}")  # SAME

# Check membership
if "user4" in website_visitors:
    print("\nuser4 is a visitor")

# Set operations
premium_users = {"user2", "user5", "user7"}
common_users = website_visitors.intersection(premium_users)
print(f"\nVisitors who are premium: {common_users}")

print(f"\nFinal memory address: {id(website_visitors)}")  # STILL SAME

#Example 4: Shopping Cart with Shared References
print("="*70)
print("EXAMPLE 4: MUTABLE OBJECTS - SHARED REFERENCES")
print("="*70)

# Demonstrate how mutable objects are shared
cart = ["Laptop", "Mouse"]
print(f"Original cart: {cart}")
print(f"Cart memory address: {id(cart)}")

# Multiple references to same mutable object
customer_view = cart  # Reference, not copy
admin_view = cart     # Another reference
backup_cart = cart.copy()  # Actual copy

print(f"\nCustomer view address: {id(customer_view)}")  # SAME as cart
print(f"Admin view address: {id(admin_view)}")          # SAME as cart
print(f"Backup cart address: {id(backup_cart)}")        # DIFFERENT

# Customer adds item
customer_view.append("Keyboard")
print(f"\nAfter customer adds Keyboard:")
print(f"  Cart: {cart}")                # Changed!
print(f"  Customer view: {customer_view}")  # Changed!
print(f"  Admin view: {admin_view}")        # Changed!
print(f"  Backup: {backup_cart}")           # Unchanged!

# Admin removes item
admin_view.remove("Mouse")
print(f"\nAfter admin removes Mouse:")
print(f"  Cart: {cart}")                # Changed!
print(f"  Customer view: {customer_view}")  # Changed!
print(f"  Admin view: {admin_view}")        # Changed!
print(f"  Backup: {backup_cart}")           # Unchanged!

print("\nAll references (cart, customer_view, admin_view) point to SAME object")
print("Backup is a separate copy, so it remains unchanged")


#Example 5: Nested Mutable Structures
print("="*70)
print("EXAMPLE 5: NESTED MUTABLE STRUCTURES")
print("="*70)

# Complex mutable structure - list of dictionaries
students = [
    {"name": "Ali", "grades": [85, 90, 78], "passed": False},
    {"name": "Sara", "grades": [92, 88, 95], "passed": False},
    {"name": "Ahmed", "grades": [70, 75, 72], "passed": False}
]

print("Initial student data:")
for student in students:
    print(f"  {student}")

print(f"\nMemory address of students list: {id(students)}")
print(f"Memory address of first student dict: {id(students[0])}")
print(f"Memory address of first student's grades: {id(students[0]['grades'])}")

# Modify nested structures
print("\n" + "-"*70)
print("MODIFICATIONS:")
print("-"*70)

# Add grade to Ali
students[0]["grades"].append(88)
print(f"Added grade to Ali: {students[0]}")

# Calculate average and update passed status for all
for student in students:
    average = sum(student["grades"]) / len(student["grades"])
    student["passed"] = average >= 75
    student["average"] = round(average, 2)

print("\nAfter calculating averages:")
for student in students:
    status = "PASSED" if student["passed"] else "❌ FAILED"
    print(f"  {student['name']}: {student['average']} - {status}")
    print(f"    Grades: {student['grades']}")

# Add new student
students.append({"name": "Fatima", "grades": [95, 98, 92], "passed": True, "average": 95.0})
print(f"\nAfter adding Fatima: {len(students)} students total")

# Verify memory addresses haven't changed
print(f"\nMemory address of students list: {id(students)}")  # SAME
print(f"Memory address of first student dict: {id(students[0])}")  # SAME
print(f"Memory address of first student's grades: {id(students[0]['grades'])}")  # SAME

print("\nAll nested structures modified in place - same memory addresses!")


##5 Examples of Immutable Objects
#Example 1: String Immutability
print("="*70)
print("EXAMPLE 1: IMMUTABLE STRING")
print("="*70)

# Strings cannot be modified in place
message = "Hello"
print(f"Original: {message}")
print(f"Memory address: {id(message)}")

# Attempting to "modify" creates NEW string
message_upper = message.upper()
print(f"\nUppercase: {message_upper}")
print(f"Memory address: {id(message_upper)}")  # DIFFERENT address

# Original unchanged
print(f"\nOriginal still: {message}")
print(f"Original address: {id(message)}")  # SAME as before

# Concatenation creates new string
greeting = message + " World"
print(f"\nConcatenated: {greeting}")
print(f"Memory address: {id(greeting)}")  # DIFFERENT address

# Replace creates new string
new_message = message.replace("H", "J")
print(f"\nReplaced: {new_message}")
print(f"Memory address: {id(new_message)}")  # DIFFERENT address

# Slicing creates new string
substring = message[1:4]
print(f"\nSubstring: {substring}")
print(f"Memory address: {id(substring)}")  # DIFFERENT address

print("\n Every operation created a NEW string object!")
print(" Original string never modified!")



#Example 2: Tuple as Safe Configuration
print("="*70)
print("EXAMPLE 2: IMMUTABLE TUPLE - SAFE CONFIGURATION")
print("="*70)

# Tuples protect critical configuration
database_config = ("localhost", 5432, "production_db", "admin")
print(f"Database config: {database_config}")
print(f"Memory address: {id(database_config)}")

# Access individual values
host = database_config[0]
port = database_config[1]
db_name = database_config[2]
user = database_config[3]

print(f"\nHost: {host}")
print(f"Port: {port}")
print(f"Database: {db_name}")
print(f"User: {user}")

# Try to modify (will cause error)
print("\nAttempting to modify tuple:")
try:
    database_config[1] = 3306  # Try to change port
except TypeError as e:
    print(f"Error: {e}")
    print("Configuration protected from accidental changes!")

# "Modifying" requires creating new tuple
new_config = (database_config[0], 3306, database_config[2], database_config[3])
print(f"\nNew config: {new_config}")
print(f"Memory address: {id(new_config)}")  # DIFFERENT address

print(f"\nOriginal config unchanged: {database_config}")
print(f"Original address: {id(database_config)}")  # SAME as before

# Tuples can be dictionary keys (because immutable)
configs = {
    ("localhost", 5432): "Development",
    ("192.168.1.100", 5432): "Production",
    ("192.168.1.101", 5433): "Backup"
}

print(f"\nConfigurations:")
for config_key, environment in configs.items():
    print(f"  {config_key}: {environment}")


#Example 3: Integer Immutability
print("="*70)
print("EXAMPLE 3: IMMUTABLE INTEGER")
print("="*70)

# Numbers are immutable
balance = 1000
print(f"Initial balance: {balance}")
print(f"Memory address: {id(balance)}")

# "Modifying" creates new integer object
balance = balance + 500
print(f"\nAfter adding 500: {balance}")
print(f"Memory address: {id(balance)}")  # DIFFERENT

# Another operation
balance = balance - 200
print(f"\nAfter subtracting 200: {balance}")
print(f"Memory address: {id(balance)}")  # DIFFERENT again

# Integer references
x = 100
y = x  # y refers to same object
print(f"\nx = {x}, Memory: {id(x)}")
print(f"y = {y}, Memory: {id(y)}")  # SAME as x

x = x + 1  # Creates NEW object
print(f"\nAfter x = x + 1:")
print(f"x = {x}, Memory: {id(x)}")  # DIFFERENT
print(f"y = {y}, Memory: {id(y)}")  # UNCHANGED

print("\ny still refers to original 100 object")
print("x now refers to new 101 object")

# Small integer caching (Python optimization)
a = 5
b = 5
print(f"\na = {a}, Memory: {id(a)}")
print(f"b = {b}, Memory: {id(b)}")  # SAME! (cached by Python)
print("Python caches small integers (-5 to 256)")

