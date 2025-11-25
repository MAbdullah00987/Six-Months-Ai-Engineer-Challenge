

#Purpose of Aliasing in Python
#What is Aliasing?
#Aliasing occurs when two or more variable names (references) point to the same object in memory. When you create an alias, you're not copying the object - you're creating another name that refers to the exact same object.
#Why Do We Need Aliasing?
#Benefits:

#Memory efficiency - Share large data structures without duplicating them
#Performance - Avoid expensive copying operations
#Data synchronization - Multiple parts of code can work with the same data
#Simplified code - Use shorter or more descriptive names for the same object
#Function parameters - Pass objects to functions without copying

#Potential Issues (Why You Need to Understand It):

#Unintended modifications - Changing data through one alias affects all aliases
#Debugging difficulty - Hard to track where changes originated
#Side effects - Functions might modify original data unexpectedly

#
#Three Examples of Aliasing
#Example 1: List Aliasing (Most Common Case)
# Original list
shopping_cart = ['apple', 'banana', 'orange']

# Creating an alias - both names point to the SAME list
cart_backup = shopping_cart

print(f"Original cart: {shopping_cart}")
print(f"Backup cart: {cart_backup}")

# Modify through the alias
cart_backup.append('grapes')

# BOTH variables show the change!
print(f"\nAfter modification:")
print(f"Original cart: {shopping_cart}")  # Output: ['apple', 'banana', 'orange', 'grapes']
print(f"Backup cart: {cart_backup}")      # Output: ['apple', 'banana', 'orange', 'grapes']

# Verify they're the same object
print(f"\nAre they the same object? {shopping_cart is cart_backup}")  # Output: True
print(f"Original ID: {id(shopping_cart)}")
print(f"Backup ID: {id(cart_backup)}")    # Same memory address

#Example 2: Dictionary Aliasing
# Original dictionary
student_grades = {
    'Alice': 85,
    'Bob': 90,
    'Charlie': 78
}

# Create alias with a shorter name
grades = student_grades

print(f"Original: {student_grades}")
print(f"Alias: {grades}")

# Modify through the alias
grades['Alice'] = 95
grades['David'] = 88

# Both references show the changes
print(f"\nAfter modification:")
print(f"Original: {student_grades}")  
# Output: {'Alice': 95, 'Bob': 90, 'Charlie': 78, 'David': 88}
print(f"Alias: {grades}")
# Output: {'Alice': 95, 'Bob': 90, 'Charlie': 78, 'David': 88}

# Check if they're the same object
print(f"\nSame object? {student_grades is grades}")  # Output: True
#Example 3: Function Parameter Aliasing
def update_employee_salary(employee_dict, employee_name, new_salary):
    """
    employee_dict is an ALIAS to the original dictionary
    Changes made here affect the original!
    """
    employee_dict[employee_name] = new_salary
    print(f"Inside function: {employee_dict}")

# Original data
company_employees = {
    'John': 50000,
    'Sarah': 60000,
    'Mike': 55000
}

print(f"Before function call: {company_employees}")

# Pass the dictionary to function
# employee_dict parameter becomes an alias to company_employees
update_employee_salary(company_employees, 'John', 75000)

# Original dictionary is modified!
print(f"After function call: {company_employees}")
# Output: {'John': 75000, 'Sarah': 60000, 'Mike': 55000}

#Three Examples of Avoiding Aliasing (Creating Copies)
#Example 1: Shallow Copy for Lists
import copy

# Original list
original = [1, 2, 3, 4, 5]

# Creating a COPY (not an alias)
copied = original.copy()  # or original[:] or list(original)

print(f"Original: {original}")
print(f"Copied: {copied}")

# Modify the copy
copied.append(6)
copied[0] = 100

# Only the copy changes, original remains unchanged
print(f"\nAfter modification:")
print(f"Original: {original}")  # Output: [1, 2, 3, 4, 5] - UNCHANGED
print(f"Copied: {copied}")      # Output: [100, 2, 3, 4, 5, 6] - CHANGED

# They are DIFFERENT objects
print(f"\nAre they the same object? {original is copied}")  # Output: False
print(f"Original ID: {id(original)}")
print(f"Copied ID: {id(copied)}")  # Different memory address

#Example 2: Deep Copy for Nested Structures
import copy

# Nested list (list within a list)
original_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Shallow copy - ONLY copies outer list
shallow = original_matrix.copy()

# Deep copy - copies EVERYTHING including nested lists
deep = copy.deepcopy(original_matrix)

print(f"Original: {original_matrix}")
print(f"Shallow: {shallow}")
print(f"Deep: {deep}")

# Modify nested element through shallow copy
shallow[0][0] = 999

print(f"\nAfter shallow copy modification:")
print(f"Original: {original_matrix}")  # Output: [[999, 2, 3], ...] - CHANGED!
print(f"Shallow: {shallow}")            # Output: [[999, 2, 3], ...] - CHANGED!
print(f"Deep: {deep}")                  # Output: [[1, 2, 3], ...] - UNCHANGED

# Why? Inner lists are still ALIASED in shallow copy!
print(f"\nInner lists same? {original_matrix[0] is shallow[0]}")  # True
print(f"Inner lists same? {original_matrix[0] is deep[0]}")       # False

#Example 3: Dictionary Copy to Avoid Aliasing
# Original configuration
default_config = {
    'theme': 'dark',
    'language': 'en',
    'notifications': True
}

# Create a copy for user settings
user_config = default_config.copy()

print(f"Default: {default_config}")
print(f"User: {user_config}")

# Modify user config
user_config['theme'] = 'light'
user_config['font_size'] = 14

print(f"\nAfter user changes:")
print(f"Default: {default_config}")  
# Output: {'theme': 'dark', 'language': 'en', 'notifications': True} - UNCHANGED
print(f"User: {user_config}")
# Output: {'theme': 'light', 'language': 'en', 'notifications': True, 'font_size': 14}

# Verify they're different objects
print(f"\nSame object? {default_config is user_config}")  # Output: False

#Practical Comparison: Aliasing vs Copying
# Example showing the difference
print("=== ALIASING ===")
list1 = [10, 20, 30]
list2 = list1           # Alias - points to same object
list2.append(40)
print(f"list1: {list1}")  # Output: [10, 20, 30, 40]
print(f"list2: {list2}")  # Output: [10, 20, 30, 40]
print(f"Same? {list1 is list2}")  # True

print("\n=== COPYING ===")
list3 = [10, 20, 30]
list4 = list3.copy()    # Copy - creates new object
list4.append(40)
print(f"list3: {list3}")  # Output: [10, 20, 30]
print(f"list4: {list4}")  # Output: [10, 20, 30, 40]
print(f"Same? {list3 is list4}")  # False

