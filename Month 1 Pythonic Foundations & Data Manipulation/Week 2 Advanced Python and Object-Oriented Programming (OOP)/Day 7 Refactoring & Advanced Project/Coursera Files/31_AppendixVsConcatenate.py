
#Difference Between Append and Concatenate in Python
#Key Differences
# Feature              append()                                   Concatenation (+)
# Modifies original?   Yes (in-place)                             No (creates new list)
# What it adds         Adds single element (even if it's a list)  Combines/merges lists element by element
# Return value         None                                       New list
# Syntax               list.append(item)                          list1 + list2 or list += other
# Use case             Adding one item at a time                  Merging multiple lists

#APPEND - Three Examples

#Example 1: Append adds ONE item (even if it's a list)
# Append adds the entire item as a single element
my_list = [1, 2, 3]
print(f"Original: {my_list}")

# Append a single number
my_list.append(4)
print(f"After append(4): {my_list}")
# Output: [1, 2, 3, 4]

# Append a LIST - adds the whole list as ONE element!
my_list.append([5, 6])
print(f"After append([5, 6]): {my_list}")
# Output: [1, 2, 3, 4, [5, 6]] - Notice the nested list!

# Append a string - adds whole string as ONE element
my_list.append("hello")
print(f"After append('hello'): {my_list}")
# Output: [1, 2, 3, 4, [5, 6], 'hello']
#Example 2: Append modifies original list (in-place)
# Append changes the original list and returns None
fruits = ['apple', 'banana']
print(f"Original fruits: {fruits}")
print(f"ID of fruits: {id(fruits)}")

# Append returns None
result = fruits.append('orange')
print(f"Return value of append: {result}")  # Output: None

print(f"Fruits after append: {fruits}")     # Output: ['apple', 'banana', 'orange']
print(f"ID after append: {id(fruits)}")     # Same ID - modified in place!

# You can't chain append operations
# This WON'T work: fruits.append('grape').append('mango')
# Because append returns None

#Example 3: Building a list with append
# Common pattern: build a list incrementally
scores = []
print(f"Starting list: {scores}")

# Add items one by one
for i in range(1, 6):
    score = i * 10
    scores.append(score)
    print(f"Added {score}: {scores}")

# Final result
print(f"\nFinal scores: {scores}")
# Output: [10, 20, 30, 40, 50]

# Append dictionary to list
students = []
students.append({'name': 'Alice', 'grade': 90})
students.append({'name': 'Bob', 'grade': 85})
print(f"\nStudents: {students}")

#CONCATENATE - Three Examples
#Example 1: Concatenate combines lists element by element
# Concatenation merges lists together
list1 = [1, 2, 3]
list2 = [4, 5, 6]

print(f"List 1: {list1}")
print(f"List 2: {list2}")

# Using + operator - creates NEW list
result = list1 + list2
print(f"Concatenated (list1 + list2): {result}")
# Output: [1, 2, 3, 4, 5, 6] - All elements merged!

# Original lists unchanged
print(f"List 1 after concat: {list1}")  # Still [1, 2, 3]
print(f"List 2 after concat: {list2}")  # Still [4, 5, 6]

# Can concatenate multiple lists
list3 = [7, 8, 9]
combined = list1 + list2 + list3
print(f"Three lists combined: {combined}")
# Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]
#Example 2: Concatenate creates NEW list (doesn't modify original)
# Concatenation always creates a new list
colors1 = ['red', 'blue']
colors2 = ['green', 'yellow']

print(f"Colors1: {colors1}")
print(f"ID of colors1: {id(colors1)}")

# Create new list with +
all_colors = colors1 + colors2
print(f"All colors: {all_colors}")
print(f"ID of all_colors: {id(all_colors)}")  # Different ID - new list!

# Originals unchanged
print(f"Colors1 still: {colors1}")  # Output: ['red', 'blue']
print(f"Colors2 still: {colors2}")  # Output: ['green', 'yellow']

# Using += (augmented assignment) - modifies original
print(f"\nBefore +=, colors1 ID: {id(colors1)}")
colors1 += colors2
print(f"After colors1 += colors2: {colors1}")
print(f"After +=, colors1 ID: {id(colors1)}")  # Same ID - modified in place!
#Example 3: Practical concatenation scenarios
# Scenario: Combining data from different sources
morning_sales = [100, 150, 120]
afternoon_sales = [200, 180, 210]
evening_sales = [90, 110, 95]

# Combine all sales data
total_sales = morning_sales + afternoon_sales + evening_sales
print(f"Total sales: {total_sales}")
# Output: [100, 150, 120, 200, 180, 210, 90, 110, 95]

# Calculate total
print(f"Sum of all sales: ${sum(total_sales)}")

# Can concatenate with empty list
extra = []
all_data = total_sales + extra
print(f"With empty list: {all_data}")  # Same as total_sales

# Concatenate strings in list
first_name = ['John']
last_name = ['Doe']
full_name = first_name + last_name
print(f"Full name list: {full_name}")  # Output: ['John', 'Doe']

#Side-by-Side Comparison
#Comparison 1: Adding a list to another list
print("=== APPEND vs CONCATENATE ===\n")

# Using APPEND
list_a = [1, 2, 3]
list_a.append([4, 5])
print(f"With append: {list_a}")
# Output: [1, 2, 3, [4, 5]] - List nested inside!
print(f"Length: {len(list_a)}")  # Length is 4 (4 elements)

# Using CONCATENATE
list_b = [1, 2, 3]
list_b = list_b + [4, 5]
print(f"With concatenate: {list_b}")
# Output: [1, 2, 3, 4, 5] - All elements at same level!
print(f"Length: {len(list_b)}")  # Length is 5 (5 elements)

#Comparison 2: Performance and memory
import time

# APPEND - Efficient for building lists
start = time.time()
append_list = []
for i in range(10000):
    append_list.append(i)
append_time = time.time() - start
print(f"Append time: {append_time:.6f} seconds")

# CONCATENATE - Less efficient (creates new list each time)
start = time.time()
concat_list = []
for i in range(10000):
    concat_list = concat_list + [i]  # Creates new list every iteration!
concat_time = time.time() - start
print(f"Concatenate time: {concat_time:.6f} seconds")

print(f"Concatenate is {concat_time/append_time:.2f}x slower")

# BETTER: Use += for concatenation in loops
start = time.time()
better_list = []
for i in range(10000):
    better_list += [i]  # More efficient than +
better_time = time.time() - start
print(f"Concatenate with += time: {better_time:.6f} seconds")

#Comparison 3: When to use which?
print("\n=== USE CASES ===\n")

# Use APPEND when: Adding ONE item at a time
shopping_cart = []
shopping_cart.append('milk')
shopping_cart.append('bread')
shopping_cart.append('eggs')
print(f"Cart with append: {shopping_cart}")

# Use CONCATENATE when: Merging multiple lists
breakfast = ['eggs', 'toast']
lunch = ['sandwich', 'salad']
dinner = ['pasta', 'chicken']
full_menu = breakfast + lunch + dinner
print(f"Menu with concatenate: {full_menu}")

# Use EXTEND when: Adding multiple items from another list
favorites = ['pizza']
more_favorites = ['burger', 'tacos', 'sushi']
favorites.extend(more_favorites)  # Better than concatenate for this!
print(f"Favorites with extend: {favorites}")

#Important Notes
#Why append() adds lists as single elements:
# This is BY DESIGN - append adds whatever you give it as ONE item
my_list = [1, 2, 3]

# Appending different types
my_list.append(4)           # Adds integer
my_list.append([5, 6])      # Adds list (as one element)
my_list.append((7, 8))      # Adds tuple (as one element)
my_list.append({'key': 9})  # Adds dict (as one element)

print(my_list)
# Output: [1, 2, 3, 4, [5, 6], (7, 8), {'key': 9}]
print(f"Length: {len(my_list)}")  # Length is 7 (7 elements)

#Alternative: extend() method
# If you want to add multiple elements individually, use extend()
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# Using extend (modifies in place, like append)
list1.extend(list2)
print(f"After extend: {list1}")
# Output: [1, 2, 3, 4, 5, 6] - Same result as concatenation!

# extend() vs append() vs concatenate()
a = [1, 2]
b = [1, 2]
c = [1, 2]

a.append([3, 4])      # Result: [1, 2, [3, 4]]
b.extend([3, 4])      # Result: [1, 2, 3, 4]
c = c + [3, 4]        # Result: [1, 2, 3, 4]

print(f"append: {a}")
print(f"extend: {b}")
print(f"concat: {c}")

#Quick Reference
# APPEND - adds ONE item
my_list = [1, 2, 3]
my_list.append(4)        # [1, 2, 3, 4]
my_list.append([5, 6])   # [1, 2, 3, 4, [5, 6]] - nested!

# CONCATENATE - merges lists
list1 = [1, 2, 3]
list2 = [4, 5, 6]
result = list1 + list2   # [1, 2, 3, 4, 5, 6] - flat!

# EXTEND - adds multiple items (like concatenate but in-place)
list1 = [1, 2, 3]
list1.extend([4, 5, 6])  # [1, 2, 3, 4, 5, 6] - flat!

