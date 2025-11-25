

#List Methods in Python
#Python lists have many built-in methods for manipulating data. Here's a comprehensive guide with three examples for each method.

#1. append() - Add single element to the end
#Purpose: Adds one item to the end of the list. Modifies the original list.
#Example 1: Building a to-do list
tasks = ['Buy groceries', 'Clean room']
print(f"Initial tasks: {tasks}")

# Add new tasks one by one
tasks.append('Pay bills')
tasks.append('Call mom')
tasks.append('Exercise')

print(f"Updated tasks: {tasks}")
# Output: ['Buy groceries', 'Clean room', 'Pay bills', 'Call mom', 'Exercise']
#Example 2: Collecting user input
scores = []

# Simulate collecting scores
test_scores = [85, 90, 78, 92, 88]

for score in test_scores:
    if score >= 80:
        scores.append(score)
        
print(f"Scores above 80: {scores}")
# Output: [85, 90, 92, 88]

#Example 3: Building dynamic data
fibonacci = [0, 1]

# Generate fibonacci sequence
for i in range(8):
    next_num = fibonacci[-1] + fibonacci[-2]
    fibonacci.append(next_num)

print(f"Fibonacci sequence: {fibonacci}")
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

#2. extend() - Add multiple elements from iterable
#Purpose: Adds all elements from another iterable (list, tuple, string, etc.) to the end.
#Example 1: Combining lists
fruits = ['apple', 'banana']
more_fruits = ['orange', 'grape', 'mango']

print(f"Original: {fruits}")

fruits.extend(more_fruits)
print(f"After extend: {fruits}")
# Output: ['apple', 'banana', 'orange', 'grape', 'mango']

# Compare with append (adds the whole list as one element)
fruits2 = ['apple', 'banana']
fruits2.append(more_fruits)
print(f"With append: {fruits2}")
# Output: ['apple', 'banana', ['orange', 'grape', 'mango']]

#Example 2: Extending with different iterables
numbers = [1, 2, 3]

# Extend with tuple
numbers.extend((4, 5, 6))
print(f"After tuple: {numbers}")

# Extend with range
numbers.extend(range(7, 10))
print(f"After range: {numbers}")

# Extend with string (adds each character)
letters = ['a', 'b']
letters.extend('cde')
print(f"Letters: {letters}")
# Output: ['a', 'b', 'c', 'd', 'e']

#Example 3: Building a playlist
my_playlist = ['Song A', 'Song B']
rock_songs = ['Rock Song 1', 'Rock Song 2']
pop_songs = ['Pop Song 1', 'Pop Song 2', 'Pop Song 3']

print(f"Initial playlist: {my_playlist}")

# Add multiple songs at once
my_playlist.extend(rock_songs)
print(f"After rock: {my_playlist}")

my_playlist.extend(pop_songs)
print(f"Final playlist: {my_playlist}")
# Output: ['Song A', 'Song B', 'Rock Song 1', 'Rock Song 2', 'Pop Song 1', 'Pop Song 2', 'Pop Song 3']

#3. insert() - Add element at specific position
#Purpose: Inserts an element at a specified index. Syntax: list.insert(index, element)
#Example 1: Insert at beginning
students = ['Bob', 'Charlie', 'David']
print(f"Original: {students}")

# Insert at the beginning (index 0)
students.insert(0, 'Alice')
print(f"After insert: {students}")
# Output: ['Alice', 'Bob', 'Charlie', 'David']

#Example 2: Insert in the middle
priorities = ['Low', 'High']
print(f"Before: {priorities}")

# Insert 'Medium' at index 1
priorities.insert(1, 'Medium')
print(f"After: {priorities}")
# Output: ['Low', 'Medium', 'High']

# Insert another at specific position
priorities.insert(2, 'Medium-High')
print(f"Final: {priorities}")
# Output: ['Low', 'Medium', 'Medium-High', 'High']

#Example 3: Insert with boundary handling
queue = [10, 20, 30]
print(f"Original queue: {queue}")

# Insert at valid position
queue.insert(1, 15)
print(f"After insert at 1: {queue}")

# Insert beyond length (adds to end)
queue.insert(100, 40)
print(f"Insert at 100: {queue}")
# Output: [10, 15, 20, 30, 40]

# Insert at negative index
queue.insert(-1, 35)
print(f"Insert at -1: {queue}")
# Output: [10, 15, 20, 30, 35, 40] (before last element)

#4. remove() - Remove first occurrence of value
#Purpose: Removes the first occurrence of a specified value. Raises ValueError if not found.
#Example 1: Remove specific item
colors = ['red', 'blue', 'green', 'blue', 'yellow']
print(f"Original: {colors}")

# Remove first 'blue'
colors.remove('blue')
print(f"After removing blue: {colors}")
# Output: ['red', 'green', 'blue', 'yellow'] (only first removed)
#Example 2: Remove with error handling
inventory = ['laptop', 'mouse', 'keyboard', 'monitor']
print(f"Inventory: {inventory}")

# Try to remove item
item_to_remove = 'mouse'
try:
    inventory.remove(item_to_remove)
    print(f"Removed {item_to_remove}: {inventory}")
except ValueError:
    print(f"{item_to_remove} not found in inventory")

# Try to remove non-existent item
item_to_remove = 'printer'
try:
    inventory.remove(item_to_remove)
    print(f"Removed {item_to_remove}")
except ValueError:
    print(f"{item_to_remove} not found in inventory")
# Output: printer not found in inventory

#Example 3: Remove all occurrences
pythonscores = [85, 90, 85, 78, 85, 92]
print(f"Original scores: {scores}")

# Remove all occurrences of 85
score_to_remove = 85
while score_to_remove in scores:
    scores.remove(score_to_remove)

print(f"After removing all 85s: {scores}")
# Output: [90, 78, 92]

#5. pop() - Remove and return element
#Purpose: Removes and returns element at index (default is last). Syntax: list.pop(index=-1)
#Example 1: Pop from end (stack behavior)
stack = [1, 2, 3, 4, 5]
print(f"Stack: {stack}")

# Pop last element
last = stack.pop()
print(f"Popped: {last}")
print(f"Stack now: {stack}")
# Output: Popped: 5, Stack: [1, 2, 3, 4]

# Pop again
last = stack.pop()
print(f"Popped: {last}")
print(f"Stack now: {stack}")
# Output: Popped: 4, Stack: [1, 2, 3]

#Example 2: Pop from beginning (queue behavior)
queue = ['first', 'second', 'third', 'fourth']
print(f"Queue: {queue}")

# Pop from front (index 0)
served = queue.pop(0)
print(f"Serving: {served}")
print(f"Queue now: {queue}")
# Output: Serving: first, Queue: ['second', 'third', 'fourth']

served = queue.pop(0)
print(f"Serving: {served}")
print(f"Queue now: {queue}")
# Output: Serving: second, Queue: ['third', 'fourth']
#Example 3: Pop from specific position
cards = ['Ace', 'King', 'Queen', 'Jack', '10']
print(f"Cards: {cards}")

# Pop from middle
drawn = cards.pop(2)
print(f"Drew: {drawn}")
print(f"Remaining: {cards}")
# Output: Drew: Queen, Remaining: ['Ace', 'King', 'Jack', '10']

# Pop with error handling
try:
    card = cards.pop(10)  # Invalid index
except IndexError:
    print("No card at that position!")

#6. clear() - Remove all elements
#Purpose: Removes all elements from the list, making it empty.
#Example 1: Clear shopping cart
cart = ['item1', 'item2', 'item3', 'item4']
print(f"Cart before: {cart}")
print(f"Cart length: {len(cart)}")

# Clear all items
cart.clear()
print(f"Cart after clear: {cart}")
print(f"Cart length: {len(cart)}")
# Output: Cart after clear: [], Cart length: 0

#Example 2: Reset game state
player_inventory = ['sword', 'shield', 'potion', 'key']
enemy_list = ['goblin', 'orc', 'dragon']

print(f"Inventory: {player_inventory}")
print(f"Enemies: {enemy_list}")

# Game over - clear everything
player_inventory.clear()
enemy_list.clear()

print(f"\n--- GAME RESET ---")
print(f"Inventory: {player_inventory}")
print(f"Enemies: {enemy_list}")

#Example 3: Clear vs reassignment
list1 = [1, 2, 3]
list2 = list1  # Alias

print(f"list1: {list1}")
print(f"list2: {list2}")

# Using clear() - affects all references
list1.clear()
print(f"\nAfter clear():")
print(f"list1: {list1}")  # Output: []
print(f"list2: {list2}")  # Output: [] (also empty!)

list3 = [4, 5, 6]
list4 = list3

# Using reassignment - doesn't affect references
list3 = []
print(f"\nAfter reassignment:")
print(f"list3: {list3}")  # Output: []
print(f"list4: {list4}")  # Output: [4, 5, 6] (still has data!)

#7. index() - Find position of element
#Purpose: Returns the index of first occurrence. Syntax: list.index(value, start, end)
#Example 1: Find element position
fruits = ['apple', 'banana', 'orange', 'grape', 'banana']
print(f"Fruits: {fruits}")

# Find index of 'orange'
position = fruits.index('orange')
print(f"Orange is at index: {position}")
# Output: 2

# Find first 'banana'
position = fruits.index('banana')
print(f"First banana at index: {position}")
# Output: 1
#Example 2: Search with start position
numbers = [10, 20, 30, 20, 40, 20, 50]
print(f"Numbers: {numbers}")

# Find first 20
first = numbers.index(20)
print(f"First 20 at index: {first}")

# Find next 20 (start searching after first)
second = numbers.index(20, first + 1)
print(f"Second 20 at index: {second}")

# Find third 20
third = numbers.index(20, second + 1)
print(f"Third 20 at index: {third}")
# Output: First: 1, Second: 3, Third: 5

#Example 3: Error handling with index
students = ['Alice', 'Bob', 'Charlie', 'David']
print(f"Students: {students}")

# Safe search
def find_student(name):
    try:
        position = students.index(name)
        print(f"{name} found at position {position}")
        return position
    except ValueError:
        print(f"{name} not found in list")
        return -1

find_student('Charlie')  # Output: Charlie found at position 2
find_student('Eve')      # Output: Eve not found in list

#8. count() - Count occurrences of element
#Purpose: Returns the number of times a value appears in the list.
#Example 1: Count specific values
grades = ['A', 'B', 'A', 'C', 'A', 'B', 'A', 'D']
print(f"Grades: {grades}")

# Count each grade
print(f"A's: {grades.count('A')}")  # Output: 4
print(f"B's: {grades.count('B')}")  # Output: 2
print(f"C's: {grades.count('C')}")  # Output: 1
print(f"F's: {grades.count('F')}")  # Output: 0 (not found)
#Example 2: Find most common element
votes = ['Alice', 'Bob', 'Alice', 'Charlie', 'Alice', 'Bob', 'Alice']
print(f"Votes: {votes}")

# Count votes for each candidate
candidates = list(set(votes))  # Get unique candidates
results = {}

for candidate in candidates:
    results[candidate] = votes.count(candidate)

print(f"\nVote counts:")
for candidate, count in results.items():
    print(f"{candidate}: {count} votes")

winner = max(results, key=results.get)
print(f"\nWinner: {winner}")
#Example 3: Data analysis
survey_responses = [1, 2, 3, 1, 2, 1, 4, 5, 1, 2, 3, 1]
print(f"Survey responses: {survey_responses}")

# Analyze response distribution
print("\nResponse Distribution:")
for rating in range(1, 6):
    count = survey_responses.count(rating)
    percentage = (count / len(survey_responses)) * 100
    print(f"Rating {rating}: {count} responses ({percentage:.1f}%)")

#9. sort() - Sort list in place
#Purpose: Sorts the list in ascending order (modifies original). Syntax: list.sort(reverse=False, key=None)
#Example 1: Basic sorting
numbers = [45, 12, 78, 23, 90, 34]
print(f"Original: {numbers}")

# Sort ascending
numbers.sort()
print(f"Ascending: {numbers}")
# Output: [12, 23, 34, 45, 78, 90]

# Sort descending
numbers.sort(reverse=True)
print(f"Descending: {numbers}")
# Output: [90, 78, 45, 34, 23, 12]
#Example 2: Sort strings
names = ['charlie', 'Alice', 'bob', 'David']
print(f"Original: {names}")

# Default sort (case-sensitive)
names.sort()
print(f"Default sort: {names}")
# Output: ['Alice', 'David', 'bob', 'charlie'] (uppercase first)

# Case-insensitive sort
names.sort(key=str.lower)
print(f"Case-insensitive: {names}")
# Output: ['Alice', 'bob', 'charlie', 'David']

# Sort by length
names.sort(key=len)
print(f"By length: {names}")
# Output: ['bob', 'Alice', 'David', 'charlie']
#Example 3: Sort complex objects
students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Charlie', 'grade': 78},
    {'name': 'David', 'grade': 90}
]

print("Original:")
for s in students:
    print(f"  {s['name']}: {s['grade']}")

# Sort by grade
students.sort(key=lambda x: x['grade'])
print("\nSorted by grade:")
for s in students:
    print(f"  {s['name']}: {s['grade']}")

# Sort by name
students.sort(key=lambda x: x['name'])
print("\nSorted by name:")
for s in students:
    print(f"  {s['name']}: {s['grade']}")

#10. reverse() - Reverse list in place
#Purpose: Reverses the order of elements in the list (modifies original).
#Example 1: Simple reverse
numbers = [1, 2, 3, 4, 5]
print(f"Original: {numbers}")

numbers.reverse()
print(f"Reversed: {numbers}")
# Output: [5, 4, 3, 2, 1]

# Reverse again
numbers.reverse()
print(f"Reversed again: {numbers}")
# Output: [1, 2, 3, 4, 5]
#Example 2: Reverse strings
word = list("python")
print(f"Original: {word}")
print(f"As string: {''.join(word)}")

word.reverse()
print(f"Reversed: {word}")
print(f"As string: {''.join(word)}")
# Output: nohtyp

#Example 3: Palindrome checker
def is_palindrome(lst):
    """Check if list is a palindrome"""
    reversed_list = lst.copy()
    reversed_list.reverse()
    return lst == reversed_list

# Test with different lists
test1 = [1, 2, 3, 2, 1]
test2 = ['a', 'b', 'c']
test3 = ['r', 'a', 'c', 'e', 'c', 'a', 'r']

print(f"{test1} is palindrome: {is_palindrome(test1)}")  # True
print(f"{test2} is palindrome: {is_palindrome(test2)}")  # False
print(f"{test3} is palindrome: {is_palindrome(test3)}")  # True

#11. copy() - Create shallow copy
#Purpose: Returns a shallow copy of the list (new list, same elements).
#Example 1: Basic copy
original = [1, 2, 3, 4, 5]
copied = original.copy()

print(f"Original: {original}")
print(f"Copied: {copied}")

# Modify copy
copied.append(6)
copied[0] = 100

print(f"\nAfter modifying copy:")
print(f"Original: {original}")  # Output: [1, 2, 3, 4, 5] - unchanged
print(f"Copied: {copied}")      # Output: [100, 2, 3, 4, 5, 6]

#Example 2: Copy vs alias
list1 = [10, 20, 30]

# Create alias (not a copy!)
alias = list1
# Create actual copy
copy = list1.copy()

print(f"Original: {list1}")
print(f"Alias: {alias}")
print(f"Copy: {copy}")

# Modify original
list1.append(40)

print(f"\nAfter modifying original:")
print(f"Original: {list1}")  # Output: [10, 20, 30, 40]
print(f"Alias: {alias}")     # Output: [10, 20, 30, 40] - also changed!
print(f"Copy: {copy}")       # Output: [10, 20, 30] - unchanged!

#Example 3: Shallow copy limitation
import copy

# Nested list
original = [[1, 2], [3, 4], [5, 6]]

# Shallow copy
shallow = original.copy()

# Deep copy
deep = copy.deepcopy(original)

print(f"Original: {original}")

# Modify inner list through shallow copy
shallow[0][0] = 999

print(f"\nAfter modifying shallow[0][0]:")
print(f"Original: {original}")  # Output: [[999, 2], ...] - changed!
print(f"Shallow: {shallow}")    # Output: [[999, 2], ...] - changed!
print(f"Deep: {deep}")          # Output: [[1, 2], ...] - unchanged!

#Quick Reference Table
# Creating a demo list for testing
demo = [1, 2, 3, 4, 5]

# Modifying methods (change original list)
demo.append(6)          # Add to end: [1, 2, 3, 4, 5, 6]
demo.extend([7, 8])     # Add multiple: [1, 2, 3, 4, 5, 6, 7, 8]
demo.insert(0, 0)       # Insert at index: [0, 1, 2, 3, 4, 5, 6, 7, 8]
demo.remove(5)          # Remove value: [0, 1, 2, 3, 4, 6, 7, 8]
demo.pop()              # Remove & return last: [0, 1, 2, 3, 4, 6, 7]
demo.pop(0)             # Remove & return index: [1, 2, 3, 4, 6, 7]
demo.clear()            # Empty list: []
demo = [3, 1, 4, 1, 5]
demo.sort()             # Sort: [1, 1, 3, 4, 5]
demo.reverse()          # Reverse: [5, 4, 3, 1, 1]

# Non-modifying methods (return values)
demo.index(4)           # Returns: 1 (position of 4)
demo.count(1)           # Returns: 2 (occurrences of 1)
new = demo.copy()       # Returns: new list [5, 4, 3, 1, 1]