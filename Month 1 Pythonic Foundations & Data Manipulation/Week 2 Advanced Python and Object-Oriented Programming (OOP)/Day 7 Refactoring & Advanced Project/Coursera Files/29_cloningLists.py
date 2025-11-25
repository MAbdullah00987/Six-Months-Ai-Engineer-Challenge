

#Purpose of Cloning Lists in Python
#What is Cloning Lists?
#Cloning (or copying) a list means creating a new, independent list with the same elements as the original. Unlike aliasing (where two names point to the same list), cloning creates a separate object in memory.
#Why Do We Need List Cloning?
#Key Reasons:

#Preserve original data - Keep the original list unchanged while modifying a copy
#Avoid side effects - Prevent unintended modifications when passing lists to functions
#Create variations - Make modified versions without affecting the original
#Data backup - Keep a snapshot of data before processing
#Parallel processing - Work with independent copies in different parts of code
#Undo functionality - Store previous states for rollback operations

#When You Need Cloning:

#Before sorting or modifying a list
#When passing lists to functions that might modify them
#Creating different versions of data for comparison
#Implementing undo/redo features
#Testing code without affecting original data


#Three Types of Cloning with Examples Each
#1. Shallow Cloning (For Simple Lists)
#Shallow cloning creates a new list, but if the list contains mutable objects (like nested lists), those inner objects are still shared.
#Example 1: Using slice notation [:]
# Original list
original = [1, 2, 3, 4, 5]

# Clone using slice notation
cloned = original[:]

print(f"Original: {original}")
print(f"Cloned: {cloned}")

# Modify the clone
cloned.append(6)
cloned[0] = 100

print(f"\nAfter modifying clone:")
print(f"Original: {original}")  # Output: [1, 2, 3, 4, 5] - UNCHANGED
print(f"Cloned: {cloned}")      # Output: [100, 2, 3, 4, 5, 6] - CHANGED

# Verify they are different objects
print(f"\nSame object? {original is cloned}")  # Output: False
print(f"Original ID: {id(original)}")
print(f"Cloned ID: {id(cloned)}")              # Different address

#Example 2: Using .copy() method
# Original shopping list
shopping_list = ['milk', 'bread', 'eggs', 'butter']

# Clone using .copy() method
my_list = shopping_list.copy()

print(f"Shopping list: {shopping_list}")
print(f"My list: {my_list}")

# Modify my_list
my_list.append('cheese')
my_list.remove('bread')

print(f"\nAfter modifications:")
print(f"Shopping list: {shopping_list}")  # Output: ['milk', 'bread', 'eggs', 'butter']
print(f"My list: {my_list}")              # Output: ['milk', 'eggs', 'butter', 'cheese']

# They are independent
print(f"\nIndependent? {shopping_list is not my_list}")  # Output: True

#Example 3: Using list() constructor
# Original scores
student_scores = [85, 90, 78, 92, 88]

# Clone using list() constructor
backup_scores = list(student_scores)

print(f"Original scores: {student_scores}")
print(f"Backup scores: {backup_scores}")

# Sort the clone (sorting modifies the list)
backup_scores.sort(reverse=True)

print(f"\nAfter sorting backup:")
print(f"Original scores: {student_scores}")  # Output: [85, 90, 78, 92, 88] - Unsorted
print(f"Backup scores: {backup_scores}")    # Output: [92, 90, 88, 85, 78] - Sorted

print(f"\nOriginal preserved? {student_scores == [85, 90, 78, 92, 88]}")  # True

#2. Deep Cloning (For Nested Lists)
##Deep cloning creates a completely independent copy, including all nested objects. Use this when your list contains other lists, dictionaries, or mutable objects.
#Example 1: Problem with shallow clone on nested lists
import copy

# Nested list (list of lists)
original = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Shallow clone - only copies outer list
shallow = original.copy()

# Deep clone - copies everything
deep = copy.deepcopy(original)

print(f"Original: {original}")
print(f"Shallow: {shallow}")
print(f"Deep: {deep}")

# Modify inner list through shallow clone
shallow[0][0] = 999

print(f"\n=== After modifying shallow[0][0] = 999 ===")
print(f"Original: {original}")  # Output: [[999, 2, 3], ...] - CHANGED! (Problem!)
print(f"Shallow: {shallow}")    # Output: [[999, 2, 3], ...] - CHANGED!
print(f"Deep: {deep}")          # Output: [[1, 2, 3], ...] - UNCHANGED (Safe!)

# Why? Inner lists in shallow copy are still references
print(f"\nInner list shared? {original[0] is shallow[0]}")  # True
print(f"Inner list shared? {original[0] is deep[0]}")       # False

#Example 2: Deep cloning game state
import copy

# Game state with nested structures
game_state = {
    'player': {'name': 'Hero', 'health': 100, 'inventory': ['sword', 'shield']},
    'enemies': [{'name': 'Goblin', 'health': 50}, {'name': 'Orc', 'health': 80}],
    'level': 1
}

# Save game (deep clone)
saved_game = copy.deepcopy(game_state)

print("Current game state:")
print(f"Player health: {game_state['player']['health']}")
print(f"Inventory: {game_state['player']['inventory']}")

# Continue playing - modify current state
game_state['player']['health'] = 30
game_state['player']['inventory'].append('potion')
game_state['level'] = 2

print("\n=== After playing more ===")
print(f"Current game - Health: {game_state['player']['health']}, Level: {game_state['level']}")
print(f"Current inventory: {game_state['player']['inventory']}")

print(f"\nSaved game - Health: {saved_game['player']['health']}, Level: {saved_game['level']}")
print(f"Saved inventory: {saved_game['player']['inventory']}")
# Saved game remains unchanged!

#Example 3: Deep cloning for data processing
import copy

# Original data structure
student_records = [
    {'name': 'Alice', 'grades': [85, 90, 88]},
    {'name': 'Bob', 'grades': [78, 82, 80]},
    {'name': 'Charlie', 'grades': [92, 95, 91]}
]

# Create a working copy for processing
processed_data = copy.deepcopy(student_records)

# Process the copy - add bonus points
for student in processed_data:
    student['grades'] = [grade + 5 for grade in student['grades']]
    student['average'] = sum(student['grades']) / len(student['grades'])

print("=== Original Data ===")
for student in student_records:
    print(f"{student['name']}: {student['grades']}")

print("\n=== Processed Data (with bonus) ===")
for student in processed_data:
    print(f"{student['name']}: {student['grades']}, Average: {student['average']:.2f}")

# Original data remains intact

#3. Practical Cloning Scenarios
#Example 1: Implementing Undo Functionality
class TextEditor:
    def __init__(self):
        self.lines = []
        self.history = []  # Store clones for undo
    
    def add_line(self, text):
        # Save current state before modifying
        self.history.append(self.lines.copy())
        self.lines.append(text)
    
    def undo(self):
        if self.history:
            self.lines = self.history.pop()
            return True
        return False
    
    def show(self):
        return '\n'.join(self.lines)

# Using the text editor
editor = TextEditor()
editor.add_line("First line")
editor.add_line("Second line")
editor.add_line("Third line")

print("Current text:")
print(editor.show())
print()

# Undo last action
editor.undo()
print("After undo:")
print(editor.show())

#Example 2: Cloning for A/B Testing
# Original user list
all_users = [
    {'id': 1, 'name': 'Alice'},
    {'id': 2, 'name': 'Bob'},
    {'id': 3, 'name': 'Charlie'},
    {'id': 4, 'name': 'David'}
]

# Clone for Group A (gets feature X)
group_a = all_users[:2].copy()

# Clone for Group B (gets feature Y)
group_b = all_users[2:].copy()

# Add feature flags
for user in group_a:
    user_copy = user.copy()
    user_copy['feature'] = 'X'
    print(f"Group A: {user_copy}")

for user in group_b:
    user_copy = user.copy()
    user_copy['feature'] = 'Y'
    print(f"Group B: {user_copy}")

# Original list remains unchanged
print(f"\nOriginal users: {all_users}")

#Example 3: Cloning for Safe Function Parameters
def process_and_sort(data_list):
    """
    This function sorts the list.
    We clone it to avoid modifying the original.
    """
    # Clone the list before modifying
    sorted_list = data_list.copy()
    sorted_list.sort()
    return sorted_list

# Original data
temperatures = [25, 18, 30, 22, 28, 19]

print(f"Original temperatures: {temperatures}")

# Process the data
sorted_temps = process_and_sort(temperatures)

print(f"Sorted temperatures: {sorted_temps}")
print(f"Original after function: {temperatures}")  # Still in original order!

# Without cloning, this would have modified the original

#Quick Reference: When to Use Which Method
# For simple lists (no nested structures)
simple_list = [1, 2, 3, 4, 5]
clone1 = simple_list[:]           # Slice notation
clone2 = simple_list.copy()       # .copy() method
clone3 = list(simple_list)        # list() constructor
# All three work the same for simple lists

# For nested lists/complex structures
nested_list = [[1, 2], [3, 4]]
import copy
deep_clone = copy.deepcopy(nested_list)  # Only this creates truly independent copy