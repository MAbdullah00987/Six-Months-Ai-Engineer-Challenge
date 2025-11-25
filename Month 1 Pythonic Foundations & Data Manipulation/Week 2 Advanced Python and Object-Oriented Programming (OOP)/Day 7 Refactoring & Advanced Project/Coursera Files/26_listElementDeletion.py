

#Purpose of List Element Deletion in Python
#List element deletion allows you to remove items from a list, which is essential for:

#Memory management - Removing unnecessary data frees up memory
#Data manipulation - Updating lists by removing outdated or invalid entries
#Algorithm implementation - Many algorithms require removing elements during processing
#User interaction - Allowing users to delete items from shopping carts, to-do lists, etc.
#Data cleaning - Removing duplicates, errors, or unwanted values from datasets

#Three Main Methods for Deleting List Elements
#1. del statement
#Removes elements by index or slices. Useful when you know the position of the element.

#Example 1: Delete by index
fruits = ['apple', 'banana', 'cherry', 'date']
del fruits[1]  # Removes 'banana'
print(fruits)  # Output: ['apple', 'cherry', 'date']

#Example 2: Delete slice
numbers = [1, 2, 3, 4, 5, 6]
del numbers[1:4]  # Removes elements at index 1, 2, 3
print(numbers)  # Output: [1, 5, 6]

#Example 3: Delete entire list
temp_list = [10, 20, 30]
del temp_list
# print(temp_list)  # This would raise NameError - list no longer exists

#2. remove() method
#Removes the first occurrence of a specific value. Useful when you know the value but not its position.
#Example 1: Remove specific value
colors = ['red', 'blue', 'green', 'blue']
colors.remove('blue')  # Removes first 'blue'
print(colors)  # Output: ['red', 'green', 'blue']

#Example 2: Remove with error handling
animals = ['cat', 'dog', 'bird']
try:
    animals.remove('fish')
except ValueError:
    print("Item not found")  # Output: Item not found
print(animals)  # Output: ['cat', 'dog', 'bird']

#Example 3: Remove in loop (careful approach)
scores = [85, 90, 85, 95, 85]
# Remove all occurrences of 85
while 85 in scores:
    scores.remove(85)
print(scores)  # Output: [90, 95]
#3. pop() method
#Removes and returns an element by index (default is last element). Useful when you need the removed value.
#Example 1: Pop last element
stack = [1, 2, 3, 4, 5]
last = stack.pop()
print(f"Removed: {last}")  # Output: Removed: 5
print(stack)  # Output: [1, 2, 3, 4]

#Example 2: Pop by index
tasks = ['wake up', 'exercise', 'work', 'sleep']
task = tasks.pop(1)  # Removes 'exercise'
print(f"Completed: {task}")  # Output: Completed: exercise
print(tasks)  # Output: ['wake up', 'work', 'sleep']

#Example 3: Using pop() in a queue
queue = ['first', 'second', 'third']
served = queue.pop(0)  # Remove from front
print(f"Serving: {served}")  # Output: Serving: first
print(queue)  # Output: ['second', 'third']

