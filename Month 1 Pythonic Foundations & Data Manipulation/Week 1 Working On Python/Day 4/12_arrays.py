
#Python Arrays

#What is an Array?
#An array is a special variable, which can hold more than one value at a time.
#If you have a list of items (a list of car names, for example), storing the cars in single variables could look like this:
##car1 = "Ford"
#car2 = "Volvo"
##car3 = "BMW"
#However, what if you want to loop through the cars and find a specific one? And what if you had not 3 cars, but 300?
#The solution is an array!
#An array can hold many values under a single name, and you can access the values by referring to an index number.
#Access the Elements of an Array
#You refer to an array element by referring to the index number.

#Examples
#1. Creating a List and Accessing Elements
#This example shows how to create a list and get a specific item from it using its index. Indices start at 0.

# A list of numbers
numbers = [10, 20, 30, 40, 50]

# Access the first element (index 0)
first_number = numbers[0]

# Access the third element (index 2)
third_number = numbers[2]

print(f"The list is: {numbers}")
print(f"The first number is: {first_number}")  # Output: 10
print(f"The third number is: {third_number}") # Output: 30

#2. Modifying Elements
#You can change an item in a list by accessing it by its index and assigning a new value.
# A list of fruits
fruits = ["apple", "banana", "cherry"]
print(f"Original list: {fruits}")

# Change the second item (index 1) from "banana" to "orange"
fruits[1] = "orange"

print(f"Modified list: {fruits}") # Output: ['apple', 'orange', 'cherry']

#3. Adding and Removing Elements
#You can easily add items to the end of a list with .append() or remove items by their value with .remove().

# A list of colors
colors = ["red", "green", "blue"]
print(f"Original list: {colors}")

# Add a new color to the end
colors.append("yellow")
print(f"After adding: {colors}") # Output: ['red', 'green', 'blue', 'yellow']

# Remove a specific color
colors.remove("green")
print(f"After removing: {colors}") # Output: ['red', 'blue', 'yellow']

#4. Looping Through a List
#This is a very common operation. You can use a for loop to perform an action on every single item in the list.

# A list of guest names
guests = ["Alice", "Bob", "Charlie"]

print("Sending invitations to:")
# Loop through each guest and print their name
for name in guests:
    print(f"- {name}")


#5. Slicing a List
#Slicing is a powerful way to get a sub-section of a list, rather than just one element.

# A list of letters
letters = ['a', 'b', 'c', 'd', 'e', 'f']

# Get the first three elements (from index 0 up to, but not including, index 3)
first_three = letters[0:3]
# A shorter way to write "from the start"
# first_three = letters[:3]

# Get the elements from index 2 to the end
from_c_onwards = letters[2:]

# Get the middle elements (from index 1 up to index 4)
middle = letters[1:4]

print(f"First three: {first_three}") # Output: ['a', 'b', 'c']
print(f"From 'c' on: {from_c_onwards}")  # Output: ['c', 'd', 'e', 'f']
print(f"Middle: {middle}")          # Output: ['b', 'c', 'd']

