
#what are data types in python?
#A data type is a classification that tells the computer what kind of value a variable holds and what kind of operations (like math or text joining) can be performed on it. Think of it as a label for different kinds of information, like "this is a number," "this is text," or "this is a list of items.


#How Many Data Types Are in Python?
#Python has several powerful built-in data types. There isn't a fixed "total number" because you can also create your own custom data types (called classes) or import more from libraries.
#However, the main built-in types are grouped into these categories:
#Numeric Types: int, float, complex (for complex numbers)
#Sequence Types (ordered collections): str (string), list, tuple
#Mapping Type (key-value pairs): dict (dictionary)
#Set Types (unordered, unique items): set, frozenset
#Boolean Type (True/False): bool
#None Type (a special null value): NoneType


#Practical Examples

#1. Integer (int)
#This is for whole numbers, both positive and negative, without any decimal points.
#What it is: A whole number.

age = 30
items_in_cart = 5
temperature = -10

# You can check the type with the type() function
print(type(age)) 
# Output: <class 'int'>

#2. String (str)

#This is for text. A string is a sequence of characters, enclosed in either single quotes (') or double quotes (").
#What it is: Text.

name = "Ali"
greeting = 'Hello, world!'
user_id = "user-123" # Even though it has numbers, it's text

print(type(name))
# Output: <class 'str'>

#3. List (list)
#This is a versatile, ordered collection of items. You can store different data types in the same list, and you can change, add, or remove items. Lists are defined with square brackets [].
#What it is: An ordered, changeable list of items.

scores = [95, 88, 72, 100]
shopping_list = ["apples", "milk", "bread"]
mixed_data = ["Ali", 25, True, 78.5]

# You can access items by their position (index)
print(shopping_list[0]) # Prints 'apples'

print(type(scores))
# Output: <class 'list'>