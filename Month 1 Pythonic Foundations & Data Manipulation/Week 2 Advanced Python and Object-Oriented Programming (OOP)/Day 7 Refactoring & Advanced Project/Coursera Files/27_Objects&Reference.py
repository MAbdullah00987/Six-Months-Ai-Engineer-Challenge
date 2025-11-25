

#Purpose of Objects and References in Python
#What Are Objects and References?
#In Python, everything is an object - numbers, strings, lists, functions, even classes themselves. An object is a piece of data stored in memory with:

#A type (class)
#A value
#An identity (memory address)

#A reference is a name (variable) that points to an object in memory. Multiple references can point to the same object.
#Why Do We Need Them?
#Objects:

#Encapsulation - Bundle data and behavior together
#Code reusability - Create multiple instances from one blueprint
#Memory efficiency - Objects can be shared through references
#Real-world modeling - Represent real entities in code

#References:
#Memory efficiency - Multiple variables can share the same object instead of duplicating data
#Dynamic behavior - Variables can point to different objects at runtime
#Mutable object sharing - Changes through one reference affect all references
#Function parameter passing - Objects are passed by reference, not copied


#Three Examples of Objects
#Example 1: Custom Class Object
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says Woof!"

# Creating objects (instances)
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(dog1.bark())  # Output: Buddy says Woof!
print(dog2.bark())  # Output: Max says Woof!
print(f"dog1 is at memory: {id(dog1)}")  # Shows unique memory address
print(f"dog2 is at memory: {id(dog2)}")  # Different memory address

#Example 2: Built-in Objects (List, String, Integer)
# Different types of objects
my_list = [1, 2, 3]        # List object
my_string = "Hello"         # String object
my_number = 42              # Integer object

print(type(my_list))        # Output: <class 'list'>
print(type(my_string))      # Output: <class 'str'>
print(type(my_number))      # Output: <class 'int'>

# Each has its own methods
print(my_list.append(4))    # List method
print(my_string.upper())    # String method: HELLO

#Example 3: Function Objects
def greet(name):
    return f"Hello, {name}!"

# Functions are objects too!
print(type(greet))          # Output: <class 'function'>

# You can assign functions to variables
say_hello = greet
print(say_hello("Alice"))   # Output: Hello, Alice!

# You can pass functions as arguments
def execute_function(func, value):
    return func(value)

result = execute_function(greet, "Bob")
print(result)               # Output: Hello, Bob!

#Three Examples of References
#Example 1: Multiple References to Same Object
# Creating a list object
original_list = [1, 2, 3]

# Creating another reference to the SAME object
reference_list = original_list

print(f"original_list: {id(original_list)}")
print(f"reference_list: {id(reference_list)}")  # Same memory address!

# Modifying through one reference affects the other
reference_list.append(4)

print(original_list)   # Output: [1, 2, 3, 4]
print(reference_list)  # Output: [1, 2, 3, 4]

# Both changed because they reference the same object
print(original_list is reference_list)  # Output: True

#Example 2: Immutable Objects and References
# With immutable objects (like integers, strings)
x = 10
y = x  # y references the same object as x

print(f"x id: {id(x)}")
print(f"y id: {id(y)}")  # Same address initially

# When you "change" x, it creates a NEW object
x = 20

print(f"x id after change: {id(x)}")  # Different address now
print(f"y id after x changes: {id(y)}")  # Still points to 10

print(f"x = {x}")  # Output: x = 20
print(f"y = {y}")  # Output: y = 10 (unchanged)

#Example 3: Function Parameters as References
def modify_list(lst):
    # lst is a reference to the original list
    lst.append(100)
    print(f"Inside function: {lst}")

def modify_number(num):
    # num is a reference, but integers are immutable
    num = num + 10
    print(f"Inside function: {num}")

# Mutable object (list)
my_list = [1, 2, 3]
modify_list(my_list)
print(f"After function: {my_list}")  # Output: [1, 2, 3, 100] - CHANGED!

# Immutable object (integer)
my_number = 5
modify_number(my_number)             # Output: Inside function: 15
print(f"After function: {my_number}") # Output: 5 - UNCHANGED!

#Visual Comparison: Objects vs References
# Creating objects and references
a = [1, 2, 3]     # 'a' references a list object at memory location X
b = a             # 'b' also references the SAME object at location X
c = [1, 2, 3]     # 'c' references a DIFFERENT list object at location Y

print(a is b)     # Output: True (same object)
print(a is c)     # Output: False (different objects)
print(a == c)     # Output: True (same content, different objects)

# Checking identity
print(f"a: {id(a)}")
print(f"b: {id(b)}")  # Same as a
print(f"c: {id(c)}")  # Different from a and b