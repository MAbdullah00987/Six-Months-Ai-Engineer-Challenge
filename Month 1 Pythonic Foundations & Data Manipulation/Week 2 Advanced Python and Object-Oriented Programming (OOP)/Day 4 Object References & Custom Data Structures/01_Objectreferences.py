#
#Object References in Python 
#Why We Need Object References?
#In Python, everything is an object, and variables are just labels/names that point to objects in memory. Understanding object references is crucial because:

#Memory Efficiency - Multiple variables can reference the same object instead of copying it
##Mutability Management - Understanding when changes affect other references
#Performance - Passing references is faster than copying entire objects
#Shared State - Multiple parts of code can work with the same data

#Rules for Assigning Object References Properly
#Rule 1: Assignment Creates References, Not Copies
#python# The variable name is just a label pointing to an object
a = [1, 2, 3]  # 'a' points to a list object
b = a          # 'b' points to the SAME list object, not a copy
#Rule 2: Immutable vs Mutable Objects Behave Differently
#Immutable Objects (int, float, str, tuple):

#Cannot be changed after creation
#Reassignment creates a new object

#Mutable Objects (list, dict, set, custom objects):

#Can be modified in place
#All references see the change

#Rule 3: Use .copy() or copy.deepcopy() When Needed
#pythonimport copy

# Shallow copy - creates new object, but nested objects are still references
#shallow = original.copy()

# Deep copy - creates completely independent copy
#deep = copy.deepcopy(original)
#Rule 4: Function Arguments Are Passed by Reference
#pythondef modify_list(lst):
#    lst.append(4)  # Modifies the original list

my_list = [1, 2, 3]
#modify_list(my_list)
print(my_list)  # [1, 2, 3, 4] - Original is changed!
#Rule 5: Use is vs == Correctly

#is checks if two variables reference the same object (identity)
#== checks if two objects have the same value (equality)



# Simple Examples
#Example 1: Immutable Objects (Numbers & Strings)
#python# Immutable objects - Safe from unexpected changes
x = 10
y = x          # y points to the same object as x
print(x is y)  # True - same object

y = 20         # y now points to a NEW object
print(x)       # 10 - x unchanged
print(y)       # 20
print(x is y)  # False - different objects now
#Example 2: Mutable Objects (Lists) - The Tricky Part!
#python# Mutable objects - Changes affect all references!
list1 = [1, 2, 3]
list2 = list1        # list2 points to SAME object as list1

list2.append(4)      # Modify through list2

print(list1)         # [1, 2, 3, 4] - list1 changed too! 😲
print(list2)         # [1, 2, 3, 4]
print(list1 is list2)  # True - same object
#Example 3: Creating Independent Copies
#python# To avoid shared references, create a copy
list1 = [1, 2, 3]
list2 = list1.copy()  # Create a NEW independent list

list2.append(4)

print(list1)         # [1, 2, 3] - unchanged ✅
print(list2)         # [1, 2, 3, 4]
print(list1 is list2)  # False - different objects


