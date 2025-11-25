
#Variables
#Python has no command for declaring a variable.
#A variable is created the moment you first assign a value to it.


#Simple Variables Examples

#Example 1:
x = str(3)    # x will be '3'
y = int(3)    # y will be 3
z = float(3)  # z will be 3.0


#Variable Names 

#A variable can have a short name (like x and y) or a more descriptive name (age, carname, total_volume).

#Rules for Python variables:

#A variable name must start with a letter or the underscore character
# A variable name cannot start with a number
#A variable name can only contain alpha-numeric characters and underscores (A-z, 0-9, and _ )
#Variable names are case-sensitive (age, Age and AGE are three different variables)
#A variable name cannot be any of the Python keywords.


# An integer variable
user_age = 30
print(f"The user's age is: {user_age}")

# A string variable
first_name = "Alice"
print(f"The user's first name is: {first_name}")

# A boolean (True/False) variable
is_authenticated = True
print(f"Is the user authenticated? {is_authenticated}")


#Camel Case
#Each word, except the first, starts with a capital letter:

myVariableName = "John"


#Pascal Case
#Each word starts with a capital letter:

MyVariableName = "John"

#Snake Case
#Each word is separated by an underscore character:

my_variable_name = "John"




#1. Assigning Different Values to Different Variables
#This is the most common method, often called "unpacking." You match a sequence of variables on the left with a sequence of values on the right.
# Assigns "Hassan" to name, 25 to age, and "Python" to skill

name, age, main_skill = "Hassan", 25, "Python"

print(f"Name: {name}")
print(f"Age: {age}")
print(f"Main Skill: {main_skill}")

#2. Assigning the Same Value to All Variables
#This is called "chaining" and is useful for initializing several variables to the same starting value.
# Assigns the value 0 to all three variables

x = y = z = 0

print(f"x: {x}")
print(f"y: {y}")
print(f"z: {z}")


#3. Unpacking from a List or Tuple
#This is a very practical use of unpacking, where you assign the items of a list (or tuple) to separate variables.



# A list containing a user's ID and email
user_data = [101, "user@example.com"]

# Unpack the list into two variables
user_id, email = user_data

print(f"User ID: {user_id}")
print(f"Email: {email}")

