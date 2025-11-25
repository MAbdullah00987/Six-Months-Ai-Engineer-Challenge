

#Python Casting

#Specify a Variable Type
#There may be times when you want to specify a type on to a variable. This can be done with casting. Python is an object-orientated language, and as such it uses classes to define data types, including its primitive types.

#Casting in python is therefore done using constructor functions:

#int() - constructs an integer number from an integer literal, a float literal (by removing all decimals), or a string literal (providing the string represents a whole number)
#float() - constructs a float number from an integer literal, a float literal or a string literal (providing the string represents a float or an integer)
#str() - constructs a string from a wide variety of data types, including strings, integer literals and float literals


#Example
#Integers:
x = int(1)   # x will be 1
y = int(2.8) # y will be 2
z = int("3") # z will be 3

#Example
#Floats:
x = float(1)     # x will be 1.0
y = float(2.8)   # y will be 2.8
z = float("3")   # z will be 3.0
w = float("4.2") # w will be 4.2


#Example
#Strings:
x = str("s1") # x will be 's1'
y = str(2)    # y will be '2'
z = str(3.0)  # z will be '3.0'


#Complex Examples
#1. Casting to an Integer (int)
#This converts a string or a float (decimal) into a whole number. When casting a float, it truncates (cuts off) the decimal; it does not round.

# Original value is a string
age_string = "30"
print(f"Original value: {age_string}, Type: {type(age_string)}")

# Cast to int
age_number = int(age_string)
print(f"Casted value: {age_number}, Type: {type(age_number)}")



#2. Casting to a String (str)
#This converts any other data type (like a number or a list) into a string. This is most often used when you want to combine a number with text.

# Original value is an integer
user_id = 101
print(f"Original value: {user_id}, Type: {type(user_id)}")

# Cast to str
user_id_string = str(user_id)
print(f"Casted value: {user_id_string}, Type: {type(user_id_string)}")

# Now you can combine it with other strings
print("The User ID is: " + user_id_string)


#3. Casting to a Float (float)
#This converts a string or an integer into a floating-point number (a number with a decimal).

# Original value is an integer
price = 50
print(f"Original value: {price}, Type: {type(price)}")

# Cast to float
precise_price = float(price)
print(f"Casted value: {precise_price}, Type: {type(precise_price)}")