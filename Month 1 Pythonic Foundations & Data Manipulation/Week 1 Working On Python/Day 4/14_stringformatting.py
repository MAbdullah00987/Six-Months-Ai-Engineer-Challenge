

#Python String Formatting

#Placeholders and Modifiers
#To format values in an f-string, add placeholders {}, a placeholder can contain variables, operations, functions, and modifiers to format the value.

#Add a placeholder for the price variable:

price = 59
txt = f"The price is {price} dollars"
print(txt)
#A placeholder can also include a modifier to format the value.
#A modifier is included by adding a colon : followed by a legal formatting type, like .2f which means fixed point number with 2 decimals:
#Display the price with 2 decimals:

price = 59
txt = f"The price is {price:.2f} dollars"
print(txt)
#You can also format a value directly without keeping it in a variable:

#Display the value 95 with 2 decimals:
txt = f"The price is {95:.2f} dollars"
print(txt)


#1. F-Strings (Formatted String Literals) 🚀
#This is the newest, fastest, and most readable way to format strings (available since Python 3.6). You simply prefix the string with an f and place your variables inside curly braces {}.

name = "Alex"
age = 28

# The expression inside the braces is evaluated and placed in the string.
greeting = f"Hello, my name is {name} and I am {age} years old."

print(greeting)
# Output: Hello, my name is Alex and I am 28 years old.

#2. The str.format() Method
#This method is very powerful and was the standard before f-strings. You use curly braces {} as placeholders in the string and then call the .format() method on the string to pass in the variables.

item = "laptop"
price = 1200

# Variables are inserted in the order they appear.
order_details = "The item is a {} and its price is ${}.".format(item, price)

print(order_details)
# Output: The item is a laptop and its price is $1200.
#You can also use numbered or named placeholders for more control:



# Using numbered placeholders
print("The {1} costs ${0}.".format(price, item))
# Output: The laptop costs $1200.

# Using named placeholders
print("The {product} costs ${cost}.".format(product=item, cost=price))
# Output: The laptop costs $1200.

#3. Old Style %-Formatting
#This is the original formatting method from C. While it works, it's less flexible and generally not recommended for new code. You use format specifiers like %s for strings and %d for integers.


language = "Python"
version = 3

# %s is a placeholder for a string, %d for a decimal integer.
info = "I am programming in %s version %d." % (language, version)

print(info)
# Output: I am programming in Python version 3.

#4. Formatting with Alignment and Padding
#String formatting is great for creating neatly aligned text, like tables. You can control the alignment and padding within the curly braces.
#Here's an example using f-strings:

header1 = "Fruit"
header2 = "Quantity"

# :<10 means left-align ( < ) within a space of 10 characters.
# :>10 means right-align ( > ) within a space of 10 characters.
print(f"{header1:<10} | {header2:>10}")
print("-" * 24) # A separator line

fruit1 = "Apples"
qty1 = 5
fruit2 = "Bananas"
qty2 = 12

print(f"{fruit1:<10} | {qty1:>10}")
print(f"{fruit2:<10} | {qty2:>10}")

#5. Number Formatting 🔢
#You can also control the appearance of numbers, such as setting the number of decimal places for floats or adding comma separators for large numbers.
#Here's an example using f-strings:

pi = 3.14159265
large_number = 1000000

# Format the float to have only two decimal places (:.2f)
formatted_pi = f"The value of Pi is approximately {pi:.2f}."

# Format the integer with comma separators (:,)
formatted_number = f"A large number is {large_number:,}."

print(formatted_pi)
# Output: The value of Pi is approximately 3.14.

print(formatted_number)
# Output: A large number is 1,000,000.
