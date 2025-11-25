
#Strings

#Strings in python are surrounded by either single quotation marks, or double quotation marks.
#'hello' is the same as "hello".
#You can display a string literal with the print() function


#Quotes Inside Quotes
#You can use quotes inside a string, as long as they don't match the quotes surrounding the string:
#Example

print("It's alright")
print("He is called 'Johnny'")
print('He is called "Johnny"')

#Assign String to a Variable
#Assigning a string to a variable is done with the variable name followed by an equal sign and the string:
#Example
a = "Hello"
print(a)


#Multiline Strings
#You can assign a multiline string to a variable by using three quotes:
#Example
#You can use three double quotes:
a = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""
print(a)

#Or three single quotes:
#Example
a = '''Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua.'''
print(a)


#Modify Strings
#Python has a set of built-in methods that you can use on strings.


#Upper Case
#ExampleGet your own Python Server
#The upper() method returns the string in upper case:
a = "Hello, World!"
print(a.upper())


#Lower Case
#Example
#he lower() method returns the string in lower case:
a = "Hello, World!"
print(a.lower())


#Remove Whitespace
#Whitespace is the space before and/or after the actual text, and very often you want to remove this space.
#Example
#The strip() method removes any whitespace from the beginning or the end:
a = " Hello, World! "
print(a.strip()) # returns "Hello, World!"


#String Concatenation
#To concatenate, or combine, two strings you can use the + operator.

#ExampleGet your own Python Server
#Merge variable a with variable b into variable c:
a = "Hello"
b = "World"
c = a + b
print(c)

#Example
#To add a space between them, add a " ":
a = "Hello"
b = "World"
c = a + " " + b
print(c)


#String Format
#We can combine strings and numbers by using f-strings or the format() method!

#F-Strings
#F-String was introduced in Python 3.6, and is now the preferred way of formatting strings.

#To specify a string as an f-string, simply put an f in front of the string literal, and add curly brackets {} as placeholders for variables and other operations.

#Example
#Create an f-string:
age = 36
txt = f"My name is John, I am {age}"
print(txt)


#Placeholders and Modifiers
#A placeholder can contain variables, operations, functions, and modifiers to format the value.
#Example
#Add a placeholder for the price variable:
price = 59
txt = f"The price is {price} dollars"
print(txt)


#Escape Character
#To insert characters that are illegal in a string, use an escape character.
#An escape character is a backslash \ followed by the character you want to insert.
#An example of an illegal character is a double quote inside a string that is surrounded by double quotes:

#\'	Single Quote	
#\\	Backslash	
#\n	New Line	
#\r	Carriage Return	
#\t	Tab	
#\b	Backspace	
#\f	Form Feed	
#\ooo	Octal value	
#\xhh	Hex value


#String Methods
#Python has a set of built-in methods that you can use on strings.

#Method	Description
#capitalize()	Converts the first character to upper case
#casefold()	    Converts string into lower case
#center()	    Returns a centered string
#count()	    Returns the number of times a specified value occurs in a string
#encode()	    Returns an encoded version of the string
#endswith()	    Returns true if the string ends with the specified value
#expandtabs()	Sets the tab size of the string
#find()	        Searches the string for a specified value and returns the position of where it was found
#format()	    Formats specified values in a string
#format_map()	Formats specified values in a string
#index()	    Searches the string for a specified value and returns the position of where it was found
#isalnum()	    Returns True if all characters in the string are alphanumeric
#isalpha()	    Returns True if all characters in the string are in the alphabet
#isascii()	    Returns True if all characters in the string are ascii characters
#isdecimal()	Returns True if all characters in the string are decimals
#isdigit()	    Returns True if all characters in the string are digits
#isidentifier()	Returns True if the string is an identifier
#islower()	    Returns True if all characters in the string are lower case
#isnumeric()	Returns True if all characters in the string are numeric
#isprintable()	Returns True if all characters in the string are printable
#isspace()	    Returns True if all characters in the string are whitespaces
#istitle()	    Returns True if the string follows the rules of a title
#isupper()	    Returns True if all characters in the string are upper case
#join()	        Joins the elements of an iterable to the end of the string
#ljust()	    Returns a left justified version of the string
#lower()	    Converts a string into lower case
#lstrip()	    Returns a left trim version of the string
#maketrans()	Returns a translation table to be used in translations
#partition()	Returns a tuple where the string is parted into three parts
#replace()	    Returns a string where a specified value is replaced with a specified value
#rfind()	    Searches the string for a specified value and returns the last position of where it was found
#rindex()	    Searches the string for a specified value and returns the last position of where it was found
#rjust()	    Returns a right justified version of the string
#rpartition()	Returns a tuple where the string is parted into three parts
#rsplit()	    Splits the string at the specified separator, and returns a list
#rstrip()	    Returns a right trim version of the string
#split()	    Splits the string at the specified separator, and returns a list
#splitlines()	Splits the string at line breaks and returns a list
#startswith()	Returns true if the string starts with the specified value
#strip()	    Returns a trimmed version of the string
#swapcase()	    Swaps cases, lower case becomes upper case and vice versa
#title()	    Converts the first character of each word to upper case
#translate()	Returns a translated string
#upper()	    Converts a string into upper case
#zfill()	    Fills the string with a specified number of 0 values at the beginning


#Here are three slightly more complex, practical examples of Python strings.

##1. String Slicing (Reversing a String)
#You can use advanced slicing with a "step" value to do more than just get a substring. A step of -1 is a classic trick to reverse a string.
# Slicing syntax is [start:stop:step]
text = "Python"

# By leaving start and stop blank and using -1, it steps backward
# from the end to the beginning.
reversed_text = text[::-1]

print(f"Original: {text}")
print(f"Reversed: {reversed_text}")

#2. Method Chaining
#You can "chain" multiple string methods together in one line. The output of one method becomes the input for the next (from left to right).
# Raw user input with messy whitespace and case
raw_input = "   hElLo, WorlD!   "

# Chain of methods to clean the input
# 1. .strip(): Removes leading/trailing whitespace
# 2. .title(): Capitalizes the first letter of each word
# 3. .replace(): Replaces the comma with nothing
cleaned_output = raw_input.strip().title().replace(",", "")

print(f"Raw: '{raw_input}'")
print(f"Cleaned: '{cleaned_output}'")

#3. Splitting and Joining
#This is a very common pattern for processing data. You split a string into a list to work with its parts, then join it back together into a new string.
# A string of keywords, separated by commas and spaces
tags = "python, data science, machine learning"

# 1. Split the string into a list, using ", " as the delimiter
tag_list = tags.split(", ")

# 2. Create a new string by joining the list items
#    We use a new delimiter, in this case a " | "
formatted_tags = " | ".join(tag_list)

print(f"Original Tags: {tags}")
print(f"Tag List: {tag_list}")
print(f"Formatted Tags: {formatted_tags}")

