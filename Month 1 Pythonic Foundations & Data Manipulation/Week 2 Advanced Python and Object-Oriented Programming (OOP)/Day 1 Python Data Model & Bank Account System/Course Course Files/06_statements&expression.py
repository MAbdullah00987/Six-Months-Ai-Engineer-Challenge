
#Expression
#An Expression is a piece of code that produces a value. It's like a phrase in English (e.g., "the blue car").

#Examples of Expressions:
#10 + 5 (evaluates to the value 15)
#age >= 18 (evaluates to the value True or False)
#"Hello" (a simple string value is an expression)
#user_name (a variable that evaluates to the value it stores)
#my_function() (a function call that returns a value)

#What is a Statement?
#A statement is a complete line of code that tells Python to do something. Statements are the main building blocks of your program.
#Statements are often made of expressions.

#Examples of Statements:
#Assignment Statement: age = 30
#The action is storing a value.
#The 30 part is an expression.
#Function Call Statement: print("Hello, world!")
#The action is to display text.
#The "Hello, world!" part is an expression.
#Conditional Statement: if age > 18:
#The action is to start a conditional block.
#The age > 18 part is an expression.
#Import Statement: import math
#The action is to load a library.
#You cannot assign a statement to a variable. This code will fail: result = (age = 30) <-- This is a syntax error.

#How to Work with Them Perfectly

#1. Keep Expressions Simple and Clear

#power = 2 ** 3          # 8
#multiply = 5 * power    # 40
#divide = multiply / 4   # 10
#result = 10 + divide    # 20
#Or, if you must keep it on one line, use parentheses: result = 10 + (5 * (2**3) / 4)

#2. Make Statements Do One Logical Thing
#Each statement should generally be on its own line and perform one clear action.

x = 10
y = 20
print(x + y)

#3. Master Indentation (This is Critical)
#In Python, indentation (the spaces at the beginning of a line) is not just for style; it's part of the syntax.

#Statements like if, for, def, and while control other blocks of code. You must indent the code "inside" them.


age = 20
if age > 18:
 print("You can vote.") # Error! Missing indentation
#Perfect (this is how Python works):


age = 20
if age > 18:  # This 'if' is a statement
    # This 'print' statement is INSIDE the 'if' block
    print("You can vote.") 

print("This line runs no matter what.")

