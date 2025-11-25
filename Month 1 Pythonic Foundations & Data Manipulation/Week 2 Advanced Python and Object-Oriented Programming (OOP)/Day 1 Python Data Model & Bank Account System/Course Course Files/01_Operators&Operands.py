
#By Following along Cousera Course:
#University of Michigan's "Python for Everybody" on Coursera (Modules 1-4)

#Today I am Trying to Module 1 of this course about python Basics:

#Operators and operands

#What are Operators and Operands?
#An operator is a symbol (like +, *, or >) that tells the computer to perform a specific action or operation.
#An operand is the data or value that the operator acts upon. e.g 5 + 3

#Types of Operators in Python
#Operators are grouped into types based on the job they do. Here is a complete list of the main types of operators in Python.

#1. Arithmetic Operators
#What they do: Perform standard mathematical calculations.
#Operators: + (add), - (subtract), * (multiply), / (divide), % (modulus/remainder), ** (exponent/power), // (floor division).
#2. Assignment Operators
#What they do: Assign a value to a variable.
#Operators: = (assign), += (add and assign), -= (subtract and assign), *= (multiply and assign), etc.
#3. Comparison (Relational) Operators
#What they do: Compare two values and return a Boolean result (True or False).
#Operators: == (equal to), != (not equal to), > (greater than), < (less than), >= (greater than or equal to), <= (less than or equal to).
#4. Logical Operators
#What they do: Combine or modify conditional (True/False) statements.
#Operators: and, or, not.
#5. Membership Operators
#What they do: Test if a value is present in a sequence (like a list, tuple, or string).
#Operators: in, not in.
#6. Identity Operators
#What they do: Compare if two operands are the exact same object in memory (this is different from ==, which just checks if their values are equal).
#Operators: is, is not.
#7. Bitwise Operators
#hat they do: Perform operations on the raw binary (0s and 1s) representation of numbers. These are more advanced.
#Operators: & (AND), | (OR), ^ (XOR), ~ (NOT), << (left shift), >> (right shift).

#Here are three examples showing different types of operators and their operands.
#Example 1: Arithmetic and Assignment
#Expression: total_cost = 100 + 5
#Operator(s):
#+ (Arithmetic operator)
#= (Assignment operator)
#Operand(s): 100, 5, and total_cost (the variable receiving the final value).
#Example 2: Comparison
#Expression: is_adult = age >= 18
#Operator(s):
#>= (Comparison operator)
#= (Assignment operator)
#Operand(s): age (a variable), 18 (a value), and is_adult (a variable). The result of age >= 18 (which will be True or False) becomes an operand for the = operator.
#Example 3: Logical
#Expression: can_enter = has_ticket and not is_banned
#Operator(s):
#and (Logical operator)
#not (Logical operator)
#Operand(s): has_ticket and is_banned (these are variables that hold True or False values).

#Practical Examples About operators and Operands:

#1. Arithmetic Operators (Doing Math)
#These operators perform calculations.

# --- Example: Calculating a shopping cart total ---

item_price = 150  # A variable as an operand
tax = 25          # A literal value as an operand

# Operator: +
# Operands: item_price, tax
total_cost = item_price + tax

print(f"Total cost is: {total_cost}")

# Operator: * (multiplication)
# Operands: total_cost, 2
bill_for_two = total_cost * 2

print(f"Bill for two people is: {bill_for_two}")
#Explanation: item_price and tax are the operands (the data) that the + operator (the action) uses to create a new value.

#2. Assignment Operators (Storing Values)
#These operators assign or "save" a value into a variable.

# --- Example: Keeping score in a game ---

# Operator: = (simple assignment)
# Operands: score, 0
score = 0  # Assigns the value 0 to the variable 'score'

# The player finds a coin
# Operator: += (add and assign)
# Operands: score, 10
score += 10  # This is a shortcut for: score = score + 10

print(f"Current score: {score}")

# The player gets a power-up
# Operator: *= (multiply and assign)
# Operands: score, 2
score *= 2   # This is a shortcut for: score = score * 2

print(f"Score after power-up: {score}")
#Explanation: The most common operator is =. The compound operators like += and *= are very practical shortcuts for modifying a variable's existing value.

#3. Comparison Operators (Asking True/False)
#These operators compare two values and give you a True or False answer. They are the heart of all decision-making in code (like if statements).

# --- Example: Checking user access ---

age = 22
access_level = "guest"

# Operator: >= (greater than or equal to)
# Operands: age, 18
is_adult = age >= 18
print(f"Is user an adult? {is_adult}")

# Operator: == (equal to)
# Operands: access_level, "admin"
is_admin = access_level == "admin"
print(f"Is user an admin? {is_admin}")

# Operator: != (not equal to)
# Operands: access_level, "banned"
is_not_banned = access_level != "banned"
print(f"Is user not banned? {is_not_banned}")