

#Python Operators
#Operators are used to perform operations on variables and values.

#In the example below, we use the + operator to add together two values:


#Although the + operator is often used to add together two values, like in the example above, it can also be used to add together a variable and a value, or two variables:

#Example
sum1 = 100 + 50      # 150 (100 + 50)
sum2 = sum1 + 250    # 400 (150 + 250)
sum3 = sum2 + sum2   # 800 (400 + 400)


#Python divides the operators in the following groups:

#Arithmetic operators
#Assignment operators
#Comparison operators
#Logical operators
#Identity operators
#Membership operators
#Bitwise operators


#--------------------------------------------------------------------------------Arithmetic operators------------------------------------------------------------------------

#Arithmetic operators are used with numeric values to perform common mathematical operations:

#   Operator	Name	          Example	
#     +	        Addition	      x + y	
#     -	        Subtraction	      x - y	
#     *	        Multiplication	  x * y	
#     /	        Division	      x / y	
#     %	        Modulus	          x % y	
#     **	    Exponentiation	  x ** y	
#     //	    Floor division	  x // y


#
#Examples
#Here is an example using different arithmetic operators:

#ExampleGet your own Python Server
x = 15
y = 4

print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x % y)
print(x ** y)
print(x // y)


#Division in Python
#Python has two division operators:

#/ - Division (returns a float)
#// - Floor division (returns an integer)


#Example
#Division always returns a float:

x = 12
y = 5

print(x / y)


#Example
#Floor division always returns an integer.

#It rounds DOWN to the nearest integer:

x = 12
y = 5

print(x // y)


#Examples about all arithmetic operators

#1. Addition (+)
#Used for summing numbers or concatenating strings.
# Example 1: Calculating a total price

item_price = 45.50
tax = 3.64
total_cost = item_price + tax
print(f"Total cost: {total_cost}")

# Example 2: Incrementing a score counter
score = 100
score = score + 5 # or the shortcut: score += 5
print(f"New score: {score}")


#2. Subtraction (-)
#Used for finding the difference or working with negative values.

# Example 1: Calculating remaining stock
total_stock = 200
items_sold = 35
remaining_stock = total_stock - items_sold
print(f"Stock left: {remaining_stock}")

# Example 2: Calculating a budget deficit
income = 2500
expenses = 2750
balance = income - expenses
print(f"Monthly balance: {balance}") # Result is a negative number



#3. Multiplication (*)
#Used for scaling numbers or repeating strings.

# Example 1: Calculating total cost for multiple items
item_count = 15
price_per_item = 4.99
subtotal = item_count * price_per_item
print(f"Subtotal: {subtotal}")

# Example 2: Repeating a string to create a divider
divider = "-" * 20
print(divider) # Prints 20 hyphens



#4. Division (/)
#Used for standard division; it always returns a float (a number with a decimal).

# Example 1: Calculating an average score
total_score = 435
num_students = 50
average_score = total_score / num_students
print(f"Average score: {average_score}")

# Example 2: Showing the float result, even with whole numbers
result = 10 / 2
print(f"10 / 2 = {result}") # Result is 5.0, not 5


#5. Floor Division (//)
#Used for integer division; it divides and discards the remainder.

# Example 1: Converting total seconds into whole minutes
total_seconds = 140
minutes = total_seconds // 60
print(f"Minutes: {minutes}") # Result is 2

# Example 2: Finding how many full groups can be made
total_people = 30
group_size = 7
full_groups = total_people // group_size
print(f"Full groups: {full_groups}") # Result is 4



#6. Modulo (%)
#Used to get the remainder of a division.

# Example 1: Checking if a number is even or odd
number = 17
if number % 2 == 0:
    print("Even")
else:
    print("Odd") # 17 % 2 is 1 (the remainder)

# Example 2: Finding "leftover" seconds
total_seconds = 140
seconds_left_over = total_seconds % 60
print(f"Seconds left: {seconds_left_over}") # Result is 20


#7. Exponentiation (**)
#Used for raising a number to a power.
# Example 1: Calculating the area of a square
side_length = 8
area = side_length ** 2 # 8 to the power of 2
print(f"Area: {area}")

# Example 2: Finding a square root (using a fractional exponent)
number = 64
square_root = number ** 0.5 # Same as 8
print(f"Square root: {square_root}")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------Assignment Operators--------------------------------------------------------------------------------

#Assignment operators are used to assign values to variables:

# Operator	  Example	       Same As	
#    =	       x = 5	        x = 5	
#    +=	       x += 3	        x = x + 3	
#    -=	       x -= 3	        x = x - 3	
#    *=	       x *= 3	        x = x * 3	
#    /=	       x /= 3	        x = x / 3	
#    %=	       x %= 3	        x = x % 3	
#    //=	   x //= 3	        x = x // 3	
#    **=	   x **= 3	        x = x ** 3	
#    &=	       x &= 3	        x = x & 3	
#    |=	       x |= 3	        x = x | 3	
#    ^=	       x ^= 3	        x = x ^ 3	
#    >>=	   x >>= 3	        x = x >> 3	
#    <<=	   x <<= 3	        x = x << 3	
#    :=	       print(x := 3)	x = 3 print(x)
                

#The Walrus Operator
#Python 3.8 introduced the := operator, known as the "walrus operator". It assigns values to variables as part of a larger expression:

#ExampleGet your own Python Server
numbers = [1, 2, 3, 4, 5]
count = len(numbers)
if count > 3:
    print(f"List has {count} elements")

if (count := len(numbers)) > 3:
    print(f"List has {count} elements")


#Examples

#1. Add and Assign (+=)
#This is perfect for accumulating a total in a loop. Instead of total = total + number, you just use total += number.


# Use case: Summing all numbers in a list
total_cost = 0
items = [10.50, 5.25, 15.00]

for item in items:
    total_cost += item  # Adds the item price to the total_cost

print(f"Total cost: ${total_cost}")


#2. Multiply and Assign (*=)
#This is useful for applying a repeated multiplier, like calculating compound interest or applying a discount.
# Use case: Calculating an investment after 3 years of 5% growth

principal = 1000.0
growth_rate = 1.05  # 100% of original + 5% growth

for year in range(3):
    principal *= growth_rate  # Multiplies principal by 1.05 each year

print(f"Investment after 3 years: ${principal:.2f}")


#3. Modulo and Assign (%=)
#This is a bit more advanced and is great for "wrapping around" or cycling through a limited set of items, like days of the week or players in a game.
# Use case: Assigning 10 tasks to 3 team members cyclically

team = ["Alice", "Bob", "Charlie"]
num_members = len(team)
current_member_index = 0

for task_number in range(1, 11):
    member = team[current_member_index]
    print(f"Task {task_number} is assigned to: {member}")
    
    current_member_index += 1
    current_member_index %= num_members # Wraps index back to 0 when it hits 3
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------Comparison Operators--------------------------------------------------------------------------------

#Comparison Operators
#Comparison operators are used to compare two values:

#  Operator	    Name	                     Example	
#    ==	        Equal	                     x == y	
#    !=	        Not equal	                 x != y	
#     >	        Greater than                 x > y	
#     <	        Less than	                 x < y	
#    >=	        Greater than or equal to	 x >= y	
#    <=	        Less than or equal to	     x <= y
    
    

#Examples
#Comparison operators return True or False based on the comparison:

#Example
#Get your own Python Server

x = 5
y = 3

print(x == y)
print(x != y)
print(x > y)
print(x < y)
print(x >= y)
print(x <= y)


#Chaining Comparison Operators
#Python allows you to chain comparison operators:

#Example

x = 5

print(1 < x < 10)

print(1 < x and x < 10)


#Examples 

#1. Equal To (==)
#This operator checks if the values on both sides are exactly the same. It is one of the most common operators, often used in if statements to check a specific condition.

# Use case: Checking if a user typed a specific command
user_command = "quit"

if user_command == "quit":
    print("Shutting down...")
else:
    print("Processing command...")


#2. Not Equal To (!=)
#This operator checks if the values on both sides are not the same. It's very useful for finding errors or filtering out a specific value.

# Use case: Checking for an error status
http_status_code = 404

# We only want to run the code if the status is NOT 200 (which means "OK")
if http_status_code != 200:
    print(f"Error: Received status code {http_status_code}")
else:
    print("Success: Page loaded.")
    

#3. Greater Than or Equal To (>=)
#This operator checks if the value on the left is larger than or equal to the value on the right. It's essential for any logic involving thresholds.

# Use case: Checking if a user meets the minimum requirement
user_age = 18
min_age_to_vote = 18

if user_age >= min_age_to_vote:
    print("You are eligible to vote.")
else:
    print("You are not yet eligible to vote.")
    
    
#---------------------------------------------------------------------------Logical Operators------------------------------------------------------------------------------------

#Logical Operators
#Logical operators are used to combine conditional statements:

#  Operator	    Description	                                                Example	
#  and 	        Returns True if both statements are true	                x < 5 and  x < 10	
#  or	        Returns True if one of the statements is true	            x < 5 or x < 4	
#  not	        Reverse the result, returns False if the result is true	    not(x < 5 and x < 10)


#Examples

#Use of and operator
#1. Checking a Number Range
#This is a classic use case. You check if a number is between two other numbers.



age = 25

# Checks if age is greater than 18 AND less than 65
if age > 18 and age < 65:
    print("You are in the working-age group.")
else:
    print("You are not in the working-age group.")


#2. Validating User Credentials
#You need both a correct username and a correct password to log in.

username_is_correct = True
password_is_correct = False

if username_is_correct and password_is_correct:
    print("Login successful!")
else:
    print("Invalid credentials.")


#3. Checking Multiple Permissions
#This is useful for checking if a user has all the necessary rights to perform an action.

is_admin = True
is_active_user = True

if is_admin and is_active_user:
    print("Displaying Admin Dashboard...")
else:
    print("Access denied.")
    
    
#use of or operator

#1. Checking for One of Two Conditions
#This is great for allowing multiple "correct" inputs or statuses.

# Use case: A weekend day check
day = "Sunday"

if day == "Saturday" or day == "Sunday":
    print("It's the weekend!")
else:
    print("It's a weekday.")


#2. Checking for Guest or Admin Access
#A user can access a resource if they are an admin or if they are the owner of the post.

is_admin = False
is_post_owner = True

if is_admin or is_post_owner:
    print("You can edit this post.")
else:
    print("You do not have permission to edit.")


#3. Checking for Multiple Error Codes
#If any one of several error codes appears, you run the same error-handling logic.

status_code = 404 # "Not Found"

if status_code == 404 or status_code == 403: # "Forbidden"
    print("Error: Page not found or access denied.")
else:
    print("Page loaded successfully.")
    
    
#use of not operator

#1. Checking if a Condition is False
#This makes the code read more like plain English.


is_logged_in = False

# 'not is_logged_in' is True because is_logged_in is False
if not is_logged_in:
    print("Please log in to continue.")


#2. Checking for an Empty List (or String)
#In Python, empty things (like [] or "") are "falsy." Using not is a clean, "Pythonic" way to check if a list is empty.


my_list = []

# An empty list evaluates to False, so 'not my_list' is True
if not my_list:
    print("The list is empty.")
else:
    print(f"The list has {len(my_list)} items.")


#3. Toggling a Boolean Value
#This is a common pattern for turning something on or off, like a light switch.

light_is_on = True
print(f"Light is on: {light_is_on}")

# Invert the value
light_is_on = not light_is_on

print(f"Light is on: {light_is_on}")



#----------------------------------------------------------------------------------Identify Operators-----------------------------------------------------------------------------

#Identity Operators
#Identity operators are used to compare the objects, not if they are equal, but if they are actually the same object, with the same memory location: 


# Operator	      Description	                                           Example	
#  is 	          Returns True if both variables are the same object	   x is y	
#  is not	      Returns True if both variables are not the same object   x is not y



#Example
#Get your own Python Server
#The is operator returns True if both variables point to the same object:

x = ["apple", "banana"]
y = ["apple", "banana"]
z = x

print(x is z)
print(x is y)
print(x == y)

#Example
#The is not operator returns True if both variables do not point to the same object:

x = ["apple", "banana"]
y = ["apple", "banana"]

print(x is not y)



#Difference Between is and ==
#is - Checks if both variables point to the same object in memory
#== - Checks if the values of both variables are equal


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------Membership Operators---------------------------------------------------------------------------------


# Operator	         Description	Example	
#  in 	             Returns True if a sequence with the specified value is present in the object	x in y	
#  not in	         Returns True if a sequence with the specified value is not present in the object	x not in y


#Example
#Get your own Python Server
#Check if "banana" is present in a list:

fruits = ["apple", "banana", "cherry"]

print("banana" in fruits)

#Example
#Check if "pineapple" is NOT present in a list:

fruits = ["apple", "banana", "cherry"]

print("pineapple" not in fruits)


#Membership in Strings
#The membership operators also work with strings:

#Example
text = "Hello World"

print("H" in text)
print("hello" in text)
print("z" not in text)

#Examples 

#in Operator
#Checks if a value is present in a sequence (like a list, string, or dictionary).

#1. Checking an Item in a List#
#This is the most common use. You check if an item exists within a list.

# Use case: Checking for permissions

allowed_users = ["Alice", "Bob", "Charlie"]
user = "Bob"

if user in allowed_users:
    print("Access granted.")
else:
    print("Access denied.")


#2. Checking a Substring in a String
#This checks if a sequence of characters exists inside another string. It is case-sensitive.

# Use case: Simple keyword detection
email_subject = "Important: Urgent meeting required"

if "Urgent" in email_subject:
    print("This email is marked as urgent.")


#3. Checking a Key in a Dictionary
# When used on a dictionary, in checks if the value is one of the keys (not the values).

# Use case: Checking if a setting exists in a config
user_settings = {
    "theme": "dark",
    "font_size": 14
}

# This checks for the KEY "theme"
if "theme" in user_settings:
    print(f"User's theme is: {user_settings['theme']}")



#not in Operator
#Checks if a value is not present in a sequence. It is the exact opposite of in.

#1. Checking if an Item is Missing from a List
#This is useful for filtering or validation.

# Use case: Validating user input
banned_words = ["spam", "advert"]
user_comment = "This is a great post!"

if user_comment not in banned_words:
    print("Comment is clean and can be posted.")
else:
    print("Comment contains banned words.")


#2. Checking for a Missing Substring
#This confirms that a specific string is absent.

# Use case: Checking for a valid file extension
filename = "document.pdf"

if ".exe" not in filename:
    print("File does not appear to be an executable.")
else:
    print("Warning: Executable file detected.")


#3. Checking for a Missing Key in a Dictionary
#This is commonly used to set a default value if one doesn't already exist.

# Use case: Setting a default value
user_prefs = {"username": "Alice"}

# This checks if the KEY "language" is missing
if "language" not in user_prefs:
    user_prefs["language"] = "en" # Set a default

print(f"User preferences: {user_prefs}")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------Bitwise Operators--------------------------------------------------------------------------------------
#Bitwise operators
#Bitwise operators are used to compare (binary) numbers:


#  Operator	  Name	                    Description	                                                                                              Example	
#   & 	      AND	                    Sets each bit to 1 if both bits are 1	                                                                   x & y	
#   |	      OR	                    Sets each bit to 1 if one of two bits is 1	                                                               x | y	
#   ^	      XOR	                    Sets each bit to 1 if only one of two bits is 1	                                                           x ^ y	
#   ~	      NOT	                    Inverts all the bits	                                                                                    ~x	
#   <<	      Zero fill left shift	    Shift left by pushing zeros in from the right and let the leftmost bits fall off	                       x << 2	
#   >>	      Signed right shift	    Shift right by pushing copies of the leftmost bit in from the left, and let the rightmost bits fall off	   x >> 2



#Example
#Get your own Python Server
#The & operator compares each bit and set it to 1 if both are 1, otherwise it is set to 0:

print(6 & 3)

#The binary representation of 6 is 0110
#The binary representation of 3 is 0011

#Then the & operator compares the bits and returns 0010, which is 2 in decimal.

#Example
#The | operator compares each bit and set it to 1 if one or both is 1, otherwise it is set to 0:

print(6 | 3)
#The binary representation of 6 is 0110
#The binary representation of 3 is 0011

#Then the | operator compares the bits and returns 0111, which is 7 in decimal.

#Example
#The ^ operator compares each bit and set it to 1 if only one is 1, otherwise (if both are 1 or both are 0) it is set to 0:

print(6 ^ 3)
#The binary representation of 6 is 0110
#The binary representation of 3 is 0011

#Then the ^ operator compares the bits and returns 0101, which is 5 in decimal.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------Operator Presedence-----------------------------------------------------------------------------


# Precedence Order
# The precedence order is described in the table below, starting with the highest precedence at the top:

#   Operator	                                          Description	
#    ()	                                                  Parentheses	
#    **	                                                  Exponentiation	
#    +x  -x  ~x	                                          Unary plus, unary minus, and bitwise NOT	
#    *  /  //  %                                          Multiplication, division, floor division, and modulus	
#    +  -	                                              Addition and subtraction	
#    <<  >>	                                              Bitwise left and right shifts	
#    &	                                                  Bitwise AND	
#    ^	                                                  Bitwise XOR	
#    |	                                                  Bitwise OR	
#    ==  !=  >  >=  <  <=  is  is not  in  not in 	      Comparisons, identity, and membership operators	
#    not	                                              Logical NOT	
#    and	                                              AND	
#    or	                                                  OR


#Left-to-Right Evaluation
#If two operators have the same precedence, the expression is evaluated from left to right.

#Example
#Addition + and subtraction - has the same precedence, and therefore we evaluate the expression from left to right:

print(5 + 4 - 7 + 3)


#Examples

#1. Multiplication (*) before Addition (+)
#This is the most common example, just like in standard math (PEMDAS). Multiplication and division are always a higher priority than addition and subtraction.

# 5 * 2 is calculated FIRST (10)
# Then 10 + 10 is calculated
result = 10 + 5 * 2

print(f"Result: {result}")
# Output: Result: 20 (not 30)


#2. and before or
#In logical operations, and is evaluated before or. This can be a common source of bugs if you're not careful.

# (False and False) is evaluated FIRST (False)
# Then True or False is evaluated
result = True or False and False

print(f"Result: {result}")
# Output: Result: True 
# (A common mistake is thinking it's (True or False) first, which would be False)

#3. Parentheses () to Force Order
#Parentheses () have the highest precedence of all. You can use them to override any default behavior and force your code to evaluate a specific part first.

# Without parentheses (from Example 1):
result_a = 10 + 5 * 2
print(f"Default: {result_a}") # Output: 20

# With parentheses:
# (10 + 5) is calculated FIRST (15)
# Then 15 * 2 is calculated
result_b = (10 + 5) * 2

print(f"Forced: {result_b}")
# Output: Forced: 30

