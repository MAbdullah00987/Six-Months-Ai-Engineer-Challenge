
#What is type conversion
#Type conversion (also known as "type casting") is the process of changing a value from one data type to another.
#Type conversion functions are the built-in Python functions that let you explicitly (manually) force a value to change its type. The most common ones are:
#int(): Converts a value to an integer (a whole number).
#float(): Converts a value to a float (a number with a decimal).
#str(): Converts a value to a string (text).
#list(): Converts a value (like a string or tuple) into a list.

#Practical Examples

#1. Example: Converting String (str) to Integer (int)
#This is most common when getting input from a user, because the input() function always gives you a string.

# The input() function gives us a string, even if the user types a number
user_age_str = input("What is your age? ") # Let's say user types "30"

# This would cause an error: print(user_age_str + 5)
# You can't add a number (5) to text ("30")
# --- We use int() to convert it ---
user_age_int = int(user_age_str)

# Now we can do math!
age_in_five_years = user_age_int + 5

print(f"You will be {age_in_five_years} in five years.")
#Function: int()
#Conversion: "30" (str) → 30 (int)

#. Example: Converting Integer (int) to String (str)
#This is most common when you want to join a number with text (called "concatenation").

score = 100
message = "Your score is: "

# This would cause an error: print(message + score)
# You can't "add" text ("Your score is: ") to a number (100)

# --- We use str() to convert it ---
score_str = str(score)

# Now we can join the two strings
full_message = message + score_str

print(full_message)

# Note: Using an f-string (like in the first example)
# does this conversion for you automatically!
# print(f"Your score is: {score}") is the modern shortcut.
#Function: str()

#Conversion: 100 (int) → "100" (str)

#3. Example: Converting Float (float) to Integer (int)
#This is used when you want to get just the whole number part of a decimal. Note: It truncates (cuts off) the decimal; it does not round.


price = 49.95 # This is a float
points_earned = 135.8 # This is also a float

# We only want to see the main dollar amount
dollar_amount = int(price)
print(f"The dollar amount is: {dollar_amount}")

# We only want to give whole points
whole_points = int(points_earned)
print(f"Points added: {whole_points}")


