

#IN and NOT IN Operators
#Definition of IN Operator
#The in operator is a membership operator that checks whether a specific value exists within a sequence (like a list, tuple, string, set, or dictionary). It returns True if the value is found, and False if it is not found.
#Definition of NOT IN Operator
#The not in operator is the opposite of the in operator. It checks whether a specific value does NOT exist within a sequence. It returns True if the value is not found, and False if the value is found.

#Examples of IN Operator
#Example 1: Checking if a fruit is in a list
pythonfruits = ["apple", "banana", "mango", "orange"]
result = "banana" # in fruits
print(result)  # Output: True

#Example 2: Checking if a letter is in a string
pythonmessage = "Hello World"
result = "o" # in message
print(result)  # Output: True

#Example 3: Checking student enrollment
pythonenrolled_students = ["Ali", "Sara", "Ahmed", "Fatima"]
student_name = "Sara"
is_enrolled = student_name  #in enrolled_students
print(is_enrolled)  # Output: True

#Example 4: Checking if a number is in a tuple
pythonlucky_numbers = (7, 13, 21, 42)
number = 21
is_lucky = number  #in lucky_numbers
print(is_lucky)  # Output: True

#Example 5: Checking if a key exists in a dictionary
pythonstudent_grades = {"Ali": 85, "Sara": 92, "Ahmed": 78}
result = "Sara"  #in student_grades
print(result)  # Output: True

#Examples of NOT IN Operator
#Example 1: Checking if a fruit is NOT in a list
pythonfruits = ["apple", "banana", "mango", "orange"]
result = "grape"  #not in fruits
print(result)  # Output: True

#Example 2: Checking if a character is NOT in a string
pythonpassword = "MySecurePass123"
has_no_space = " " # not in  password
print(has_no_space)  # Output: True

#Example 3: Blocked users check
pythonblocked_users = ["user123", "spammer99", "fake_account"]
current_user = "john_doe"
can_post = current_user # not in blocked_users
print(can_post)  # Output: True

#Example 4: Checking if a number is NOT in a range (using list)
pythonvalid_ages = [18, 19, 20, 21, 22]
age = 25
is_invalid_age = age #not in valid_ages
print(is_invalid_age)  # Output: True

#Example 5: Checking banned words in a comment
pythonbanned_words = ["spam", "hate", "offensive"]
user_comment = "This is a nice picture"
is_clean = "spam" not in user_comment and "hate" not in user_comment
print(is_clean)  # Output: True

#Practical Combined Example

# Restaurant menu system
menu_items = ["burger", "pizza", "pasta", "salad", "fries"]
order = "sushi"

if order in menu_items:
    print(f"{order} is available. Order placed!")
else:
    print(f"Sorry, {order} is not on our menu.")

# Output: Sorry, sushi is not on our menu.

# Checking for allergies
allergens_in_dish = ["peanuts", "dairy", "gluten"]
customer_allergies = ["shellfish", "soy"]

is_safe = "peanuts" not in customer_allergies
print(f"Dish is safe: {is_safe}")  # Output: Dish is safe: True

