
#Logical Operators - Detailed Section
#Definition of Logical Operators
#Logical operators are symbols used to combine or modify conditional statements, allowing you to create complex boolean expressions. They evaluate multiple conditions and return a single boolean result (true or false).
#Three Simple Definitions:

#Logical operators connect multiple conditions to make decisions based on whether all, any, or none of the conditions are true.
#They are boolean operators that work with true/false values and return true or false as output.
#They enable complex decision-making by combining simple comparisons into more sophisticated logical expressions.


#All Types of Logical Operators
#1. AND Operator (&&, and)
#Returns true only if both conditions are true.
#Truth Table:

#True AND True = True
#True AND False = False
#False AND True = False
#False AND False = False

#Examples:
#python# Example 1: Checking age and license
#age = 20
#has_license = True
#can_drive = (age >= 18) and has_license  # True (both conditions are true)

# Example 2: Password validation
#password_length = 10
#has_special_char = False
#valid_password = (password_length >= 8) and has_special_char  # False

# Example 3: Shopping eligibility
#account_balance = 500
#item_price = 300
#in_stock = True
#can_purchase = (account_balance >= item_price) and in_stock  # True
#2. OR Operator (||, or)
#Returns true if at least one condition is true.
#Truth Table:

#True OR True = True
#True OR False = True
#False OR True = True
#False OR False = False

#Examples:
#python# Example 1: Weekend check
day = "Saturday"
is_weekend = (day == "Saturday") or (day == "Sunday")  # True

# Example 2: Payment methods
has_credit_card = False
has_cash = True
can_pay = has_credit_card or has_cash  # True

# Example 3: Emergency contact
is_parent = False
is_guardian = True
is_emergency_contact = is_parent or is_guardian  # True
#3. NOT Operator (!, not)
#Reverses the boolean value (true becomes false, false becomes true).
#Truth Table:

#NOT True = False
#NOT False = True

#xamples:
#python# Example 1: Access denied
is_logged_in = False
access_denied = not is_logged_in  # True

# Example 2: Opposite status
is_raining = True
is_sunny = not is_raining  # False

# Example 3: Availability check
is_busy = False
is_available = not is_busy  # True

#Combined Example Using All Three Logical Operators
#python# Student grade eligibility system
attendance = 85
exam_score = 72
submitted_project = True
is_suspended = False

# Complex condition using AND, OR, and NOT
passes_course = (
    (attendance >= 75 and exam_score >= 60) or submitted_project
) and not is_suspended

# Result: True (meets attendance and exam OR has project, AND is not suspended)


#Example: School Permission Slip
#python# A student can go on a field trip if they have permission AND paid the fee
has_permission = True
paid_fee = True

can_go_on_trip = has_permission and paid_fee
print(can_go_on_trip)  # Output: True

# If either is missing, they cannot go
has_permission = True
paid_fee = False
can_go_on_trip = has_permission and paid_fee
print(can_go_on_trip)  # Output: False

#Example: Movie Theater Entry
#python# You can watch an R-rated movie if you are 18+ OR have parent with you
age = 16
with_parent = True

can_watch_movie = (age >= 18) or with_parent
print(can_watch_movie)  # Output: True

# Without parent and under 18
age = 16
with_parent = False
can_watch_movie = (age >= 18) or with_parent
print(can_watch_movie)  # Output: False

#Example: Door Lock System
#python# Door opens if it's NOT locked
is_locked = False

door_opens = not is_locked
print(door_opens)  # Output: True

# If door is locked
is_locked = True
door_opens = not is_locked
print(door_opens)  # Output: False