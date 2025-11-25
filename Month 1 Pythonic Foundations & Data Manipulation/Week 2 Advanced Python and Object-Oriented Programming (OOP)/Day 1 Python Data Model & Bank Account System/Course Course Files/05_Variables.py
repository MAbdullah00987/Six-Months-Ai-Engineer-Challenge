

#What Are Variables?
#A variable is a container for storing a value. Think of it as a labeled box where you can put information. You can then refer to that information just by using the box's label (the variable name).

#You create a variable by giving it a name and using the equals sign (=) to assign it a value.
#The value inside the "box" can be changed or updated later.

#Rules for Python Variable Names
#This is how to perfectly use names (i.e., follow the rules so your code doesn't break).
#Must start with a letter (a-z, A-Z) or an underscore (_).

#Good: name, _internal_var
#Bad: 1name (cannot start with a number)
#Can only contain letters, numbers, and underscores (A-z, 0-9, and _).
#Good: user_name_1
#Bad: user-name (no dashes), user@ (no special symbols)
#Names are case-sensitive.
#age, Age, and AGE are three different variables.
#You cannot use a Python keyword.
#Bad: for = 5 (this will cause an error)
#Bad: class = "History" (this will also cause an error)

#Python Keywords

#Keywords (or "reserved words") are the special, protected words that have a specific meaning in the Python language. You cannot use them as variable names, function names, or any other identifier.

# | False   | await    | else    | import   | pass
# | None    | break    | except  | in       | raise
# | True    | class    | finally | is       | return
# | and     | continue | for     | lambda   | try
# | as      | def      | from    | nonlocal | while
# | assert  | del      | global  | not      | with
# | async   | elif     | if      | or       | yield


#How to Choose the Right Variable Name (Best Practices)
#Following the rules above makes your code work. Following these best practices makes your code readable and easy for you and others to understand.

#Be Descriptive: The name should instantly tell you what's inside.
#Bad: x = 10
#Good: item_count = 10
#Use snake_case: This is the official Python style guide (PEP 8) for variable names. It means you separate words with an underscore (_).
#Bad: totalScore (this is "camelCase," used in other languages)
#Good: total_score
#Avoid Single Letters: Don't use l, O, or I as they can be confused with numbers (1, 0) or each other. The only common exception is for simple loop counters (e.g., for i in range(5):).
#Bad: u = "Ali"
#Good: user_name = "Ali"


#Three Examples of Good vs. Bad Names

#Example 1: Storing a User's Name
#Bad: s = "Sara" (Not descriptive)
#Bad: UserName = "Sara" (Uses "PascalCase," which is reserved for Classes)
#Good: user_name = "Sara" (Descriptive and uses snake_case)

#Example 2: Counting Active Users
#Bad: au = 150 (Too short, what is "au"?)
#Bad: activeusers = 150 (Hard to read)
#Good: active_user_count = 150 (Clear, descriptive, and readable)
#Example 3: A List of Products

#Bad: l = ["apple", "banana"] ('l' is a terrible name)
#Bad: list = ["apple", "banana"] (This is very bad! list is a Python keyword/function name. Using it as a variable name will break things.)
#Good: product_list = ["apple", "banana"] (Clear and doesn't conflict with Python's built-in names)

#1.  Storing User Profile Information
#Variables are perfect for holding information about a user in an application.

# Variables storing different data types for a user profile
user_name = "aisha_k"
first_name = "Aisha"
age = 22
is_verified = True  # Has this user confirmed their email?
login_count = 14

# --- How they are used ---
print(f"Welcome back, {first_name}!")

if is_verified:
    print("Your account is verified.")
else:
    print("Please verify your email address.")

print(f"This is your {login_count}th login.")