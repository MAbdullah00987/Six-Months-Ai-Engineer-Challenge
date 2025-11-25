

#A function is a reusable, named block of code that performs a specific task.
#You need to use functions for three main reasons:
#Reusability (DRY Principle): "Don't Repeat Yourself." If you find yourself writing the same piece of code over and over, you can put it inside a function.
# Then, you just "call" the function by its name whenever you need it, instead of retyping all the code.
#Organization: Functions break down a large, complex program into smaller, manageable pieces. It's much easier to read and understand a program that says calculate_average()
# or print_report() than to read one giant, 500-line script.
#Abstraction: A function hides the complex details of how it does something. You just need to know what it does. When you call print("Hello"), you don't need to know how Python 
# makes the letters appear on the screen; you just trust that the print function will do its job.

#why we call functionsa reusable and also iff so how reuse any complex code without getting any errir to it
#I ask this question to gemini ai

#Why Functions are Reusable 🚀
#Functions are "reusable" because you write the complex logic once, give it a name, and then you can call (or "run") that entire block of code as many times as you want, just by using its name.

#This follows the DRY principle: Don't Repeat Yourself.

#Before Functions: If you need to greet a user 3 times, you write the code 3 times.

#With Functions: You write the "greet" code once (inside a function) and just call the function 3 times.

#Think of it like a recipe. You write the recipe for "Bake a Cake" once. You don't have to rewrite the entire recipe every single time you want a cake. You just "call" the recipe and follow it.

#How to Reuse Complex Code Without Errors
#This is the key to good programming. You make your function reliable by building it like a "black box."

#A black box has clear inputs and clear outputs, and you don't have to worry about the complex machinery inside.

#To build a good "black box" function, you must follow three rules:

#1. Give It Clear Inputs (Parameters)
#A reliable function gets all the data it needs from its parameters (the variables in the parentheses). It should not depend on global variables outside the function.

#2. Give It Clear Outputs (Return Values)
#A reusable function should return its result. It should not just print() the result. This lets the code that called the function decide what to do with the result (print it, save it, pass it to another function, etc.).

#3. Make It Do One Thing (Single Responsibility)
#A function should do one job and do it well. Don't write one giant function called get_user_and_process_data_and_save_to_file().

#When you follow these rules, you can reuse the function with any data, and you'll always know what to expect.

#Example: Bad vs. Good Reusable Function
#Here is a perfect example of how to make complex code reusable.

#❌ The "Bad" Way (Not Reusable)
#This function is complex and will cause errors.

#It relies on a global variable user_data. What if that variable doesn't exist?

#It prints the result instead of returning it, so you can't use the total for anything else.


# BAD: Not reusable
user_data = [10, 20, 5] # This is a global variable

def calculate_total():
    # This complex logic is tied to a specific variable
    # that exists OUTSIDE the function.
    
    total = sum(user_data)
    total = total + (total * 0.05) # Add 5% tax
    print(f"The total is: {total}") # Just prints, doesn't return

# This works ONCE...
calculate_total()

# ...but what if 'user_data' changes or has a different name?
# The function is broken. It's not reusable.


#The "Good" Way (Reusable and Safe)
#This function is a "black box." It's complex inside, but it's safe, predictable, and 100% reusable.

#It gets its data from a parameter (item_prices).

#It returns the final value.

# GOOD: Reusable and safe
def calculate_total_with_tax(item_prices):
    """
    Takes a list of prices and returns the total
    with a 5% tax added.
    """
    # This complex logic is self-contained.
    # It ONLY uses the 'item_prices' list it was given.
    
    subtotal = sum(item_prices)
    tax = subtotal * 0.05
    total = subtotal + tax
    
    return total # Returns the result

# --- Now we can reuse it with ANY data, without errors ---

# Use it for one list
list1 = [10, 20, 5]
total1 = calculate_total_with_tax(list1)
print(f"List 1 Total: {total1}")

# Reuse it for a different list
list2 = [150, 300, 50.75]
total2 = calculate_total_with_tax(list2)
print(f"List 2 Total: {total2}")

#By making the function a "black box" with clear inputs (parameters) and outputs (return values), you can reuse your complex code infinitely without worrying about it causing errors.

#Examples About Functions

#1. Simple Function (No Inputs, No Output) 🛠️
#This type of function is like a "shortcut" for a task. It doesn't take any input or send any value back; it just performs an action.

# 1. Define the function
def greet():
    """This function just prints a greeting."""
    print("Hello! Welcome.")
    print("This is a reusable function.")

# 2. Call the function to make it run
print("--- Calling the function ---")
greet()

#2. Function with Inputs (Parameters) and Output (Return Value) 🔢
#This is the most common type. It takes data in (as parameters), processes it, and sends a result back using the return keyword.


# 'num1' and 'num2' are parameters (inputs)
def add_numbers(num1, num2):
    """Takes two numbers and returns their sum."""
    total = num1 + num2
    return total  # Sends the 'total' value back

# Call the function and store its return value in a variable
sum_result = add_numbers(5, 10)

print(f"The result is: {sum_result}")
print(f"8 + 3 is: {add_numbers(8, 3)}")

#3. Function with Logic (Decision-Making) ⚖️
#This function takes an input and uses if/else logic to decide what to return. This connects functions with the conditional steps you learned.


def check_age_to_vote(age):
    """Checks an age and returns a status string."""
    if age >= 18:
        return "Eligible to vote."
    else:
        return "Not eligible to vote."

# Call the function and store the result
status1 = check_age_to_vote(25)
status2 = check_age_to_vote(16)

print(f"A 25-year-old is: {status1}")
print(f"A 16-year-old is: {status2}")


#This function takes an age (as a string) and calculates the year the person was born.

def find_birth_year(age_as_string):
    """
    Takes an age (as a string) and returns the birth year.
    This function must convert the string to an integer.
    """
    try:
        # 1. This is the Type Conversion
        # We convert the string to an integer to do math
        age_as_int = int(age_as_string)
        
        # 2. Now we can use the integer in a calculation
        current_year = 2025 # (Assuming the current year is 2025)
        birth_year = current_year - age_as_int
        
        return birth_year
        
    except ValueError:
        # This 'except' block runs if the conversion in int() fails
        return "Error: You did not enter a valid number."

# --- How We Use the Function ---

# 1. Get user input. This will be a string (e.g., "30")
age_from_user = input("How old are you? ")

# 2. Call the function, passing in the string
calculated_year = find_birth_year(age_from_user)

# 3. Print the result
print(f"You were born in or around: {calculated_year}")


#This function takes a username and an integer ID and combines them into a single string.

def create_user_tag(username, user_id):
    """
    Takes a username (str) and a user_id (int)
    and returns a combined tag (str).
    """
    
    # 1. This is the Type Conversion
    # We convert the integer 'user_id' to a string
    # so we can join it with other strings.
    id_as_string = str(user_id)
    
    # 2. Now we can safely combine the strings
    # This would fail if user_id was still an integer:
    # "Error: can only concatenate str (not "int") to str"
    tag = username + "_" + id_as_string
    
    return tag

# --- How We Use the Function ---

# 1. Define our variables
user_name = "hassan"
user_number = 101

# 2. Call the function
user_tag = create_user_tag(user_name, user_number)

# 3. Print the result
print(f"The new user tag is: {user_tag}")
print(f"The type of the new tag is: {type(user_tag)}")

