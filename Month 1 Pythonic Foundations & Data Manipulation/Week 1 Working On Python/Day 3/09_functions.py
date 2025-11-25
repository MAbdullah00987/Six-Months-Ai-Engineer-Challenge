

#1. Function with *args (Variable Arguments)
#The *args syntax allows a function to accept any number of positional arguments. Inside the function, args becomes a tuple containing all the arguments that were passed in.
#This is complex because the function doesn't know in advance how many inputs it will get.
# Use case: A function that calculates the average of any amount of numbers
def calculate_average(*numbers):
    # 'numbers' is a tuple, e.g., (10, 20, 30)
    
    # Handle the edge case where no numbers are given
    if not numbers:
        return 0
        
    total = sum(numbers)
    count = len(numbers)
    return total / count

# Call the function with different numbers of arguments
print(f"Average of 10, 20, 30: {calculate_average(10, 20, 30)}")
print(f"Average of 5, 10: {calculate_average(5, 10)}")
print(f"Average of 1, 2, 3, 4, 5, 6, 7: {calculate_average(1, 2, 3, 4, 5, 6, 7)}")
#2. Function with **kwargs (Keyword Arguments)
#The **kwargs (keyword arguments) syntax allows a function to accept any number of named arguments. Inside the function, kwargs becomes a dictionary mapping the argument names (keywords) to their values.
#This is essential for AI and data science libraries, as it lets you pass optional settings.
# Use case: A function that builds a user profile
def build_profile(**user_info):
    # 'user_info' is a dictionary, e.g., {'name': 'Alice', 'age': 30}
    
    # .get() is a safe way to access a dictionary key
    # It returns None if the key doesn't exist, instead of crashing
    name = user_info.get('name')
    age = user_info.get('age')
    
    print(f"--- Creating Profile for: {name} ---")
    if age:
        print(f"Age: {age}")
        
    # Loop through any other information that was passed
    for key, value in user_info.items():
        if key not in ['name', 'age']:
            print(f"{key.title()}: {value}")

# Call the function with different named arguments
build_profile(name="Alice", age=30, location="New York")
print("-" * 20)
build_profile(name="Hassan", skills=["Python", "AI"], team="R&D")

#3. A Recursive Function
#A recursive function is one that calls itself to solve a problem. It's complex because you must define a base case (when the function stops) and a recursive step (when it calls itself with a smaller version of the problem).
# Use case: Calculating the factorial of a number (e.g., 5! = 5 * 4 * 3 * 2 * 1)
def factorial(n):
    # 1. Base Case: This is what stops the loop.
    # The factorial of 1 (or 0) is 1.
    if n == 1 or n == 0:
        return 1
    
    # 2. Recursive Step: The function calls itself with a smaller number
    # e.g., 5 * factorial(4)
    else:
        return n * factorial(n - 1)

# Let's trace factorial(4):
# 1. returns 4 * factorial(3)
# 2.   returns 3 * factorial(2)
# 3.     returns 2 * factorial(1)
# 4.       returns 1 (Base Case)
# Result: 4 * 3 * 2 * 1 = 24

result = factorial(5)
print(f"The factorial of 5 is: {result}")