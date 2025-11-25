
#What are functions?
#A function is a reusable block of code that performs a specific task. Think of it as a recipe: you define it once (write the recipe) and then you can "call" it (follow the recipe) whenever you need to, as many times as you want.
#A function call is the act of executing or running the code inside a function. You do this simply by writing the function's name followed by parentheses ().
#How to Call a Function
#To call a function, you just type its name and add parentheses: function_name().
#If the function needs information to do its job (these inputs are called arguments or parameters), you put that information inside the parentheses.

#Practical Examples

#Example 1: A Simple Function (No Inputs, No Output)
#This function just performs an action (printing to the screen). It doesn't need any information to do its job.

# 1. DEFINE the function (writing the "recipe")
def greet_user():
    print("Hello! Welcome to the program.")
    print("Have a great day.")

# 2. CALL the function (running the "recipe")
greet_user()

# You can call it again
greet_user()
#Function Call: greet_user()
#What it does: Runs the code inside greet_user, printing two lines to the screen.


#Example 2: A Function with Inputs (Arguments)
#This function needs data (an input) to work. We pass the data (like "Ali" or "Sara") inside the parentheses when we call it.

# 1. DEFINE the function
#    'name' is a parameter (a placeholder for the input)
def greet_person(name):
    print(f"Hi, {name}! It's nice to meet you.")

# 2. CALL the function with an argument
#    "Ali" is the argument (the actual value)
greet_person("Ali")

# Call it again with a different argument
greet_person("Sara")
#Function Call: greet_person("Ali")
#What it does: The value "Ali" is passed into the function and stored in the name variable. The function then uses that variable to print a personalized message.


#Example 3: A Function That Returns a Value (Has an Output)
#This function takes inputs, performs a calculation, and then uses the return keyword to send a value back to the code that called it.

# 1. DEFINE the function
def add_numbers(num1, num2):
    total = num1 + num2
    return total  # Sends the 'total' value back

# 2. CALL the function and store the result
#    We need a variable (like 'sum_result') to catch the returned value.
sum_result = add_numbers(5, 3)

print(f"The result of the addition is: {sum_result}")
print(f"10 + 15 is: {add_numbers(10, 15)}")
#Function Call: add_numbers(5, 3)

#What it does: It passes 5 and 3 into the function. The function calculates total = 8. The return total line sends the 8 back. The main code then stores that 8 in the sum_result variable.