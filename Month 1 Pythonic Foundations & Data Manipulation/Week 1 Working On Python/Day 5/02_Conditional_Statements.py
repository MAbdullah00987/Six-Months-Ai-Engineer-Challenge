

#-----------------------------------------------------------------------Conditional Statements in Python------------------------------------------------------------------------

#Conditional steps, more commonly known as conditional statements, are how your Python program makes decisions.
#They allow you to run a specific block of code only if a certain condition is True. The main keywords for this
# are if, elif, and else.

#1. if (The Primary Condition)
#This is the main check. If the condition is True, the code indented under it runs. If it's False, the code is skipped.

age = 20
# This is the condition
if age >= 18:
    # This block only runs if 'age' is 18 or more
    print("You are eligible to vote.") 

#2. elif (The "Else If" Condition)
#This is an optional second check. It only runs if the first if statement was False, and this new condition is True. You can have as many elif statements as you need.


temperature = 15

if temperature > 25:
    print("It's hot.")
elif temperature > 10:
    # This runs because 15 > 10 (and the first 'if' was False)
    print("It's a pleasant day.") 
elif temperature > 0:
    print("It's cold.")

#3. else (The "Catch-All" Condition)
# This is an optional final block. It runs if all of the preceding if and elif conditions were False. It's the default action if no other condition is met.

user_command = "save"

if user_command == "run":
    print("Running the program...")
elif user_command == "quit":
    print("Quitting...")
else:
    # This runs because 'user_command' was not 'run' or 'quit'
    print("Unknown command.")
    
    
#Comparison operators:

x = 5
if x == 5:  # Equal to
    print('Equals 5')
if x > 4:   # Greater than
    print('Greater than 4')
if x >= 5:  # Greater than or equal to
  print('Greater than or equal to 5')
if x < 6:   # Less than
    print('Less than 6')
if x <= 5:
    print('Less than or equal to 5')
if x != 6: # Not equal to
    print('Not equal to 6')
        
#One Way Decision
x = 5  # assign the value 5 to variable x

print('Before 5')  # print a message that comes before checking if x is 5

if x == 5:  # test whether x is equal to 5
    print('Is 5')  # this runs only if the condition (x == 5) is True
    print('Is still 5')  # another line that runs inside the same True branch
    print('Third 5')  # a third line inside the same True branch

print('Afterwords 5')  # this runs regardless of the previous if (it's outside the if block)
print('Before 6')  # print a message that comes before checking if x is 6

if x == 6:  # test whether x is equal to 6 (this will be False for x = 5)
    print('Is 6')  # would run only if x == 6 (not executed here)
    print('Is still 6')  # would also run only inside the True branch for x == 6
    print('Third 6')  # another conditional print for the x == 6 branch
    print('Afterwords 6')  # final line inside the x == 6 branch (not executed here)

#Two Way Decision

age = 17

# The 'if' statement is the decision point
if age >= 18:
    # Path 1: Runs if the condition (age >= 18) is True
    print("You are eligible to vote.")
else:
    # Path 2: Runs if the condition is False
    print("You are not old enough to vote.")
    
#Two Way decision with else statement

x = 4

if x > 2:
    print('Bigger than 2')  # This runs if x is greater than 2
else:
    print('smaller')
    
print('All done')  



    
        
