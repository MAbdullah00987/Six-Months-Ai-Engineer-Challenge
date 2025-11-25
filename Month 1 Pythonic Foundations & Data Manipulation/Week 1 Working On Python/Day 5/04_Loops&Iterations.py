
#What are Loops and Iteration? 🌀
#A loop is a programming structure that repeats a block of code over and over.

#An iteration is a single one-time pass through that block of code.

#Think of it like this: A loop is like having a "To-Do" list. Iteration is the act of doing the first task on the list. The loop is the whole process of going through every single task until the list is done.

#Why We Need Loops
#The main reason we use loops is automation. They let us perform repetitive tasks without repeating our code. This follows the DRY principle: Don't Repeat Yourself.

#Here are the two main reasons you'll use a loop:

#1. To Do Something a Specific Number of Times
#When you know exactly how many times you want to do something, you use a for loop.

#Example: Imagine you need to print "Hello" 3 times.
#
#❌ Without a loop (The "Bad" Way): You have to copy and paste your code. This is inefficient and hard to change.


print("Hello")
print("Hello")
print("Hello")
#✅ With a for loop (The "Good" Way): You tell Python to run the print command 3 times. If you need to change it to 100 times, you only change one number.



for i in range(3):
    print("Hello")

#2. To Do Something Until a Condition is Met
#When you don't know how many times you need to loop, but you know when to stop, you use a while loop.

#Example: Imagine you need to keep asking a user for a password until they get it right.

#❌ Without a loop: This is impossible. You don't know if the user will get it right on the 1st try or the 10th.

#✅ With a while loop: The loop will continue running as long as the password is wrong.


password = "" # Start with an empty password

while password != "secret":
    password = input("What is the password? ")

# This line only runs AFTER the loop is finished
print("Welcome in!")



#Examples of for Loops
#A for loop is used when you know how many times you want to loop, like "for every item in this list."
#1. Looping Over a List
#This is the most common use. It performs an action for each item in a list.

skills = ["Python", "Git", "AI"]

for skill in skills:
    print(f"I am learning: {skill}")

#2. Looping a Specific Number of Times
#Use the range() function to loop a set number of times.
# range(5) gives you numbers 0, 1, 2, 3, 4
for i in range(5):
    print(f"This is loop number: {i}")


#3. Looping Over a String
#You can treat a string as a list of characters.

for char in "Hello":
    print(char)


#4. Looping Over a Dictionary
#Use the .items() method to get both the key and the value from a dictionary.

user_profile = {
    "name": "Hassan",
    "skill": "Python"
}

for key, value in user_profile.items():
    print(f"{key}: {value}")


#5. Looping with range() (Start, Stop, Step)
#You can give range() a start, a stop (it stops before this number), and a "step" (how much to count by).

# Counts down from 10 to 2, by 2s
for num in range(10, 0, -2):
    print(f"Countdown: {num}")


#Examples of while Loops 
#A while loop is used when you don't know how many times you need to loop. It continues running as long as a certain condition is True.

#1. A Simple Counter
#This is the while loop version of for i in range(5):.

count = 0
while count < 5:
    print(f"Count is: {count}")
    count += 1  # Don't forget this, or it's an infinite loop!

#2. User Input Validation
#This is a classic example. The loop keeps asking for input until the user gives a valid answer.

password = ""
while password != "secret":
    password = input("Enter the password: ")

print("Welcome in!")

#3. Running Until a Condition Changes
#This is common in games. The main game loop runs as long as the player is alive.

stamina = 10
while stamina > 0:
    print("You swing your sword.")
    stamina -= 1 # You get a little more tired

print("You are out of stamina!")

#4. Processing Items from a List
#You can use a while loop to process items in a list until the list is empty.

tasks = ["Task 1", "Task 2", "Task 3"]
while tasks:  # This is True as long as the list is not empty
    task = tasks.pop(0) # .pop(0) removes and returns the first item
    print(f"Processing: {task}")

print("All tasks are done!")

#5. Simple Guessing Game
#The loop continues as long as the guess is wrong.

secret_number = 7
guess = 0 # Initialize guess to a non-secret number

while guess != secret_number:
    guess = int(input("Guess a number between 1 and 10: "))

print(f"You got it! The number was {secret_number}.")

#How to Use Statements Inside Loops 
#You almost always use statements inside loops to make them useful. The most common are if, break, and continue.
#1. if (To Make Decisions)
#Use if to do different things inside the loop for different items.

# Use case: Find all even numbers in a list
numbers = [1, 2, 3, 4, 5, 6, 7]

for num in numbers:
    # Use an 'if' statement to check a condition
    if num % 2 == 0:
        print(f"Found an even number: {num}")

#2. break (To Stop the Loop Early)
#Use break to exit the loop immediately, even if the main condition is still True.

# Use case: Stop searching a list once you find what you need
numbers = [1, 5, 12, 45, 100, 3]

for num in numbers:
    print(f"Checking: {num}")
    if num == 12:
        print("Found 12! Stopping the loop.")
        break  # This exits the 'for' loop

#3. continue (To Skip an Iteration)
#Use continue to stop the current iteration and immediately jump to the next item in the loop.

# Use case: Process only valid data and skip invalid entries
data = [10, 20, "INVALID", 30, "SKIP", 40]

for item in data:
    if not isinstance(item, int):
        continue  # Skips this item and goes to the next
    
    # This line will only run if the 'continue' was not hit
    print(f"Processing data: {item}")
    
    
#definite loop is a loop where you know exactly how many times it will run before it starts.
#The for loop is Python's definite loop. The most common example is looping over a list:

# The list 'skills' has 3 items.
# We know this loop will run exactly 3 times.
skills = ["Python", "Git", "AI"]

print("--- Starting Loop ---")
for skill in skills:
    print(f"I am learning: {skill}")

print("--- Loop Finished ---")


#Find the largest value

#Example 1: Finding the Largest Number in a List with Mixed Data
#This is "complex" because the list contains data that isn't a number. We must use if statements to check the type of data before we can compare it.
#We'll initialize largest_so_far to None. This helps us handle the first number we find.

data_list = ["apple", 25, 10, "banana", 99, 15.5, "grape", 50]
largest_so_far = None

print(f"Checking list: {data_list}")

for item in data_list:
    # 1. Skip non-numeric data
    if not isinstance(item, (int, float)):
        print(f"Skipping '{item}' (not a number)")
        continue # Goes to the next item

    # 2. Handle the very first number we find
    if largest_so_far is None:
        largest_so_far = item
        print(f"First number found: {largest_so_far}")
    
    # 3. Compare the current item to our champion
    elif item > largest_so_far:
        print(f"Found new largest: {item} (was {largest_so_far})")
        largest_so_far = item

print(f"\nFinished! The largest number is: {largest_so_far}")

#Example 2: Finding the Student with the Highest Score
#This is "complex" because we are looping through a list of dictionaries. We need to find the student who has the highest score.
#We'll track two variables: highest_score and top_student.


students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78},
    {"name": "David", "score": 95},
    {"name": "Eve", "score": 89}
]

# We can initialize by assuming the first student is the winner
top_student = students[0]["name"]
highest_score = students[0]["score"]

print(f"Checking student scores...")
print(f"Starting with {top_student} (Score: {highest_score})")

# We loop through the rest of the students (skipping the first one)
for student in students[1:]:
    current_name = student["name"]
    current_score = student["score"]
    
    # Compare the current student's score to our champion's score
    if current_score > highest_score:
        print(f"New top student! {current_name} beat {top_student} with {current_score}")
        # Update both champion variables
        highest_score = current_score
        top_student = current_name
    else:
        print(f"Checking {current_name}... (Score: {current_score}, not higher than {highest_score})")

print(f"\nFinished! The top student is: {top_student} with a score of {highest_score}")
    
    
    
    