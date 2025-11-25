
#Debugging is the process of finding and fixing errors (or "bugs") in your computer code.

#A "bug" can be one of three main types:
#Syntax Errors: The code is "written" incorrectly. You broke a grammar rule of the Python language (e.g., misspelled a keyword, missed a colon :). 
#Runtime Errors: The code's syntax is correct, but it fails "during" the program's execution (e.g., trying to divide by zero, or trying to use a variable that doesn't exist yet).
#Logical Errors: The code runs perfectly, but it does the wrong thing. This is the hardest to find. For example, you told it to add price + tax but you meant to write price * (1 + tax). The computer did exactly what you said, but what you said was wrong.


#1. Syntax Error (Grammar Error)
#This is an error in the "grammar" of the code. 

# We forgot the colon ':' at the end of the line
age = 30
if age > 18 :
    print("You are an adult.")
#Error: SyntaxError: expected ':'

#Why it fails: The if statement in Python must end with a colon (:). Because the grammar rule was broken, Python can't execute the file.

#2. Runtime Error (Crash Error)
#This is an error that happens while the program is running. The syntax was correct, but the code was asked to do something impossible.

# The syntax is correct, but the math is impossible
numerator = 100
denominator = 0

result = numerator / denominator

print(result)
#Error: ZeroDivisionError: division by zero

#Why it fails: The code runs successfully until it hits the division line. It's mathematically impossible to divide by zero, so the program "crashes" and raises a ZeroDivisionError.

#3. Semantic (Logical) Error (Wrong Result)
#This is the most dangerous type of error. The syntax is correct, the program runs without crashing, but the result is wrong. The code did exactly what you told it to do, but what you told it was not what you meant to do.


# GOAL: Find the average of three scores
score1 = 80
score2 = 90
score3 = 70

# LOGICAL ERROR: We found the sum, but forgot to divide by 3
average = score1 + score2 + score3

print(f"The average score is: {average}")
#Output: The average score is: 240
#Why it fails: The program ran perfectly, but 240 is not the average. The logic was wrong. The correct code should have been average = (score1 + score2 + score3) / 3. Debugging this requires you to notice that the output is not what you expected.


