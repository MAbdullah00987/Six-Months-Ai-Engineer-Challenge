

#Conditional Execution in Python
#Definition of Conditional Execution
#Conditional execution is a programming concept where certain blocks of code are executed only when specific conditions are met. It allows the program to make decisions and choose different paths of execution based on whether conditions evaluate to True or False.
#Simple explanation: It's like making decisions in real life - "IF it's raining, THEN take an umbrella, ELSE wear sunglasses."

#Why We Need Conditional Execution

#Decision Making: Programs need to make choices based on different situations
#Control Flow: Directs the program to execute specific code blocks based on conditions
##User Interaction: Responds differently based on user input or actions
#Error Handling: Handles different scenarios including errors gracefully
#Logic Implementation: Implements business rules and complex logic
#Efficiency: Executes only necessary code, saving time and resources

#Example: An ATM machine needs to check if you have enough balance before allowing withdrawal - this requires conditional execution.

#Types of Conditional Execution in Python
#1. IF Statement
#2. IF-ELSE Statement
#3. IF-ELIF-ELSE Statement (Multiple Conditions)
#4. Nested IF Statement
#5. Ternary Operator (Conditional Expression)

#Type 1: IF Statement
#Executes code only if the condition is True.
#Example 1: Age Verification
age = 20

if age >= 18:
    print("You are an adult")
    print("You can vote")

# Output: You are an adult
#         You can vote

#Example 2: Password Length Check
password = "secure123"

if len(password) >= 8:
    print("Password is strong enough")

# Output: Password is strong enough

#Example 3: Positive Number Check
number = 15

if number > 0:
    print(f"{number} is a positive number")

# Output: 15 is a positive number

#Example 4: Discount Eligibility
purchase_amount = 5000

if purchase_amount > 3000:
    print("Congratulations! You get 10% discount")
    discount = purchase_amount * 0.10
    print(f"Discount amount: {discount}")

# Output: Congratulations! You get 10% discount
#         Discount amount: 500.0
#Example 5: List Membership Check
username = "admin"
admin_users = ["admin", "superuser", "root"]

if username in admin_users:
    print("Access granted to admin panel")

# Output: Access granted to admin panel

#Type 2: IF-ELSE Statement
#Executes one block if condition is True, another block if False.

#Example 1: Even or Odd Number
number = 7

if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")

# Output: 7 is odd
#Example 2: Temperature Check

temperature = 15

if temperature > 25:
    print("It's hot outside. Wear light clothes.")
else:
    print("It's cold outside. Wear warm clothes.")

# Output: It's cold outside. Wear warm clothes.

#Example 3: Login Authentication
entered_password = "pass123"
correct_password = "secure456"

if entered_password == correct_password:
    print("Login successful!")
else:
    print("Incorrect password. Access denied.")

# Output: Incorrect password. Access denied.

#Example 4: Grade Pass/Fail
marks = 55

if marks >= 60:
    print("Congratulations! You passed.")
else:
    print("Sorry, you failed. Better luck next time.")

# Output: Sorry, you failed. Better luck next time.

#Example 5: Voting Eligibility
pythonage = 16

if age >= 18:
    print("You are eligible to vote")
else:
    print(f"You need to wait {18 - age} more years to vote")

# Output: You need to wait 2 more years to vote


#Type 3: IF-ELIF-ELSE Statement
#Checks multiple conditions in sequence.

#Example 1: Grade System
marks = 75

if marks >= 90:
    print("Grade: A+ (Excellent)")
elif marks >= 80:
    print("Grade: A (Very Good)")
elif marks >= 70:
    print("Grade: B (Good)")
elif marks >= 60:
    print("Grade: C (Average)")
else:
    print("Grade: F (Fail)")

# Output: Grade: B (Good)

#Example 2: Traffic Light System
light = "yellow"

if light == "red":
    print("STOP! Do not move.")
elif light == "yellow":
    print("SLOW DOWN! Prepare to stop.")
elif light == "green":
    print("GO! You can move.")
else:
    print("Invalid signal")

# Output: SLOW DOWN! Prepare to stop.

#Example 3: Movie Ticket Pricing
age = 12

if age < 5:
    ticket_price = 0
    print("Free entry for toddlers")
elif age < 13:
    ticket_price = 200
    print(f"Child ticket: {ticket_price} PKR")
elif age < 60:
    ticket_price = 500
    print(f"Adult ticket: {ticket_price} PKR")
else:
    ticket_price = 300
    print(f"Senior citizen ticket: {ticket_price} PKR")

# Output: Child ticket: 200 PKR

#Example 4: BMI Calculator
bmi = 22.5

if bmi < 18.5:
    print("Underweight - Eat more nutritious food")
elif bmi < 25:
    print("Normal weight - Keep it up!")
elif bmi < 30:
    print("Overweight - Consider exercise")
else:
    print("Obese - Consult a doctor")

# Output: Normal weight - Keep it up!
#Example 5: Day of Week
day = 3

if day == 1:
    print("Monday - Start of work week")
elif day == 2:
    print("Tuesday")
elif day == 3:
    print("Wednesday - Mid week")
elif day == 4:
    print("Thursday")
elif day == 5:
    print("Friday - Last working day")
elif day == 6:
    print("Saturday - Weekend!")
elif day == 7:
    print("Sunday - Weekend!")
else:
    print("Invalid day number")

# Output: Wednesday - Mid week

#Type 4: Nested IF Statement
#IF statements inside other IF statements.
#Example 1: Login with Username and Password
username = "john"
password = "pass123"

if username == "john":
    if password == "pass123":
        print("Login successful!")
    else:
        print("Wrong password")
else:
    print("Username not found")

# Output: Login successful!

#Example 2: Scholarship Eligibility
pythonmarks = 85
attendance = 92

if marks >= 80:
    if attendance >= 90:
        print("Eligible for scholarship!")
        print("Congratulations!")
    else:
        print("Good marks but attendance is low")
else:
    print("Marks are below scholarship requirement")

# Output: Eligible for scholarship!
#         Congratulations!

#Example 3: Age and License Check for Driving
age = 20
has_license = True

if age >= 18:
    if has_license:
        print("You can drive")
    else:
        print("You need to get a license first")
else:
    print("You are too young to drive")

# Output: You can drive

#Example 4: Online Shopping Validation
is_logged_in = True
has_items_in_cart = True
has_payment_method = False

if is_logged_in:
    if has_items_in_cart:
        if has_payment_method:
            print("Proceeding to checkout...")
        else:
            print("Please add a payment method")
    else:
        print("Your cart is empty")
else:
    print("Please login first")

# Output: Please add a payment method
#Example 5: Weather-Based Activity Suggestion
is_raining = False
temperature = 28

if is_raining:
    print("Stay indoors")
else:
    if temperature > 30:
        print("It's hot - Go swimming")
    elif temperature > 20:
        print("Pleasant weather - Go for a walk")
    else:
        print("It's cold - Wear warm clothes")

# Output: Pleasant weather - Go for a walk

#Type 5: Ternary Operator (Conditional Expression)
#A shorthand way to write simple if-else statements in one line.
#Syntax: value_if_true if condition else value_if_false

#Example 1: Maximum of Two Numbers
a = 10
b = 20

maximum = a if a > b else b
print(f"Maximum: {maximum}")

# Output: Maximum: 20

#Example 2: Adult or Minor
age = 17
status = "Adult" if age >= 18 else "Minor"
print(status)

# Output: Minor
#Example 3: Pass or Fail
marks = 65
result = "Pass" if marks >= 60 else "Fail"
print(f"Result: {result}")

# Output: Result: Pass

#Example 4: Discount Calculation
purchase = 4000
discount = 10 if purchase > 3000 else 0
print(f"Discount: {discount}%")

# Output: Discount: 10%
#Example 5: Greeting Message
time_hour = 14
greeting = "Good Morning" if time_hour < 12 else "Good Afternoon" if time_hour < 18 else "Good Evening"
print(greeting)

# Output: Good Afternoon

#Comparison Table
#TypeWhen to UseComplexityIFSingle condition checkSimpleIF-ELSETwo possible outcomesSimpleIF-ELIF-ELSEMultiple conditionsMediumNested IFComplex multi-level logicComplexTernarySimple one-line conditionsSimple

#Real-World Complete Example
#python# Restaurant Billing System
order_total = 2500
is_member = True
payment_method = "card"
tip_percentage = 15

# Check membership discount
if is_member:
    discount = 0.10
    print("Member discount: 10%")
else:
    discount = 0
    print("No membership discount")

# Calculate discounted amount
discounted_amount = order_total * (1 - discount)

# Check payment method
if payment_method == "cash":
    if discounted_amount >= 2000:
        print("Additional 5% cash discount!")
        discounted_amount *= 0.95
    else:
        print("No cash discount for orders under 2000")
elif payment_method == "card":
    print("Card payment accepted")
else:
    print("Invalid payment method")

# Add tip
tip = discounted_amount * (tip_percentage / 100)
final_amount = discounted_amount + tip

# Final output
print(f"\nOrder Total: {order_total} PKR")
print(f"After Discount: {discounted_amount} PKR")
print(f"Tip ({tip_percentage}%): {tip} PKR")
print(f"Final Amount: {final_amount} PKR")

# Output:
# Member discount: 10%
# Card payment accepted
# 
# Order Total: 2500 PKR
# After Discount: 2250.0 PKR
# Tip (15%): 337.5 PKR
# Final Amount: 2587.5 PKR