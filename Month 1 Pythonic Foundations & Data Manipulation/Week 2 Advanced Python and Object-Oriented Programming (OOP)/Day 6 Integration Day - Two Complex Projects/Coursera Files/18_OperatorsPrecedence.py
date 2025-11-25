

#Operator Precedence
#Definition of Operator Precedence
#Operator precedence determines the order in which operators are evaluated in an expression when multiple operators are present. Operators with higher precedence are evaluated before operators with lower precedence. When operators have the same precedence, the order of evaluation is determined by their associativity (left-to-right or right-to-left).
#Simple analogy: Just like in mathematics, multiplication is performed before addition (e.g., 2 + 3 × 4 = 2 + 12 = 14), programming languages follow a specific order of operations.

#Operator Precedence Order (Highest to Lowest)

#Parentheses ()
#Exponentiation **
#Unary operators +x, -x, not
#Multiplication, Division, Modulus *, /, //, %
#Addition, Subtraction +, -
#Comparison operators <, <=, >, >=, ==, !=
#Logical NOT not
#Logical AND and
#Logical OR or
#Assignment operators =, +=, -=, etc.


#5 Examples of Operator Precedence
#Example 1: Arithmetic Operators (Multiplication before Addition)
pythonresult = 5 + 3 * 2
print(pythonresult)  # Output: 11

# Explanation:
# 3 * 2 is evaluated first (= 6)
# Then 5 + 6 = 11

# With parentheses to change order:
result = (5 + 3) * 2
print(result)  # Output: 16

#Example 2: Exponentiation before Multiplication
pythonresult = 2 * 3 ** 2
print(result)  # Output: 18

# Explanation:
# 3 ** 2 is evaluated first (= 9)
# Then 2 * 9 = 18

# With parentheses:
result = (2 * 3) ** 2
print(result)  # Output: 36

#Example 3: Comparison before Logical AND
age = 25
salary = 50000

result = age > 18 and salary > 30000
print(result)  # Output: True

# Explanation:
# age > 18 is evaluated first (= True)
# salary > 30000 is evaluated next (= True)
# Then True and True = True

#Example 4: Logical NOT before Logical AND
is_student = False
has_discount = True

result = not is_student and has_discount
print(result)  # Output: True

# Explanation:
# not is_student is evaluated first (= True)
# Then True and has_discount (= True and True = True)

# With parentheses to change meaning:
result = not (is_student and has_discount)
print(result)  # Output: True (different logic)

#Example 5: Complex Expression with Multiple Operators
result = 10 + 5 * 2 - 8 / 4
print(result)  # Output: 18.0

# Explanation (step by step):
# 1. 5 * 2 = 10 (multiplication first)
# 2. 8 / 4 = 2.0 (division next)
# 3. 10 + 10 = 20 (addition left to right)
# 4. 20 - 2.0 = 18.0 (subtraction last)

# With parentheses to control order:
result = (10 + 5) * (2 - 8) / 4
print(result)  # Output: -22.5

# More Practical Examples
#Example 6: Shopping Cart Calculation
item_price = 100
quantity = 3
discount = 10
tax_rate = 0.05

# Without parentheses - wrong calculation
total = item_price * quantity - discount + item_price * tax_rate
print(total)  # Output: 295.0

# Explanation:
# 100 * 3 = 300
# 100 * 0.05 = 5
# 300 - 10 + 5 = 295.0

# With parentheses - correct calculation
total = (item_price * quantity - discount) * (1 + tax_rate)
print(total)  # Output: 304.5

#Example 7: Boolean Logic with Comparisons
x = 10
y = 5
z = 15

result = x > y and z > x or y == 5
print(result)  # Output: True

# Explanation:
# 1. x > y = True (10 > 5)
# 2. z > x = True (15 > 10)
# 3. y == 5 = True
# 4. True and True = True
# 5. True or True = True

#Example 8: Unary Operators with Arithmetic
x = 5
result = -x * 2 + 3
print(result)  # Output: -7

# Explanation:
# 1. -x = -5 (unary minus first)
# 2. -5 * 2 = -10 (multiplication)
# 3. -10 + 3 = -7 (addition)

#Example 9: Modulus with Addition
result = 17 + 10 % 3 * 2
print(result)  # Output: 19

# Explanation:
# 1. 10 % 3 = 1 (modulus first)
# 2. 1 * 2 = 2 (multiplication)
# 3. 17 + 2 = 19 (addition)

# With parentheses:
result = (17 + 10) % (3 * 2)
print(result)  # Output: 3

#Example 10: Membership Operator with Logical AND
numbers = [1, 2, 3, 4, 5]
x = 3
y = 6

result = x in numbers and y not in numbers
print(result)  # Output: True

# Explanation:
# 1. x in numbers = True (3 is in list)
# 2. y not in numbers = True (6 is not in list)
# 3. True and True = True

#Key Takeaway
#Use parentheses () to make your code clear and control the order of evaluation! Even when you know the precedence rules, parentheses improve readability and prevent bugs.
#python# Unclear without parentheses:
#result = a + b * c / d - e

# Clear with parentheses:
#result = a + ((b * c) / d) - e
