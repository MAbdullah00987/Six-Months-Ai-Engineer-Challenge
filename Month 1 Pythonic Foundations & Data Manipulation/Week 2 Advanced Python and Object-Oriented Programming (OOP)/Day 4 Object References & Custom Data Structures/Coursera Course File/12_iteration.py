

#ITERATION DEFINITION:
#  Iteration is the process of repeating a set of operations multiple times.
#  In programming, iteration means executing a block of code repeatedly,
#  either for a specific number of times or until a condition is met.

#TYPES OF ITERATION:
#  1. Definite Iteration (for loop): Repeats a specific number of times
#  2. Indefinite Iteration (while loop): Repeats until a condition is false

#KEY TERMS:
#  - Loop: A control structure that repeats a block of code
#  - Iteration: One complete execution of the loop body
#  - Iterator: An object that can be iterated over
#  - Iterable: A sequence that can be looped through (string, list, tuple, etc.)



#EXAMPLE 1: BASIC FOR LOOP ITERATION"


print("\n Simple counting with for loop:")
print("-" * 80)

print("\nCode:")
print("for i in range(5):")
print("    print(f'Iteration {i}')")

print("\nOutput:")
for i in range(5):
    print(f'Iteration {i}')

print("\nExplanation:")
print("  - The loop runs 5 times (0, 1, 2, 3, 4)")
print("  - 'i' is the loop variable that changes each iteration")
print("  - range(5) generates numbers from 0 to 4")


print("\n\n" + "=" * 80)
print("EXAMPLE 2: FLOW OF EXECUTION USING FOR LOOP")
print("=" * 80)

print("\n Understanding step-by-step execution:")
print("-" * 80)

print("\nCode:")
print("numbers = [10, 20, 30]")
print("for num in numbers:")
print("    print(f'Processing: {num}')")
print("    result = num * 2")
print("    print(f'Result: {result}')")
print("print('Loop finished!')")

print("\nOutput with Execution Flow:")
numbers = [10, 20, 30]
print("\n>>> Loop starts")
iteration_count = 1

for num in numbers:
    print(f"\n  [Iteration {iteration_count}]")
    print(f"  → Loop variable 'num' = {num}")
    print(f"  → Processing: {num}")
    result = num * 2
    print(f"  → Result: {result}")
    iteration_count += 1

print("\n>>> Loop finished!")

print("\nFlow of Execution:")
print("""
  Step 1: Loop variable 'num' takes the first value (10)
  Step 2: Execute all statements in the loop body
  Step 3: Loop variable 'num' takes the next value (20)
  Step 4: Execute all statements again
  Step 5: Loop variable 'num' takes the last value (30)
  Step 6: Execute all statements again
  Step 7: No more values, exit the loop
  Step 8: Continue with code after the loop
""")


print("\n" + "=" * 80)
print("EXAMPLE 3: STRINGS AND LOOPS")
print("=" * 80)

print("\n📌 Iterating through a string:")
print("-" * 80)

word = "Python"
print(f"\nOriginal String: '{word}'")

print("\nCode:")
print("word = 'Python'")
print("for letter in word:")
print("    print(letter)")

print("\nOutput:")
for letter in word:
    print(letter)

print("\nMore string iteration examples:")
print("-" * 80)

# Example 1: Count vowels
text = "Hello World"
vowels = "aeiouAEIOU"
vowel_count = 0

print(f"\nText: '{text}'")
print("Counting vowels...")

for char in text:
    if char in vowels:
        vowel_count += 1
        print(f"  Found vowel: '{char}'")

print(f"Total vowels: {vowel_count}")

# Example 2: Reverse a string using loop
original = "Python"
reversed_str = ""

print(f"\nReversing '{original}':")
for char in original:
    reversed_str = char + reversed_str
    print(f"  Step: '{char}' + '{reversed_str[1:]}' = '{reversed_str}'")

print(f"Final result: '{reversed_str}'")

# Example 3: Character positions
message = "Code"
print(f"\nCharacter positions in '{message}':")
position = 0
for char in message:
    print(f"  Position {position}: '{char}'")
    position += 1


print("\n\n" + "=" * 80)
print("EXAMPLE 4: LISTS AND FOR LOOPS")
print("=" * 80)

print("\n Iterating through a list:")
print("-" * 80)

fruits = ["Apple", "Banana", "Cherry", "Date"]
print(f"\nFruits list: {fruits}")

print("\nCode:")
print("fruits = ['Apple', 'Banana', 'Cherry', 'Date']")
print("for fruit in fruits:")
print("    print(fruit)")

print("\nOutput:")
for fruit in fruits:
    print(f"  {fruit}")

print("\nMore list iteration examples:")
print("-" * 80)

# Example 1: Sum of numbers
numbers = [10, 20, 30, 40, 50]
total = 0

print(f"\nNumbers: {numbers}")
print("Calculating sum:")

for num in numbers:
    total += num
    print(f"  Adding {num}, total so far: {total}")

print(f"Final sum: {total}")

# Example 2: Finding maximum
numbers = [45, 23, 67, 12, 89, 34]
max_num = numbers[0]

print(f"\nNumbers: {numbers}")
print("Finding maximum:")

for num in numbers:
    if num > max_num:
        max_num = num
        print(f"  New max found: {max_num}")
    else:
        print(f"  {num} is not greater than current max ({max_num})")

print(f"Maximum number: {max_num}")

# Example 3: List with index using enumerate
colors = ["Red", "Green", "Blue", "Yellow"]
print(f"\nColors: {colors}")
print("Using enumerate to get index and value:")

for index, color in enumerate(colors):
    print(f"  Index {index}: {color}")

# Example 4: Modifying list elements
prices = [100, 200, 300, 400]
print(f"\nOriginal prices: {prices}")
print("Applying 10% discount:")

discounted_prices = []
for price in prices:
    new_price = price * 0.9
    discounted_prices.append(new_price)
    print(f"  ${price} → ${new_price}")

print(f"Discounted prices: {discounted_prices}")


print("\n\n" + "=" * 80)
print("EXAMPLE 5: NESTED LOOPS (ITERATION WITHIN ITERATION)")
print("=" * 80)

print("\n Simple nested loop example:")
print("-" * 80)

print("\nCode:")
print("for i in range(3):")
print("    for j in range(2):")
print("        print(f'i={i}, j={j}')")

print("\nOutput:")
for i in range(3):
    for j in range(2):
        print(f"  i={i}, j={j}")

print("\n Explanation:")
print("  - Outer loop runs 3 times (i = 0, 1, 2)")
print("  - For each outer iteration, inner loop runs 2 times (j = 0, 1)")
print("  - Total iterations: 3 × 2 = 6")


print("\n\n" + "=" * 80)
print("EXAMPLE 6: PRACTICAL ITERATION EXAMPLES")
print("=" * 80)

print("\nExample A: Shopping Cart Total")
print("-" * 80)

cart = [
    ("Apple", 2.50, 3),
    ("Banana", 1.20, 5),
    ("Orange", 3.00, 2)
]

print("Shopping Cart:")
total = 0

for item, price, quantity in cart:
    item_total = price * quantity
    total += item_total
    print(f"  {item}: ${price} × {quantity} = ${item_total:.2f}")

print(f"\nTotal: ${total:.2f}")


print("\n Example B: Grade Calculator")
print("-" * 80)

students = ["Alice", "Bob", "Charlie", "Diana"]
grades = [85, 92, 78, 95]

print("Student Grades:")
for i in range(len(students)):
    student = students[i]
    grade = grades[i]
    
    if grade >= 90:
        letter = "A"
    elif grade >= 80:
        letter = "B"
    else:
        letter = "C"
    
    print(f"  {student}: {grade} ({letter})")


print("\n Example C: Pattern Printing")
print("-" * 80)

print("\nPrinting a triangle:")
for i in range(1, 6):
    print("  " + "* " * i)


print("\n\n" + "=" * 80)
print("ITERATION SUMMARY")
print("=" * 80)

'''
KEY CONCEPTS:

1. ITERATION DEFINITION:
   - Repeating a process multiple times
   - Each repeat is called an iteration

2. FOR LOOP SYNTAX:
   for variable in iterable:
       # code block to repeat

3. FLOW OF EXECUTION:
   a) Initialize loop variable with first value
   b) Execute loop body
   c) Get next value for loop variable
   d) Repeat until no more values
   e) Exit loop

4. STRINGS AND LOOPS:
   - Each character is accessed one by one
   - Loop variable holds one character per iteration
   - Can process, count, or modify characters

5. LISTS AND LOOPS:
   - Each element is accessed one by one
   - Loop variable holds one element per iteration
   - Can process, sum, find, or transform elements

6. COMMON PATTERNS:
   - range(n): Iterate n times
   - enumerate(): Get index and value
   - Nested loops: Loop within loop
   - Accumulation: Build result during iteration

7. BEST PRACTICES:
   - Use meaningful variable names
   - Keep loop body simple
   - Avoid modifying list while iterating
   - Use built-in functions when possible


'''