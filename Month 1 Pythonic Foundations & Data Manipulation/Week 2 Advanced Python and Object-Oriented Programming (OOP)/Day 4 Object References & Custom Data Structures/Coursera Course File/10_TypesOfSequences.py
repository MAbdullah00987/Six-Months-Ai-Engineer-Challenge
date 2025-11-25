
#Three types of Sequences are discussed in this course:
#1 Strings
#2 Lists
#3 Tuples
#Start with strings 

#Strings (str)
#A String is an ordered, immutable sequence of characters.
#It's used to store all text data, like words, sentences, or symbols.
#Immutable means that once you create the string, you cannot change the characters inside it.
#They are created using single quotes ('...') or double quotes ("...").

#Examples of Strings
#1: A simple word:
my_name = "Mirza"

#2: A sentence with spaces and punctuation:
greeting = "Hello, I am learning AI!"

#3: A string that includes numbers and symbols (like an ID):
#user_id = "ai_dev_001"


#Lists (list)
#A List is an ordered, mutable sequence of items.
#It's a very flexible container that can hold items of any data type (numbers, strings, or even other lists).
#Mutable means you can freely add, remove, or change the items inside the list after it has been created.
#They are created using square brackets [...].

#Examples of Lists

#1:A list of numbers (e.g., scores):
scores = [95, 88, 76, 100]

#2: A list of strings (e.g., tasks):
tasks_today = ["Learn Python", "Setup Git", "Post on LinkedIn"]

#3:A mixed list with different data types:
project_info = ["AI Avatar", 2025, 3.14, True]



#Tuples (tuple)
#A Tuple is an ordered, immutable sequence of items.
#It is very similar to a list, but like a string, it is immutable. Once you create a tuple, you cannot add, remove, or change its items.
#They are often used for data that should not change, like coordinates or fixed records.
#They are created using parentheses (...).

#Examples of Tuples

#1: A tuple of (x, y) coordinates:
coordinates = (10, 20)

#2: A tuple of configuration settings that shouldn't change:
db_config = ("localhost", 5432, "admin_user")

#3: A mixed-type tuple (e.g., a person's record):
person = ("Aisha", 28, "AI Engineer")

#Examples

'''
#Example 1: Student Grade Management System

print("=" * 70)
print("EXAMPLE 1: Student Grade Management System")
print("=" * 70)

# Using tuples for student data (immutable)
students = [
    ("Alice Johnson", 85, 92, 88, 90),
    ("Bob Smith", 78, 85, 82, 88),
    ("Charlie Brown", 92, 95, 89, 94),
    ("Diana Prince", 88, 91, 87, 93),
    ("Eve Wilson", 76, 82, 79, 85)
]

# String manipulation for formatting
print("\n STUDENT GRADE REPORT")
print("-" * 70)

for student in students:
    name = student[0]  # Index operator to get name
    grades = student[1:]  # Slicing to get all grades
    
    # Calculate average using length
    average = sum(grades) / len(grades)
    
    # String formatting and indexing
    first_name = name.split()[0]  # Get first name
    last_initial = name.split()[1][0]  # Index operator for last initial
    
    # Find highest and lowest grade using indexing
    highest = max(grades)
    lowest = min(grades)
    highest_index = grades.index(highest)
    lowest_index = grades.index(lowest)
    
    subjects = ["Math", "Science", "English", "History"]
    
    print(f"\nStudent: {name}")
    print(f"  Nickname: {first_name} {last_initial}.")
    print(f"  Grades: {list(grades)}")
    print(f"  Average: {average:.2f}")
    print(f"  Highest: {highest} in {subjects[highest_index]}")
    print(f"  Lowest: {lowest} in {subjects[lowest_index]}")
    print(f"  Total subjects: {len(grades)}")
    
    # Grade classification using string comparison
    if average >= 90:
        grade = "A"
        comment = "Excellent work!"
    elif average >= 80:
        grade = "B"
        comment = "Good job!"
    else:
        grade = "C"
        comment = "Keep improving!"
    
    print(f"  Final Grade: {grade} - {comment}")

# Advanced string and list operations
print("\n" + "=" * 70)
print("CLASS STATISTICS")
print("-" * 70)

all_grades = [grade for student in students for grade in student[1:]]
class_average = sum(all_grades) / len(all_grades)
print(f"Class Average: {class_average:.2f}")
print(f"Total Students: {len(students)}")
print(f"Total Grades Recorded: {len(all_grades)}")

# Find top performer using indexing
top_student = max(students, key=lambda s: sum(s[1:]) / len(s[1:]))
print(f"Top Performer: {top_student[0]} with average {sum(top_student[1:]) / len(top_student[1:]):.2f}")

'''

'''
#Example 2: Text Analysis and Word Processing System")


print("\n\n" + "=" * 70)
print("EXAMPLE 2: Text Analysis and Word Processing System")
print("=" * 70)

# Long text for analysis
text = """Python is an amazing programming language. It is versatile and powerful.
Python supports multiple programming paradigms including object-oriented,
functional, and procedural programming. Python has a rich ecosystem of libraries."""

print("\n ORIGINAL TEXT:")
print(text)

# String operations and indexing
print("\n" + "-" * 70)
print(" TEXT ANALYSIS")
print("-" * 70)

# Basic statistics using length
total_chars = len(text)
total_chars_no_spaces = len(text.replace(" ", ""))
total_lines = len(text.split("\n"))

print(f"Total Characters: {total_chars}")
print(f"Characters (no spaces): {total_chars_no_spaces}")
print(f"Total Lines: {total_lines}")

# Word analysis using lists
words = text.lower().replace(".", "").replace(",", "").split()
print(f"Total Words: {len(words)}")

# Find longest and shortest words using indexing
longest_word = max(words, key=len)
shortest_word = min(words, key=len)
print(f"Longest Word: '{longest_word}' ({len(longest_word)} characters)")
print(f"Shortest Word: '{shortest_word}' ({len(shortest_word)} characters)")

# Word frequency using dictionary and indexing
word_count = {}
for word in words:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

# Sort and display top 5 words
sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
print(f"\n TOP 5 MOST FREQUENT WORDS:")
for i in range(min(5, len(sorted_words))):
    word, count = sorted_words[i]
    print(f"  {i+1}. '{word}': {count} times")

# Sentence analysis using tuples (immutable data)
sentences = text.replace("\n", " ").split(".")
sentences = [s.strip() for s in sentences if s.strip()]

print(f"\n SENTENCE ANALYSIS:")
print(f"Total Sentences: {len(sentences)}")

for i, sentence in enumerate(sentences):
    word_list = sentence.split()
    first_word = word_list[0] if len(word_list) > 0 else ""
    last_word = word_list[-1] if len(word_list) > 0 else ""
    
    print(f"\n  Sentence {i+1}:")
    print(f"    Text: {sentence[:50]}..." if len(sentence) > 50 else f"    Text: {sentence}")
    print(f"    Words: {len(word_list)}")
    print(f"    Characters: {len(sentence)}")
    print(f"    First word: '{first_word}'")
    print(f"    Last word: '{last_word}'")

# Character frequency analysis
char_freq = {}
for char in text.lower():
    if char.isalpha():
        char_freq[char] = char_freq.get(char, 0) + 1

print(f"\n TOP 5 MOST COMMON LETTERS:")
sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
for i in range(min(5, len(sorted_chars))):
    char, count = sorted_chars[i]
    print(f"  {i+1}. '{char}': {count} times")
'''

#Example 3: Shopping Cart and Inventory Management System

print("\n\n" + "=" * 70)
print("EXAMPLE 3: Shopping Cart and Inventory Management System")
print("=" * 70)

# Product catalog using tuples (product_id, name, price, category)
products = [
    ("P001", "Laptop", 899.99, "Electronics"),
    ("P002", "Mouse", 25.50, "Electronics"),
    ("P003", "Keyboard", 75.00, "Electronics"),
    ("P004", "Monitor", 299.99, "Electronics"),
    ("P005", "Desk Chair", 199.99, "Furniture"),
    ("P006", "Desk Lamp", 45.00, "Furniture"),
    ("P007", "Notebook", 3.99, "Stationery"),
    ("P008", "Pen Set", 12.50, "Stationery"),
]

# Shopping cart as list of tuples (product_tuple, quantity)
shopping_cart = [
    (products[0], 1),  # Laptop x1
    (products[1], 2),  # Mouse x2
    (products[2], 1),  # Keyboard x1
    (products[6], 5),  # Notebook x5
]

print("\n🛒 SHOPPING CART SUMMARY")
print("-" * 70)

# Calculate totals using indexing and length
subtotal = 0
total_items = 0

for i in range(len(shopping_cart)):
    product, quantity = shopping_cart[i]
    product_id = product[0]
    name = product[1]
    price = product[2]
    category = product[3]
    
    item_total = price * quantity
    subtotal += item_total
    total_items += quantity
    
    print(f"{i+1}. {name}")
    print(f"   ID: {product_id} | Category: {category}")
    print(f"   Price: ${price:.2f} x {quantity} = ${item_total:.2f}")
    print()

# Tax calculation
tax_rate = 0.08
tax = subtotal * tax_rate
total = subtotal + tax

print("-" * 70)
print(f"Subtotal: ${subtotal:.2f}")
print(f"Tax (8%): ${tax:.2f}")
print(f"TOTAL: ${total:.2f}")
print(f"Total Items: {total_items}")

# Category breakdown
print("\nPURCHASE BY CATEGORY")
print("-" * 70)

categories = {}
for product, quantity in shopping_cart:
    category = product[3]
    price = product[2]
    
    if category not in categories:
        categories[category] = {"count": 0, "total": 0}
    
    categories[category]["count"] += quantity
    categories[category]["total"] += price * quantity

for category, data in categories.items():
    print(f"{category}:")
    print(f"  Items: {data['count']}")
    print(f"  Total: ${data['total']:.2f}")

# Find most expensive item in cart using indexing
most_expensive = max(shopping_cart, key=lambda x: x[0][2])
print(f"\nMost Expensive Item: {most_expensive[0][1]} at ${most_expensive[0][2]:.2f}")

# Product search by name substring (string indexing)
print("\n🔍 PRODUCT SEARCH")
print("-" * 70)
search_term = "Desk"
print(f"Searching for products containing '{search_term}':")

found_products = []
for product in products:
    product_name = product[1]
    if search_term.lower() in product_name.lower():
        found_products.append(product)

if len(found_products) > 0:
    for product in found_products:
        print(f"  - {product[1]} (${product[2]:.2f}) - {product[3]}")
else:
    print(f"  No products found containing '{search_term}'")

# Inventory statistics
print("\nINVENTORY STATISTICS")
print("-" * 70)
print(f"Total Products in Catalog: {len(products)}")

# Price analysis using indexing
prices = [product[2] for product in products]
avg_price = sum(prices) / len(prices)
max_price = max(prices)
min_price = min(prices)

print(f"Average Product Price: ${avg_price:.2f}")
print(f"Most Expensive: ${max_price:.2f}")
print(f"Least Expensive: ${min_price:.2f}")

# Category count
unique_categories = list(set([product[3] for product in products]))
print(f"Number of Categories: {len(unique_categories)}")
print(f"Categories: {', '.join(unique_categories)}")

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED!")
print("=" * 70)


