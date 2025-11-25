

#What are Special Methods (Magic Methods/Dunder Methods) in OOP? ✨
#Special methods (also called magic methods or dunder methods) are methods with double underscores (__) before and after their names. They allow you to define how objects behave with Python's built-in operations.
#Dunder = Double UNDERscore (e.g., __init__, __str__, __add__)

#Why Do We Need Special Methods?
# Make classes behave like built-in types (int, str, list)
# Customize operator behavior (+, -, *, /, ==, <, >)
# Control object creation and deletion
# Define string representation
# Enable iteration and indexing
# Make code more Pythonic and intuitive


#1. __init__: The "Constructor"
#This is the most common special method. It runs automatically when you create a new object from your class. Its job is to "initialize" the object by setting up its starting attributes.

class Book:
    # __init__ is called when we do Book(...)
    def __init__(self, title, author, pages):
        print("A new book is being created!")
        self.title = title
        self.author = author
        self.pages = pages

# --- Step-by-Step ---
# 1. We "call" the class.
# 2. Python automatically runs __init__.
# 3. '1984', 'Orwell', 328 are passed to __init__
book1 = Book("1984", "George Orwell", 328)

# 4. The attributes are now set
print(f"Title: {book1.title}")
print(f"Pages: {book1.pages}")


#A new book is being created!
Title: 1984
Pages: 328

#2. __str__ & __repr__: String Representation
#Right now, if you try to print your book1 object, you get an ugly, useless message:


print(book1)
# Output: <__main__.Book object at 0x10a4f3a90>

#We can fix this by adding __str__, which defines the "user-friendly" string that print() will use.
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
        
    # __str__ is called by print()
    def __str__(self):
        # Return a nice, human-readable string
        return f"'{self.title}' by {self.author}"
    
    # __repr__ is the "developer" representation
    # Its goal is to be an unambiguous, official string
    def __repr__(self):
        return f"Book(title='{self.title}', author='{self.author}')"

# --- Step-by-Step ---
book2 = Book("The Hobbit", "J.R.R. Tolkien", 310)

# 1. We call print() on the object
# 2. Python looks for a __str__ method. It finds ours!
print(book2) 

# 3. If you just type the variable name in a shell (or debugger)
#    Python looks for __repr__
# >>> book2
# Book(title='The Hobbit', author='J.R.R. Tolkien')


#3. __len__: Behaving Like a Built-in
#What if you want to use the len() function on your book to get the page count? By default, it fails:

#We can fix this by adding the __len__ method.
#Let's add it to our class:

class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
        
    def __str__(self):
        return f"'{self.title}' by {self.author}"
        
    # __len__ is called by len()
    def __len__(self):
        # We decide what "length" means. For a book,
        # the number of pages makes sense.
        return self.pages

# --- Step-by-Step ---
book3 = Book("Dune", "Frank Herbert", 412)

# 1. We call len() on the object
# 2. Python looks for a __len__ method. It finds ours!
# 3. It runs our method and returns self.pages
page_count = len(book3)

print(f"The book has {page_count} pages.")


#4. __add__ & __eq__: Custom Operator Behavior
#What if we want to use operators like + or ==?
#Let's use a new, simple Vector class, as math operators make more sense here.

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
        
    # __add__ is called by the '+' operator
    def __add__(self, other_vector):
        # Return a NEW Vector object with the added values
        new_x = self.x + other_vector.x
        new_y = self.y + other_vector.y
        return Vector(new_x, new_y)
        
    # __eq__ is called by the '==' operator
    def __eq__(self, other_vector):
        # Return True if they are the same, False otherwise
        return self.x == other_vector.x and self.y == other_vector.y

# --- Step-by-Step for __add__ ---
v1 = Vector(2, 4)
v2 = Vector(5, 1)

# 1. Python sees 'v1 + v2'
# 2. It calls v1.__add__(v2)
# 3. Our __add__ method runs, calculates (2+5) and (4+1)
# 4. It returns a new Vector(7, 5)
v3 = v1 + v2

print(f"{v1} + {v2} = {v3}")


# --- Step-by-Step for __eq__ ---
v4 = Vector(7, 5)

# 1. Python sees 'v3 == v4'
# 2. It calls v3.__eq__(v4)
# 3. Our __eq__ method runs.
# 4. It checks: 7 == 7 (True) AND 5 == 5 (True)
# 5. It returns True
are_equal = (v3 == v4)

print(f"Are {v3} and {v4} the same? {are_equal}")
