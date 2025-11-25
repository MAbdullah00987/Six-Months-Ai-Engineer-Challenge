
#Custom Data Structures in Python 🏗️Why We Need Custom Data Structures?Custom data structures allow you to create specialized containers that perfectly fit your application's needs. Here's why they're essential:
#Domain-Specific Logic - Model real-world entities with appropriate behavior
#Data Encapsulation - Group related data and operations together
#Type Safety - Enforce rules and constraints on your data
#Code Reusability - Write once, use everywhere
#Better Organization - Make code more readable and maintainable
#Performance Optimization - Design structures optimized for specific operations
#📋 Rules for Creating Custom Data Structures ProperlyRule 1: Choose the Right Base Structure

##List-like → Inherit from list or use composition with a list
#Dictionary-like → Inherit from dict or use collections.UserDict
#Set-like → Inherit from set or use collections.UserSet
#Completely Custom → Create from scratch with classes
#Rule 2: Implement Special Methods (Dunder Methods)

#__init__()      # Constructor
#__str__()       # String representation (user-friendly)
#__repr__()      # Developer representation
#__len__()       # len(object)
#__getitem__()   # object[key]
#__setitem__()   # object[key] = value
#__iter__()      # for item in object
#__contains__()  # item in object
#Rule 3: Encapsulate Data

##Use private attributes (prefix with _ or __)
#Provide getter/setter methods or properties
#Validate data in setters

#Rule 4: Follow Single Responsibility Principle
#Each data structure should have one clear purpose
#Rule 5: Make It Iterable When Appropriate
#Implement __iter__() if users need to loop through elements
#Rule 6: Provide Useful Methods
#Add methods that make sense for your data structure's purpose
#Rule 7: Handle Edge Cases

#Empty structures
#Invalid inputs
#Boundary conditions


"""
Custom Data Structures in Python - Complete Examples
Demonstrates various custom data structures with proper implementation
"""


def print_section(title):
    """Helper function to print section headers."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================================
# Example 1: Simple Stack (LIFO - Last In First Out)
# ============================================================================

class Stack:
    """
    A simple Stack implementation using a list.
    LIFO: Last In, First Out
    """
    
    def __init__(self):
        """Initialize an empty stack."""
        self._items = []  # Private attribute
    
    def push(self, item):
        """Add an item to the top of the stack."""
        self._items.append(item)
    
    def pop(self):
        """Remove and return the top item."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop()
    
    def peek(self):
        """Return the top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self._items[-1]
    
    def is_empty(self):
        """Check if stack is empty."""
        return len(self._items) == 0
    
    def size(self):
        """Return the number of items in the stack."""
        return len(self._items)
    
    def __len__(self):
        """Support len() function."""
        return len(self._items)
    
    def __str__(self):
        """String representation."""
        return f"Stack({self._items})"
    
    def __repr__(self):
        """Developer representation."""
        return f"Stack({self._items!r})"


def demo_stack():
    """Demonstrate Stack usage."""
    print_section("Example 1: Stack (LIFO)")
    
    stack = Stack()
    
    print("Pushing items: 1, 2, 3, 4, 5")
    for i in range(1, 6):
        stack.push(i)
        print(f"  Pushed {i}, Stack: {stack}")
    
    print(f"\nStack size: {len(stack)}")
    print(f"Top item (peek): {stack.peek()}")
    
    print("\nPopping all items:")
    while not stack.is_empty():
        item = stack.pop()
        print(f"  Popped: {item}, Remaining: {stack}")


# ============================================================================
# Example 2: Queue (FIFO - First In First Out)
# ============================================================================

class Queue:
    """
    A simple Queue implementation using a list.
    FIFO: First In, First Out
    """
    
    def __init__(self):
        """Initialize an empty queue."""
        self._items = []
    
    def enqueue(self, item):
        """Add an item to the rear of the queue."""
        self._items.append(item)
    
    def dequeue(self):
        """Remove and return the front item."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self._items.pop(0)
    
    def front(self):
        """Return the front item without removing it."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self._items[0]
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self._items) == 0
    
    def size(self):
        """Return the number of items in the queue."""
        return len(self._items)
    
    def __len__(self):
        """Support len() function."""
        return len(self._items)
    
    def __str__(self):
        """String representation."""
        return f"Queue({self._items})"


def demo_queue():
    """Demonstrate Queue usage."""
    print_section("Example 2: Queue (FIFO)")
    
    queue = Queue()
    
    print("Enqueueing: Alice, Bob, Charlie, David")
    for name in ["Alice", "Bob", "Charlie", "David"]:
        queue.enqueue(name)
        print(f"  Enqueued {name}, Queue: {queue}")
    
    print(f"\nQueue size: {len(queue)}")
    print(f"Front person: {queue.front()}")
    
    print("\nServing customers (dequeue):")
    while not queue.is_empty():
        person = queue.dequeue()
        print(f"  Serving: {person}, Remaining: {queue}")


# ============================================================================
# Example 3: Contact Book (Dictionary-based)
# ============================================================================

class ContactBook:
    """
    A custom contact book to store and manage contacts.
    """
    
    def __init__(self):
        """Initialize an empty contact book."""
        self._contacts = {}
    
    def add_contact(self, name, phone, email=None):
        """Add a new contact."""
        if name in self._contacts:
            print(f"  Contact '{name}' already exists. Use update_contact() instead.")
            return False
        
        self._contacts[name] = {
            'phone': phone,
            'email': email
        }
        print(f" Added contact: {name}")
        return True
    
    def remove_contact(self, name):
        """Remove a contact."""
        if name not in self._contacts:
            print(f"  Contact '{name}' not found.")
            return False
        
        del self._contacts[name]
        print(f"Removed contact: {name}")
        return True
    
    def update_contact(self, name, phone=None, email=None):
        """Update an existing contact."""
        if name not in self._contacts:
            print(f"  Contact '{name}' not found.")
            return False
        
        if phone:
            self._contacts[name]['phone'] = phone
        if email:
            self._contacts[name]['email'] = email
        
        print(f" Updated contact: {name}")
        return True
    
    def get_contact(self, name):
        """Get a contact's information."""
        return self._contacts.get(name, None)
    
    def search(self, query):
        """Search contacts by name (case-insensitive)."""
        query = query.lower()
        results = [name for name in self._contacts.keys() if query in name.lower()]
        return results
    
    def list_all(self):
        """List all contacts."""
        if not self._contacts:
            print("Contact book is empty.")
            return
        
        print(f"\n All Contacts ({len(self._contacts)}):")
        for name, info in self._contacts.items():
            email = info['email'] if info['email'] else 'N/A'
            print(f"  • {name}: {info['phone']} | {email}")
    
    def __len__(self):
        """Return number of contacts."""
        return len(self._contacts)
    
    def __contains__(self, name):
        """Support 'in' operator."""
        return name in self._contacts
    
    def __getitem__(self, name):
        """Support bracket notation: book[name]"""
        return self._contacts[name]
    
    def __iter__(self):
        """Make the contact book iterable."""
        return iter(self._contacts.items())


def demo_contact_book():
    """Demonstrate ContactBook usage."""
    print_section("Example 3: Contact Book")
    
    book = ContactBook()
    
    # Add contacts
    print("\nAdding contacts:")
    book.add_contact("Alice Johnson", "555-0101", "alice@email.com")
    book.add_contact("Bob Smith", "555-0102", "bob@email.com")
    book.add_contact("Charlie Brown", "555-0103")
    
    # Try adding duplicate
    book.add_contact("Alice Johnson", "555-9999")
    
    # List all
    book.list_all()
    
    # Search
    print("\nSearching for 'alice':")
    results = book.search("alice")
    print(f"  Found: {results}")
    
    # Update
    print("\nUpdating Bob's email:")
    book.update_contact("Bob Smith", email="bob.smith@email.com")
    
    # Check if contact exists
    print(f"\nIs 'Bob Smith' in contacts? {'Bob Smith' in book}")
    
    # Access contact
    print(f"Bob's info: {book['Bob Smith']}")
    
    # Remove
    print("\nRemoving Charlie:")
    book.remove_contact("Charlie Brown")
    
    book.list_all()
    
    # Iterate
    print("\nIterating through contacts:")
    for name, info in book:
        print(f"  {name}: {info['phone']}")


# ============================================================================
# Example 4: Shopping Cart with Items
# ============================================================================

class CartItem:
    """Represents an item in the shopping cart."""
    
    def __init__(self, name, price, quantity=1):
        """Initialize a cart item."""
        self.name = name
        self.price = price
        self.quantity = quantity
    
    def total_price(self):
        """Calculate total price for this item."""
        return self.price * self.quantity
    
    def __str__(self):
        """String representation."""
        return f"{self.name} x{self.quantity} @ ${self.price:.2f} = ${self.total_price():.2f}"


class ShoppingCart:
    """
    A shopping cart that stores items and calculates totals.
    """
    
    def __init__(self):
        """Initialize an empty cart."""
        self._items = []
    
    def add_item(self, name, price, quantity=1):
        """Add an item to the cart."""
        # Check if item already exists
        for item in self._items:
            if item.name == name:
                item.quantity += quantity
                print(f" Updated {name} quantity to {item.quantity}")
                return
        
        # Add new item
        new_item = CartItem(name, price, quantity)
        self._items.append(new_item)
        print(f" Added {name} to cart")
    
    def remove_item(self, name):
        """Remove an item from the cart."""
        for i, item in enumerate(self._items):
            if item.name == name:
                removed = self._items.pop(i)
                print(f" Removed {removed.name} from cart")
                return True
        print(f"  Item '{name}' not found in cart")
        return False
    
    def update_quantity(self, name, quantity):
        """Update the quantity of an item."""
        for item in self._items:
            if item.name == name:
                item.quantity = quantity
                print(f" Updated {name} quantity to {quantity}")
                return True
        print(f"  Item '{name}' not found in cart")
        return False
    
    def get_total(self):
        """Calculate the total price of all items."""
        return sum(item.total_price() for item in self._items)
    
    def get_item_count(self):
        """Get total number of items (including quantities)."""
        return sum(item.quantity for item in self._items)
    
    def clear(self):
        """Clear all items from the cart."""
        self._items.clear()
        print(" Cart cleared")
    
    def show_cart(self):
        """Display all items in the cart."""
        if not self._items:
            print("🛒 Your cart is empty")
            return
        
        print(f"\n🛒 Shopping Cart ({self.get_item_count()} items):")
        print("-" * 60)
        for item in self._items:
            print(f"  {item}")
        print("-" * 60)
        print(f"  Total: ${self.get_total():.2f}")
    
    def __len__(self):
        """Return number of unique items."""
        return len(self._items)
    
    def __iter__(self):
        """Make cart iterable."""
        return iter(self._items)
    
    def __str__(self):
        """String representation."""
        return f"ShoppingCart({len(self)} items, total: ${self.get_total():.2f})"


def demo_shopping_cart():
    """Demonstrate ShoppingCart usage."""
    print_section("Example 4: Shopping Cart")
    
    cart = ShoppingCart()
    
    print("\nAdding items to cart:")
    cart.add_item("Laptop", 999.99, 1)
    cart.add_item("Mouse", 29.99, 2)
    cart.add_item("Keyboard", 79.99, 1)
    cart.add_item("Mouse", 29.99, 1)  # Add more of the same item
    
    cart.show_cart()
    
    print("\nUpdating Mouse quantity to 5:")
    cart.update_quantity("Mouse", 5)
    cart.show_cart()
    
    print("\nRemoving Keyboard:")
    cart.remove_item("Keyboard")
    cart.show_cart()
    
    print(f"\nCart summary: {cart}")
    print(f"Total items (with quantities): {cart.get_item_count()}")
    print(f"Unique items: {len(cart)}")


# ============================================================================
# Example 5: Student Grade Book
# ============================================================================

class Student:
    """Represents a student with grades."""
    
    def __init__(self, name, student_id):
        """Initialize a student."""
        self.name = name
        self.student_id = student_id
        self._grades = []  # Private list of grades
    
    def add_grade(self, grade):
        """Add a grade (0-100)."""
        if 0 <= grade <= 100:
            self._grades.append(grade)
            return True
        else:
            print(f"  Invalid grade: {grade}. Must be between 0-100.")
            return False
    
    def get_average(self):
        """Calculate average grade."""
        if not self._grades:
            return 0
        return sum(self._grades) / len(self._grades)
    
    def get_letter_grade(self):
        """Get letter grade based on average."""
        avg = self.get_average()
        if avg >= 90:
            return 'A'
        elif avg >= 80:
            return 'B'
        elif avg >= 70:
            return 'C'
        elif avg >= 60:
            return 'D'
        else:
            return 'F'
    
    def __str__(self):
        """String representation."""
        avg = self.get_average()
        letter = self.get_letter_grade()
        return f"{self.name} (ID: {self.student_id}): Avg: {avg:.1f} ({letter})"


class GradeBook:
    """Manages multiple students and their grades."""
    
    def __init__(self, course_name):
        """Initialize a grade book."""
        self.course_name = course_name
        self._students = {}
    
    def add_student(self, name, student_id):
        """Add a new student."""
        if student_id in self._students:
            print(f"  Student ID {student_id} already exists.")
            return None
        
        student = Student(name, student_id)
        self._students[student_id] = student
        print(f" Added student: {name} (ID: {student_id})")
        return student
    
    def add_grade(self, student_id, grade):
        """Add a grade for a student."""
        if student_id not in self._students:
            print(f"  Student ID {student_id} not found.")
            return False
        
        return self._students[student_id].add_grade(grade)
    
    def get_student(self, student_id):
        """Get a student by ID."""
        return self._students.get(student_id)
    
    def get_class_average(self):
        """Calculate the class average."""
        if not self._students:
            return 0
        averages = [s.get_average() for s in self._students.values()]
        return sum(averages) / len(averages)
    
    def show_all_students(self):
        """Display all students and their grades."""
        if not self._students:
            print(" No students in grade book.")
            return
        
        print(f"\n {self.course_name} - Grade Book")
        print("-" * 60)
        for student in self._students.values():
            print(f"  {student}")
        print("-" * 60)
        print(f"  Class Average: {self.get_class_average():.1f}")
    
    def __len__(self):
        """Return number of students."""
        return len(self._students)
    
    def __contains__(self, student_id):
        """Support 'in' operator."""
        return student_id in self._students


def demo_grade_book():
    """Demonstrate GradeBook usage."""
    print_section("Example 5: Grade Book")
    
    grade_book = GradeBook("Python Programming 101")
    
    print("\nAdding students:")
    grade_book.add_student("Alice Johnson", "S001")
    grade_book.add_student("Bob Smith", "S002")
    grade_book.add_student("Charlie Brown", "S003")
    
    print("\nAdding grades:")
    # Alice's grades
    grade_book.add_grade("S001", 95)
    grade_book.add_grade("S001", 88)
    grade_book.add_grade("S001", 92)
    
    # Bob's grades
    grade_book.add_grade("S002", 78)
    grade_book.add_grade("S002", 82)
    grade_book.add_grade("S002", 85)
    
    # Charlie's grades
    grade_book.add_grade("S003", 65)
    grade_book.add_grade("S003", 70)
    grade_book.add_grade("S003", 68)
    
    grade_book.show_all_students()
    
    print(f"\nTotal students: {len(grade_book)}")
    print(f"Is S001 enrolled? {'S001' in grade_book}")
    print(f"Is S999 enrolled? {'S999' in grade_book}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n" + "🏗️ " * 25)
    print("CUSTOM DATA STRUCTURES IN PYTHON - COMPLETE DEMONSTRATION")
    print("🏗️ " * 25)
    
    demo_stack()
    demo_queue()
    demo_contact_book()
    demo_shopping_cart()
    demo_grade_book()
    
    print("\n" + "=" * 70)
    print(" All demonstrations completed successfully!")
    print("=" * 70)
    
    print("\n KEY TAKEAWAYS:")
    print("1. Custom data structures encapsulate data and behavior")
    print("2. Implement dunder methods for Python integration")
    print("3. Use private attributes (_) to protect internal data")
    print("4. Provide meaningful methods for your use case")
    print("5. Handle edge cases (empty, invalid input, etc.)")
    print("6. Make structures iterable when appropriate")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()