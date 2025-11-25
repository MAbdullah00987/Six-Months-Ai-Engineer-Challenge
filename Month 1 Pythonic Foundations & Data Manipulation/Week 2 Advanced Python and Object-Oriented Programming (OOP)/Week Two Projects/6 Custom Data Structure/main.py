
#Stack and Queue Data Structures in Python 
#Custom Data Structure: Implement a Stack or Queue using a Python class.

"""
Complete Stack and Queue Implementation in Python
Demonstrates LIFO and FIFO data structures with full functionality
"""

# ============================================
# STACK IMPLEMENTATION (LIFO)
# ============================================

class Stack:
    """
    Stack: Last In, First Out (LIFO) data structure
    Like a stack of plates - add and remove from the top
    """
    
    def __init__(self):
        """
        Initialize an empty stack
        Uses a Python list internally to store items
        """
        self.items = []  # Internal storage using list
    
    def push(self, item):
        """
        Add an item to the top of the stack
        Time Complexity: O(1)
        """
        self.items.append(item)
        print(f"Pushed: {item}")
    
    def pop(self):
        """
        Remove and return the top item from the stack
        Time Complexity: O(1)
        Raises: IndexError if stack is empty
        """
        if self.is_empty():
            raise IndexError("Cannot pop from empty stack!")
        item = self.items.pop()
        print(f"Popped: {item}")
        return item
    
    def peek(self):
        """
        Return the top item without removing it
        Time Complexity: O(1)
        """
        if self.is_empty():
            raise IndexError("Cannot peek at empty stack!")
        return self.items[-1]  # -1 gets the last element
    
    def is_empty(self):
        """Check if stack is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in stack"""
        return len(self.items)
    
    def clear(self):
        """Remove all items from stack"""
        self.items = []
        print("✓ Stack cleared")
    
    def __str__(self):
        """String representation of stack (for print statement)"""
        return f"Stack: {self.items} (top → bottom)"
    
    def __len__(self):
        """Allow len(stack) to work"""
        return len(self.items)


# ============================================
# QUEUE IMPLEMENTATION (FIFO)
# ============================================

class Queue:
    """
    Queue: First In, First Out (FIFO) data structure
    Like a line at a store - first come, first served
    """
    
    def __init__(self):
        """
        Initialize an empty queue
        Uses a Python list internally
        """
        self.items = []
    
    def enqueue(self, item):
        """
        Add an item to the back of the queue
        Time Complexity: O(1)
        """
        self.items.append(item)
        print(f"Enqueued: {item}")
    
    def dequeue(self):
        """
        Remove and return the front item from queue
        Time Complexity: O(n) due to list shifting
        For better performance, use collections.deque
        """
        if self.is_empty():
            raise IndexError("Cannot dequeue from empty queue!")
        item = self.items.pop(0)  # Remove from front
        print(f"✓ Dequeued: {item}")
        return item
    
    def peek(self):
        """
        Return the front item without removing it
        Time Complexity: O(1)
        """
        if self.is_empty():
            raise IndexError("Cannot peek at empty queue!")
        return self.items[0]
    
    def is_empty(self):
        """Check if queue is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in queue"""
        return len(self.items)
    
    def clear(self):
        """Remove all items from queue"""
        self.items = []
        print("✓ Queue cleared")
    
    def __str__(self):
        """String representation of queue"""
        return f"Queue: {self.items} (front → back)"
    
    def __len__(self):
        """Allow len(queue) to work"""
        return len(self.items)


# ============================================
# DEMONSTRATION AND EXAMPLES
# ============================================

def demonstrate_stack():
    """Show how Stack works with practical examples"""
    print("\n" + "="*50)
    print("STACK DEMONSTRATION (LIFO)")
    print("="*50)
    
    # Create a new stack
    stack = Stack()
    
    print("\n1. PUSHING items onto stack:")
    stack.push("Book 1")
    stack.push("Book 2")
    stack.push("Book 3")
    print(f"   Current stack: {stack}")
    print(f"   Size: {stack.size()}")
    
    print("\n2. PEEKING at top item:")
    top = stack.peek()
    print(f"   Top item: {top}")
    print(f"   Stack unchanged: {stack}")
    
    print("\n3. POPPING items (Last In, First Out):")
    stack.pop()
    stack.pop()
    print(f"   Remaining: {stack}")
    
    print("\n4. CHECK if empty:")
    print(f"   Is empty? {stack.is_empty()}")
    
    stack.clear()
    print(f"   After clear, is empty? {stack.is_empty()}")


def demonstrate_queue():
    """Show how Queue works with practical examples"""
    print("\n" + "="*50)
    print("QUEUE DEMONSTRATION (FIFO)")
    print("="*50)
    
    # Create a new queue
    queue = Queue()
    
    print("\n1. ENQUEUING customers:")
    queue.enqueue("Customer A")
    queue.enqueue("Customer B")
    queue.enqueue("Customer C")
    print(f"   Current queue: {queue}")
    print(f"   Size: {queue.size()}")
    
    print("\n2. PEEKING at front customer:")
    front = queue.peek()
    print(f"   Front customer: {front}")
    
    print("\n3. DEQUEUING customers (First In, First Out):")
    queue.dequeue()
    queue.dequeue()
    print(f"   Remaining: {queue}")
    
    print("\n4. ADD more and check size:")
    queue.enqueue("Customer D")
    print(f"   Queue: {queue}")
    print(f"   Size using len(): {len(queue)}")


def practical_examples():
    """Real-world use cases"""
    print("\n" + "="*50)
    print("PRACTICAL EXAMPLES")
    print("="*50)
    
    # Example 1: Browser History (Stack)
    print("\n📱 BROWSER HISTORY (Stack):")
    history = Stack()
    history.push("google.com")
    history.push("youtube.com")
    history.push("github.com")
    print(f"   Current page: {history.peek()}")
    print("   Press back button...")
    history.pop()
    print(f"   Now on: {history.peek()}")
    
    # Example 2: Print Queue (Queue)
    print("\nPRINT QUEUE (Queue):")
    printer = Queue()
    printer.enqueue("Document1.pdf")
    printer.enqueue("Photo.jpg")
    printer.enqueue("Report.docx")
    print(f"   {printer}")
    print("   Printing documents in order...")
    printer.dequeue()
    printer.dequeue()
    print(f"   {printer}")


def error_handling_demo():
    """Demonstrate error handling"""
    print("\n" + "="*50)
    print("ERROR HANDLING")
    print("="*50)
    
    stack = Stack()
    print("\nTrying to pop from empty stack:")
    try:
        stack.pop()
    except IndexError as e:
        print(f"   Error caught: {e}")
    
    queue = Queue()
    print("\nTrying to peek at empty queue:")
    try:
        queue.peek()
    except IndexError as e:
        print(f"   Error caught: {e}")


# ============================================
# RUN ALL DEMONSTRATIONS
# ============================================

if __name__ == "__main__":
    print("\n🎓 STACK AND QUEUE DATA STRUCTURES IN PYTHON")
    
    demonstrate_stack()
    demonstrate_queue()
    practical_examples()
    error_handling_demo()
    
    print("\n" + "="*50)
    print("All demonstrations complete!")
    print("="*50)