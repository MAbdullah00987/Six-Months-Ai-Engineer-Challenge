

#Personal Library Catalog Project 
#Personal Library Catalog: Use OOP to model Book and Library objects to manage a collection of books.

class Book:
    """Represents a single book in the library"""
    
    def __init__(self, title, author, isbn, year):
        """
        Initialize a new book
        
        Parameters:
        - title: Book title (string)
        - author: Author name (string)
        - isbn: ISBN number (string)
        - year: Publication year (int)
        """
        self.title = title
        self.author = author
        self.isbn = isbn
        self.year = year
        self.is_available = True  # Book starts as available
    
    def borrow(self):
        """Mark book as borrowed"""
        if self.is_available:
            self.is_available = False
            return True
        return False
    
    def return_book(self):
        """Mark book as returned"""
        if not self.is_available:
            self.is_available = True
            return True
        return False
    
    def __str__(self):
        """String representation of the book"""
        status = "Available" if self.is_available else "Borrowed"
        return f"'{self.title}' by {self.author} ({self.year}) - ISBN: {self.isbn} [{status}]"


class Library:
    """Manages a collection of books"""
    
    def __init__(self, name):
        """
        Initialize a new library
        
        Parameters:
        - name: Library name (string)
        """
        self.name = name
        self.books = []  # Empty list to store Book objects
    
    def add_book(self, book):
        """
        Add a book to the library
        
        Parameters:
        - book: Book object to add
        """
        # Check if ISBN already exists
        for existing_book in self.books:
            if existing_book.isbn == book.isbn:
                print(f"Book with ISBN {book.isbn} already exists!")
                return False
        
        self.books.append(book)
        print(f"Added: {book.title}")
        return True
    
    def remove_book(self, isbn):
        """
        Remove a book by ISBN
        
        Parameters:
        - isbn: ISBN of book to remove (string)
        """
        for i, book in enumerate(self.books):
            if book.isbn == isbn:
                removed_book = self.books.pop(i)
                print(f"🗑️ Removed: {removed_book.title}")
                return True
        
        print(f"Book with ISBN {isbn} not found!")
        return False
    
    def search_by_title(self, title):
        """Search for books by title (partial match)"""
        results = []
        for book in self.books:
            if title.lower() in book.title.lower():
                results.append(book)
        return results
    
    def search_by_author(self, author):
        """Search for books by author (partial match)"""
        results = []
        for book in self.books:
            if author.lower() in book.author.lower():
                results.append(book)
        return results
    
    def search_by_isbn(self, isbn):
        """Search for a book by exact ISBN"""
        for book in self.books:
            if book.isbn == isbn:
                return book
        return None
    
    def borrow_book(self, isbn):
        """Borrow a book by ISBN"""
        book = self.search_by_isbn(isbn)
        if book:
            if book.borrow():
                print(f"You borrowed: {book.title}")
                return True
            else:
                print(f"'{book.title}' is already borrowed!")
                return False
        else:
            print(f"Book with ISBN {isbn} not found!")
            return False
    
    def return_book(self, isbn):
        """Return a borrowed book by ISBN"""
        book = self.search_by_isbn(isbn)
        if book:
            if book.return_book():
                print(f"You returned: {book.title}")
                return True
            else:
                print(f"'{book.title}' was not borrowed!")
                return False
        else:
            print(f"Book with ISBN {isbn} not found!")
            return False
    
    def display_all_books(self):
        """Display all books in the library"""
        if not self.books:
            print("📭 The library is empty!")
            return
        
        print(f"\n{self.name} - Complete Catalog ({len(self.books)} books)")
        print("=" * 80)
        for i, book in enumerate(self.books, 1):
            print(f"{i}. {book}")
        print("=" * 80)
    
    def display_available_books(self):
        """Display only available books"""
        available = [book for book in self.books if book.is_available]
        
        if not available:
            print("No available books!")
            return
        
        print(f"\nAvailable Books ({len(available)})")
        print("=" * 80)
        for i, book in enumerate(available, 1):
            print(f"{i}. {book}")
        print("=" * 80)
    
    def get_statistics(self):
        """Display library statistics"""
        total = len(self.books)
        available = sum(1 for book in self.books if book.is_available)
        borrowed = total - available
        
        print(f"\n{self.name} - Statistics")
        print("=" * 40)
        print(f"Total Books: {total}")
        print(f"Available: {available}")
        print(f"Borrowed: {borrowed}")
        print("=" * 40)


# ============================================
# DEMO: Using the Library System
# ============================================

def main():
    # Create a library
    my_library = Library("My Personal Library")
    
    # Create some books
    book1 = Book("The Great Gatsby", "F. Scott Fitzgerald", "978-0743273565", 1925)
    book2 = Book("To Kill a Mockingbird", "Harper Lee", "978-0061120084", 1960)
    book3 = Book("1984", "George Orwell", "978-0451524935", 1949)
    book4 = Book("Pride and Prejudice", "Jane Austen", "978-0141439518", 1813)
    book5 = Book("The Catcher in the Rye", "J.D. Salinger", "978-0316769174", 1951)
    
    # Add books to library
    print("\nAdding Books...")
    my_library.add_book(book1)
    my_library.add_book(book2)
    my_library.add_book(book3)
    my_library.add_book(book4)
    my_library.add_book(book5)
    
    # Display all books
    my_library.display_all_books()
    
    # Borrow some books
    print("\nBorrowing Books...")
    my_library.borrow_book("978-0743273565")  # The Great Gatsby
    my_library.borrow_book("978-0451524935")  # 1984
    
    # Display available books
    my_library.display_available_books()
    
    # Search for books
    print("\nSearching for 'Pride'...")
    results = my_library.search_by_title("Pride")
    for book in results:
        print(f"  Found: {book}")
    
    print("\nSearching for author 'Orwell'...")
    results = my_library.search_by_author("Orwell")
    for book in results:
        print(f"  Found: {book}")
    
    # Return a book
    print("\nReturning Books...")
    my_library.return_book("978-0743273565")  # The Great Gatsby
    
    # Show statistics
    my_library.get_statistics()
    
    # Try to remove a book
    print("\nRemoving a Book...")
    my_library.remove_book("978-0316769174")  # The Catcher in the Rye
    
    # Final display
    my_library.display_all_books()


if __name__ == "__main__":
    main()