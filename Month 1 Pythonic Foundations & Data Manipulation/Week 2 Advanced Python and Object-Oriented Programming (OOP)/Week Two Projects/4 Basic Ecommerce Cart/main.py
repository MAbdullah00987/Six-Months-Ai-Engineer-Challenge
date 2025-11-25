
#E-commerce Shopping Cart Project - Complete Guide
#Basic E-commerce Cart: Create Product and ShoppingCart classes to simulate adding items to a cart.

class Product:
    """
    Represents a product in the store.
    Each product has a name, price, and optional description.
    """
    
    def __init__(self, name, price, description=""):
        """
        Initialize a new product.
        
        Args:
            name (str): Product name
            price (float): Product price
            description (str): Optional product description
        """
        self.name = name
        self.price = price
        self.description = description
    
    def __str__(self):
        """
        Returns a string representation of the product.
        This method is called when you use print() on a Product object.
        """
        if self.description:
            return f"{self.name} - ${self.price:.2f} ({self.description})"
        return f"{self.name} - ${self.price:.2f}"
    
    def __repr__(self):
        """
        Returns a detailed representation for debugging.
        """
        return f"Product('{self.name}', {self.price}, '{self.description}')"


class ShoppingCart:
    """
    Represents a shopping cart that holds products.
    Tracks items and their quantities, calculates totals.
    """
    
    def __init__(self):
        """
        Initialize an empty shopping cart.
        Uses a dictionary to store products and their quantities.
        """
        # Dictionary: key = Product object, value = quantity
        self.items = {}
    
    def add_item(self, product, quantity=1):
        """
        Add a product to the cart.
        
        Args:
            product (Product): The product to add
            quantity (int): How many to add (default: 1)
        """
        if quantity <= 0:
            print("Quantity must be positive!")
            return
        
        # If product already in cart, increase quantity
        if product in self.items:
            self.items[product] += quantity
            print(f"Added {quantity} more {product.name}(s) to cart.")
        else:
            # Otherwise, add new product
            self.items[product] = quantity
            print(f"Added {quantity} {product.name}(s) to cart.")
    
    def remove_item(self, product, quantity=None):
        """
        Remove a product from the cart.
        
        Args:
            product (Product): The product to remove
            quantity (int): How many to remove (None = remove all)
        """
        if product not in self.items:
            print(f"{product.name} is not in the cart.")
            return
        
        # Remove all of this product
        if quantity is None:
            del self.items[product]
            print(f"Removed all {product.name}(s) from cart.")
        else:
            # Remove specific quantity
            if quantity >= self.items[product]:
                del self.items[product]
                print(f"Removed all {product.name}(s) from cart.")
            else:
                self.items[product] -= quantity
                print(f"Removed {quantity} {product.name}(s) from cart.")
    
    def get_total(self):
        """
        Calculate the total price of all items in the cart.
        
        Returns:
            float: Total price
        """
        total = 0
        for product, quantity in self.items.items():
            total += product.price * quantity
        return total
    
    def get_item_count(self):
        """
        Get the total number of items in the cart.
        
        Returns:
            int: Total quantity of all items
        """
        return sum(self.items.values())
    
    def display_cart(self):
        """
        Display all items in the cart with details.
        """
        if not self.items:
            print("\nYour cart is empty.")
            return
        
        print("\n" + "="*50)
        print("SHOPPING CART")
        print("="*50)
        
        for product, quantity in self.items.items():
            subtotal = product.price * quantity
            print(f"{product.name}")
            print(f"  Price: ${product.price:.2f} x {quantity} = ${subtotal:.2f}")
            if product.description:
                print(f"  Description: {product.description}")
            print("-" * 50)
        
        print(f"\nTotal Items: {self.get_item_count()}")
        print(f"Total Price: ${self.get_total():.2f}")
        print("="*50)
    
    def clear_cart(self):
        """
        Remove all items from the cart.
        """
        self.items.clear()
        print("Cart cleared!")


# ==================== DEMO USAGE ====================

def main():
    """
    Demonstration of the shopping cart system.
    """
    print("Welcome to the E-Commerce Store!")
    print("="*50)
    
    # Create some products
    laptop = Product("Laptop", 999.99, "15-inch, 16GB RAM")
    mouse = Product("Wireless Mouse", 29.99, "Ergonomic design")
    keyboard = Product("Mechanical Keyboard", 89.99, "RGB backlit")
    monitor = Product("4K Monitor", 399.99, "27-inch display")
    usb_cable = Product("USB-C Cable", 12.99)
    
    # Create a shopping cart
    cart = ShoppingCart()
    
    # Add items to cart
    print("\n--- Adding items to cart ---")
    cart.add_item(laptop, 1)
    cart.add_item(mouse, 2)
    cart.add_item(keyboard, 1)
    cart.add_item(usb_cable, 3)
    
    # Display cart
    cart.display_cart()
    
    # Add more of an existing item
    print("\n--- Adding more mice ---")
    cart.add_item(mouse, 1)
    
    # Remove some items
    print("\n--- Removing 2 USB cables ---")
    cart.remove_item(usb_cable, 2)
    
    # Display updated cart
    cart.display_cart()
    
    # Remove all of an item
    print("\n--- Removing all keyboards ---")
    cart.remove_item(keyboard)
    
    # Add another product
    print("\n--- Adding monitor ---")
    cart.add_item(monitor, 1)
    
    # Final cart display
    cart.display_cart()
    
    # Get specific information
    print(f"\nYou have {cart.get_item_count()} items in your cart.")
    print(f"Your total is: ${cart.get_total():.2f}")


# Run the demo
if __name__ == "__main__":
    main()