
#Shape Calculator Project - Complete Guide
#Shape Calculator: Define a base Shape class and inherit from it to create Circle, Rectangle, etc., with methods to calculate area and perimeter.


import math
from abc import ABC, abstractmethod

# ============================================
# BASE CLASS - Shape (Abstract)
# ============================================
class Shape(ABC):
    """
    Abstract base class for all shapes.
    This class defines the blueprint that all shapes must follow.
    """
    
    def __init__(self, name):
        """Constructor - called when creating a new shape object"""
        self.name = name
    
    @abstractmethod
    def area(self):
        """Abstract method - must be implemented by child classes"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Abstract method - must be implemented by child classes"""
        pass
    
    def display_info(self):
        """Common method available to all shapes"""
        print(f"\n{'='*40}")
        print(f"Shape: {self.name}")
        print(f"Area: {self.area():.2f} square units")
        print(f"Perimeter: {self.perimeter():.2f} units")
        print(f"{'='*40}")


# ============================================
# CHILD CLASS 1 - Circle
# ============================================
class Circle(Shape):
    """Circle class inherits from Shape"""
    
    def __init__(self, radius):
        # Call parent class constructor
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        """Override area method for circle"""
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        """Override perimeter method for circle (circumference)"""
        return 2 * math.pi * self.radius
    
    def diameter(self):
        """Circle-specific method"""
        return 2 * self.radius


# ============================================
# CHILD CLASS 2 - Rectangle
# ============================================
class Rectangle(Shape):
    """Rectangle class inherits from Shape"""
    
    def __init__(self, length, width):
        super().__init__("Rectangle")
        self.length = length
        self.width = width
    
    def area(self):
        """Override area method for rectangle"""
        return self.length * self.width
    
    def perimeter(self):
        """Override perimeter method for rectangle"""
        return 2 * (self.length + self.width)
    
    def is_square(self):
        """Rectangle-specific method"""
        return self.length == self.width


# ============================================
# CHILD CLASS 3 - Square
# ============================================
class Square(Shape):
    """Square class inherits from Shape"""
    
    def __init__(self, side):
        super().__init__("Square")
        self.side = side
    
    def area(self):
        """Override area method for square"""
        return self.side ** 2
    
    def perimeter(self):
        """Override perimeter method for square"""
        return 4 * self.side
    
    def diagonal(self):
        """Square-specific method"""
        return self.side * math.sqrt(2)


# ============================================
# CHILD CLASS 4 - Triangle
# ============================================
class Triangle(Shape):
    """Triangle class inherits from Shape"""
    
    def __init__(self, side_a, side_b, side_c):
        super().__init__("Triangle")
        self.side_a = side_a
        self.side_b = side_b
        self.side_c = side_c
    
    def area(self):
        """Override area method using Heron's formula"""
        s = self.perimeter() / 2  # semi-perimeter
        return math.sqrt(s * (s - self.side_a) * (s - self.side_b) * (s - self.side_c))
    
    def perimeter(self):
        """Override perimeter method for triangle"""
        return self.side_a + self.side_b + self.side_c
    
    def is_valid(self):
        """Triangle-specific method to check validity"""
        return (self.side_a + self.side_b > self.side_c and
                self.side_b + self.side_c > self.side_a and
                self.side_a + self.side_c > self.side_b)


# ============================================
# DEMONSTRATION - How to use the classes
# ============================================
def main():
    """Main function to demonstrate the shape calculator"""
    
    print("SHAPE CALCULATOR - DEMONSTRATION")
    print("="*40)
    
    # Create different shape objects
    circle = Circle(radius=5)
    rectangle = Rectangle(length=10, width=6)
    square = Square(side=7)
    triangle = Triangle(side_a=3, side_b=4, side_c=5)
    
    # Display information for each shape
    shapes = [circle, rectangle, square, triangle]
    
    for shape in shapes:
        shape.display_info()
    
    # Demonstrate shape-specific methods
    print("\n" + "="*40)
    print("SHAPE-SPECIFIC METHODS:")
    print("="*40)
    print(f"Circle diameter: {circle.diameter():.2f} units")
    print(f"Is rectangle a square? {rectangle.is_square()}")
    print(f"Square diagonal: {square.diagonal():.2f} units")
    print(f"Is triangle valid? {triangle.is_valid()}")
    
    # Interactive menu
    print("\n" + "="*40)
    print("INTERACTIVE MODE")
    print("="*40)
    
    while True:
        print("\nChoose a shape to calculate:")
        print("1. Circle")
        print("2. Rectangle")
        print("3. Square")
        print("4. Triangle")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            radius = float(input("Enter radius: "))
            c = Circle(radius)
            c.display_info()
            
        elif choice == '2':
            length = float(input("Enter length: "))
            width = float(input("Enter width: "))
            r = Rectangle(length, width)
            r.display_info()
            
        elif choice == '3':
            side = float(input("Enter side length: "))
            s = Square(side)
            s.display_info()
            
        elif choice == '4':
            a = float(input("Enter side a: "))
            b = float(input("Enter side b: "))
            c = float(input("Enter side c: "))
            t = Triangle(a, b, c)
            if t.is_valid():
                t.display_info()
            else:
                print("\nInvalid triangle! Sum of any two sides must be greater than the third.")
                
        elif choice == '5':
            print("\nThank you for using Shape Calculator!")
            break
        else:
            print("\nInvalid choice! Please try again.")


# ============================================
# RUN THE PROGRAM
# ============================================
if __name__ == "__main__":
    main()