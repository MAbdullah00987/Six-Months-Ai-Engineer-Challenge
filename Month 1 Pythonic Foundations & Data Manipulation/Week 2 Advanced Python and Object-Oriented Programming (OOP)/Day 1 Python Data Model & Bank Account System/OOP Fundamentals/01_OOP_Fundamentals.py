
#What is OOP in Python?
#Object-Oriented Programming (OOP) is a programming paradigm that organizes code around "objects" rather than functions and logic. An object combines data (attributes) and behavior (methods) into a single unit.
#Core Concepts of OOP
#1. Classes and Objects

#Class: A blueprint or template for creating objects
#Object: An instance of a class

#2. Encapsulation
#Bundling data and methods together, hiding internal details from the outside world.

#3. Inheritance
#Creating new classes based on existing ones, reusing code.

#4. Polymorphism
#Different classes can be used interchangeably if they share the same interface.


#Why Do We Need OOP in Python?
#1. Code Reusability
#Write code once, use it many times through inheritance and object creation.


#2. Better Organization
#Large projects become manageable when code is organized into classes.


#3. Real-World Modeling
#OOP naturally maps to real-world entities and relationships.

#4. Easier Maintenance
#Changes in one class don't affect others if designed properly.

#5. Data Security
#Encapsulation protects data from unintended modification.

#6. Scalability
#Easy to add new features without breaking existing code.

#7. Team Collaboration
#Multiple developers can work on different classes simultaneously.

#When to Use OOP vs Procedural Programming?
#Use OOP when:

#Building complex applications
#Code needs to be reused
#Modeling real-world entities
#Working in teams
#Building libraries/frameworks

#Use Procedural when:

#Simple scripts
#Quick automation tasks
#Small utilities
#Linear data processing

#
#Practical Example: With vs Without OOP
#Without OOP (Procedural):

def calculate_circle_area(radius):
    return 3.14 * radius * radius

def calculate_rectangle_area(length, width):
    return length * width

circle_area = calculate_circle_area(5)
rect_area = calculate_rectangle_area(4, 6)


#With OOP:
#pythonclass Shape:
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        """Calculate and return the area of the shape"""
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width
    
    def area(self):
        return self.length * self.width

# Easy to extend and maintain
shapes = [Circle(5), Rectangle(4, 6)]

print("Solution 1: Using ABC")
for shape in shapes:
    print(f"Area: {shape.area()}")