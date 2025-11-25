

#Classes in OOP Python
#What is a Class?
#A class is a blueprint or template for creating objects. Think of it like an architectural blueprint for a house - the blueprint itself isn't a house, but you can build many houses from it.

# Class = Blueprint
class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def drive(self):
        return f"{self.brand} {self.model} is driving"

# Objects = Actual instances built from the blueprint
car1 = Car("Toyota", "Camry")
car2 = Car("Honda", "Civic")

print(car1.drive())  # Toyota Camry is driving
print(car2.drive())  # Honda Civic is driving

#The 4 Fundamental Pillars of OOP in Python
#There are 4 main fundamentals (also called pillars or principles) of OOP:


# | Pillar          | Purpose                   | Real-World Example
# | Encapsulation   | Data hiding & protection  | ATM machine - you can't directly access the cash, only through authorized operations
# | Inheritance     | Code reuasability         | Family tree - children inherit traits from parents
# | Polymorphism    | Multiple forms            | Remote control - same "power" button works differently for TV, AC, etc.
# | Abstraction     | Hide complexity           | Car - you don't need to know how the engine works to drive it

# 1:What is Encapsulation? Encapsulation is the practice of:

#Bundling data (attributes) and methods together in a class
#Hiding internal details from the outside world
#Controlling access to data through methods
#Think of it like a capsule/pill  - the medicine is wrapped inside, and you can't directly touch the medicine, only take the whole pill.

#1. The Bank Account (Data Protection)

# --- GOOD: With Encapsulation ---
class BankAccount:
    def __init__(self, owner, initial_balance):
        self.owner = owner
        self.__balance = initial_balance  # Private variable

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited {amount}. New balance: {self.__balance}")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if amount > self.__balance:
            print("Insufficient funds.")
        elif amount <= 0:
            print("Withdrawal amount must be positive.")
        else:
            self.__balance -= amount
            print(f"Withdrew {amount}. New balance: {self.__balance}")

    def get_balance(self):
        return self.__balance

# Now, you are forced to use the safe public methods:
my_account = BankAccount("Ali", 100)
my_account.deposit(50)
my_account.withdraw(200)  # Prints "Insufficient funds."

# You can't do this! This line will cause an error or have no effect.
# my_account.__balance = -5000

#2: The Car (Hiding Complexity)

class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model
        self.__engine_running = False # Private

    # --- Private Methods (The Complex Internals) ---
    def __check_fuel(self):
        print("Checking fuel...")
        return True # Assume we have fuel

    def __ignite_spark_plugs(self):
        print("Igniting spark plugs...")
        return True
    
    def __engage_starter(self):
        print("Engaging starter motor...")
        return True

    # --- Public Method (The Simple Interface) ---
    def start_engine(self):
        if self.__engine_running:
            print("Engine is already running.")
            return

        # The public method calls all the complex private ones
        if self.__check_fuel() and self.__engage_starter() and self.__ignite_spark_plugs():
            self.__engine_running = True
            print("Vroom! Engine started.")
        else:
            print("Engine failed to start.")

# The user only needs to know this one, simple command:
my_car = Car("Toyota", "Camry")
my_car.start_engine()


#2: Inheritance
#What is Inheritance? 
#Inheritance is when a new class (child) is created based on an existing class (parent), automatically getting all the parent's attributes and methods.
#Think of it like family inheritance - children inherit traits from their parents (eye color, height, etc.) but can also have their own unique traits.Why Do We Need Inheritance?✅ Code Reusability - Don't write the same code twice
# Logical Hierarchy - Model real-world relationships
# Easy Maintenance - Update parent class, all children benefit
# Extensibility - Add new features without changing existing code
# Organize Code - Group related classes together

#The Animal Kingdom (Basic Inheritance)

# 1. PARENT CLASS (Superclass)
class Animal:
    def __init__(self, name):
        self.name = name
    
    def eat(self):
        print(f"{self.name} is eating.")

# 2. CHILD CLASS (Subclass)
#    Notice (Animal) in the definition. This is how we inherit.
class Dog(Animal):
    def bark(self):
        print(f"{self.name} says Woof!")

# --- How it's used ---
my_dog = Dog("Buddy")

# This method was INHERITED from the Animal class
my_dog.eat()  

# This method is specific to the Dog class
my_dog.bark()

#Example No.2: The User System (Adding New Functionality)

# 1. PARENT CLASS
class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email
    
    def login(self):
        print(f"User {self.username} has logged in.")

# 2. CHILD CLASS
class Admin(User):
    def ban_user(self, user_to_ban):
        print(f"--- ADMIN ACTION ---")
        print(f"{self.username} is banning {user_to_ban.username}.")

# --- How it's used ---
basic_user = User("ali", "ali@example.com")
admin_user = Admin("sara_admin", "sara@example.com")

# Both objects can use the inherited login() method
basic_user.login()
admin_user.login()

# But only the Admin object can use the special method
admin_user.ban_user(basic_user)


#Polymorphism

#What is Polymorphism? 
#Polymorphism means "many forms" - it's the ability of different objects to respond to the same method call in their own unique way.
#Think of it like a TV remote 📺 - the "power" button works on TV, AC, fan, etc., but each device responds differently to the same button press.Why Do We Need Polymorphism?✅ Flexibility - Same interface, different behaviors
# Code Simplicity - Write less code, handle multiple types
# Extensibility - Add new types without changing existing code
# Cleaner Code - More intuitive and readable
# Duck Typing - "If it walks like a duck and quacks like a duck, it's a duck"

#Types of Polymorphism
#Method Overriding (Runtime) - Child class changes parent's method
#Method Overloading (Compile-time) - Same name, different parameters
#Operator Overloading - Change how operators (+, -, *, etc.) work

#Example 1: Polymorphism with Inheritance (Method Overriding)
class Animal:
    def speak(self):
        print("This animal makes a sound.")

class Dog(Animal):
    # This 'speak' OVERRIDES the parent's 'speak'
    def speak(self):
        print("Woof!")

class Cat(Animal):
    # This 'speak' ALSO OVERRIDES the parent's 'speak'
    def speak(self):
        print("Meow!")

# --- This is polymorphism in action ---
my_dog = Dog()
my_cat = Cat()

# We put different objects in a list
animals = [my_dog, my_cat]

# We call the SAME .speak() method on each object
# but get DIFFERENT behavior.
for animal in animals:
    animal.speak()

#Example 2: Polymorphism with "Duck Typing" (No Inheritance)

class Bird:
    def fly(self):
        print("The bird is flapping its wings.")

class Airplane:
    def fly(self):
        print("The airplane is engaging its engines.")

class Dragon:
    def fly(self):
        print("The dragon is soaring on magical winds.")

# --- This function is polymorphic ---
# It doesn't care if 'thing' is a Bird, Airplane, or Dragon.
# It only cares that 'thing' has a .fly() method.
def make_it_fly(thing):
    thing.fly()

# Create objects from different, unrelated classes
my_bird = Bird()
my_plane = Airplane()
my_dragon = Dragon()

# We can pass all of them to the SAME function
make_it_fly(my_bird)
make_it_fly(my_plane)
make_it_fly(my_dragon)

#Abstraction
#What is Abstraction? 
#Abstraction is the process of:
#Hiding complex implementation details
#Showing only essential features to the user
#Defining "what to do" but not "how to do it"
#Think of it like driving a car  - you use the steering wheel, pedals, and gears without knowing how the engine, transmission, or fuel injection works internally.Why Do We Need Abstraction?✅ Simplicity - Hide complexity, show only what's needed
# Security - Hide implementation details
# Maintainability - Change internals without affecting users
# Focus - Users focus on WHAT to do, not HOW it's done
# Standardization - Force consistent interface across implementations

#Example1: Abstraction as a Function

# --- Calculating for item 1 ---
price1 = 100
discount1 = 10
tax_rate = 0.05

discounted_price1 = price1 - discount1
tax1 = discounted_price1 * tax_rate
final_price1 = discounted_price1 + tax1
print(f"Item 1 total: {final_price1}")

# --- Calculating for item 2 ---
price2 = 50
discount2 = 5
# tax_rate is the same

discounted_price2 = price2 - discount2
tax2 = discounted_price2 * tax_rate
final_price2 = discounted_price2 + tax2
print(f"Item 2 total: {final_price2}")



#Additional OOP Concepts (Beyond the 4 Pillars)

#5. Class vs Instance Variables

#6. Static Methods & Class Methods

#7. Magic/Dunder Methods
