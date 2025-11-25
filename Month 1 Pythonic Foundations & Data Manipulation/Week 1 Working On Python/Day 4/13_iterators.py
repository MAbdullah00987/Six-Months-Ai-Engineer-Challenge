

#Python Iterators

#An iterator is an object that contains a countable number of values.
#An iterator is an object that can be iterated upon, meaning that you can traverse through all the values.
#Technically, in Python, an iterator is an object which implements the iterator protocol, which consist of the methods __iter__() and __next__().
#Iterator vs Iterable
#Lists, tuples, dictionaries, and sets are all iterable objects. They are iterable containers which you can get an iterator from.
#All these objects have a iter() method which is used to get an iterator:
#ExampleGet your own Python Server
#Return an iterator from a tuple, and print each value:

mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit))
print(next(myit))
print(next(myit))


#Create an Iterator
#To create an object/class as an iterator you have to implement the methods __iter__() and __next__() to your object.

#As you will learned in the Python Classes/Objects chapter, all classes have a function called __init__(), which allows you to do some initializing when the object is being created.

#The __iter__() method acts similar, you can do operations (initializing etc.), but must always return the iterator object itself.

#The __next__() method also allows you to do operations, and must return the next item in the sequence.

#Example
#Create an iterator that returns numbers, starting with 1, and each sequence will increase by one (returning 1,2,3,4,5 etc.):

class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self

  def __next__(self):
    x = self.a
    self.a += 1
    return x

myclass = MyNumbers()
myiter = iter(myclass)

print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))


#Examples:

#1. The for loop (Implicit Iterator)
#This is the most common way to use an iterator in Python. The for loop automatically gets an iterator from the list and calls next() on it until it's empty.


# The list 'fruits' is the ITERABLE
fruits = ["apple", "banana", "cherry"]

# The 'for' loop automatically creates an ITERATOR
# and loops through it.
print("Looping with a for loop:")
for fruit in fruits:
    print(fruit)


#2. Manual Iteration (Explicit Iterator)
#You can manually get an iterator from an iterable using the iter() function. You then get each item using the next() function. This shows what the for loop does behind the scenes.

# Get the iterator manually
my_list = ['a', 'b', 'c']
my_iterator = iter(my_list)

print(f"The iterator object is: {my_iterator}")

# Get items one by one
print(next(my_iterator))  # Output: a
print(next(my_iterator))  # Output: b
print(next(my_iterator))  # Output: c

# If you call next() again, it raises a 'StopIteration' error
# because the iterator is empty.
# print(next(my_iterator))  # Raises StopIteration

#3. Iterating Over a String
#Strings are iterables, just like lists. When you iterate over a string, you get one character at a time.

my_string = "hello"

# The for loop gets an iterator for the string
for char in my_string:
    print(char)
    

#4. Iterating Over a Dictionary 📖
#When you iterate over a dictionary, the iterator gives you the keys by default. To get keys and values, you use the .items() method, which returns an iterator of (key, value) pairs.


person = {
    "name": "Alice",
    "age": 30
}

# .items() returns an iterator
print("Iterating over items:")
for key, value in person.items():
    print(f"{key}: {value}")


#5. map() Function (A Lazy Iterator)
#Some functions, like map(), don't compute all their results at once. They return an iterator that generates each result only when you ask for it. This is very memory-efficient.

numbers = [1, 2, 3, 4]

# map() returns an iterator. 
# It doesn't create a new list immediately.
squared_iterator = map(lambda x: x * x, numbers)

print(f"The iterator object: {squared_iterator}")

# The code inside map() runs when next() is called
print(next(squared_iterator))  # Output: 1 (1*1)
print(next(squared_iterator))  # Output: 4 (2*2)

# You can also use a for loop, which will call next() for you
print("Using a loop for the rest:")
for item in squared_iterator:
    print(item)

