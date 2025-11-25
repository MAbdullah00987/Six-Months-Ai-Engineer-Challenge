


#List
#Lists are used to store multiple items in a single variable.
#Lists are one of 4 built-in data types in Python used to store collections of data, the other 3 are Tuple, Set, and Dictionary, all with different qualities and usage.
#Lists are created using square brackets:


#ExampleGet your own Python Server
#Create a List:

thislist = ["apple", "banana", "cherry"]
print(thislist)


#List Items
#List items are ordered, changeable, and allow duplicate values.
#List items are indexed, the first item has index [0], the second item has index [1] etc.
#Ordered
#When we say that lists are ordered, it means that the items have a defined order, and that order will not change.
#f you add new items to a list, the new items will be placed at the end of the list.


'''
#Looping Through an Entire List 

magicians = ['alice', 'david', 'carolina']
for magician in magicians:
 print(magician)
 
'''
 
 
#Doing More Work Within a for Loop

'''
magicians = ['alice', 'david', 'carolina']
for magician in magicians:
 print(f"{magician.title()}, that was a great trick!")
 
 
 
magicianss = ['hassig','qasim','ammar']
 
for magician in magicianss:
     print(f"{magician.title()}, that makes him the best magician team")
    
     
     
simalarities = ['smart work','possibilities','best practices']

for similarity in simalarities:
 print(f"{similarity.title()} ,that makes him special.")
 
 
 
# A Closer Look at Looping

cats = ['Chinku','minku','temu']
for cat in cats:
    print(f"{cat.title()},my favourites pet names.")
    
    
    dogs= ['dogy','bob','dobby']
for dog in dogs:
    print(f"{dog.title()}, my favourite dog names.")
    
    
    list_of_items = ['python','django','numpy']
for item in list_of_items:
    print(f"{item.title()}, my favourite list of items.")
    

'''

#Doing More Work Within a for Loop
'''
Animals= ['civet cat', 'leopard', 'cheetah']
for Animal in Animals:
 print(f"{Animal.title()}, wild animals that can kill people.\n")
 print(f"{Animal.title()}, stay away from wild animals.")
 
 #Let’s add a second line to our message, telling each magician that we’re
#looking forward to their next trick:

'''

#Doing Something After a for Loop

'''
Cats = ['Billy', 'Bill', 'Drill']
for Cat in Cats:
 print(f"{Cat.title()}, that was some favourite cat names!.")
 print(f"I can't wait to see your next trick, {Cat.title()}.\n")
 
print("Thank you, everyone. That was a great magic show!")

Dogs = ['Doggy', 'Scoob', 'Scooby']
for Dog in Dogs:
 print(f"{Dog.title()}, that was some favourite cat names!.")
 print(f"I can't wait to see your next trick, {Dog.title()}.\n")
 
print("Thank you, everyone. That was a great magic show!")

The first two calls to print() are repeated once for each magician in the
list, as you saw earlier. However, because the last line is not indented, it’s
printed only once:

Doggy, that was some favourite cat names!.
Scoob, can't wait to see your next trick.
Scooby, that was some favourite cat names!.  

'''
#Avoiding Indentation Errors
#First Error
#Forgetting to Indent

'''
magicians = ['alice', 'david', 'carolina']
for magician in magicians:
print(magician)


#File "magicians.py", line 3
# print(magician)
# ^
#IndentationError: expected an indented block after 'for' statement on line 2

#print(magician.title()) Solution

'''

#Forgetting to Indent Additional Lines

'''
magicians = ['alice', 'david', 'carolina']
for magician in magicians:
 print(f"{magician.title()}, that was a great trick!")
#print(f"I can't wait to see your next trick, {magician.title()}.\n")
#Solution this is logical error so python does gives an error
#print(f"{magician.title()},I can't wait to see your next trick.")
'''

#Indenting Unnecessarily
'''
message = "Hello Python world!"
print(message)

#You can avoid unexpected indentation errors by indenting only when
##you have a specific reason to do so. In the programs you’re writing at this
#point, the only lines you should indent are the actions you want to repeat
#for each item in a for loop.

print(message) #we cant print our message with spaces before writing print

'''
#Indenting Unnecessarily After the Loop
'''

magicians = ['alice', 'david', 'carolina']
for magician in magicians:
 print(f"{magician.title()}, that was a great trick!")
print(f"I can't wait to see your next trick, {magician.title()}.\n")
print("Thank you everyone, that was a great magic show!")


#Because the last line 1 is indented, it’s printed once for each person in
#the list:
#Alice, that was a great trick!
#I can't wait to see your next trick, Alice.
#Thank you everyone, that was a great magic show!
#David, that was a great trick!
#I can't wait to see your next trick, David.
#Thank you everyone, that was a great magic show!
#Carolina, that was a great trick!
#I can't wait to see your next trick, Carolina.
#Thank you everyone, that was a great magic show!

#This is another logical error, similar to the one in “Forgetting to Indent
##Additional Lines” on page 54. Because Python doesn’t know what you’re
#trying to accomplish with your code, it will run all code that is written in
#valid syntax. If an action is repeated many times when it should be executed
#only once, you probably need to unindent the code for that action

'''

#Forgetting the Colon

'''
magicians = ['alice', 'david', 'carolina']
for magician in magicians
 print(magician)
#56   Chapter 4
#If you accidentally forget the colon 1, you’ll get a syntax error because
#Python doesn’t know exactly what you’re trying to do:
# File "magicians.py", line 2
# for magician in magicians
# ^
#SyntaxError: expected ':'

'''

#Making Numerical Lists
'''
#Many reasons exist to store a set of numbers. For example, you’ll need to
#keep track of the positions of each character in a game, and you might want
#Working with Lists   57
#to keep track of a player’s high scores as well. In data visualizations, you’ll
#almost always work with sets of numbers, such as temperatures, distances,
#population sizes, or latitude and longitude values, among other types of
#numerical sets.

#Using the range() Function

for value in range(1,5):
    print(value)

for value in range(1,6):
    print(value)
    '''

#Using range() to Make a List of Numbers

'''
#Simple Range
numbers = list(range(10))
print(numbers)

#Specifying Start and End
numbers = list(range(5, 10))
print(numbers)

#Counting with a Step
even_numbers = list(range(2, 11, 2))
print(even_numbers)

#Counting Backwards
countdown = list(range(5, 0, -1))
print(countdown)

#Creating a List of Squares
squares = [value**2 for value in range(1, 11)]
print(squares)



squares = []
for value in range(1, 11):
 square = value ** 2
 squares.append(square)
 print(squares)



#To write this code more concisely, omit the temporary variable square
#and append each new value directly to the list:  

squares = []
for value in range(1,11):
 squares.append(value**2)
print(squares)
'''

#Simple Statistics with a List of Numbers
'''
A few Python functions are helpful when working with lists of numbers. For
example, you can easily find the minimum, maximum, and sum of a list of
numbers:
 digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
 min(digits)
0
 max(digits)
9
 sum(digits)
45
'''

#List Comprehensions
'''
#The following example builds the same list of square numbers you saw
#earlier but uses a list comprehension:

squares = [value**2 for value in range(1, 11)]
print(squares)
'''

#Working with Part of a List
#In Chapter 3 you learned how to access single elements in a list, and in this
##chapter you’ve been learning how to work through all the elements in a
#list. You can also work with a specific group of items in a list, called a slice in
#Python.

#Slicing a List
''''
players = ['charles', 'martina', 'michael', 'florence', 'eli']
print(players[2:3])
'''
#Copying a List
''''
my_foods = ['pizza', 'falafel', 'carrot cake']
friend_foods = my_foods[:]

print("My favorite foods are:")
print(my_foods)
print("\nMy friend's favorite foods are:")
print(friend_foods)

'''

#Defining a Tuple

#A tuple looks just like a list, except you use parentheses instead of square
#brackets. Once you define a tuple, you can access individual elements by
#using each item’s index, just as you would for a list.

#For example, if we have a rectangle that should always be a certain size, we
#can ensure that its size doesn’t change by putting the dimensions into a tuple:
dimensions = (200, 50)
print(dimensions[0])
print(dimensions[1])

#We define the tuple dimensions, using parentheses instead of square
##brackets. Then we print each element in the tuple individually, using the
#same syntax we’ve been using to access elements in a list:


def get_user_info():
    name = "Alice"
    age = 30
    city = "London"
    return (name, age, city) # The function returns a single tuple

info = get_user_info()
print(info)

def get_user_info():
    name = "abby"
    age = 40
    city = "London"
    return(name,age,city)

info = get_user_info()
print(info)
    
    
 
   
    

