
#link = https://py3.umsi.education/ns/books/published/fopp/PythonTurtle/OurFirstTurtleProgram.html
#link = https://www.w3schools.com/colors/default.asp

'''
import turtle #allow us to use turtle library
wn = turtle.Screen() #Create a graphics window 
alex = turtle.Turtle() #Create a turtle named alex
alex.forward(150) #Tell alex to move forward by 150 units
alex.left(90) #Tell alex to turn left by 90 degrees
alex.forward(75) #Compolete the second side side of rectangle
alex.salary = 50000
print(alex.salary)
alex.left(90)
alex.forward(150)
alex.left(90)
alex.forward(75)
'''
#Explanation:
#Here are a couple of things you’ll need to understand about this program.

#The first line tells Python to load a module named turtle. That module brings us two new types that we can use: the Turtle type, and the Screen type. The dot notation turtle.Turtle means “The Turtle type that is defined within the turtle module”. (Remember that Python is case sensitive, so the module name, turtle, with a lowercase t, is different from the type Turtle because of the uppercase T.)
#We then create and open what the turtle module calls a screen (we would prefer to call it a window, or in the case of this web version of Python simply a canvas), which we assign to variable wn. Every window contains a canvas, which is the area inside the window on which we can draw.
#In line 3 we create a turtle. The variable alex is made to refer to this turtle. These first three lines set us up so that we are ready to do some drawing.
#In lines 4-6, we instruct the object alex to move and to turn. We do this by invoking or activating alex’s methods — these are the instructions that all turtles know how to respond to.

'''
import turtle
wn = turtle.Screen()
ella = turtle.Turtle()
ella.right(90)
ella.forward(150)
ella.left(90)
ella.forward(75)



import turtle
wn= turtle.Screen()
maria=turtle.Turtle()
maria.right(45)
maria.forward(75)
maria.left(90)
maria.forward(90)
maria.left(90)
maria.forward(90)
maria.left(90)
maria.forward(90)

'''

'''
import turtle

wn = turtle.Screen()
wn.bgcolor("blue")        # set the window background color

tess = turtle.Turtle()
tess.color("red")              # make tess blue
tess.pensize(3)                 # set the width of her pen

tess.forward(50)
tess.left(120)
tess.forward(50)
tess.left(120)
tess.forward(50)
tess.left(120)


wn.exitonclick()                # wait for a user click on the canvas
'''

import turtle 
wn= turtle.Screen()
wn.bgcolor("teal")
bob=turtle.Turtle()
bob.color("yellow")
bob.pensize(10)

bob.forward(200)
bob.left(90)
bob.forward(200)
bob.left(90)
bob.forward(200)
bob.left(90)
bob.forward(200)
wn.exitonclick()

