
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
'''

'''
import turtle  # import the turtle graphics module
wn = turtle.Screen()  # create and return a new drawing window (screen)
wn.bgcolor("lightgreen")  # set the window background color to light green

# create a turtle named tess for drawing triangular shapes
tess = turtle.Turtle()  # instantiate a new Turtle object and assign to tess
tess.color("purple")  # set tess's pen color to purple
tess.pensize(5)  # set tess's pen thickness to 5

# create a turtle named alex for drawing square shapes
alex = turtle.Turtle()  # instantiate a new Turtle object and assign to alex
alex.color("orange")  # set alex's pen color to orange
alex.pensize(4)  # set alex's pen thickness to 4

# create a turtle named bob for repetitive angled movements
bob = turtle.Turtle()  # instantiate a new Turtle object and assign to bob
bob.color("blue")  # set bob's pen color to blue
bob.pensize(3)  # set bob's pen thickness to 3

# draw an equilateral triangle with tess
tess.forward(80)  # move tess forward by 80 units (first side)
tess.left(120)  # turn tess left by 120 degrees (angle for equilateral triangle)
tess.forward(80)  # move tess forward by 80 units (second side)
tess.left(120)  # turn tess left by 120 degrees
tess.forward(80)  # move tess forward by 80 units (third side)
tess.left(120)  # turn tess left by 120 degrees to complete orientation

# reposition tess to draw a short line in the opposite direction
tess.right(180)  # rotate tess 180 degrees to face the opposite direction
tess.forward(90)  # move tess forward by 90 units

# draw a square with alex
alex.forward(50)  # move alex forward by 50 units (first side)
alex.left(90)  # turn alex left by 90 degrees
alex.forward(50)  # move alex forward by 50 units (second side)
alex.left(90)  # turn alex left by 90 degrees
alex.forward(50)  # move alex forward by 50 units (third side)
alex.left(90)  # turn alex left by 90 degrees
alex.forward(50)  # move alex forward by 50 units (fourth side)
alex.left(90)  # turn alex left by 90 degrees to restore starting orientation

# perform a repeated angled movement pattern with bob (series of left/right/forward)
bob.left(80)  # rotate bob left by 80 degrees
bob.right(120)  # rotate bob right by 120 degrees
bob.forward(140)  # move bob forward by 140 units
bob.left(80)  # rotate bob left by 80 degrees
bob.right(120)  # rotate bob right by 120 degrees
bob.forward(140)  # move bob forward by 140 units
bob.left(80)  # rotate bob left by 80 degrees
bob.right(120)  # rotate bob right by 120 degrees
bob.forward(140)  # move bob forward by 140 units
bob.left(80)  # rotate bob left by 80 degrees
bob.right(120)  # rotate bob right by 120 degrees
bob.forward(140)  # move bob forward by 140 units
bob.left(80)  # rotate bob left by 80 degrees
bob.right(120)  # rotate bob right by 120 degrees
bob.forward(140)  # move bob forward by 140 units
bob.left(80)  # rotate bob left by 80 degrees
bob.right(120)  # rotate bob right by 120 degrees
bob.forward(140)  # move bob forward by 140 units
bob.left(80)  # rotate bob left by 80 degrees
bob.right(120)  # rotate bob right by 120 degrees
bob.forward(140)  # move bob forward by 140 units
bob.left(80)  # rotate bob left by 80 degrees
bob.right(120)  # rotate bob right by 120 degrees
bob.forward(140)  # move bob forward by 140 units
bob.left(80)  # rotate bob left by 80 degrees
bob.right(120)  # rotate bob right by 120 degrees
bob.forward(140)  # move bob forward by 140 units

wn.exitonclick()  # wait for the user to click inside the window, then close it


'''
'''
import turtle
wn = turtle.Screen()
wn.bgcolor("lightblue")

jamal = turtle.Turtle()
jamal.color("green")
jamal.pensize(6)

jamal.right(90)
jamal.forward(150)


jamal.left(90)
jamal.forward(75)

tina = turtle.Turtle()
tina.color("purple")
tina.pensize(4)

tina.left(180)
tina.forward(75)

wn.exitonclick()

'''
'''
import turtle
wn = turtle.Screen()
wn.bgcolor("lightyellow")

jamal = turtle.Turtle()
jamal.color("red")
jamal.pensize(5)

jamal.left(90)
jamal.forward(200)

tina = turtle.Turtle()
tina.color("blue")
tina.pensize(3)
tina.forward(150)

wn.exitonclick()
'''

'''
print("This will execute first ")

for i in range(5):
    print("This item will repeat five times")
    print("This will also item will repeat five times")
    
print("Now we are outside of the loop")
'''
'''
import turtle
wn= turtle.Screen()
wn.bgcolor("lightgrey")

elan = turtle.Turtle()
elan.color("black")
elan.pensize(2)

distance = 50
for i in range(50):
    elan.forward(distance)
    elan.left(90)
    distance = distance + 20
    '''
    
'''
import turtle
wn= turtle.Screen()
wn.bgcolor("lightgrey")

elan = turtle.Turtle()
elan.color("black")
elan.pensize(2)

distance = 1
angle = 90
for i in range(16):
    elan.forward(distance)
    elan.right(angle)
    distance = distance + 2
    angle = angle - 3
    
'''

'''
import turtle
wn = turtle.Screen()
wn.bgcolor("lightgreen")


tess = turtle.Turtle()
tess.color("purple")
tess.shape("turtle")
tess.pensize(5)

dist = 5
tess.up()

for _ in range(50):
    tess.stamp()
    tess.forward(dist)
    tess.right(24)
    dist = dist + 1
wn.exitonclick()
'''

'''
import turtle  # import the turtle graphics module
wn = turtle.Screen()  # create and return a new drawing window (screen) 

jose = turtle.Turtle()  # instantiate a new Turtle object and assign to jose
jose.color("brown")  # set jose's pen color to brown
jose.shape("turtle")  # set jose's shape to turtle
jose.pensize(4)  # set jose's pen thickness to 4
jose.penup()

for size in range(10):
    jose.stamp()
    jose.right(36)
    jose.forward(50)
    jose.forward(-50)
    
wn.exitonclick()
'''

import random
prob = random.random()
print(prob)

diceThrow = random.randrange(1,7)
print(diceThrow)




import turtle
wn = turtle.Screen()
wn.title("The Slicing Operator in Python")
wn.bgcolor("lightblue")
wn.setup(width=800, height=600)
acolor = turtle.Turtle()
acolor.speed(1)
acolor.color("darkblue")
acolor.pensize(3)

for aColor in ["Yellow", "Red", "Green", "Purple", "Orange"]:
    acolor.color(aColor)
    acolor.forward(150)
    acolor.right(90)
    acolor.forward(196)
    acolor.right(70)
    acolor.forward(160)
    acolor.right(80)
    
    wn.exitonclick()





#Summary of Turtle Methods

#Method        Parameters       Description
#Turtle          None           Creates and returns a new turtle object
#forward       distance         Moves the turtle forward
#backward      distance         Moves the turle backward
#right         angle            Turns the turtle clockwise
#left          angle            Turns the turtle counter clockwise
#up            None             Picks up the turtles tail
#down          None             Puts down the turtles tail
#color         color name       Changes the color of the turtle’s tail
#fillcolor     color name       Changes the color of the turtle will use to fill a polygon
#heading       None             Returns the current heading
#position     None              Returns the current position
#goto         x,y               Move the turtle to position x,y
#begin_fill   None              Remember the starting point for a filled polygon
#end_fill     None              Close the polygon and fill with the current fill color
#dot          None              Leave a dot at the current position
#stamp        None              Leaves an impression of a turtle shape at the current location
#shape     shapename            Should be ‘arrow’, ‘triangle’, ‘classic’, ‘turtle’, ‘circle’, or ‘square’
#speed      integer             0 = no animation, fastest; 1 = slowest; 10 = very fast






