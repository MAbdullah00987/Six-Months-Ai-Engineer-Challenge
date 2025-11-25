

#Intro: What we can do with Turtles and Conditionals
#So far, our programs have either been a series of statements which always execute sequentially or
# operations that are applied to each item in an iterable. Yet programs frequently need to be more
# subtle with their behavior. For example, a messaging app might only set a message’s title bold if it has not been read by the user. Or a video game needs to update the position of all the characters that are not asleep. This is done with something called a selection or a conditional statement.

#Examples
'''
import turtle
wn = turtle.Screen()

amy = turtle.Turtle()
amy.pencolor("Pink")
amy.forward(50)
if amy.pencolor() == "Pink":
    amy.right(60)
    amy.forward(100)
else:
    amy.left(60)
    amy.forward(100)

kenji = turtle.Turtle()
kenji.forward(60)
if kenji.pencolor() == "Pink":
    kenji.right(60)
    kenji.forward(100)
else:
    kenji.left(60)
    kenji.forward(100)
'''
#Second Example

import turtle
wn = turtle.Screen()

amy = turtle.Turtle()
amy.pencolor("Pink")
amy.right(170)

colors = ["Purple", "Yellow", "Orange", "Pink", "Orange", "Yellow", "Purple", "Orange", "Pink", "Pink", "Orange", "Yellow", "Purple", "Orange", "Purple", "Yellow", "Orange", "Pink", "Orange", "Purple", "Purple", "Yellow", "Orange", "Pink", "Orange", "Yellow", "Purple", "Yellow"]


for color in colors:
    if amy.pencolor() == "Purple":
        amy.forward(50)
        amy.right(59)
    elif amy.pencolor() == "Yellow":
        amy.forward(65)
        amy.left(98)
    elif amy.pencolor() == "Orange":
        amy.forward(30)
        amy.left(60)
    elif amy.pencolor() == "Pink":
        amy.forward(50)
        amy.right(57)

    amy.pencolor(color)

#The above example combines a for loop with a set of conditional statements. 
# Here, we loop through a list of colors and each iteration checks to see what amy’s pen color is.
# Depending on the pen color, the turtle will move in a certain direction, for a certain distance.
# Before the for loop iterates, amy’s pen color is changed to whatever color is in the for loop and it continues. 
# Note how the color doesn’t change until the end, so that we can start using whatever color amy is set to initally. 
# This means that the last color in the list colors will not be used, though you can see how the icon changes to the appropriate color.

