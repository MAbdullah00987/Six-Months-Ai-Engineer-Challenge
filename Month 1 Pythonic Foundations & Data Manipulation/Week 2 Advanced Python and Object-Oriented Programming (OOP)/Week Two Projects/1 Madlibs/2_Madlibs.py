

#Adventure Story MadLibs
# MADLIBS PROJECT about ADVENTURE STORY
# This program creates a fun adventure story using user inputs


print("=" * 120)
print("Welcome to Adventure Story Madlibs!")
print("=" * 120)

# Collect inputs from the user - Nouns

print("First, lets get some nouns (person, place or thing):")

name = input("Enter a name: ")
animal = input("Enter a animal: ")
place = input("Enter a place: ")
food = input("Enter a food: ")

# Collect inputs - Adjectives (describing words)

print("\nNow, give me some adjectives (describing words): ")

# Ask for first adjective
adjective1 = input("Enter an adjective: ")

# Ask for first adjective
adjective2 = input("Enter another adjective: ")

# Ask for first adjective
adjective3 = input("Enter one more adjective: ")

# Collect inputs - Verbs (action words)

print("\nGreat! I need some verbs(action verbs):")

# Ask for verb ending in -ing
verb_ing = input("Enter a verb ending in -ing: ")

# Ask for past tense verb
verb_past = input("Enter a verb in the past tense:")

# Collect inputs - Numbers and miscellaneous
print("\nAlmost done! A few more things: ")

number = input("Enter a number: ")

color = input("Enter a color: ")

body_part = input("Enter a body part: ")

# Create the story using f-string with all the inputs

print("\n" + "=" * 120)
print("HERE'S MINE ADVENTURE STORY!")
print("=" * 120)

# Print the complete story with user inputs inserted

story = f"""
Once upon a time, there was a {adjective1} person named {name}. 
{name} love {verb_ing} with their pet {animal} named Fluffy.

One day, they decided to visit the {adjective2} {place}.
On the way, they found {number} pieces of {color} {food}.

"{food} is my favourite!" shouted {name}, jumping up and down on one {body_part}.

Suddenly a {adjective3} {animal} appeared and {verb_past} right past them! 
It was the most exciting day ever at the {place}!

The End!
"""

# Display the completed story
print(story)

# Print closing message
print("=" * 120)
print("Thanks for playing MadLibs!")
print("=" * 120)
 