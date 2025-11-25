

#Slicing Operator
#The slicing operator [ : ] is a feature in Python used to access a range of items from a sequence, such as a string, list, or tuple.


"""
Python Slice Operator [:] - Complete Guide
Syntax: sequence[start:stop:step]
"""

print("=" * 80)
print("SLICE OPERATOR [:] - COMPLETE GUIDE")
print("=" * 80)

print("\n" + "=" * 80)
print("SLICE OPERATOR SYNTAX AND RULES")
print("=" * 80)

print("""
SYNTAX: sequence[start:stop:step]

PARAMETERS:
  - start: Starting index (inclusive) - default is 0
  - stop:  Ending index (exclusive) - default is end of sequence
  - step:  Step/increment value - default is 1

RULES:
  1. slice[start:stop]       → from start to stop-1
  2. slice[:stop]            → from beginning to stop-1
  3. slice[start:]           → from start to end
  4. slice[:]                → entire sequence (creates a copy)
  5. slice[start:stop:step]  → from start to stop-1, every step elements
  6. slice[::step]           → entire sequence, every step elements
  7. Negative indices count from the end: -1 is last element
  8. Negative step reverses the sequence
  9. Out of range indices don't cause errors (returns what's available)
""")

print("\n" + "=" * 80)
print("BASIC SLICING EXAMPLES")
print("=" * 80)

# Example string
text = "Hello World"
print(f"\nOriginal String: '{text}'")
print(f"Length: {len(text)}")
print(f"Indices: 0={text[0]}, 1={text[1]}, 2={text[2]}, ..., -1={text[-1]}")

print("\n" + "-" * 80)
print("BASIC SLICING:")
print("-" * 80)

print(f"text[0:5]   = '{text[0:5]}'     # From index 0 to 4 (5 is excluded)")
print(f"text[:5]    = '{text[:5]}'     # From start to index 4")
print(f"text[6:]    = '{text[6:]}'     # From index 6 to end")
print(f"text[:]     = '{text[:]}'     # Entire string (copy)")
print(f"text[0:11:2] = '{text[0:11:2]}'    # Every 2nd character from 0 to 10")
print(f"text[::2]   = '{text[::2]}'    # Every 2nd character")
print(f"text[::-1]  = '{text[::-1]}'  # Reverse the string")

print("\n" + "-" * 80)
print("NEGATIVE INDEXING:")
print("-" * 80)

print(f"text[-5:]   = '{text[-5:]}'     # Last 5 characters")
print(f"text[:-5]   = '{text[:-5]}'      # All except last 5")
print(f"text[-5:-1] = '{text[-5:-1]}'    # From -5 to -2")
print(f"text[-1::-1] = '{text[-1::-1]}' # Reverse using negative index")

# Example list
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"\n\nOriginal List: {numbers}")
print(f"Length: {len(numbers)}")

print("\n" + "-" * 80)
print("LIST SLICING:")
print("-" * 80)

print(f"numbers[2:7]    = {numbers[2:7]}       # Elements from index 2 to 6")
print(f"numbers[:4]     = {numbers[:4]}           # First 4 elements")
print(f"numbers[6:]     = {numbers[6:]}           # From index 6 to end")
print(f"numbers[::3]    = {numbers[::3]}          # Every 3rd element")
print(f"numbers[1::2]   = {numbers[1::2]}         # Odd numbers")
print(f"numbers[::2]    = {numbers[::2]}          # Even numbers")
print(f"numbers[::-1]   = {numbers[::-1]}   # Reversed list")
print(f"numbers[7:2:-1] = {numbers[7:2:-1]}       # From 7 to 3 backwards")


print("\n\n" + "=" * 80)
print("SLICE ASSIGNMENT RULES (LISTS ONLY)")
print("=" * 80)

print("""
IMPORTANT: Slice assignment works ONLY with MUTABLE sequences (lists)
           Strings and tuples are IMMUTABLE - they cannot be modified

RULES FOR SLICE ASSIGNMENT:
  1. Can replace a slice with any iterable
  2. The replacement can be longer or shorter than the slice
  3. Can insert elements by using empty slice
  4. Can delete elements by assigning empty list
  5. Strings and tuples DO NOT support slice assignment
  6. Step slicing (with step != 1) requires same length replacement
""")

print("\n" + "-" * 80)
print("SLICE ASSIGNMENT EXAMPLES:")
print("-" * 80)

# Example 1: Basic replacement
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"\nOriginal list: {my_list}")

my_list[2:5] = [20, 30, 40]
print(f"After my_list[2:5] = [20, 30, 40]:  {my_list}")

# Example 2: Replace with different length
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
my_list[2:5] = [100, 200]  # Replacing 3 elements with 2
print(f"After my_list[2:5] = [100, 200]:    {my_list}")

# Example 3: Insert elements
my_list = [0, 1, 2, 3, 4, 5]
print(f"\nOriginal list: {my_list}")
my_list[2:2] = [10, 20, 30]  # Insert at index 2
print(f"After my_list[2:2] = [10, 20, 30]:  {my_list}")

# Example 4: Delete elements
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"\nOriginal list: {my_list}")
my_list[2:5] = []  # Delete elements from index 2 to 4
print(f"After my_list[2:5] = []:           {my_list}")

# Example 5: Replace every 2nd element
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"\nOriginal list: {my_list}")
my_list[::2] = [10, 20, 30, 40, 50]  # Must be same length!
print(f"After my_list[::2] = [10, 20, 30, 40, 50]: {my_list}")

print("\n" + "-" * 80)
print("SLICE ASSIGNMENT ERRORS:")
print("-" * 80)

print("\n# ERROR: Strings are immutable")
print("text = 'Hello'")
print("# text[0:2] = 'Hi'  #  TypeError: 'str' object does not support item assignment")

print("\n# ERROR: Tuples are immutable")
print("my_tuple = (1, 2, 3, 4, 5)")
print("# my_tuple[0:2] = (10, 20)  #  TypeError: 'tuple' object does not support item assignment")

print("\n# ERROR: Step slicing requires same length")
print("my_list = [0, 1, 2, 3, 4, 5]")
print("# my_list[::2] = [10, 20]  #  ValueError: attempt to assign sequence of size 2 to extended slice of size 3")


print("\n\n" + "=" * 80)
print("EXAMPLE 1: TEXT PROCESSOR - Extract and Manipulate Text")
print("=" * 80)

# Original text
message = "Python Programming is Amazing!"
print(f"\n Original Message: '{message}'")
print(f"Length: {len(message)}")

print("\n" + "-" * 80)
print("🔍 EXTRACTING PARTS:")
print("-" * 80)

# Extract parts using slicing
first_word = message[0:6]
second_word = message[7:18]
third_word = message[19:21]
fourth_word = message[22:]

print(f"First word:  message[0:6]   = '{first_word}'")
print(f"Second word: message[7:18]  = '{second_word}'")
print(f"Third word:  message[19:21] = '{third_word}'")
print(f"Fourth word: message[22:]   = '{fourth_word}'")

# Every other character
every_other = message[::2]
print(f"\nEvery other character: message[::2] = '{every_other}'")

# Reverse the message
reversed_msg = message[::-1]
print(f"Reversed message: message[::-1] = '{reversed_msg}'")

# Extract first and last 5 characters
first_five = message[:5]
last_five = message[-5:]
print(f"\nFirst 5 chars: message[:5]  = '{first_five}'")
print(f"Last 5 chars:  message[-5:] = '{last_five}'")

# Remove first and last word
middle_part = message[7:-9]
print(f"\nMiddle part: message[7:-9] = '{middle_part}'")

print("\n" + "-" * 80)
print("CREATING NEW STRINGS:")
print("-" * 80)

# Create new strings using slices
greeting = message[:6] + " " + "World"
print(f"New greeting: '{greeting}'")

# Create acronym
words = message.split()
acronym = "".join([word[0] for word in words])
print(f"Acronym: '{acronym}'")

# Reverse each word
reversed_words = " ".join([word[::-1] for word in words])
print(f"Reversed words: '{reversed_words}'")


print("\n\n" + "=" * 80)
print("EXAMPLE 2: PLAYLIST MANAGER - Organize Songs")
print("=" * 80)

# Original playlist
playlist = ["Song1", "Song2", "Song3", "Song4", "Song5", "Song6", "Song7", "Song8", "Song9", "Song10"]
print(f"\nOriginal Playlist: {playlist}")
print(f"Total songs: {len(playlist)}")

print("\n" + "-" * 80)
print("📋 EXTRACTING SUBLISTS:")
print("-" * 80)

# Get first 3 songs
top_three = playlist[:3]
print(f"Top 3 songs:      playlist[:3]   = {top_three}")

# Get last 3 songs
bottom_three = playlist[-3:]
print(f"Bottom 3 songs:   playlist[-3:]  = {bottom_three}")

# Get middle songs
middle_songs = playlist[3:7]
print(f"Middle songs:     playlist[3:7]  = {middle_songs}")

# Get every 2nd song
every_second = playlist[::2]
print(f"Every 2nd song:   playlist[::2]  = {every_second}")

# Get odd position songs
odd_songs = playlist[1::2]
print(f"Odd position songs: playlist[1::2] = {odd_songs}")

# Reverse playlist
reversed_playlist = playlist[::-1]
print(f"Reversed playlist: playlist[::-1] = {reversed_playlist}")

print("\n" + "-" * 80)
print("MODIFYING PLAYLIST (SLICE ASSIGNMENT):")
print("-" * 80)

# Make a copy for modifications
modified_playlist = playlist.copy()
print(f"\nOriginal: {modified_playlist}")

# Replace first 3 songs with new songs
modified_playlist[:3] = ["NewSong1", "NewSong2", "NewSong3"]
print(f"After replacing first 3: {modified_playlist}")

# Make another copy
modified_playlist = playlist.copy()

# Insert songs at position 5
modified_playlist[5:5] = ["Bonus1", "Bonus2"]
print(f"\nAfter inserting at position 5: {modified_playlist}")

# Make another copy
modified_playlist = playlist.copy()

# Delete songs 3 to 5
modified_playlist[3:6] = []
print(f"\nAfter deleting songs 3-5: {modified_playlist}")

# Make another copy
modified_playlist = playlist.copy()

# Replace every 3rd song
modified_playlist[::3] = ["Hit1", "Hit2", "Hit3", "Hit4"]
print(f"\nAfter replacing every 3rd: {modified_playlist}")

print("\n" + "-" * 80)
print("PLAYLIST STATISTICS:")
print("-" * 80)

print(f"Total songs in original: {len(playlist)}")
print(f"First song: {playlist[0]}")
print(f"Last song: {playlist[-1]}")
print(f"First half: {playlist[:len(playlist)//2]}")
print(f"Second half: {playlist[len(playlist)//2:]}")

# Create a shuffled version (simplified)
import random
shuffled = playlist[:]  # Create a copy using slice
random.shuffle(shuffled)
print(f"Shuffled: {shuffled}")


print("\n\n" + "=" * 80)
print("SUMMARY: SLICE OPERATOR KEY POINTS")
print("=" * 80)

print("""
1. Syntax: sequence[start:stop:step]
   - start: inclusive, default = 0
   - stop: exclusive, default = len(sequence)
   - step: increment, default = 1

2. Common Patterns:
   - sequence[:n]     → First n elements
   - sequence[n:]     → From n to end
   - sequence[-n:]    → Last n elements
   - sequence[:-n]    → All except last n
   - sequence[::-1]   → Reverse sequence
   - sequence[::2]    → Every 2nd element

3. Slice Assignment (Lists Only):
   - Can replace, insert, or delete elements
   - Replacement can be different length
   - Step slicing requires same length
   - Strings and tuples are immutable

4. Best Practices:
   - Use slicing for readable code
   - Remember stop is exclusive
   - Negative indices count from end
   - Slicing never raises IndexError
   - Use [:] to create shallow copies
""")

print("\n" + "=" * 80)
print("SLICE OPERATOR GUIDE COMPLETE!")
print("=" * 80)

