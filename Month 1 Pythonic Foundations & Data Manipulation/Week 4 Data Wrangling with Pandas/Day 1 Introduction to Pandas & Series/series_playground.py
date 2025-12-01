import pandas as pd
import numpy as np

# ==========================================
# CONCEPT 1: Series = Data + Labels (Index)
# ==========================================
print("--- CONCEPT 1: Creating a Series ---")
# Think of it like a list, but each item has a sticky note (label) attached to it.
data = [10, 20, 30, 40]
labels = ['a', 'b', 'c', 'd']

s = pd.Series(data, index=labels)
print("Series s:\n", s)
print("\nValues:", s.values)
print("Index:", s.index)


# ==========================================
# CONCEPT 2: It behaves like a LIST (Array)
# ==========================================
print("\n--- CONCEPT 2: Array-like Behavior ---")
# You can slice it by position (integer index)
print("First two elements (s[:2]):\n", s[:2])

# You can do math on the whole thing at once (vectorization)
print("s * 2:\n", s * 2)

# You can use boolean filtering
print("s > 20:\n", s[s > 20])


# ==========================================
# CONCEPT 3: It behaves like a DICTIONARY
# ==========================================
print("\n--- CONCEPT 3: Dictionary-like Behavior ---")
# You can look up values by their label
print("Value at label 'b':", s['b'])

# You can check if a label exists
print("Is 'c' in s?", 'c' in s)
print("Is 'z' in s?", 'z' in s)


# ==========================================
# CONCEPT 4: Automatic Alignment (The "Magic")
# ==========================================
print("\n--- CONCEPT 4: Alignment ---")
# When you combine two Series, pandas matches them by LABEL, not position.

s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['b', 'c', 'd'])

print("s1:\n", s1)
print("s2:\n", s2)

# Notice:
# 'a' is in s1 but not s2 -> NaN (Not a Number)
# 'b' matches 'b' (2 + 10 = 12)
# 'c' matches 'c' (3 + 20 = 23)
# 'd' is in s2 but not s1 -> NaN

print("s1 + s2 (Aligned by index):\n", s1 + s2)
