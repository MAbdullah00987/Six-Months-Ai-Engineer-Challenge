
#on-mutating Methods on Strings


#There are a wide variety of methods for string objects. Try the following program.


ss = "Hello, World"
print(ss.upper())

tt = ss.lower()
print(tt)
print(ss)


#In this example, upper is a method that can be invoked on any string object to create a new string in which all the characters are in uppercase. lower works in a similar fashion changing all characters in the string to lowercase. (The original string ss remains unchanged. A new string tt is created.)
#You’ve already seen a few methods, such as count and index, that work with strings and are non-mutating. In addition to those and upper and lower, the following table provides a summary of some other useful string methods. There are a few activecode examples that follow so that you can try them out.

#Method              Parameters               Description
#upper                none                    Returns a string in all uppercase
#lower                none                    Returns a string in all lowercase
#count                item                    Returns the number of occurrences of item
#index                item                    Returns the leftmost index where the substring item is found and causes a runtime error if item is not found
#strip                none                    Returns a string with the leading and trailing whitespace removed
#replace              old, new               Replaces all occurrences of old substring with new
#format               substitutions          Involved! See String Format Method, below

