

#Started Python for Everybody Course On Coursera
#Course Instructor: Dr. Charles Severance

# Start With Coments 
#Code tells the computer what to do, but comments tell other people (and your future self) why you are doing it.

print(float(99) + 100)  # convert 99 to a float (99.0), add 100, and print the result (199.0)

# Output: 199.0 

i = 42  # store the integer value 42 in the variable i
type(i)  # get the type of i (this returns <class 'int'> but does not print it)

# Output: <class 'int'>  # this shows what type(i) would be if you printed it

f = float(i)  # convert the integer i to a float and store it in f (now f == 42.0)
print(f)  # print the float value stored in f

# Output: 42.0  # expected printed value from the line above

type(f)  # get the type of f (this returns <class 'float'> but does not print it)

# Output: <class 'float'>  # this shows what type(f) would be if you printed it

#Comments Helps to read and understand behaivior of code easily.


#Get the name of file and open it

name = input("Enter file:")
handle = open(name, 'r')

#Count word frequency

counts = dict()
for line in handle:
    words = line.split()
    for word in words:
        counts[word] = counts.get(word, 0) + 1
        
#Find the most common word

bigcount = None
bigword = None
for word, count in counts.items():
    if bigcount is None or count > bigcount:
        bigword = word
        bigcount = count

#All done
print(bigword, bigcount)
