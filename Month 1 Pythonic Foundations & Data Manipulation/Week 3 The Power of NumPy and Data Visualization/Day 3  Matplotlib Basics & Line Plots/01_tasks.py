

#Day 3: Matplotlib Basics & Line Plots

'''
#Part 1: Understanding the Basics

import matplotlib.pyplot as plt
import numpy as np

# Simple line plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.show()


#plt.plot() creates a line plot
#plt.show() displays the plot



x = [0.1,0.2,0.3,0.4,0.5,0.6]
y = [1,2,3,4,5,6]

plt.plot(x,y)
plt.show()

plt.plot(x, y, color='red', linewidth=2, linestyle='--', marker='o')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.title('My First Customized Plot')
plt.grid(True)
plt.show()
'''

#Part 2: Working with Figures and Axes
#The Figure-Axes Structure

import matplotlib.pyplot as plt
import numpy as np

'''
x = [1,2,3,4,5,6]
y = [0.1,0.2,0.3,0.4,0.5,0.6]

# Proper way to create plots
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y, label='Linear Growth')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Understanding Figure and Axes')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#Why use fig, ax?
#More control over plot elements
#Better for complex visualizations
#Easier to modify specific parts


# Multiple Lines on One Plot
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
ax.plot(x, y2, label='cos(x)', color='red', linewidth=2)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Sine and Cosine Functions')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.show()
'''

#Part 3: Working with Subplots

#import matplotlib.pyplot as plt
#import numpy as np
'''
x = np.array([1,2,3,4,5,6])
y = np.array([0.1,0.2,0.3,0.4,0.5,0.6])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left
axes[0, 0].plot(x, np.sin(x), 'b-')
axes[0, 0].set_title('Sine')

# Top-right
axes[0, 1].plot(x, np.cos(x), 'r-')
axes[0, 1].set_title('Cosine')

# Bottom-left
axes[1, 0].plot(x, np.tan(x), 'g-')
axes[1, 0].set_title('Tangent')
axes[1, 0].set_ylim(-5, 5)

# Bottom-right
axes[1, 1].plot(x, x**2, 'm-')
axes[1, 1].set_title('Quadratic')

plt.tight_layout()
plt.show()


#Subplots with Different Sizes
x = np.array([1,2,3,4,5,6])

fig = plt.figure(figsize=(12, 8))

# Create grid: 3 rows, 3 columns
ax1 = plt.subplot(2, 2, 1)  # Top-left
ax2 = plt.subplot(2, 2, 2)  # Top-right
ax3 = plt.subplot(2, 1, 2)  # Bottom (spans full width)

ax1.plot(x, np.sin(x))
ax1.set_title('Plot 1')

ax2.plot(x, np.cos(x))
ax2.set_title('Plot 2')

ax3.plot(x, np.sin(x) * np.cos(x))
ax3.set_title('Wide Plot')

plt.tight_layout()
plt.show()



#Grids and Styling

x = np.array([1,2,3,4,5,6])
y = np.sin(x)  # Add some data to plot

fig, ax = plt.subplots(figsize=(10, 6))

# Plot your data first
ax.plot(x, y, 'b-', linewidth=2)

# Then set up the grid (choose ONE approach)
# Option 1: Basic customized grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')

# OR Option 2: Major and minor grid
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-', alpha=0.5)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)

plt.show()

#Another Example 
x = np.array([1,2,3,4,5,6])
y = np.sin(x)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Basic grid
axes[0, 0].plot(x, y)
axes[0, 0].grid(True)
axes[0, 0].set_title('Basic Grid')

# Customized grid
axes[0, 1].plot(x, y)
axes[0, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
axes[0, 1].set_title('Customized Grid')

# X-axis only
axes[1, 0].plot(x, y)
axes[1, 0].grid(True, axis='x')
axes[1, 0].set_title('X-axis Grid Only')

# Major and minor
axes[1, 1].plot(x, y)
axes[1, 1].minorticks_on()
axes[1, 1].grid(True, which='major', linestyle='-', alpha=0.5)
axes[1, 1].grid(True, which='minor', linestyle=':', alpha=0.3)
axes[1, 1].set_title('Major & Minor Grid')

plt.tight_layout()
plt.show()



#Annotations

x = np.array([1,2,3,4,5,6])
y = np.sin(x)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Basic grid
axes[0, 0].plot(x, y)
axes[0, 0].grid(True)
axes[0, 0].set_title('Basic Grid')

# Customized grid
axes[0, 1].plot(x, y)
axes[0, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
axes[0, 1].set_title('Customized Grid')

# X-axis only
axes[1, 0].plot(x, y)
axes[1, 0].grid(True, axis='x')
axes[1, 0].set_title('X-axis Grid Only')

# Major and minor
axes[1, 1].plot(x, y)
axes[1, 1].minorticks_on()
axes[1, 1].grid(True, which='major', linestyle='-', alpha=0.5)
axes[1, 1].grid(True, which='minor', linestyle=':', alpha=0.3)
axes[1, 1].set_title('Major & Minor Grid')

plt.tight_layout()
plt.show()
'''

#Saving Plots

import matplotlib.pyplot as plt
import numpy as np

# 1. CREATE A PLOT FIRST
x = np.array([1, 2, 3, 4, 5, 6])
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

# 2. NOW SAVE IT
# Basic save
plt.savefig('my_plot.png')

# High quality save
plt.savefig('my_plot_hq.png', dpi=300, bbox_inches='tight')

# Different formats
plt.savefig('my_plot.pdf')  # PDF format
plt.savefig('my_plot.svg')  # SVG format (vector)
plt.savefig('my_plot.jpg', dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})  # ✅ FIXED

# Transparent background
plt.savefig('my_plot_transparent.png', transparent=True, dpi=300)

# With specific settings
plt.savefig('my_plot_final.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

plt.show()