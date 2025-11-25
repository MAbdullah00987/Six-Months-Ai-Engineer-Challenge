
#Day 5: Advanced Plotting Techniques 

#Part 1: Understanding Subplots Basics

import matplotlib.pyplot as plt
import numpy as np
'''
# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = x**2

# METHOD 1: Using plt.subplot()
# Creates a figure with 2x2 grid of subplots
plt.figure(figsize=(12, 10))

# subplot(rows, columns, index)
plt.subplot(2, 2, 1)  # First subplot (top-left)
plt.plot(x, y1, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)  # Second subplot (top-right)
plt.plot(x, y2, 'r-', linewidth=2)
plt.title('Cosine Wave')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)  # Third subplot (bottom-left)
plt.plot(x, y3, 'g-', linewidth=2)
plt.title('Tangent Wave')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.ylim(-5, 5)  # Limit y-axis for tangent
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)  # Fourth subplot (bottom-right)
plt.plot(x, y4, 'm-', linewidth=2)
plt.title('Quadratic Function')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True, alpha=0.3)

# Adjust spacing to prevent overlap
plt.tight_layout()
plt.savefig('basic_subplots.png', dpi=300, bbox_inches='tight')
plt.show()

print("Created 2x2 grid of subplots using plt.subplot()")


#Part 2: Better Method - plt.subplots()
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)

# METHOD 2: Using plt.subplots() - More flexible and cleaner
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# axes is a 2D array: axes[row, column]
# Top-left: axes[0, 0]
axes[0, 0].plot(x, np.sin(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sine Wave', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('X axis')
axes[0, 0].set_ylabel('Y axis')
axes[0, 0].grid(True, alpha=0.3)

# Top-right: axes[0, 1]
axes[0, 1].plot(x, np.cos(x), 'r-', linewidth=2)
axes[0, 1].set_title('Cosine Wave', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('X axis')
axes[0, 1].set_ylabel('Y axis')
axes[0, 1].grid(True, alpha=0.3)

# Bottom-left: axes[1, 0]
axes[1, 0].plot(x, np.exp(x/5), 'g-', linewidth=2)
axes[1, 0].set_title('Exponential Growth', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('X axis')
axes[1, 0].set_ylabel('Y axis')
axes[1, 0].grid(True, alpha=0.3)

# Bottom-right: axes[1, 1]
axes[1, 1].plot(x, np.log(x + 1), 'm-', linewidth=2)
axes[1, 1].set_title('Logarithmic Function', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('X axis')
axes[1, 1].set_ylabel('Y axis')
axes[1, 1].grid(True, alpha=0.3)

# Add overall title
fig.suptitle('Mathematical Functions Comparison', fontsize=16, fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.savefig('subplots_method.png', dpi=300, bbox_inches='tight')
plt.show()

print("Created 2x2 grid using plt.subplots() - cleaner syntax!")


#Part 3: Mixing Different Chart Types

import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
categories = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
sales = [250, 180, 320, 210, 290]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
revenue = [50000, 55000, 52000, 60000, 65000, 70000]
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5

# Create figure with mixed chart types
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. BAR CHART - Sales by Product
axes[0, 0].bar(categories, sales, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'], 
               edgecolor='black', linewidth=1.5)
axes[0, 0].set_title('Product Sales Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Products')
axes[0, 0].set_ylabel('Sales (units)')
axes[0, 0].grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for i, v in enumerate(sales):
    axes[0, 0].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

# 2. LINE CHART - Revenue Over Time
axes[0, 1].plot(months, revenue, marker='o', linewidth=2.5, markersize=8, 
                color='#E74C3C', markerfacecolor='white', markeredgewidth=2)
axes[0, 1].set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Revenue ($)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].fill_between(months, revenue, alpha=0.2, color='#E74C3C')

# 3. SCATTER PLOT - Correlation Analysis
axes[1, 0].scatter(x_scatter, y_scatter, alpha=0.6, s=50, c=y_scatter, 
                   cmap='viridis', edgecolors='black', linewidth=0.5)
axes[1, 0].set_title('Data Correlation Analysis', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Variable X')
axes[1, 0].set_ylabel('Variable Y')
axes[1, 0].grid(True, alpha=0.3)
# Add trend line
z = np.polyfit(x_scatter, y_scatter, 1)
p = np.poly1d(z)
axes[1, 0].plot(sorted(x_scatter), p(sorted(x_scatter)), "r--", linewidth=2, label='Trend')
axes[1, 0].legend()

# 4. PIE CHART - Market Share
axes[1, 1].pie(sales, labels=categories, autopct='%1.1f%%', startangle=90,
               colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
               explode=(0.05, 0, 0, 0, 0), shadow=True)
axes[1, 1].set_title('Market Share Distribution', fontsize=14, fontweight='bold')

# Overall title and layout
fig.suptitle('Sales Dashboard - Q2 2024', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('mixed_charts.png', dpi=300, bbox_inches='tight')
plt.show()

print("Created dashboard with 4 different chart types!")


#Part 4: Sharing Axes Between Subplots
import matplotlib.pyplot as plt
import numpy as np

# Generate time series data
time = np.linspace(0, 10, 200)
signal1 = np.sin(2 * np.pi * time) + np.random.randn(200) * 0.1
signal2 = np.sin(2 * np.pi * time + np.pi/4) + np.random.randn(200) * 0.1
signal3 = np.sin(2 * np.pi * time + np.pi/2) + np.random.randn(200) * 0.1

# EXAMPLE 1: Sharing X-axis (useful for time series)
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(time, signal1, 'b-', linewidth=1.5)
axes[0].set_title('Signal 1', fontweight='bold')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, signal2, 'r-', linewidth=1.5)
axes[1].set_title('Signal 2', fontweight='bold')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, signal3, 'g-', linewidth=1.5)
axes[2].set_title('Signal 3', fontweight='bold')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Time Series with Shared X-Axis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('shared_xaxis.png', dpi=300, bbox_inches='tight')
plt.show()

# EXAMPLE 2: Sharing Y-axis (useful for comparing magnitudes)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

categories1 = ['A', 'B', 'C', 'D']
values1 = [23, 45, 56, 78]
values2 = [34, 56, 42, 67]
values3 = [45, 32, 61, 54]

axes[0].bar(categories1, values1, color='#3498DB')
axes[0].set_title('Dataset 1', fontweight='bold')
axes[0].set_ylabel('Values')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(categories1, values2, color='#E74C3C')
axes[1].set_title('Dataset 2', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].bar(categories1, values3, color='#2ECC71')
axes[2].set_title('Dataset 3', fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

fig.suptitle('Datasets with Shared Y-Axis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('shared_yaxis.png', dpi=300, bbox_inches='tight')
plt.show()

# EXAMPLE 3: Sharing both axes
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

x = np.linspace(0, 10, 100)

axes[0, 0].plot(x, np.sin(x), 'b-')
axes[0, 0].set_title('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, np.cos(x), 'r-')
axes[0, 1].set_title('cos(x)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(x, np.sin(2*x), 'g-')
axes[1, 0].set_title('sin(2x)')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x, np.cos(2*x), 'm-')
axes[1, 1].set_title('cos(2x)')
axes[1, 1].set_xlabel('X')
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Trigonometric Functions with Shared Axes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('shared_both_axes.png', dpi=300, bbox_inches='tight')
plt.show()

print("Created visualizations with shared axes!")
print("  sharex=True: All subplots share the same X-axis")
print("  sharey=True: All subplots share the same Y-axis")
print("  Both: Consistent scaling across all plots")


#Part 5: Advanced Layout Techniques
import matplotlib.pyplot as plt
import numpy as np

# TECHNIQUE 1: Using subplot2grid for irregular layouts
fig = plt.figure(figsize=(14, 10))

# Large plot spanning 2 rows
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), 'b-', linewidth=2)
ax1.set_title('Main Plot (2x2 grid space)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Smaller plots on the right
ax2 = plt.subplot2grid((3, 3), (0, 2))
ax2.bar(['A', 'B', 'C'], [10, 15, 12], color='#FF6B6B')
ax2.set_title('Small Bar')
ax2.grid(True, alpha=0.3, axis='y')

ax3 = plt.subplot2grid((3, 3), (1, 2))
ax3.scatter(np.random.randn(20), np.random.randn(20), alpha=0.6)
ax3.set_title('Small Scatter')
ax3.grid(True, alpha=0.3)

# Bottom plots
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax4.hist(np.random.randn(1000), bins=30, color='#4ECDC4', edgecolor='black')
ax4.set_title('Histogram 1')
ax4.grid(True, alpha=0.3, axis='y')

ax5 = plt.subplot2grid((3, 3), (2, 1))
ax5.plot(x, np.cos(x), 'r-', linewidth=2)
ax5.set_title('Line Plot')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot2grid((3, 3), (2, 2))
ax6.pie([30, 40, 30], labels=['X', 'Y', 'Z'], autopct='%1.0f%%')
ax6.set_title('Pie Chart')

fig.suptitle('Irregular Grid Layout using subplot2grid', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('irregular_layout.png', dpi=300, bbox_inches='tight')
plt.show()

# TECHNIQUE 2: Using GridSpec for more control
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Create subplots with different sizes
ax1 = fig.add_subplot(gs[0, :])  # Top row, all columns
ax1.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)), 'b-', linewidth=2)
ax1.set_title('Wide Plot Across Top', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1:, 0])  # Left column, bottom 2 rows
ax2.imshow(np.random.rand(10, 10), cmap='viridis')
ax2.set_title('Tall Plot\nOn Left', fontsize=14, fontweight='bold')

ax3 = fig.add_subplot(gs[1, 1:])  # Middle right
categories = ['Q1', 'Q2', 'Q3', 'Q4']
values = [23, 45, 56, 78]
ax3.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax3.set_title('Quarterly Results', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

ax4 = fig.add_subplot(gs[2, 1:])  # Bottom right
x = np.random.randn(50)
y = np.random.randn(50)
ax4.scatter(x, y, s=100, alpha=0.6, c=np.arange(50), cmap='plasma', edgecolors='black')
ax4.set_title('Scatter Analysis', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

fig.suptitle('Complex Layout using GridSpec', fontsize=16, fontweight='bold')
plt.savefig('gridspec_layout.png', dpi=300, bbox_inches='tight')
plt.show()

print(" Created advanced layouts!")
print("   subplot2grid: For irregular grid patterns")
print("   GridSpec: For maximum control over spacing and layout")
'''

#Part 6: Complete Dashboard Project
import matplotlib.pyplot as plt
import numpy as np

# TECHNIQUE 1: Using subplot2grid for irregular layouts
fig = plt.figure(figsize=(14, 10))

# Large plot spanning 2 rows
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), 'b-', linewidth=2)
ax1.set_title('Main Plot (2x2 grid space)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Smaller plots on the right
ax2 = plt.subplot2grid((3, 3), (0, 2))
ax2.bar(['A', 'B', 'C'], [10, 15, 12], color='#FF6B6B')
ax2.set_title('Small Bar')
ax2.grid(True, alpha=0.3, axis='y')

ax3 = plt.subplot2grid((3, 3), (1, 2))
ax3.scatter(np.random.randn(20), np.random.randn(20), alpha=0.6)
ax3.set_title('Small Scatter')
ax3.grid(True, alpha=0.3)

# Bottom plots
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax4.hist(np.random.randn(1000), bins=30, color='#4ECDC4', edgecolor='black')
ax4.set_title('Histogram 1')
ax4.grid(True, alpha=0.3, axis='y')

ax5 = plt.subplot2grid((3, 3), (2, 1))
ax5.plot(x, np.cos(x), 'r-', linewidth=2)
ax5.set_title('Line Plot')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot2grid((3, 3), (2, 2))
ax6.pie([30, 40, 30], labels=['X', 'Y', 'Z'], autopct='%1.0f%%')
ax6.set_title('Pie Chart')

fig.suptitle('Irregular Grid Layout using subplot2grid', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('irregular_layout.png', dpi=300, bbox_inches='tight')
plt.show()

# TECHNIQUE 2: Using GridSpec for more control
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Create subplots with different sizes
ax1 = fig.add_subplot(gs[0, :])  # Top row, all columns
ax1.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)), 'b-', linewidth=2)
ax1.set_title('Wide Plot Across Top', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1:, 0])  # Left column, bottom 2 rows
ax2.imshow(np.random.rand(10, 10), cmap='viridis')
ax2.set_title('Tall Plot\nOn Left', fontsize=14, fontweight='bold')

ax3 = fig.add_subplot(gs[1, 1:])  # Middle right
categories = ['Q1', 'Q2', 'Q3', 'Q4']
values = [23, 45, 56, 78]
ax3.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax3.set_title('Quarterly Results', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

ax4 = fig.add_subplot(gs[2, 1:])  # Bottom right
x = np.random.randn(50)
y = np.random.randn(50)
ax4.scatter(x, y, s=100, alpha=0.6, c=np.arange(50), cmap='plasma', edgecolors='black')
ax4.set_title('Scatter Analysis', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

fig.suptitle('Complex Layout using GridSpec', fontsize=16, fontweight='bold')
plt.savefig('gridspec_layout.png', dpi=300, bbox_inches='tight')
plt.show()

print(" Created advanced layouts!")
print("   subplot2grid: For irregular grid patterns")
print("   GridSpec: For maximum control over spacing and layout")