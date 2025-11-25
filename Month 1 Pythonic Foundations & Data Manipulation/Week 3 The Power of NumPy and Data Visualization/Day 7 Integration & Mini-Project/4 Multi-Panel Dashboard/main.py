import matplotlib.pyplot as plt
import numpy as np

# Step 1: Prepare the data
# Line chart data - Monthly sales trend
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [45, 52, 48, 65, 70, 78]

# Bar chart data - Product categories
categories = ['Electronics', 'Clothing', 'Food', 'Books']
revenue = [85, 65, 45, 35]

# Scatter plot data - Price vs Units Sold
np.random.seed(42)
prices = np.random.uniform(10, 110, 50)
units_sold = np.random.uniform(20, 220, 50)

# Step 2: Create the figure with subplots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sales Analytics Dashboard', fontsize=20, fontweight='bold')

# Step 3: Create Line Chart (Top Left - axes[0, 0])
axes[0, 0].plot(months, sales, marker='o', linewidth=2, markersize=8, color='#3498db')
axes[0, 0].set_title('Line Chart - Sales Trend', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Month', fontsize=12)
axes[0, 0].set_ylabel('Sales (in thousands)', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0, max(sales) + 10)

# Step 4: Create Bar Chart (Top Right - axes[0, 1])
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
bars = axes[0, 1].bar(categories, revenue, color=colors, edgecolor='black', linewidth=1.2)
axes[0, 1].set_title('Bar Chart - Product Categories', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Category', fontsize=12)
axes[0, 1].set_ylabel('Revenue (in thousands)', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].set_ylim(0, max(revenue) + 10)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}K',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Step 5: Create Scatter Plot (Bottom Left - axes[1, 0])
scatter = axes[1, 0].scatter(prices, units_sold, c=prices, cmap='viridis', 
                             s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1, 0].set_title('Scatter Plot - Price vs Units Sold', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Price ($)', fontsize=12)
axes[1, 0].set_ylabel('Units Sold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_label('Price ($)', fontsize=10)

# Step 6: Hide the fourth subplot or add another chart
# Option 1: Hide it
axes[1, 1].axis('off')

# Option 2: Add a pie chart (uncomment if you want)
# expense_categories = ['Marketing', 'Operations', 'R&D', 'Admin']
# expenses = [30, 25, 25, 20]
# axes[1, 1].pie(expenses, labels=expense_categories, autopct='%1.1f%%',
#                colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
# axes[1, 1].set_title('Pie Chart - Expense Distribution', fontsize=14, fontweight='bold')

# Step 7: Adjust layout and display
plt.tight_layout()
plt.show()