

#Day 1: Descriptive Statistics (The Shape of Data)
#Objective: Learn to summarize massive datasets into a few powerful numbers.
#Concepts: Mean vs. Median (robustness to outliers), Variance, Standard Deviation, Interquartile Range (IQR).
#Task: Project - Descriptive Statistics Report. Write a reusable script that accepts a CSV file and outputs a 
# summary report. Don't just use df.describe(); verify the math by calculating variance from scratch using NumPy.
#Key Python Function: df.describe(), np.std(), np.var().
#Foundations of Probability
#Focus: Basic probability concepts and distributions
#Read Burkov Chapter 2: Sections on probability fundamentals
#Study probability distributions (Normal, Binomial)
#Review conditional probability basics
#Coin Flip Simulator - Start with fundamentals by simulating coin flips and observing convergence to theoretical probabilities
#Dice Roll Probability - Calculate and visualize probability distributions for dice sums
#Work through probability calculation exercises
#Sketch different probability distributions by hand


#Mastering Descriptive Statistics & Probability Through Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''
#Part 1: Descriptive Statistics - Understanding the Shape of Data
#Mean and Median

# Scenario: Salaries in a small company
salaries_normal = [45000, 48000, 50000, 52000, 55000, 58000, 60000]
salaries_with_ceo = [45000, 48000, 50000, 52000, 55000, 58000, 5000000]  # CEO joins!

# Calculate manually
def calculate_mean(data):
    return sum(data) / len(data)

def calculate_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        return sorted_data[n//2]

# Compare
print("WITHOUT OUTLIER:")
print(f"Mean: ${calculate_mean(salaries_normal):,.2f}")
print(f"Median: ${calculate_median(salaries_normal):,.2f}")
print(f"NumPy Mean: ${np.mean(salaries_normal):,.2f}")
print(f"NumPy Median: ${np.median(salaries_normal):,.2f}")

print("\nWITH CEO (OUTLIER):")
print(f"Mean: ${calculate_mean(salaries_with_ceo):,.2f}")
print(f"Median: ${calculate_median(salaries_with_ceo):,.2f}")

# Visualize the impact
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Normal distribution
axes[0].bar(range(len(salaries_normal)), salaries_normal, color='skyblue', edgecolor='black')
axes[0].axhline(calculate_mean(salaries_normal), color='red', linestyle='--', linewidth=2, label=f'Mean: ${calculate_mean(salaries_normal):,.0f}')
axes[0].axhline(calculate_median(salaries_normal), color='green', linestyle='--', linewidth=2, label=f'Median: ${calculate_median(salaries_normal):,.0f}')
axes[0].set_title('Salaries Without Outlier', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Salary ($)')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# With outlier
axes[1].bar(range(len(salaries_with_ceo)), salaries_with_ceo, color='lightcoral', edgecolor='black')
axes[1].axhline(calculate_mean(salaries_with_ceo), color='red', linestyle='--', linewidth=2, label=f'Mean: ${calculate_mean(salaries_with_ceo):,.0f}')
axes[1].axhline(calculate_median(salaries_with_ceo), color='green', linestyle='--', linewidth=2, label=f'Median: ${calculate_median(salaries_with_ceo):,.0f}')
axes[1].set_title('Salaries With CEO (Outlier)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Salary ($)')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

'''

'''
#Variance & Standard Deviation - Measuring Spread

# Two datasets with same mean but different spreads
consistent_scores = [78, 79, 80, 81, 82]  # Very consistent student
variable_scores = [50, 70, 80, 90, 110]   # Unpredictable student

def calculate_variance_manual(data):
    """Calculate variance step-by-step"""
    mean = sum(data) / len(data)
    squared_diffs = [(x - mean)**2 for x in data]
    variance = sum(squared_diffs) / len(data)
    return variance

def calculate_std_manual(data):
    """Standard deviation is square root of variance"""
    return calculate_variance_manual(data) ** 0.5

# Manual calculations
print("CONSISTENT STUDENT:")
print(f"Mean: {np.mean(consistent_scores):.2f}")
print(f"Variance (Manual): {calculate_variance_manual(consistent_scores):.2f}")
print(f"Variance (NumPy): {np.var(consistent_scores):.2f}")
print(f"Std Dev (Manual): {calculate_std_manual(consistent_scores):.2f}")
print(f"Std Dev (NumPy): {np.std(consistent_scores):.2f}")

print("\nVARIABLE STUDENT:")
print(f"Mean: {np.mean(variable_scores):.2f}")
print(f"Variance (Manual): {calculate_variance_manual(variable_scores):.2f}")
print(f"Variance (NumPy): {np.var(variable_scores):.2f}")
print(f"Std Dev (Manual): {calculate_std_manual(variable_scores):.2f}")
print(f"Std Dev (NumPy): {np.std(variable_scores):.2f}")

# Visualize spread
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Consistent scores
axes[0].scatter(range(len(consistent_scores)), consistent_scores, s=200, c='green', edgecolor='black', alpha=0.7)
axes[0].axhline(np.mean(consistent_scores), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0].fill_between(range(len(consistent_scores)), 
                       np.mean(consistent_scores) - np.std(consistent_scores),
                       np.mean(consistent_scores) + np.std(consistent_scores),
                       alpha=0.2, color='green', label='±1 Std Dev')
axes[0].set_title(f'Consistent Student (σ={np.std(consistent_scores):.2f})', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Variable scores
axes[1].scatter(range(len(variable_scores)), variable_scores, s=200, c='orange', edgecolor='black', alpha=0.7)
axes[1].axhline(np.mean(variable_scores), color='red', linestyle='--', linewidth=2, label='Mean')
axes[1].fill_between(range(len(variable_scores)), 
                       np.mean(variable_scores) - np.std(variable_scores),
                       np.mean(variable_scores) + np.std(variable_scores),
                       alpha=0.2, color='orange', label='±1 Std Dev')
axes[1].set_title(f'Variable Student (σ={np.std(variable_scores):.2f})', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''


#Interquartile Range (IQR) - Robust Spread Measure

# Dataset with outliers
house_prices = [150000, 175000, 180000, 190000, 200000, 210000, 220000, 
                250000, 280000, 300000, 350000, 2000000]  # One mansion!

def calculate_iqr_manual(data):
    """Calculate IQR step-by-step"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Q1 (25th percentile)
    q1_pos = n * 0.25
    q1 = sorted_data[int(q1_pos)] if q1_pos.is_integer() else sorted_data[int(q1_pos)]
    
    # Q3 (75th percentile)
    q3_pos = n * 0.75
    q3 = sorted_data[int(q3_pos)] if q3_pos.is_integer() else sorted_data[int(q3_pos)]
    
    iqr = q3 - q1
    return q1, q3, iqr

q1_manual, q3_manual, iqr_manual = calculate_iqr_manual(house_prices)
q1_np = np.percentile(house_prices, 25)
q3_np = np.percentile(house_prices, 75)
iqr_np = q3_np - q1_np

print("HOUSE PRICES ANALYSIS:")
print(f"Q1 (Manual): ${q1_manual:,.2f}")
print(f"Q1 (NumPy): ${q1_np:,.2f}")
print(f"Q3 (Manual): ${q3_manual:,.2f}")
print(f"Q3 (NumPy): ${q3_np:,.2f}")
print(f"IQR (Manual): ${iqr_manual:,.2f}")
print(f"IQR (NumPy): ${iqr_np:,.2f}")
print(f"\nStd Dev (affected by outlier): ${np.std(house_prices):,.2f}")
print(f"IQR (robust to outlier): ${iqr_np:,.2f}")

# Boxplot visualization
fig, ax = plt.subplots(figsize=(12, 6))
box = ax.boxplot(house_prices, vert=False, patch_artist=True, widths=0.5,
                  boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=2),
                  medianprops=dict(color='red', linewidth=3),
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black', linewidth=1.5),
                  flierprops=dict(marker='o', color='red', markersize=10, alpha=0.7))

ax.set_xlabel('Price ($)', fontsize=12)
ax.set_title('House Prices Distribution - IQR Shows Outliers Clearly', fontsize=14, fontweight='bold')
ax.text(q1_np, 0.85, f'Q1: ${q1_np:,.0f}', fontsize=10, color='blue')
ax.text(q3_np, 0.85, f'Q3: ${q3_np:,.0f}', fontsize=10, color='blue')
ax.text(np.median(house_prices), 1.15, f'Median: ${np.median(house_prices):,.0f}', fontsize=10, color='red')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
