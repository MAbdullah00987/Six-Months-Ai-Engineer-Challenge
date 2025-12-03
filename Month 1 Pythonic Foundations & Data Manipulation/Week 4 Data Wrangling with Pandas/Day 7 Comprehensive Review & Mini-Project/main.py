"""
Complete E-commerce Data Analysis Pipeline
Author: Data Analysis Project
Date: December 2025
Duration: 3-4 hours

This script demonstrates a complete data analysis workflow including:
- Data generation and loading
- Data cleaning and preprocessing
- Data merging and transformation
- Groupby operations and aggregations
- Advanced filtering and indexing
- Results export and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 30)

print("="*80)
print("E-COMMERCE DATA ANALYSIS PIPELINE")
print("="*80)


# STEP 1: DATA GENERATION AND LOADING

print("\n[STEP 1] Generating and Loading Data...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate Customers Data
customers = pd.DataFrame({
    'customer_id': range(1, 101),
    'name': [f'Customer_{i}' for i in range(1, 101)],
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'signup_date': pd.date_range('2023-01-01', periods=100, freq='3D')
})

# Generate Products Data
products = pd.DataFrame({
    'product_id': range(1, 51),
    'product_name': [f'Product_{i}' for i in range(1, 51)],
    'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], 50),
    'price': np.random.uniform(10, 500, 50).round(2)
})

# Generate Orders Data (with intentional missing values)
orders = pd.DataFrame({
    'order_id': range(1, 501),
    'customer_id': np.random.choice(range(1, 101), 500),
    'product_id': np.random.choice(range(1, 51), 500),
    'quantity': np.random.randint(1, 5, 500),
    'order_date': pd.date_range('2023-01-01', periods=500, freq='H'),
    'status': np.random.choice(['Completed', 'Pending', 'Cancelled', None], 500, p=[0.7, 0.15, 0.1, 0.05])
})

# Save to CSV files
customers.to_csv('customers.csv', index=False)
products.to_csv('products.csv', index=False)
orders.to_csv('orders.csv', index=False)

print("✓ Data generated and saved to CSV files")
print(f"  - Customers: {customers.shape}")
print(f"  - Products: {products.shape}")
print(f"  - Orders: {orders.shape}")


# STEP 2: DATA EXPLORATION

print("\n[STEP 2] Exploring Data...")

# Load the data (simulating real-world scenario)
customers = pd.read_csv('customers.csv')
products = pd.read_csv('products.csv')
orders = pd.read_csv('orders.csv')

print("\n--- CUSTOMERS DATA ---")
print(customers.head())
print(f"\nShape: {customers.shape}")
print(f"Data types:\n{customers.dtypes}")

print("\n--- PRODUCTS DATA ---")
print(products.head())
print(f"\nShape: {products.shape}")
print(f"\nPrice statistics:\n{products['price'].describe()}")

print("\n--- ORDERS DATA ---")
print(orders.head())
print(f"\nShape: {orders.shape}")
print(f"\nMissing values:\n{orders.isnull().sum()}")
print(f"\nStatus distribution:\n{orders['status'].value_counts(dropna=False)}")


# STEP 3: DATA CLEANING

print("\n[STEP 3] Cleaning Data...")

# Convert date columns to datetime
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
orders['order_date'] = pd.to_datetime(orders['order_date'])

print("✓ Converted date columns to datetime format")

# Handle missing values in orders status
missing_count = orders['status'].isnull().sum()
orders['status'] = orders['status'].fillna('Unknown')
print(f"illed {missing_count} missing status values with 'Unknown'")

# Remove duplicate orders (if any)
original_count = len(orders)
orders_clean = orders.drop_duplicates(subset=['order_id'])
duplicates_removed = original_count - len(orders_clean)
print(f"Removed {duplicates_removed} duplicate orders")

# Filter out cancelled orders for main analysis
orders_active = orders_clean[orders_clean['status'] != 'Cancelled'].copy()
cancelled_count = len(orders_clean) - len(orders_active)
print(f"Filtered out {cancelled_count} cancelled orders")
print(f"Active orders for analysis: {len(orders_active)}")


# STEP 4: DATA MERGING

print("\n[STEP 4] Merging Datasets...")

# Merge orders with products
orders_products = orders_active.merge(
    products, 
    on='product_id', 
    how='left'
)
print(f"✓ Merged orders with products: {orders_products.shape}")

# Merge with customers
full_data = orders_products.merge(
    customers, 
    on='customer_id', 
    how='left'
)
print(f"✓ Merged with customers: {full_data.shape}")

# Calculate total sale amount for each order
full_data['total_amount'] = full_data['quantity'] * full_data['price']
print("✓ Calculated total_amount column")

# Verify merge quality
null_check = full_data.isnull().sum()
if null_check.sum() > 0:
    print(f"\n⚠ Warning: Found missing values after merge:")
    print(null_check[null_check > 0])
else:
    print("✓ No missing values after merge - data integrity maintained")

print("\n--- MERGED DATASET SAMPLE ---")
print(full_data[['order_id', 'customer_id', 'name', 'product_name', 
                 'category', 'quantity', 'price', 'total_amount', 'region']].head())


# STEP 5: KEY METRICS CALCULATION (GROUPBY OPERATIONS)

print("\n[STEP 5] Calculating Key Metrics...")

# 1. Total sales by category
sales_by_category = full_data.groupby('category').agg({
    'total_amount': 'sum',
    'order_id': 'count',
    'quantity': 'sum'
}).round(2)
sales_by_category.columns = ['Total_Sales', 'Order_Count', 'Units_Sold']
sales_by_category = sales_by_category.sort_values('Total_Sales', ascending=False)

print("\n--- SALES BY CATEGORY ---")
print(sales_by_category)

# 2. Total sales by region
sales_by_region = full_data.groupby('region').agg({
    'total_amount': ['sum', 'mean', 'count']
}).round(2)
sales_by_region.columns = ['Total_Sales', 'Avg_Order_Value', 'Order_Count']
sales_by_region = sales_by_region.sort_values('Total_Sales', ascending=False)

print("\n--- SALES BY REGION ---")
print(sales_by_region)

# 3. Sales over time (by month)
full_data['month'] = full_data['order_date'].dt.to_period('M')
sales_by_month = full_data.groupby('month').agg({
    'total_amount': 'sum',
    'order_id': 'count'
}).round(2)
sales_by_month.columns = ['Total_Sales', 'Order_Count']

print("\n--- MONTHLY SALES TREND ---")
print(sales_by_month)


# STEP 6: ADVANCED ANALYSIS

print("\n[STEP 6] Performing Advanced Analysis...")

# Top 10 customers by total purchase
top_customers = full_data.groupby(['customer_id', 'name', 'region']).agg({
    'total_amount': 'sum',
    'order_id': 'count'
}).round(2)
top_customers.columns = ['Total_Spent', 'Order_Count']
top_customers['Avg_Order_Value'] = (top_customers['Total_Spent'] / top_customers['Order_Count']).round(2)
top_customers = top_customers.sort_values('Total_Spent', ascending=False).head(10)

print("\n--- TOP 10 CUSTOMERS ---")
print(top_customers)

# Top 10 products by revenue
top_products = full_data.groupby(['product_id', 'product_name', 'category']).agg({
    'total_amount': 'sum',
    'quantity': 'sum',
    'order_id': 'count'
}).round(2)
top_products.columns = ['Total_Revenue', 'Units_Sold', 'Times_Ordered']
top_products['Avg_Price'] = (top_products['Total_Revenue'] / top_products['Units_Sold']).round(2)
top_products = top_products.sort_values('Total_Revenue', ascending=False).head(10)

print("\n--- TOP 10 PRODUCTS BY REVENUE ---")
print(top_products)

# Category performance by region (multi-level grouping)
category_region_performance = full_data.groupby(['region', 'category'])['total_amount'].agg(['sum', 'mean', 'count']).round(2)
category_region_performance.columns = ['Total_Sales', 'Avg_Order', 'Order_Count']

print("\n--- CATEGORY PERFORMANCE BY REGION ---")
print(category_region_performance)


# STEP 7: ADVANCED FILTERING AND INDEXING

print("\n[STEP 7] Applying Advanced Filtering and Indexing...")

# Example 1: Filter high-value orders using boolean indexing
high_value_threshold = 500
high_value_orders = full_data[full_data['total_amount'] > high_value_threshold]
print(f"\n✓ High-value orders (>${high_value_threshold}): {len(high_value_orders)}")
print(f"  Total revenue from high-value orders: ${high_value_orders['total_amount'].sum():.2f}")

# Example 2: Use .loc for label-based indexing
electronics_orders = full_data.loc[full_data['category'] == 'Electronics']
print(f"\n✓ Electronics orders: {len(electronics_orders)}")
print(f"  Total electronics revenue: ${electronics_orders['total_amount'].sum():.2f}")

# Example 3: Use .iloc for position-based indexing
first_10_orders = full_data.iloc[:10]
print(f"\n✓ Retrieved first 10 orders using .iloc")
print(f"  First order ID: {first_10_orders.iloc[0]['order_id']}")
print(f"  Last (10th) order ID: {first_10_orders.iloc[9]['order_id']}")

# Example 4: Complex boolean indexing with multiple conditions
north_electronics_high = full_data.loc[
    (full_data['region'] == 'North') & 
    (full_data['category'] == 'Electronics') & 
    (full_data['total_amount'] > 200)
]
print(f"\n✓ North region, Electronics, >$200: {len(north_electronics_high)} orders")
print(f"  Average order value: ${north_electronics_high['total_amount'].mean():.2f}")

# Example 5: Using .query() method for complex filtering
clothing_south = full_data.query("category == 'Clothing' and region == 'South' and quantity >= 3")
print(f"\n✓ Clothing orders in South with quantity >= 3: {len(clothing_south)}")

# Example 6: Multi-index for hierarchical analysis
regional_category_sales = full_data.groupby(['region', 'category'])['total_amount'].sum().round(2)
print("\n--- HIERARCHICAL SALES (Multi-Index) ---")
print(regional_category_sales.head(10))

# Access specific value using multi-index
if ('North', 'Electronics') in regional_category_sales.index:
    print(f"\n✓ North Electronics sales: ${regional_category_sales[('North', 'Electronics')]:.2f}")


# STEP 8: EXPORT RESULTS AND SUMMARY STATISTICS

print("\n[STEP 8] Exporting Results...")

# Calculate overall summary statistics
summary_stats = {
    'Total_Revenue': full_data['total_amount'].sum(),
    'Total_Orders': len(full_data),
    'Average_Order_Value': full_data['total_amount'].mean(),
    'Median_Order_Value': full_data['total_amount'].median(),
    'Total_Customers': full_data['customer_id'].nunique(),
    'Total_Products_Sold': full_data['product_id'].nunique(),
    'Total_Units_Sold': full_data['quantity'].sum(),
    'Average_Quantity_Per_Order': full_data['quantity'].mean()
}

summary_df = pd.DataFrame([summary_stats]).round(2)

print("\n" + "="*80)
print("OVERALL SUMMARY STATISTICS")
print("="*80)
for key, value in summary_stats.items():
    print(f"{key:.<30} {value:>15,.2f}")
print("="*80)

# Export all results to CSV files
sales_by_category.to_csv('sales_by_category.csv')
sales_by_region.to_csv('sales_by_region.csv')
sales_by_month.to_csv('sales_by_month.csv')
top_customers.to_csv('top_customers.csv')
top_products.to_csv('top_products.csv')
category_region_performance.to_csv('category_region_performance.csv')
summary_df.to_csv('overall_summary.csv', index=False)
full_data.to_csv('full_analysis_data.csv', index=False)

print("\n✓ All reports exported successfully!")
print("\nFiles created:")
print("  1. sales_by_category.csv")
print("  2. sales_by_region.csv")
print("  3. sales_by_month.csv")
print("  4. top_customers.csv")
print("  5. top_products.csv")
print("  6. category_region_performance.csv")
print("  7. overall_summary.csv")
print("  8. full_analysis_data.csv")

# STEP 9: DATA VISUALIZATION

print("\n[STEP 9] Creating Visualizations...")

# Create a comprehensive dashboard
fig = plt.figure(figsize=(16, 12))
fig.suptitle('E-commerce Sales Analysis Dashboard', fontsize=18, fontweight='bold', y=0.995)

# 1. Sales by Category (Bar Chart)
ax1 = plt.subplot(3, 3, 1)
sales_by_category['Total_Sales'].plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
ax1.set_title('Total Sales by Category', fontsize=12, fontweight='bold')
ax1.set_xlabel('Category', fontsize=10)
ax1.set_ylabel('Sales ($)', fontsize=10)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# 2. Sales by Region (Pie Chart)
ax2 = plt.subplot(3, 3, 2)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
sales_by_region['Total_Sales'].plot(kind='pie', ax=ax2, autopct='%1.1f%%', 
                                     colors=colors, startangle=90)
ax2.set_title('Sales Distribution by Region', fontsize=12, fontweight='bold')
ax2.set_ylabel('')

# 3. Monthly Sales Trend (Line Chart)
ax3 = plt.subplot(3, 3, 3)
sales_by_month['Total_Sales'].plot(ax=ax3, marker='o', linewidth=2, 
                                   color='green', markersize=8)
ax3.set_title('Monthly Sales Trend', fontsize=12, fontweight='bold')
ax3.set_xlabel('Month', fontsize=10)
ax3.set_ylabel('Sales ($)', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Top 5 Products (Horizontal Bar)
ax4 = plt.subplot(3, 3, 4)
top_5_products = top_products.head(5)['Total_Revenue']
ax4.barh(range(len(top_5_products)), top_5_products.values, color='coral', edgecolor='black')
ax4.set_yticks(range(len(top_5_products)))
ax4.set_yticklabels([str(idx[1]) for idx in top_5_products.index])
ax4.set_title('Top 5 Products by Revenue', fontsize=12, fontweight='bold')
ax4.set_xlabel('Revenue ($)', fontsize=10)
ax4.grid(axis='x', alpha=0.3)

# 5. Order Count by Category (Bar)
ax5 = plt.subplot(3, 3, 5)
sales_by_category['Order_Count'].plot(kind='bar', ax=ax5, color='lightblue', edgecolor='black')
ax5.set_title('Order Count by Category', fontsize=12, fontweight='bold')
ax5.set_xlabel('Category', fontsize=10)
ax5.set_ylabel('Number of Orders', fontsize=10)
ax5.tick_params(axis='x', rotation=45)
ax5.grid(axis='y', alpha=0.3)

# 6. Average Order Value by Region (Bar)
ax6 = plt.subplot(3, 3, 6)
sales_by_region['Avg_Order_Value'].plot(kind='bar', ax=ax6, color='lightgreen', edgecolor='black')
ax6.set_title('Average Order Value by Region', fontsize=12, fontweight='bold')
ax6.set_xlabel('Region', fontsize=10)
ax6.set_ylabel('Avg Order Value ($)', fontsize=10)
ax6.tick_params(axis='x', rotation=45)
ax6.grid(axis='y', alpha=0.3)

# 7. Units Sold by Category (Bar)
ax7 = plt.subplot(3, 3, 7)
sales_by_category['Units_Sold'].plot(kind='bar', ax=ax7, color='gold', edgecolor='black')
ax7.set_title('Units Sold by Category', fontsize=12, fontweight='bold')
ax7.set_xlabel('Category', fontsize=10)
ax7.set_ylabel('Units Sold', fontsize=10)
ax7.tick_params(axis='x', rotation=45)
ax7.grid(axis='y', alpha=0.3)

# 8. Top 5 Customers (Horizontal Bar)
ax8 = plt.subplot(3, 3, 8)
top_5_customers = top_customers.head(5)['Total_Spent']
ax8.barh(range(len(top_5_customers)), top_5_customers.values, color='plum', edgecolor='black')
ax8.set_yticks(range(len(top_5_customers)))
ax8.set_yticklabels([f"{idx[1]}" for idx in top_5_customers.index])
ax8.set_title('Top 5 Customers by Revenue', fontsize=12, fontweight='bold')
ax8.set_xlabel('Total Spent ($)', fontsize=10)
ax8.grid(axis='x', alpha=0.3)

# 9. Order Distribution (Histogram)
ax9 = plt.subplot(3, 3, 9)
ax9.hist(full_data['total_amount'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax9.axvline(full_data['total_amount'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: ${full_data["total_amount"].mean():.2f}')
ax9.set_title('Order Value Distribution', fontsize=12, fontweight='bold')
ax9.set_xlabel('Order Value ($)', fontsize=10)
ax9.set_ylabel('Frequency', fontsize=10)
ax9.legend()
ax9.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('sales_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'sales_dashboard.png'")

# Create additional visualization: Heatmap of Category-Region Performance
fig2, ax = plt.subplots(figsize=(10, 6))
pivot_data = full_data.pivot_table(values='total_amount', index='category', 
                                   columns='region', aggfunc='sum', fill_value=0)
im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(len(pivot_data.columns)))
ax.set_yticks(np.arange(len(pivot_data.index)))
ax.set_xticklabels(pivot_data.columns)
ax.set_yticklabels(pivot_data.index)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Total Sales ($)', rotation=270, labelpad=20)

# Add text annotations
for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        text = ax.text(j, i, f'${pivot_data.values[i, j]:.0f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title('Sales Heatmap: Category vs Region', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('Category', fontsize=12)

plt.tight_layout()
plt.savefig('category_region_heatmap.png', dpi=300, bbox_inches='tight')

