
import pandas as pd
import numpy as np

#Real-World Example: E-commerce Analysis

# Complete e-commerce dataset
np.random.seed(42)

dates = pd.date_range('2024-01-01', periods=100, freq='D')
data = {
    'Date': np.random.choice(dates, 500),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 500),
    'Category': np.random.choice(['Electronics', 'Clothing', 'Home'], 500),
    'Product': np.random.choice(['Item_A', 'Item_B', 'Item_C', 'Item_D'], 500),
    'Sales': np.random.randint(100, 1000, 500),
    'Quantity': np.random.randint(1, 10, 500),
    'Customer_Type': np.random.choice(['New', 'Returning'], 500)
}

ecommerce = pd.DataFrame(data)
ecommerce['Profit'] = ecommerce['Sales'] * 0.3
ecommerce['Date'] = pd.to_datetime(ecommerce['Date'])
ecommerce['Month'] = ecommerce['Date'].dt.month
ecommerce['Year'] = ecommerce['Date'].dt.year

print("E-commerce Dataset (first 10 rows):")
print(ecommerce.head(10))

# Analysis 1: Total sales by region
print("\n=== Analysis 1: Regional Performance ===")
regional_sales = ecommerce.groupby('Region')['Sales'].agg(['sum', 'mean', 'count'])
regional_sales.columns = ['Total_Sales', 'Avg_Sales', 'Num_Transactions']
print(regional_sales)

# Analysis 2: Category performance by region
print("\n=== Analysis 2: Category by Region ===")
category_region = ecommerce.groupby(['Region', 'Category'])['Sales'].sum().unstack()
print(category_region)

# Analysis 3: Monthly trends
print("\n=== Analysis 3: Monthly Trends ===")
monthly = ecommerce.groupby('Month').agg({
    'Sales': 'sum',
    'Quantity': 'sum',
    'Profit': 'sum'
})
print(monthly)

# Analysis 4: Customer type analysis
print("\n=== Analysis 4: Customer Type Performance ===")
customer_analysis = ecommerce.groupby(['Customer_Type', 'Category']).agg({
    'Sales': ['sum', 'mean'],
    'Quantity': 'sum'
})
print(customer_analysis)

# Analysis 5: Iterate and find top performer in each region
print("\n=== Analysis 5: Top Product per Region ===")
for region, group in ecommerce.groupby('Region'):
    top_product = group.groupby('Product')['Sales'].sum().idxmax()
    top_sales = group.groupby('Product')['Sales'].sum().max()
    print(f"{region}: {top_product} (${top_sales:,.0f})")

# Analysis 6: Using hierarchical index
ecommerce_indexed = ecommerce.set_index(['Region', 'Category', 'Product'])
print("\n=== Analysis 6: Hierarchical Index Grouping ===")
level_analysis = ecommerce_indexed.groupby(level=['Region', 'Category'])['Sales'].sum()
print(level_analysis.head(10))

# Analysis 7: Custom function grouping
def sales_tier(sales):
    if sales < 300:
        return 'Low'
    elif sales < 600:
        return 'Medium'
    else:
        return 'High'

ecommerce['Sales_Tier'] = ecommerce['Sales'].apply(sales_tier)
tier_analysis = ecommerce.groupby(['Region', 'Sales_Tier']).size().unstack(fill_value=0)
print("\n=== Analysis 7: Sales Tier Distribution by Region ===")
print(tier_analysis)

# Analysis 8: Column selection efficiency
print("\n=== Analysis 8: Efficient Column Selection ===")
# Only analyze sales and profit
efficient = ecommerce.groupby('Region')[['Sales', 'Profit']].sum()
print(efficient)

#Real-World Example: Sales Analysis

# Realistic sales dataset
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=365, freq='D')

sales_data = pd.DataFrame({
    'Date': np.random.choice(dates, 1000),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 1000),
    'Product': np.random.choice(['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E'], 1000),
    'Sales': np.random.randint(100, 2000, 1000),
    'Quantity': np.random.randint(1, 20, 1000),
    'Cost': np.random.randint(50, 1000, 1000),
    'Customer_Type': np.random.choice(['New', 'Returning', 'VIP'], 1000)
})

sales_data['Profit'] = sales_data['Sales'] - sales_data['Cost']
sales_data['Profit_Margin'] = (sales_data['Profit'] / sales_data['Sales']) * 100

print("Sales Dataset (first 10 rows):")
print(sales_data.head(10))
print("\nDataset Shape:", sales_data.shape)

# Analysis 1: Total sales by region
print("\n=== Analysis 1: Regional Performance ===")
regional_sales = sales_data.groupby('Region')['Sales'].agg(['sum', 'mean', 'count'])
regional_sales.columns = ['Total_Sales', 'Avg_Sales', 'Num_Transactions']
print(regional_sales)

# Analysis 2: Category performance by region
print("\n=== Analysis 2: Category by Region ===")
category_region = sales_data.groupby(['Region', 'Product_Category'])['Sales'].sum().unstack()
print(category_region)

# Analysis 3: Monthly trends
print("\n=== Analysis 3: Monthly Trends ===")
monthly = sales_data.groupby('Date')['Sales'].agg(['sum', 'mean', 'count'])
monthly.columns = ['Total_Sales', 'Avg_Sales', 'Num_Transactions']
print(monthly)

# Analysis 4: Customer type analysis
print("\n=== Analysis 4: Customer Type Performance ===")
customer_analysis = sales_data.groupby(['Customer_Type', 'Product_Category']).agg({
    'Sales': ['sum', 'mean'],
    'Quantity': 'sum'
})
print(customer_analysis)

# Analysis 5: Iterate and find top performer in each region
print("\n=== Analysis 5: Top Product per Region ===")
for region, group in sales_data.groupby('Region'):
    top_product = group.groupby('Product')['Sales'].sum().idxmax()
    top_sales = group.groupby('Product')['Sales'].sum().max()
    print(f"{region}: {top_product} (${top_sales:,.0f})")

# Analysis 6: Using hierarchical index
sales_indexed = sales_data.set_index(['Region', 'Product_Category', 'Product'])
print("\n=== Analysis 6: Hierarchical Index Grouping ===")
level_analysis = sales_indexed.groupby(level=['Region', 'Product_Category'])['Sales'].sum()
print(level_analysis.head(10))

# Analysis 7: Custom function grouping
def sales_tier(sales):
    if sales < 300:
        return 'Low'
    elif sales < 600:
        return 'Medium'
    else:
        return 'High'

sales_data['Sales_Tier'] = sales_data['Sales'].apply(sales_tier)
tier_analysis = sales_data.groupby(['Region', 'Sales_Tier']).size().unstack(fill_value=0)
print("\n=== Analysis 7: Sales Tier Distribution by Region ===")
print(tier_analysis)

# Analysis 8: Column selection efficiency
print("\n=== Analysis 8: Efficient Column Selection ===")
# Only analyze sales and profit
efficient = sales_data.groupby('Region')[['Sales', 'Profit']].sum()
print(efficient)

#Product Category Performance
print("\n" + "="*70)
print("ANALYSIS 2: PRODUCT CATEGORY PERFORMANCE")
print("="*70)

# Custom functions for detailed analysis
def top_product(group):
    """Find the product with highest sales in each group"""
    return group.groupby('Product')['Sales'].sum().idxmax()

def sales_concentration(group):
    """Calculate how concentrated sales are (top product % of total)"""
    total_sales = group['Sales'].sum()
    top_sales = group.groupby('Product')['Sales'].sum().max()
    return (top_sales / total_sales) * 100

category_analysis = sales_data.groupby('Product_Category').agg(
    Total_Sales=('Sales', 'sum'),
    Avg_Sale=('Sales', 'mean'),
    Total_Transactions=('Sales', 'count'),
    Total_Profit=('Profit', 'sum'),
    Avg_Profit_Margin=('Profit_Margin', 'mean'),
    Total_Units=('Quantity', 'sum'),
    Min_Sale=('Sales', 'min'),
    Max_Sale=('Sales', 'max'),
    Sales_Std=('Sales', 'std')
)

print(category_analysis)

# Add custom calculations
print("\n=== Advanced Category Metrics ===")
for category, group in sales_data.groupby('Product_Category'):
    top_prod = top_product(group)
    concentration = sales_concentration(group)
    print(f"{category:15} | Top Product: {top_prod:10} | Concentration: {concentration:.1f}%")