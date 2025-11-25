
#Stock Price Visualizer with Moving Averages 
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic stock data
np.random.seed(42)
days = 365
dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]

# Generate realistic stock price movement
initial_price = 100
returns = np.random.normal(0.001, 0.02, days)  # Daily returns
prices = initial_price * np.exp(np.cumsum(returns))

# Calculate moving averages
def moving_average(data, window):
    """Calculate simple moving average"""
    return np.convolve(data, np.ones(window), 'valid') / window

ma_20 = moving_average(prices, 20)
ma_50 = moving_average(prices, 50)
ma_100 = moving_average(prices, 100)

# Create the visualization
fig = plt.figure(figsize=(16, 10))

# Main price chart
ax1 = plt.subplot(2, 1, 1)
ax1.plot(dates, prices, label='Stock Price', color='#2E86AB', linewidth=2, alpha=0.8)
ax1.plot(dates[19:], ma_20, label='20-day MA', color='#A23B72', linewidth=2, linestyle='--')
ax1.plot(dates[49:], ma_50, label='50-day MA', color='#F18F01', linewidth=2, linestyle='--')
ax1.plot(dates[99:], ma_100, label='100-day MA', color='#C73E1D', linewidth=2, linestyle='--')

ax1.set_title('Stock Price Analysis with Moving Averages', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add shaded regions for bull/bear markets
ax1.fill_between(dates[:180], 90, 120, alpha=0.1, color='green', label='Bull Period')
ax1.fill_between(dates[180:250], 90, 120, alpha=0.1, color='red', label='Bear Period')

# Annotate key events
max_price_idx = np.argmax(prices)
min_price_idx = np.argmin(prices)

ax1.annotate(f'Peak: ${prices[max_price_idx]:.2f}',
             xy=(dates[max_price_idx], prices[max_price_idx]),
             xytext=(dates[max_price_idx], prices[max_price_idx] + 10),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=10, color='green', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', alpha=0.8))

ax1.annotate(f'Low: ${prices[min_price_idx]:.2f}',
             xy=(dates[min_price_idx], prices[min_price_idx]),
             xytext=(dates[min_price_idx], prices[min_price_idx] - 10),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))

# Volume subplot (simulated)
ax2 = plt.subplot(2, 1, 2)
volume = np.random.uniform(1000000, 5000000, days)
colors = ['green' if prices[i] > prices[i-1] else 'red' for i in range(1, days)]
colors.insert(0, 'green')

ax2.bar(dates, volume, color=colors, alpha=0.6, width=0.8)
ax2.set_title('Trading Volume', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Volume', fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

# Format y-axis for volume
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

plt.tight_layout()

# Save the figure
plt.savefig('stock_price_analysis.png', dpi=300, bbox_inches='tight')
print("Stock analysis saved as 'stock_price_analysis.png'")
print(f"Price Range: ${prices.min():.2f} - ${prices.max():.2f}")
print(f"Current Price: ${prices[-1]:.2f}")
print(f"Total Return: {((prices[-1]/prices[0] - 1) * 100):.2f}%")

plt.show()