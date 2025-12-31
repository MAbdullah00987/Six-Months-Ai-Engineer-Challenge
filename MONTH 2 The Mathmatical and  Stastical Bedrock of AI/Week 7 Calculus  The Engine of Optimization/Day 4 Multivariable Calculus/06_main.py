#Phase 2: Manual Calculation (Build Intuition)
#Let's work through a concrete example by hand first:
#Example: h(x) = sigmoid(3x + 2)

#Outer function: f(u) = sigmoid(u) = 1/(1 + e^(-u))
#Inner function: g(x) = 3x + 2

#Step-by-step:

#g'(x) = 3
#f'(u) = sigmoid(u) · (1 - sigmoid(u))
#h'(x) = f'(g(x)) · g'(x) = sigmoid(3x + 2) · (1 - sigmoid(3x + 2)) · 3

#Visualizing Gradient Flow with Seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
sns.set_palette("husl")

# Functions for analysis
def linear(x, w, b):
    return w * x + b

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

# Derivatives - FIXED to handle both arrays and scalars
def relu_derivative(x):
    x = np.asarray(x)  # Convert to array if scalar
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Create comprehensive gradient flow analysis
x = np.linspace(-5, 5, 1000)

# Data for different activation functions
data = {
    'x': np.tile(x, 3),
    'Function Value': np.concatenate([sigmoid(x), relu(x), tanh(x)]),
    'Gradient': np.concatenate([sigmoid_derivative(x), relu_derivative(x), tanh_derivative(x)]),
    'Activation': ['Sigmoid']*len(x) + ['ReLU']*len(x) + ['Tanh']*len(x)
}

df = pd.DataFrame(data)

# Create figure
fig = plt.figure(figsize=(18, 12))

# Plot 1: Activation functions
ax1 = plt.subplot(3, 3, 1)
for activation in ['Sigmoid', 'ReLU', 'Tanh']:
    subset = df[df['Activation'] == activation]
    ax1.plot(subset['x'], subset['Function Value'], label=activation, linewidth=2.5)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('Input (x)', fontsize=12)
ax1.set_ylabel('Output f(x)', fontsize=12)
ax1.set_title('Activation Functions', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Gradients (derivatives)
ax2 = plt.subplot(3, 3, 2)
for activation in ['Sigmoid', 'ReLU', 'Tanh']:
    subset = df[df['Activation'] == activation]
    ax2.plot(subset['x'], subset['Gradient'], label=activation, linewidth=2.5)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Input (x)', fontsize=12)
ax2.set_ylabel("Gradient f'(x)", fontsize=12)
ax2.set_title('Gradients (Derivatives)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Gradient magnitude comparison (heatmap) - FIXED: added observed parameter
ax3 = plt.subplot(3, 3, 3)
pivot_data = df.pivot_table(values='Gradient', index='Activation', 
                             columns=pd.cut(df['x'], bins=20), aggfunc='mean', 
                             observed=False)  # FIXED: Added to suppress warning
sns.heatmap(pivot_data, cmap='RdYlGn', center=0, ax=ax3, cbar_kws={'label': 'Gradient'})
ax3.set_xlabel('Input Range', fontsize=12)
ax3.set_ylabel('Activation', fontsize=12)
ax3.set_title('Gradient Heatmap', fontsize=14, fontweight='bold')
ax3.set_xticklabels([])

# Plot 4: Chain rule through multiple layers (sigmoid)
ax4 = plt.subplot(3, 3, 4)
layers = 5
colors = plt.cm.viridis(np.linspace(0, 1, layers))

gradient_product = np.ones_like(x)
for layer in range(layers):
    gradient_product *= sigmoid_derivative(x)
    ax4.plot(x, gradient_product, label=f'After Layer {layer+1}', 
             linewidth=2, color=colors[layer])

ax4.set_xlabel('Input (x)', fontsize=12)
ax4.set_ylabel('Cumulative Gradient', fontsize=12)
ax4.set_title('Vanishing Gradient (Sigmoid)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

# Plot 5: Chain rule through multiple layers (ReLU)
ax5 = plt.subplot(3, 3, 5)
gradient_product_relu = np.ones_like(x)
for layer in range(layers):
    gradient_product_relu *= relu_derivative(x)
    ax5.plot(x, gradient_product_relu, label=f'After Layer {layer+1}', 
             linewidth=2, color=colors[layer])

ax5.set_xlabel('Input (x)', fontsize=12)
ax5.set_ylabel('Cumulative Gradient', fontsize=12)
ax5.set_title('Stable Gradient (ReLU)', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Gradient flow comparison - FIXED: properly handle scalar values
ax6 = plt.subplot(3, 3, 6)
x_sample = 1.5
sigmoid_flow = [sigmoid_derivative(x_sample)**i for i in range(1, 11)]
relu_flow = [float(relu_derivative(x_sample))**i for i in range(1, 11)]  # FIXED: convert to float
tanh_flow = [tanh_derivative(x_sample)**i for i in range(1, 11)]

layers_range = range(1, 11)
ax6.plot(layers_range, sigmoid_flow, 'o-', label='Sigmoid', linewidth=2, markersize=8)
ax6.plot(layers_range, relu_flow, 's-', label='ReLU', linewidth=2, markersize=8)
ax6.plot(layers_range, tanh_flow, '^-', label='Tanh', linewidth=2, markersize=8)
ax6.set_xlabel('Layer Depth', fontsize=12)
ax6.set_ylabel('Gradient Magnitude', fontsize=12)
ax6.set_title(f'Gradient at x={x_sample} Through Layers', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_yscale('log')

# Plot 7: Distribution of gradients (violin plot)
ax7 = plt.subplot(3, 3, 7)
sns.violinplot(data=df, x='Activation', y='Gradient', ax=ax7, palette='Set2')
ax7.set_xlabel('Activation Function', fontsize=12)
ax7.set_ylabel('Gradient Distribution', fontsize=12)
ax7.set_title('Gradient Distribution Comparison', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Composite function example
ax8 = plt.subplot(3, 3, 8)
# h(x) = sigmoid(relu(x^2))
x_comp = np.linspace(-3, 3, 500)
y1 = x_comp**2  # g1(x)
y2 = relu(y1)   # g2(g1(x))
y3 = sigmoid(y2)  # f(g2(g1(x)))

ax8.plot(x_comp, y1/10, label='x² (scaled)', linewidth=2, alpha=0.7)
ax8.plot(x_comp, y2/10, label='ReLU(x²) (scaled)', linewidth=2, alpha=0.7)
ax8.plot(x_comp, y3, label='Sigmoid(ReLU(x²))', linewidth=2.5)
ax8.set_xlabel('x', fontsize=12)
ax8.set_ylabel('Output', fontsize=12)
ax8.set_title('Composite Function: h(x) = σ(ReLU(x²))', fontsize=14, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Summary statistics
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

stats_text = """
GRADIENT FLOW INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━

Sigmoid:
  • Saturates at extremes
  • Max gradient: ~0.25
  • Problem: Vanishing gradients

ReLU:
  • Gradient = 1 (x > 0)
  • Gradient = 0 (x ≤ 0)
  • Problem: Dying neurons

Tanh:
  • Similar to sigmoid
  • Max gradient: 1.0
  • Better than sigmoid

CHAIN RULE EFFECT:
  Deep networks multiply gradients
  → Small gradients compound
  → Vanishing/exploding gradients
"""

ax9.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10,
         family='monospace', transform=ax9.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.show()

# Print numerical analysis
print("=" * 70)
print("GRADIENT FLOW ANALYSIS")
print("=" * 70)

for activation in ['Sigmoid', 'ReLU', 'Tanh']:
    subset = df[df['Activation'] == activation]
    print(f"\n{activation}:")
    print(f"  Mean Gradient:   {subset['Gradient'].mean():.4f}")
    print(f"  Max Gradient:    {subset['Gradient'].max():.4f}")
    print(f"  Min Gradient:    {subset['Gradient'].min():.4f}")
    print(f"  Std Gradient:    {subset['Gradient'].std():.4f}")

print("\n" + "=" * 70)
print("10-LAYER GRADIENT FLOW (at x=1.5)")
print("=" * 70)

x_test = 1.5
for activation, func in [('Sigmoid', sigmoid_derivative), 
                         ('ReLU', relu_derivative), 
                         ('Tanh', tanh_derivative)]:
    gradient = float(func(x_test))  # FIXED: convert to float
    print(f"\n{activation}:")
    for depth in [1, 5, 10, 20]:
        compound_grad = gradient ** depth
        print(f"  Depth {depth:2d}: {compound_grad:.6e}")

print("\n" + "=" * 70)