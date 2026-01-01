import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sympy import symbols, diff, exp, lambdify
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 70)
print("CHAIN RULE MASTERY - PART 4: ADVANCED ANALYSIS & INSIGHTS")
print("=" * 70)

# ============================================================================
# TOPIC 14: Multi-Layer Chain Rule (Deep Network)
# ============================================================================
print("\n" + "=" * 70)
print("TOPIC 14: Deep Chain Rule - 4 Layer Network")
print("=" * 70)

def deep_chain_rule_example():
    """
    Demonstrates chain rule through 4 layers:
    y = f4(f3(f2(f1(x))))
    """
    
    # Define each layer symbolically
    x_sym = symbols('x')
    
    # Layer functions
    f1 = x_sym**2  # f1(x) = x^2
    f2 = exp(f1)   # f2 = e^(f1)
    f3 = f2 + 1    # f3 = f2 + 1
    f4 = 1/f3      # f4 = 1/f3
    
    # Compute derivatives
    df1_dx = diff(f1, x_sym)
    df2_df1 = diff(f2, x_sym) / df1_dx  # df2/df1
    df3_df2 = diff(f3, x_sym) / diff(f2, x_sym)  # df3/df2
    df4_df3 = diff(f4, x_sym) / diff(f3, x_sym)  # df4/df3
    
    # Final derivative using chain rule
    df4_dx = diff(f4, x_sym)
    
    print("Deep Composition: y = 1/(e^(x^2) + 1)")
    print("\nLayer-by-layer:")
    print(f"  f1(x) = {f1}")
    print(f"  f2(f1) = {f2}")
    print(f"  f3(f2) = {f3}")
    print(f"  f4(f3) = {f4}")
    
    print("\nDerivatives:")
    print(f"  df1/dx = {df1_dx}")
    print(f"  df2/df1 = e^(f1)")
    print(f"  df3/df2 = 1")
    print(f"  df4/df3 = -1/f3^2")
    
    print(f"\nChain Rule: dy/dx = df4/df3 * df3/df2 * df2/df1 * df1/dx")
    print(f"Result: {df4_dx}")
    
    # Numerical evaluation
    x_vals = np.linspace(-2, 2, 100)
    
    f1_num = lambdify(x_sym, f1, 'numpy')
    f2_num = lambdify(x_sym, f2, 'numpy')
    f3_num = lambdify(x_sym, f3, 'numpy')
    f4_num = lambdify(x_sym, f4, 'numpy')
    df4_dx_num = lambdify(x_sym, df4_dx, 'numpy')
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot each layer
    layer_funcs = [
        (f1_num, 'f1(x) = x²', 'blue'),
        (f2_num, 'f2 = e^(f1)', 'green'),
        (f3_num, 'f3 = f2 + 1', 'orange'),
        (f4_num, 'f4 = 1/f3', 'red'),
        (df4_dx_num, 'dy/dx (Final)', 'purple')
    ]
    
    for idx, (func, label, color) in enumerate(layer_funcs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        y_vals = func(x_vals)
        ax.plot(x_vals, y_vals, color=color, linewidth=2.5, label=label)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Remove empty subplot
    axes[1, 2].axis('off')
    
    # Add chain rule diagram
    ax = axes[1, 2]
    ax.text(0.5, 0.8, 'Chain Rule Flow:', ha='center', fontsize=14, 
            fontweight='bold', transform=ax.transAxes)
    
    chain_text = """
    x → f1 → f2 → f3 → f4 → y
    
    dy/dx = df4/df3 × df3/df2 × 
            df2/df1 × df1/dx
    
    Each arrow represents 
    one derivative multiplication!
    """
    ax.text(0.5, 0.4, chain_text, ha='center', va='center', fontsize=11,
            transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('deep_chain_rule.png', dpi=150, bbox_inches='tight')
    print("\n✓ Deep chain rule visualization saved")
    plt.show()

deep_chain_rule_example()

# ============================================================================
# TOPIC 15: Gradient Accumulation and Vanishing Gradients
# ============================================================================
print("\n" + "=" * 70)
print("TOPIC 15: Understanding Vanishing/Exploding Gradients")
print("=" * 70)

def analyze_gradient_propagation():
    """
    Shows how gradients can vanish or explode in deep networks
    """
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    # Simulate gradient flow through multiple layers
    n_layers = 10
    initial_gradient = 1.0
    
    # Test different activation values
    activation_values = np.linspace(-5, 5, 100)
    
    gradient_flows = []
    
    for activation in activation_values:
        gradient = initial_gradient
        layer_gradients = [gradient]
        
        for layer in range(n_layers):
            # Gradient gets multiplied by sigmoid derivative at each layer
            gradient *= sigmoid_derivative(activation)
            layer_gradients.append(gradient)
        
        gradient_flows.append(layer_gradients)
    
    gradient_flows = np.array(gradient_flows)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sigmoid derivative
    ax = axes[0, 0]
    ax.plot(activation_values, sigmoid_derivative(activation_values), 
            'b-', linewidth=2.5)
    ax.fill_between(activation_values, 0, sigmoid_derivative(activation_values), 
                     alpha=0.3)
    ax.axhline(y=0.25, color='r', linestyle='--', label='Max value = 0.25')
    ax.set_xlabel('Activation Value', fontsize=11)
    ax.set_ylabel("σ'(x)", fontsize=11)
    ax.set_title("Sigmoid Derivative (Always ≤ 0.25)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 2: Gradient flow through layers
    ax = axes[0, 1]
    for layer_idx in [0, 2, 5, 9]:
        ax.plot(activation_values, gradient_flows[:, layer_idx], 
                linewidth=2, label=f'Layer {layer_idx}')
    ax.set_xlabel('Activation Value', fontsize=11)
    ax.set_ylabel('Gradient Magnitude', fontsize=11)
    ax.set_title('Gradient Flow Through Layers', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 3: Heatmap of gradient decay
    ax = axes[1, 0]
    im = ax.imshow(gradient_flows.T, aspect='auto', cmap='viridis', 
                   extent=[activation_values[0], activation_values[-1], 0, n_layers])
    ax.set_xlabel('Activation Value', fontsize=11)
    ax.set_ylabel('Layer Depth', fontsize=11)
    ax.set_title('Gradient Magnitude Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Gradient Magnitude')
    
    # Plot 4: Gradient at final layer
    ax = axes[1, 1]
    final_gradients = gradient_flows[:, -1]
    ax.plot(activation_values, final_gradients, 'r-', linewidth=2.5)
    ax.fill_between(activation_values, 0, final_gradients, alpha=0.3, color='red')
    ax.set_xlabel('Activation Value', fontsize=11)
    ax.set_ylabel('Final Gradient', fontsize=11)
    ax.set_title(f'Gradient After {n_layers} Layers', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    min_gradient = np.min(final_gradients[final_gradients > 0])
    ax.text(0.5, 0.9, f'Minimum gradient: {min_gradient:.2e}', 
            transform=ax.transAxes, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('gradient_vanishing.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gradient vanishing analysis saved")
    plt.show()
    
    # Print analysis
    print("\nKey Insights:")
    print(f"  • Sigmoid derivative max value: 0.25")
    print(f"  • After 10 layers with sigmoid: gradient ≈ 0.25^10 = {0.25**10:.2e}")
    print(f"  • This is the VANISHING GRADIENT problem!")
    print(f"  • Solution: Use ReLU, ResNets, or careful initialization")

analyze_gradient_propagation()

# ============================================================================
# TOPIC 16: Computational Graph and Automatic Differentiation
# ============================================================================
print("\n" + "=" * 70)
print("TOPIC 16: Computational Graph Visualization")
print("=" * 70)

def create_computation_graph():
    """
    Visualize a computational graph for: y = (x * w1 + b) * w2
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Define node positions
    nodes = {
        'x': (0, 2),
        'w1': (0, 1),
        'mul1': (1.5, 1.5),
        'b': (1.5, 0.5),
        'add': (3, 1.5),
        'w2': (3, 0.5),
        'mul2': (4.5, 1),
        'y': (6, 1)
    }
    
    # Draw nodes
    for name, (x, y) in nodes.items():
        if name in ['x', 'w1', 'w2', 'b']:
            color = 'lightgreen'
            shape = 'round'
        elif name == 'y':
            color = 'lightcoral'
            shape = 'round'
        else:
            color = 'lightblue'
            shape = 'round'
        
        circle = plt.Circle((x, y), 0.25, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw forward edges
    edges = [
        ('x', 'mul1', 'forward'),
        ('w1', 'mul1', 'forward'),
        ('mul1', 'add', 'forward'),
        ('b', 'add', 'forward'),
        ('add', 'mul2', 'forward'),
        ('w2', 'mul2', 'forward'),
        ('mul2', 'y', 'forward')
    ]
    
    for start, end, edge_type in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        # Adjust positions to edge of circles
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        dx /= length
        dy /= length
        
        start_x = x1 + 0.25 * dx
        start_y = y1 + 0.25 * dy
        end_x = x2 - 0.25 * dx
        end_y = y2 - 0.25 * dy
        
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='blue', zorder=1)
        ax.add_patch(arrow)
    
    # Add gradient flow arrows (backprop)
    gradient_edges = [
        ('y', 'mul2', '∂y/∂mul2'),
        ('mul2', 'add', '∂y/∂add'),
        ('mul2', 'w2', '∂y/∂w2'),
        ('add', 'mul1', '∂y/∂mul1'),
        ('add', 'b', '∂y/∂b'),
        ('mul1', 'x', '∂y/∂x'),
        ('mul1', 'w1', '∂y/∂w1')
    ]
    
    for start, end, label in gradient_edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        # Draw gradient arrow below the forward arrow
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 - 0.15
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        dx /= length
        dy /= length
        
        end_x = x2 - 0.25 * dx
        end_y = y2 - 0.25 * dy - 0.15
        start_x = x1 + 0.25 * dx
        start_y = y1 + 0.25 * dy - 0.15
        
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='red', 
                               linestyle='--', zorder=1)
        ax.add_patch(arrow)
    
    # Add title and labels
    ax.text(3, 3, 'Computational Graph: y = (x × w1 + b) × w2', 
            ha='center', fontsize=16, fontweight='bold')
    
    ax.text(1, 3.2, 'Forward Pass →', color='blue', fontsize=12, fontweight='bold')
    ax.text(5, 3.2, '← Backward Pass (Gradients)', color='red', fontsize=12, fontweight='bold')
    
    # Legend
    legend_y = 0
    ax.add_patch(plt.Circle((0.5, legend_y), 0.15, color='lightgreen', ec='black'))
    ax.text(0.8, legend_y, 'Input/Parameter', fontsize=10)
    
    ax.add_patch(plt.Circle((2.5, legend_y), 0.15, color='lightblue', ec='black'))
    ax.text(2.8, legend_y, 'Operation', fontsize=10)
    
    ax.add_patch(plt.Circle((4.5, legend_y), 0.15, color='lightcoral', ec='black'))
    ax.text(4.8, legend_y, 'Output', fontsize=10)
    
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('computational_graph.png', dpi=150, bbox_inches='tight')
    print("\n✓ Computational graph saved")
    plt.show()

create_computation_graph()

# ============================================================================
# TOPIC 17: Summary Statistics
# ============================================================================
print("\n" + "=" * 70)
print("TOPIC 17: Chain Rule - Summary and Key Formulas")
print("=" * 70)

summary_data = {
    'Concept': [
        'Basic Chain Rule',
        'Two Functions',
        'Three Functions',
        'Neural Network',
        'Sigmoid Derivative',
        'ReLU Derivative',
        'Tanh Derivative',
        'General Backprop'
    ],
    'Formula': [
        'd/dx[f(g(x))] = f\'(g(x)) · g\'(x)',
        'dy/dx = dy/du · du/dx',
        'dy/dx = dy/dv · dv/du · du/dx',
        'dL/dx = dL/dz · dz/dx',
        'σ\'(x) = σ(x)(1-σ(x))',
        'ReLU\'(x) = 1 if x>0 else 0',
        'tanh\'(x) = 1 - tanh²(x)',
        'dL/dW = dL/dz · dz/dW'
    ],
    'Application': [
        'All composite functions',
        'Simple compositions',
        'Deep compositions',
        'Forward + Backward pass',
        'Binary classification',
        'Deep learning (common)',
        'Range [-1, 1] output',
        'Gradient descent'
    ]
}

df_summary = pd.DataFrame(summary_data)

print("\n" + "=" * 70)
print("KEY FORMULAS AND CONCEPTS")
print("=" * 70)
print(df_summary.to_string(index=False))

# Create a visual summary
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

title_text = "CHAIN RULE: THE ENGINE OF DEEP LEARNING"
ax.text(0.5, 0.95, title_text, ha='center', fontsize=18, 
        fontweight='bold', transform=ax.transAxes)

summary_text = """
┌─────────────────────────────────────────────────────────────────┐
│                     CORE PRINCIPLE                             │
├─────────────────────────────────────────────────────────────────┤
│  The derivative of a composite function is the product of       │
│  derivatives of each component function.                        │
│                                                                 │
│  d/dx[f(g(x))] = f'(g(x)) × g'(x)                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  NEURAL NETWORK APPLICATION                     │
├─────────────────────────────────────────────────────────────────┤
│  Forward Pass:  Input → Layer 1 → Layer 2 → ... → Output        │
│  Backward Pass: ∂L/∂output → ∂L/∂Layer2 → ∂L/∂Layer1 → ∂L/∂W    │
│                                                                 │
│  Each "→" in backprop is one chain rule multiplication!         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    KEY INSIGHTS                                  │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Backpropagation = Chain rule applied recursively             │
│  ✓ Gradient of layer n depends on gradient of layer n+1         │
│  ✓ Gradients can vanish (sigmoid) or explode (bad init)         │
│  ✓ Each layer's derivative acts as a "gate" for gradients       │
│  ✓ Choice of activation function deeply affects learning        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  PRACTICAL TIPS                                 │
├─────────────────────────────────────────────────────────────────┤
│  • Always compute forward pass first (store intermediate values)│
│  • Store activations needed for backward pass                   │
│  • Use ReLU to avoid vanishing gradients                        │
│  • Batch normalization helps stabilize gradients                │
│  • Monitor gradient magnitudes during training                  │
└─────────────────────────────────────────────────────────────────┘
"""

ax.text(0.5, 0.5, summary_text, ha='center', va='center',
        fontsize=10, family='monospace', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('chain_rule_summary.png', dpi=150, bbox_inches='tight')
print("\n✓ Summary visualization saved")
plt.show()

print("\n" + "=" * 70)
print("🎓 CONGRATULATIONS! CHAIN RULE MASTERY COMPLETE!")
print("=" * 70)
print("\nYou now understand:")
print("  ✓ Chain rule mathematical foundation")
print("  ✓ Sigmoid and activation function derivatives")
print("  ✓ Forward and backward propagation")
print("  ✓ Gradient flow in deep networks")
print("  ✓ Vanishing/exploding gradient problem")
print("  ✓ Computational graphs")
print("\nNext steps:")
print("  → Implement a multi-layer network from scratch")
print("  → Study optimization algorithms (SGD, Adam)")
print("  → Learn about advanced architectures (CNNs, RNNs)")
print("=" * 70)