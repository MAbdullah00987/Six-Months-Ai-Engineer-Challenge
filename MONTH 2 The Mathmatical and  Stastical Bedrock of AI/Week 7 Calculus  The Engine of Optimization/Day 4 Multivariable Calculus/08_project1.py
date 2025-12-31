#Phase 2: Manual Calculation (Build Intuition)
#Let's work through a concrete example by hand first:
#Example: h(x) = sigmoid(3x + 2)

#Outer function: f(u) = sigmoid(u) = 1/(1 + e^(-u))
#Inner function: g(x) = 3x + 2

#Step-by-step:

#g'(x) = 3
#f'(u) = sigmoid(u) · (1 - sigmoid(u))
#h'(x) = f'(g(x)) · g'(x) = sigmoid(3x + 2) · (1 - sigmoid(3x + 2)) · 3

#Interactive Chain Rule Explorer 


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# Define functions and derivatives
functions = {
    'x^2': (lambda x: x**2, lambda x: 2*x, 'Quadratic'),
    'sin(x)': (lambda x: np.sin(x), lambda x: np.cos(x), 'Sine'),
    'exp(x)': (lambda x: np.exp(np.clip(x, -10, 10)), lambda x: np.exp(np.clip(x, -10, 10)), 'Exponential'),
    'sigmoid': (lambda x: 1/(1+np.exp(-np.clip(x, -500, 500))), 
                lambda x: (s:=1/(1+np.exp(-np.clip(x, -500, 500)))) * (1-s), 'Sigmoid'),
    'tanh': (lambda x: np.tanh(x), lambda x: 1-np.tanh(x)**2, 'Tanh'),
    'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float), 'ReLU')
}

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# Main plots
ax_outer = fig.add_subplot(gs[0, 0])
ax_inner = fig.add_subplot(gs[0, 1])
ax_composite = fig.add_subplot(gs[0, 2])
ax_deriv_outer = fig.add_subplot(gs[1, 0])
ax_deriv_inner = fig.add_subplot(gs[1, 1])
ax_deriv_composite = fig.add_subplot(gs[1, 2])
ax_chain_viz = fig.add_subplot(gs[2, :])
ax_info = fig.add_subplot(gs[3, :])
ax_info.axis('off')

# Initial setup
x = np.linspace(-5, 5, 500)
current_outer = 'sigmoid'
current_inner = 'x^2'
eval_point = 1.0

def update_plots():
    # Get functions
    f_outer, f_outer_prime, name_outer = functions[current_outer]
    g_inner, g_inner_prime, name_inner = functions[current_inner]
    
    # Compute composite
    try:
        g_x = g_inner(x)
        h_x = f_outer(g_x)
        
        # Derivatives
        g_prime = g_inner_prime(x)
        f_prime_at_g = f_outer_prime(g_x)
        h_prime = f_prime_at_g * g_prime
        
        # Evaluation point
        g_eval = g_inner(eval_point)
        h_eval = f_outer(g_eval)
        g_prime_eval = g_inner_prime(eval_point)
        f_prime_eval = f_outer_prime(g_eval)
        h_prime_eval = f_prime_eval * g_prime_eval
        
        # Clear axes
        for ax in [ax_outer, ax_inner, ax_composite, ax_deriv_outer, ax_deriv_inner, ax_deriv_composite]:
            ax.clear()
        
        # Plot outer function
        z = np.linspace(-5, 5, 500)
        ax_outer.plot(z, f_outer(z), 'b-', linewidth=2.5, label=f'f(u) = {name_outer}')
        ax_outer.scatter([g_eval], [h_eval], color='red', s=150, zorder=5, edgecolors='black', linewidth=2)
        ax_outer.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_outer.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax_outer.set_xlabel('u', fontsize=11, fontweight='bold')
        ax_outer.set_ylabel('f(u)', fontsize=11, fontweight='bold')
        ax_outer.set_title(f'Outer: f(u) = {name_outer}', fontsize=13, fontweight='bold')
        ax_outer.legend(fontsize=10)
        ax_outer.grid(True, alpha=0.3)
        ax_outer.set_ylim([-5, 5])
        
        # Plot inner function
        ax_inner.plot(x, g_x, 'g-', linewidth=2.5, label=f'g(x) = {name_inner}')
        ax_inner.scatter([eval_point], [g_eval], color='red', s=150, zorder=5, edgecolors='black', linewidth=2)
        ax_inner.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_inner.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax_inner.set_xlabel('x', fontsize=11, fontweight='bold')
        ax_inner.set_ylabel('g(x)', fontsize=11, fontweight='bold')
        ax_inner.set_title(f'Inner: g(x) = {name_inner}', fontsize=13, fontweight='bold')
        ax_inner.legend(fontsize=10)
        ax_inner.grid(True, alpha=0.3)
        ax_inner.set_ylim([-5, 5])
        
        # Plot composite
        ax_composite.plot(x, h_x, 'purple', linewidth=2.5, label=f'h(x) = {name_outer}({name_inner})')
        ax_composite.scatter([eval_point], [h_eval], color='red', s=150, zorder=5, edgecolors='black', linewidth=2)
        ax_composite.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_composite.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax_composite.set_xlabel('x', fontsize=11, fontweight='bold')
        ax_composite.set_ylabel('h(x)', fontsize=11, fontweight='bold')
        ax_composite.set_title(f'Composite: h(x) = f(g(x))', fontsize=13, fontweight='bold')
        ax_composite.legend(fontsize=10)
        ax_composite.grid(True, alpha=0.3)
        ax_composite.set_ylim([-5, 5])
        
        # Plot derivatives
        ax_deriv_outer.plot(z, f_outer_prime(z), 'b-', linewidth=2.5, label=f"f'(u)")
        ax_deriv_outer.scatter([g_eval], [f_prime_eval], color='red', s=150, zorder=5, edgecolors='black', linewidth=2)
        ax_deriv_outer.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_deriv_outer.set_xlabel('u', fontsize=11, fontweight='bold')
        ax_deriv_outer.set_ylabel("f'(u)", fontsize=11, fontweight='bold')
        ax_deriv_outer.set_title(f"Derivative: f'(u)", fontsize=13, fontweight='bold')
        ax_deriv_outer.legend(fontsize=10)
        ax_deriv_outer.grid(True, alpha=0.3)
        ax_deriv_outer.set_ylim([-3, 3])
        
        ax_deriv_inner.plot(x, g_prime, 'g-', linewidth=2.5, label=f"g'(x)")
        ax_deriv_inner.scatter([eval_point], [g_prime_eval], color='red', s=150, zorder=5, edgecolors='black', linewidth=2)
        ax_deriv_inner.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_deriv_inner.set_xlabel('x', fontsize=11, fontweight='bold')
        ax_deriv_inner.set_ylabel("g'(x)", fontsize=11, fontweight='bold')
        ax_deriv_inner.set_title(f"Derivative: g'(x)", fontsize=13, fontweight='bold')
        ax_deriv_inner.legend(fontsize=10)
        ax_deriv_inner.grid(True, alpha=0.3)
        ax_deriv_inner.set_ylim([-3, 3])
        
        ax_deriv_composite.plot(x, h_prime, 'purple', linewidth=2.5, label=f"h'(x) = f'(g(x))·g'(x)")
        ax_deriv_composite.scatter([eval_point], [h_prime_eval], color='red', s=150, zorder=5, edgecolors='black', linewidth=2)
        ax_deriv_composite.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_deriv_composite.set_xlabel('x', fontsize=11, fontweight='bold')
        ax_deriv_composite.set_ylabel("h'(x)", fontsize=11, fontweight='bold')
        ax_deriv_composite.set_title(f"Chain Rule: h'(x)", fontsize=13, fontweight='bold')
        ax_deriv_composite.legend(fontsize=10)
        ax_deriv_composite.grid(True, alpha=0.3)
        ax_deriv_composite.set_ylim([-3, 3])
        
        # Chain visualization
        ax_chain_viz.clear()
        ax_chain_viz.set_xlim(0, 10)
        ax_chain_viz.set_ylim(0, 10)
        ax_chain_viz.axis('off')
        
        # Draw boxes and arrows
        box_props = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black', linewidth=2)
        arrow_props = dict(arrowstyle='->', lw=2.5, color='darkblue')
        
        ax_chain_viz.text(1, 7, 'INPUT\nx', ha='center', va='center', fontsize=12, 
                         bbox=box_props, fontweight='bold')
        ax_chain_viz.annotate('', xy=(2.5, 7), xytext=(1.8, 7), arrowprops=arrow_props)
        
        ax_chain_viz.text(3.5, 7, f'INNER\ng(x)\n{name_inner}', ha='center', va='center', fontsize=11, 
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='black', linewidth=2),
                         fontweight='bold')
        ax_chain_viz.annotate('', xy=(5.2, 7), xytext=(4.3, 7), arrowprops=arrow_props)
        
        ax_chain_viz.text(6.5, 7, f'OUTER\nf(g(x))\n{name_outer}', ha='center', va='center', fontsize=11, 
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='black', linewidth=2),
                         fontweight='bold')
        ax_chain_viz.annotate('', xy=(8, 7), xytext=(7.3, 7), arrowprops=arrow_props)
        
        ax_chain_viz.text(9, 7, 'OUTPUT\nh(x)', ha='center', va='center', fontsize=12, 
                         bbox=box_props, fontweight='bold')
        
        # Backward pass
        back_arrow = dict(arrowstyle='<-', lw=2.5, color='darkred')
        ax_chain_viz.text(9, 3, f"∂h/∂h\n= 1", ha='center', va='center', fontsize=10, 
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', edgecolor='red', linewidth=2))
        ax_chain_viz.annotate('', xy=(7.3, 3), xytext=(8, 3), arrowprops=back_arrow)
        
        ax_chain_viz.text(6.5, 3, f"∂h/∂g\n= {f_prime_eval:.3f}", ha='center', va='center', fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', edgecolor='red', linewidth=2))
        ax_chain_viz.annotate('', xy=(4.3, 3), xytext=(5.2, 3), arrowprops=back_arrow)
        
        ax_chain_viz.text(3.5, 3, f"∂h/∂x\n= {h_prime_eval:.3f}", ha='center', va='center', fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', edgecolor='red', linewidth=2))
        
        ax_chain_viz.text(5, 5.3, 'FORWARD PASS', ha='center', fontsize=13, fontweight='bold', color='darkblue')
        ax_chain_viz.text(5, 1.5, 'BACKWARD PASS (CHAIN RULE)', ha='center', fontsize=13, fontweight='bold', color='darkred')
        
        # Info panel
        ax_info.clear()
        ax_info.axis('off')
        info_text = f"""
CHAIN RULE COMPUTATION AT x = {eval_point:.2f}

Forward Pass:
  g({eval_point:.2f}) = {g_eval:.4f}
  f(g({eval_point:.2f})) = f({g_eval:.4f}) = {h_eval:.4f}

Backward Pass (Chain Rule):
  g'({eval_point:.2f}) = {g_prime_eval:.4f}
  f'(g({eval_point:.2f})) = f'({g_eval:.4f}) = {f_prime_eval:.4f}
  h'({eval_point:.2f}) = f'(g({eval_point:.2f})) × g'({eval_point:.2f}) = {f_prime_eval:.4f} × {g_prime_eval:.4f} = {h_prime_eval:.4f}

This is exactly how backpropagation works in neural networks!
Use the slider below to change the evaluation point.
        """
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=11, 
                    family='monospace', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=1))
        
    except Exception as e:
        ax_info.clear()
        ax_info.axis('off')
        ax_info.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', 
                    fontsize=12, color='red', transform=ax_info.transAxes)
    
    plt.draw()

# Create slider
ax_slider = plt.axes([0.2, 0.02, 0.6, 0.02])
slider = Slider(ax_slider, 'x value', -4, 4, valinit=eval_point, valstep=0.1)

def update_slider(val):
    global eval_point
    eval_point = val
    update_plots()

slider.on_changed(update_slider)

# Function selection buttons
ax_outer_radio = plt.axes([0.02, 0.6, 0.08, 0.15])
radio_outer = RadioButtons(ax_outer_radio, list(functions.keys()), active=3)

ax_inner_radio = plt.axes([0.02, 0.4, 0.08, 0.15])
radio_inner = RadioButtons(ax_inner_radio, list(functions.keys()), active=0)

def update_outer(label):
    global current_outer
    current_outer = label
    update_plots()

def update_inner(label):
    global current_inner
    current_inner = label
    update_plots()

radio_outer.on_clicked(update_outer)
radio_inner.on_clicked(update_inner)

plt.figtext(0.02, 0.77, 'Outer Function:', fontsize=10, fontweight='bold')
plt.figtext(0.02, 0.57, 'Inner Function:', fontsize=10, fontweight='bold')

# Initial plot
update_plots()
plt.show()