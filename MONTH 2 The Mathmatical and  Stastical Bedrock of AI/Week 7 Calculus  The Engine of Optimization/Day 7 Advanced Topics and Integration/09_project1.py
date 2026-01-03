
#Project 1: Local vs. Global Minima (2 hours)
#Create a function with multiple minima: f(x) = sin(x) + sin(1.5x) + 0.1x

#Run gradient descent from different starting points
#Show how algorithm gets stuck in local minima
#Create visualization showing:

#Function with multiple minima
#Different convergence paths
#Final solutions from different initializations

#Discuss implications for neural networks

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# FUNCTION DEFINITION


def f(x):
    """Multimodal function with multiple local minima"""
    return np.sin(x) + np.sin(1.5 * x) + 0.1 * x

def f_prime(x):
    """Derivative of f(x)"""
    return np.cos(x) + 1.5 * np.cos(1.5 * x) + 0.1

# ============================================
# GRADIENT DESCENT IMPLEMENTATION
# ============================================

def gradient_descent(start_x, learning_rate=0.1, max_iterations=100, tolerance=1e-6):
    """
    Perform gradient descent optimization
    
    Parameters:
    -----------
    start_x : float
        Starting point
    learning_rate : float
        Step size for updates
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence threshold
    
    Returns:
    --------
    dict : Dictionary containing path, values, and convergence info
    """
    x = start_x
    path = [x]
    values = [f(x)]
    gradients = [f_prime(x)]
    
    for i in range(max_iterations):
        grad = f_prime(x)
        
        # Check convergence
        if abs(grad) < tolerance:
            break
        
        # Update step
        x = x - learning_rate * grad
        
        # Store history
        path.append(x)
        values.append(f(x))
        gradients.append(grad)
    
    return {
        'path': np.array(path),
        'values': np.array(values),
        'gradients': np.array(gradients),
        'final_x': x,
        'final_value': f(x),
        'iterations': len(path),
        'converged': abs(grad) < tolerance
    }

# ============================================
# FIND TRUE MINIMA
# ============================================

def find_all_minima(x_range=(-10, 20), num_points=10000):
    """Find all local minima in the range"""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = f(x)
    grad = f_prime(x)
    
    # Find where gradient changes sign (critical points)
    minima = []
    maxima = []
    
    for i in range(1, len(x)-1):
        if grad[i-1] < 0 and grad[i+1] > 0:  # Minimum
            minima.append((x[i], y[i]))
        elif grad[i-1] > 0 and grad[i+1] < 0:  # Maximum
            maxima.append((x[i], y[i]))
    
    return minima, maxima


# VISUALIZATION 1: COMPREHENSIVE OVERVIEW


def create_comprehensive_visualization(starting_points, learning_rate=0.1):
    """Create main visualization showing all concepts"""
    
    # Run gradient descent from different starting points
    results = []
    for start in starting_points:
        result = gradient_descent(start, learning_rate=learning_rate)
        result['start'] = start
        results.append(result)
    
    # Find true minima
    minima, maxima = find_all_minima()
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Plot 1: Main function with all paths
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot function
    x_plot = np.linspace(-10, 20, 1000)
    y_plot = f(x_plot)
    ax1.plot(x_plot, y_plot, 'b-', linewidth=2.5, label='f(x) = sin(x) + sin(1.5x) + 0.1x', zorder=1)
    
    # Plot true minima and maxima
    if minima:
        minima_x, minima_y = zip(*minima)
        ax1.scatter(minima_x, minima_y, color='green', s=200, marker='v', 
                   edgecolor='black', linewidth=2, label='Local Minima', zorder=5)
        # Mark global minimum
        global_min_idx = np.argmin(minima_y)
        ax1.scatter(minima_x[global_min_idx], minima_y[global_min_idx], 
                   color='red', s=300, marker='*', edgecolor='black', 
                   linewidth=2, label='Global Minimum', zorder=6)
    
    if maxima:
        maxima_x, maxima_y = zip(*maxima)
        ax1.scatter(maxima_x, maxima_y, color='orange', s=150, marker='^', 
                   edgecolor='black', linewidth=2, label='Local Maxima', zorder=5, alpha=0.6)
    
    # Plot gradient descent paths
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for result, color in zip(results, colors):
        path = result['path']
        values = result['values']
        
        # Plot path
        ax1.plot(path, values, 'o-', color=color, linewidth=2, 
                markersize=6, alpha=0.7, zorder=3)
        
        # Mark start and end
        ax1.plot(path[0], values[0], 'o', color=color, markersize=15, 
                markeredgecolor='black', markeredgewidth=2, zorder=4)
        ax1.plot(path[-1], values[-1], 's', color=color, markersize=12, 
                markeredgecolor='black', markeredgewidth=2, zorder=4)
    
    ax1.set_xlabel('x', fontsize=14, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=14, fontweight='bold')
    ax1.set_title('Gradient Descent: Multiple Starting Points → Different Local Minima', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-10, 20)
    
    # Plot 2: Convergence curves
    ax2 = fig.add_subplot(gs[1, 0])
    for result, color in zip(results, colors):
        iterations = np.arange(len(result['values']))
        ax2.plot(iterations, result['values'], 'o-', color=color, 
                linewidth=2, markersize=4, alpha=0.7,
                label=f"Start: {result['start']:.1f}")
    
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('f(x)', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence History', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient magnitude
    ax3 = fig.add_subplot(gs[1, 1])
    for result, color in zip(results, colors):
        iterations = np.arange(len(result['gradients']))
        ax3.semilogy(iterations, np.abs(result['gradients']), 'o-', 
                    color=color, linewidth=2, markersize=4, alpha=0.7)
    
    ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('|Gradient| (log scale)', fontsize=12, fontweight='bold')
    ax3.set_title('Gradient Magnitude Decay', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1e-6, color='r', linestyle='--', linewidth=2, label='Tolerance')
    ax3.legend(fontsize=9)
    
    # Plot 4: Final positions distribution
    ax4 = fig.add_subplot(gs[1, 2])
    final_positions = [r['final_x'] for r in results]
    final_values = [r['final_value'] for r in results]
    
    scatter = ax4.scatter(final_positions, final_values, c=colors, 
                         s=200, edgecolor='black', linewidth=2, zorder=3)
    
    # Add true minima for reference
    if minima:
        ax4.scatter(minima_x, minima_y, color='green', s=150, marker='v', 
                   edgecolor='black', linewidth=2, alpha=0.5, zorder=2)
    
    ax4.set_xlabel('Final x', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Final f(x)', fontsize=12, fontweight='bold')
    ax4.set_title('Final Convergence Points', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Summary table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create DataFrame for results
    summary_data = []
    for i, result in enumerate(results):
        summary_data.append({
            'Start Point': f"{result['start']:.2f}",
            'Final Point': f"{result['final_x']:.4f}",
            'Final Value': f"{result['final_value']:.6f}",
            'Iterations': result['iterations'],
            'Converged': '✓' if result['converged'] else '✗',
            'Type': 'Global Min' if result['final_value'] == min(r['final_value'] for r in results) else 'Local Min'
        })
    
    df = pd.DataFrame(summary_data)
    
    # Create table
    table_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                          GRADIENT DESCENT RESULTS SUMMARY                                        ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Learning Rate: {learning_rate}  |  Tolerance: 1e-6  |  Max Iterations: 100                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

{df.to_string(index=False)}

KEY OBSERVATIONS:
─────────────────────────────────────────────────────────────────────────────────────────────────────
• Different starting points lead to DIFFERENT local minima
• Algorithm converged to {len(set(round(r['final_x'], 2) for r in results))} distinct minima
• Global minimum found from starting points: {', '.join([f"{r['start']:.1f}" for r in results if r['final_value'] == min(res['final_value'] for res in results)])}
• Some starting points got STUCK in suboptimal local minima
• Gradient magnitude decreased rapidly near convergence points
    """
    
    ax5.text(0.05, 0.95, table_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))
    
    plt.savefig('local_vs_global_minima_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, df

# VISUALIZATION 2: ANIMATED GRADIENT DESCENT

def create_animated_descent(starting_points, learning_rate=0.1, save_animation=False):
    """Create animated visualization of gradient descent"""
    
    print("Creating animated visualization...")
    
    # Run gradient descent
    results = []
    max_iterations = 0
    for start in starting_points:
        result = gradient_descent(start, learning_rate=learning_rate)
        results.append(result)
        max_iterations = max(max_iterations, result['iterations'])
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot function
    x_plot = np.linspace(-10, 20, 1000)
    y_plot = f(x_plot)
    ax.plot(x_plot, y_plot, 'b-', linewidth=3, label='f(x)', zorder=1)
    
    # Find and plot minima
    minima, maxima = find_all_minima()
    if minima:
        minima_x, minima_y = zip(*minima)
        ax.scatter(minima_x, minima_y, color='green', s=200, marker='v', 
                  edgecolor='black', linewidth=2, label='Local Minima', zorder=5)
        global_min_idx = np.argmin(minima_y)
        ax.scatter(minima_x[global_min_idx], minima_y[global_min_idx], 
                  color='red', s=300, marker='*', edgecolor='black', 
                  linewidth=2, label='Global Minimum', zorder=6)
    
    # Initialize scatter plots for moving points
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    scatters = []
    paths = []
    
    for color in colors:
        scatter = ax.scatter([], [], s=200, c=[color], edgecolor='black', 
                           linewidth=2, zorder=4)
        scatters.append(scatter)
        path, = ax.plot([], [], 'o-', color=color, linewidth=2, 
                       markersize=6, alpha=0.5, zorder=3)
        paths.append(path)
    
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=14, fontweight='bold')
    ax.set_title('Animated Gradient Descent: Watch Points Converge to Different Minima', 
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 20)
    ax.set_ylim(min(y_plot)-0.5, max(y_plot)+0.5)
    
    # Animation function
    def animate(frame):
        for i, (result, scatter, path) in enumerate(zip(results, scatters, paths)):
            if frame < len(result['path']):
                # Update current position
                scatter.set_offsets([[result['path'][frame], result['values'][frame]]])
                # Update path
                path.set_data(result['path'][:frame+1], result['values'][:frame+1])
            else:
                # Keep at final position
                scatter.set_offsets([[result['path'][-1], result['values'][-1]]])
                path.set_data(result['path'], result['values'])
        
        return scatters + paths
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=max_iterations, 
                        interval=200, blit=True, repeat=True)
    
    if save_animation:
        print("Saving animation (this may take a while)...")
        anim.save('gradient_descent_animation.gif', writer='pillow', fps=5)
        print("Animation saved as 'gradient_descent_animation.gif'")
    
    plt.show()
    
    return anim


# VISUALIZATION 3: LEARNING RATE COMPARISON


def compare_learning_rates(start_x=-5):
    """Compare different learning rates"""
    
    learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot function reference
    x_plot = np.linspace(-10, 20, 1000)
    y_plot = f(x_plot)
    
    results_summary = []
    
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        
        # Run gradient descent
        result = gradient_descent(start_x, learning_rate=lr)
        
        results_summary.append({
            'Learning Rate': lr,
            'Final x': f"{result['final_x']:.4f}",
            'Final f(x)': f"{result['final_value']:.6f}",
            'Iterations': result['iterations'],
            'Converged': result['converged']
        })
        
        # Plot function
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.3)
        
        # Plot minima
        minima, _ = find_all_minima()
        if minima:
            minima_x, minima_y = zip(*minima)
            ax.scatter(minima_x, minima_y, color='green', s=100, marker='v', 
                      edgecolor='black', linewidth=1.5, alpha=0.5)
        
        # Plot path
        ax.plot(result['path'], result['values'], 'ro-', linewidth=2, 
               markersize=6, alpha=0.7)
        ax.plot(result['path'][0], result['values'][0], 'go', markersize=12, 
               markeredgecolor='black', markeredgewidth=2, label='Start')
        ax.plot(result['path'][-1], result['values'][-1], 'rs', markersize=12, 
               markeredgecolor='black', markeredgewidth=2, label='End')
        
        ax.set_title(f'LR = {lr} ({result["iterations"]} iterations)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('f(x)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 20)
    
    plt.suptitle(f'Learning Rate Comparison (Start: x={start_x})', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    df_lr = pd.DataFrame(results_summary)
    print("\n" + "="*80)
    print("LEARNING RATE COMPARISON SUMMARY")
    print("="*80)
    print(df_lr.to_string(index=False))
    print("="*80)


# VISUALIZATION 4: LANDSCAPE EXPLORATION


def explore_landscape():
    """Detailed exploration of the optimization landscape"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    x_plot = np.linspace(-10, 20, 1000)
    y_plot = f(x_plot)
    grad_plot = f_prime(x_plot)
    
    # Plot 1: Function
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x_plot, y_plot, 'b-', linewidth=3)
    minima, maxima = find_all_minima()
    
    if minima:
        minima_x, minima_y = zip(*minima)
        ax1.scatter(minima_x, minima_y, color='green', s=200, marker='v', 
                   edgecolor='black', linewidth=2, label='Local Minima', zorder=5)
        global_min_idx = np.argmin(minima_y)
        ax1.scatter(minima_x[global_min_idx], minima_y[global_min_idx], 
                   color='red', s=300, marker='*', edgecolor='black', 
                   linewidth=2, label='Global Minimum', zorder=6)
    
    if maxima:
        maxima_x, maxima_y = zip(*maxima)
        ax1.scatter(maxima_x, maxima_y, color='orange', s=150, marker='^', 
                   edgecolor='black', linewidth=2, label='Local Maxima', alpha=0.6)
    
    ax1.set_xlabel('x', fontsize=14, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=14, fontweight='bold')
    ax1.set_title('Optimization Landscape: f(x) = sin(x) + sin(1.5x) + 0.1x', 
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gradient
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x_plot, grad_plot, 'r-', linewidth=2.5)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.fill_between(x_plot, 0, grad_plot, where=(grad_plot>0), 
                     color='red', alpha=0.2, label='Gradient > 0')
    ax2.fill_between(x_plot, 0, grad_plot, where=(grad_plot<0), 
                     color='blue', alpha=0.2, label='Gradient < 0')
    ax2.set_xlabel('x', fontsize=12, fontweight='bold')
    ax2.set_ylabel("f'(x)", fontsize=12, fontweight='bold')
    ax2.set_title('Gradient (Derivative)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Curvature (second derivative approximation)
    ax3 = fig.add_subplot(gs[1, 1])
    second_deriv = np.gradient(grad_plot, x_plot)
    ax3.plot(x_plot, second_deriv, 'g-', linewidth=2.5)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.fill_between(x_plot, 0, second_deriv, where=(second_deriv>0), 
                     color='green', alpha=0.2, label='Convex (f" > 0)')
    ax3.fill_between(x_plot, 0, second_deriv, where=(second_deriv<0), 
                     color='purple', alpha=0.2, label='Concave (f" < 0)')
    ax3.set_xlabel('x', fontsize=12, fontweight='bold')
    ax3.set_ylabel('f"(x)', fontsize=12, fontweight='bold')
    ax3.set_title('Curvature (Second Derivative)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Minima analysis
    ax4 = fig.add_subplot(gs[2, :])
    
    if minima:
        minima_df = pd.DataFrame(minima, columns=['x', 'f(x)'])
        minima_df['Type'] = 'Local'
        minima_df.loc[minima_df['f(x)'].idxmin(), 'Type'] = 'Global'
        
        colors_map = {'Local': 'green', 'Global': 'red'}
        bars = ax4.bar(range(len(minima_df)), minima_df['f(x)'], 
                      color=[colors_map[t] for t in minima_df['Type']],
                      edgecolor='black', linewidth=2)
        
        ax4.set_xticks(range(len(minima_df)))
        ax4.set_xticklabels([f"x={x:.2f}" for x in minima_df['x']], fontsize=10)
        ax4.set_ylabel('f(x)', fontsize=12, fontweight='bold')
        ax4.set_title('All Local Minima Values', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, minima_df['f(x)'])):
            ax4.text(bar.get_x() + bar.get_width()/2, val, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', edgecolor='black', label='Global Minimum'),
                          Patch(facecolor='green', edgecolor='black', label='Local Minimum')]
        ax4.legend(handles=legend_elements, fontsize=11)
    
    plt.savefig('optimization_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()


# NEURAL NETWORKS IMPLICATIONS


def discuss_neural_network_implications():
    """Print discussion about implications for neural networks"""
    
    print("\n" + "="*80)
    print(" " * 20 + "IMPLICATIONS FOR NEURAL NETWORKS")
    print("="*80)
    
    discussion = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    WHY LOCAL MINIMA MATTER IN DEEP LEARNING                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

1. THE PROBLEM:
   ─────────────────────────────────────────────────────────────────────────────
   • Neural networks have MILLIONS of parameters (weights and biases)
   • Loss landscape is extremely high-dimensional and non-convex
   • Many local minima exist - some good, some bad
   • Random initialization determines which minimum we'll find
   
2. WHAT WE LEARNED FROM THIS PROJECT:
   ─────────────────────────────────────────────────────────────────────────────
   ✓ Different starting points → Different final solutions
   ✓ Gradient descent follows the local slope and can get "stuck"
   ✓ No guarantee of finding the global optimum
   ✓ The quality of solution depends heavily on initialization
   
3. REAL-WORLD NEURAL NETWORK STRATEGIES:
   ─────────────────────────────────────────────────────────────────────────────
   
   a) RANDOM RESTARTS:
      • Train multiple models with different random initializations
      • Pick the best performing one
      • Ensemble predictions from all models
   
   b) SMART INITIALIZATION:
      • Xavier/Glorot initialization: weights ~ N(0, 1/n)
      • He initialization: weights ~ N(0, 2/n)
      • Proper initialization helps start in "better" regions
   
   c) ADVANCED OPTIMIZERS:
      • Momentum: Builds velocity to escape shallow minima
      • Adam: Adaptive learning rates per parameter
      • RMSprop: Helps navigate ravines in loss landscape
      • These help escape poor local minima
   
   d) LEARNING RATE SCHEDULING:
      • Start with larger LR (broad exploration)
      • Gradually decrease (fine-tuning)
      • Helps find better minima
   
   e) ARCHITECTURE TRICKS:
      • Batch normalization: Smooths loss landscape
      • Skip connections: Creates multiple gradient paths
      • Dropout: Prevents overfitting to specific minima
   
4. SURPRISING INSIGHT:
   ─────────────────────────────────────────────────────────────────────────────
   • Recent research shows most local minima in deep networks are "good enough"
   • High-dimensional spaces have interesting properties
   • Saddle points are actually more problematic than local minima!
   • Overparameterization helps - many paths to good solutions
   
5. PRACTICAL TAKEAWAYS:
   ─────────────────────────────────────────────────────────────────────────────
   ✓ Always try multiple random seeds
   ✓ Use proven initialization schemes
   ✓ Modern optimizers (Adam, AdamW) are your friends
   ✓ Don't expect global optimum - "good enough" is often sufficient
   ✓ Monitor training: if stuck, consider learning rate changes or restarts
   ✓ Early stopping prevents overfitting to poor minima

╔══════════════════════════════════════════════════════════════════════════════════╗
║  Bottom Line: Local minima are a challenge, but modern techniques handle them    ║
║  well enough for practical deep learning. The key is smart experimentation!     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
    """
    
    print(discussion)
    print("="*80 + "\n")


# MAIN EXECUTION


def main():
    """Run complete project"""
    
    print("="*80)
    print(" " * 25 + "PROJECT 1: LOCAL VS. GLOBAL MINIMA")
    print(" " * 20 + "Gradient Descent on Multimodal Functions")
    print("="*80)
    
    # Define starting points
    starting_points = [-8, -5, -2, 1, 5, 8, 12, 15]
    
    print(f"\nTesting {len(starting_points)} different starting points:")
    print(f"Starting points: {starting_points}")