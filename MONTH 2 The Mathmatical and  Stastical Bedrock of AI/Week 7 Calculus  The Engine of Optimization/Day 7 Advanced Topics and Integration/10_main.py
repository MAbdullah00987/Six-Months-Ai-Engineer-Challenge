

#Project 2: Visualizing Optimization (2 hours)
#Create an animated visualization of gradient descent

#Use 2D function: f(x,y) = (x-2)² + (y+1)²
#Generate animation showing:

#Contour plot of the function
#Path of gradient descent ball rolling downhill
#Current gradient vector at each step
#Iteration counter and loss value
#Save as GIF or MP4

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

# Define the function f(x,y) = (x-2)² + (y+1)²
def f(x, y):
    return (x - 2)**2 + (y + 1)**2

# Gradient of the function
def gradient(x, y):
    df_dx = 2 * (x - 2)
    df_dy = 2 * (y + 1)
    return np.array([df_dx, df_dy])

# Gradient descent algorithm
def gradient_descent(start_point, learning_rate=0.1, num_iterations=50):
    path = [start_point]
    losses = [f(start_point[0], start_point[1])]
    gradients = []
    
    current = start_point.copy()
    
    for i in range(num_iterations):
        grad = gradient(current[0], current[1])
        gradients.append(grad)
        current = current - learning_rate * grad
        path.append(current.copy())
        losses.append(f(current[0], current[1]))
    
    return np.array(path), losses, gradients

# Generate data for contour plot
x = np.linspace(-1, 5, 200)
y = np.linspace(-4, 2, 200)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Starting point for gradient descent
start = np.array([4.5, 1.5])
path, losses, gradients = gradient_descent(start, learning_rate=0.15, num_iterations=50)

# Create figure and subplots
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Plot contour on first subplot
contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
ax1.clabel(contour, inline=True, fontsize=8)

# Mark the minimum point
ax1.plot(2, -1, 'r*', markersize=20, label='Global Minimum (2, -1)')

# Initialize the ball (circle)
ball = Circle((path[0, 0], path[0, 1]), 0.15, color='red', zorder=5)
ax1.add_patch(ball)

# Initialize the path line
line, = ax1.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Descent Path')

# Initialize the gradient arrow
arrow = ax1.quiver([], [], [], [], color='blue', scale=10, width=0.005, 
                   label='Gradient Vector', zorder=4)

# Labels and title for contour plot
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Gradient Descent Visualization', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Setup loss plot
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Loss Value', fontsize=12)
ax2.set_title('Loss vs Iteration', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
loss_line, = ax2.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)

# Text annotations
iteration_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

position_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Animation function
def animate(frame):
    # Update ball position
    ball.center = (path[frame, 0], path[frame, 1])
    
    # Update path
    line.set_data(path[:frame+1, 0], path[:frame+1, 1])
    
    # Update gradient arrow
    if frame < len(gradients):
        arrow.set_offsets([path[frame, 0], path[frame, 1]])
        arrow.set_UVC(-gradients[frame][0], -gradients[frame][1])
    
    # Update loss plot
    loss_line.set_data(range(frame+1), losses[:frame+1])
    ax2.set_xlim(0, len(losses))
    ax2.set_ylim(0, max(losses) * 1.1)
    
    # Update text
    iteration_text.set_text(f'Iteration: {frame}\nLoss: {losses[frame]:.4f}')
    position_text.set_text(f'Position: ({path[frame, 0]:.3f}, {path[frame, 1]:.3f})')
    
    return ball, line, arrow, loss_line, iteration_text, position_text

# Create animation
anim = FuncAnimation(fig, animate, frames=len(path), 
                    interval=200, blit=True, repeat=True)

# ============================================
# SAVE ANIMATION OPTIONS
# ============================================
# Uncomment ONE of the following options to save the animation:

# Option 1: Save as GIF (recommended - works everywhere)
print("Saving animation as GIF...")
try:
    writer_gif = PillowWriter(fps=5)
    anim.save('gradient_descent.gif', writer=writer_gif)
    print("✓ Animation saved as 'gradient_descent.gif'")
except Exception as e:
    print(f"✗ Error saving GIF: {e}")

# Option 2: Save as MP4 (requires FFmpeg installed)
# Uncomment the lines below to save as MP4:
"""
print("Saving animation as MP4...")
try:
    writer_mp4 = FFMpegWriter(fps=5, bitrate=1800)
    anim.save('gradient_descent.mp4', writer=writer_mp4)
    print("✓ Animation saved as 'gradient_descent.mp4'")
except Exception as e:
    print(f"✗ Error saving MP4: {e}")
    print("  Note: MP4 requires FFmpeg to be installed on your system")
"""

# Option 3: Save BOTH formats
# Uncomment the lines below to save both GIF and MP4:
"""
print("Saving animations in both formats...")
# Save GIF
try:
    writer_gif = PillowWriter(fps=5)
    anim.save('gradient_descent.gif', writer=writer_gif)
    print("✓ GIF saved as 'gradient_descent.gif'")
except Exception as e:
    print(f"✗ Error saving GIF: {e}")

# Save MP4
try:
    writer_mp4 = FFMpegWriter(fps=5, bitrate=1800)
    anim.save('gradient_descent.mp4', writer=writer_mp4)
    print("✓ MP4 saved as 'gradient_descent.mp4'")
except Exception as e:
    print(f"✗ Error saving MP4: {e}")
    print("  Note: MP4 requires FFmpeg to be installed")
"""


plt.tight_layout()
plt.show()

# Print summary
print("\n" + "=" * 50)
print("GRADIENT DESCENT SUMMARY")
print("=" * 50)
print(f"Starting Point: ({start[0]:.2f}, {start[1]:.2f})")
print(f"Starting Loss: {losses[0]:.4f}")
print(f"Final Point: ({path[-1, 0]:.4f}, {path[-1, 1]:.4f})")
print(f"Final Loss: {losses[-1]:.6f}")
print(f"Global Minimum: (2.0, -1.0)")
print(f"Total Iterations: {len(path) - 1}")
print("=" * 50)
print("\nCheck your current directory for the saved animation file(s)!")
print(f"Current directory: {os.getcwd()}")