
#Sine Wave Plotter
import matplotlib.pyplot as plt
import numpy as np

# Generate x values
x = np.linspace(0, 4 * np.pi, 1000)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Trigonometric Functions Visualization', fontsize=16, fontweight='bold')

# 1. Basic Sine Wave
axes[0, 0].plot(x, np.sin(x), 'b-', linewidth=2, label='sin(x)')
axes[0, 0].set_title('Basic Sine Wave', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('x (radians)')
axes[0, 0].set_ylabel('y')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)

# 2. Multiple Frequencies
axes[0, 1].plot(x, np.sin(x), 'b-', linewidth=2, label='sin(x)')
axes[0, 1].plot(x, np.sin(2*x), 'r-', linewidth=2, label='sin(2x)')
axes[0, 1].plot(x, np.sin(3*x), 'g-', linewidth=2, label='sin(3x)')
axes[0, 1].set_title('Multiple Frequencies', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('x (radians)')
axes[0, 1].set_ylabel('y')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)

# 3. Phase Shifts
axes[1, 0].plot(x, np.sin(x), 'b-', linewidth=2, label='sin(x)')
axes[1, 0].plot(x, np.sin(x + np.pi/4), 'r--', linewidth=2, label='sin(x + π/4)')
axes[1, 0].plot(x, np.sin(x + np.pi/2), 'g-.', linewidth=2, label='sin(x + π/2)')
axes[1, 0].set_title('Phase Shifts', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('x (radians)')
axes[1, 0].set_ylabel('y')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)

# 4. Amplitude Variations
axes[1, 1].plot(x, np.sin(x), 'b-', linewidth=2, label='sin(x)')
axes[1, 1].plot(x, 2*np.sin(x), 'r-', linewidth=2, label='2·sin(x)')
axes[1, 1].plot(x, 0.5*np.sin(x), 'g-', linewidth=2, label='0.5·sin(x)')
axes[1, 1].set_title('Amplitude Variations', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('x (radians)')
axes[1, 1].set_ylabel('y')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
axes[1, 1].axhline(y=0, color='k', linewidth=0.5)

# Add annotations to first plot
axes[0, 0].annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2, 1.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold')

axes[0, 0].annotate('Trough', xy=(3*np.pi/2, -1), xytext=(3*np.pi/2, -1.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold')

plt.tight_layout()

# Save the figure
plt.savefig('sine_wave_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'sine_wave_analysis.png'")

plt.show()