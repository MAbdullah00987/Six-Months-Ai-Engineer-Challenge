
#Neural Network Training Animation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns

print("Creating Neural Network Training Animation...")
print("=" * 80)

# Set style
sns.set_style("whitegrid")

# Simple Neural Network for animation
class AnimatedNN:
    def __init__(self):
        np.random.seed(42)
        self.w1 = np.random.randn(2, 4) * 0.1
        self.b1 = np.zeros((1, 4))
        self.w2 = np.random.randn(4, 1) * 0.1
        self.b2 = np.zeros((1, 1))
        self.lr = 0.5
        self.history = []
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def train_step(self, X, y):
        m = X.shape[0]
        output = self.forward(X)
        
        # Backpropagation
        dz2 = output - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * self.a1 * (1 - self.a1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        
        loss = np.mean((output - y) ** 2)
        return loss
    
    def get_decision_boundary(self, resolution=100):
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        Z = self.forward(np.c_[xx.ravel(), yy.ravel()])
        return xx, yy, Z.reshape(xx.shape)

# Create dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize network
nn = AnimatedNN()

# Training data collection
print("Training network and collecting frames...")
epochs = 200
frames_data = []

for epoch in range(epochs):
    loss = nn.train_step(X, y)
    
    # Collect data every 5 epochs
    if epoch % 5 == 0 or epoch < 20:
        xx, yy, Z = nn.get_decision_boundary(50)
        frames_data.append({
            'epoch': epoch,
            'loss': loss,
            'xx': xx,
            'yy': yy,
            'Z': Z,
            'w1': nn.w1.copy(),
            'w2': nn.w2.copy()
        })
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

print(f"\nCollected {len(frames_data)} frames")

# Create animation
print("\nCreating animation...")
fig = plt.figure(figsize=(16, 6))

# Decision boundary subplot
ax1 = plt.subplot(1, 3, 1)
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlabel('Input 1', fontsize=12)
ax1.set_ylabel('Input 2', fontsize=12)
ax1.set_title('Decision Boundary Evolution', fontsize=14, fontweight='bold')

# Loss curve subplot
ax2 = plt.subplot(1, 3, 2)
ax2.set_xlim(0, epochs)
ax2.set_ylim(0, 0.3)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Weight heatmap subplot
ax3 = plt.subplot(1, 3, 3)
ax3.set_title('Weight Magnitudes', fontsize=14, fontweight='bold')

# Initialize plots
contour = ax1.contourf(frames_data[0]['xx'], frames_data[0]['yy'], 
                       frames_data[0]['Z'], levels=20, cmap='RdYlBu', alpha=0.8)
scatter = ax1.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=200, 
                     cmap='RdYlBu', edgecolor='black', linewidth=2, zorder=5)

loss_line, = ax2.plot([], [], 'r-', linewidth=2)
loss_point = ax2.scatter([], [], c='red', s=100, zorder=5)

# Text annotations
epoch_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def init():
    loss_line.set_data([], [])
    loss_point.set_offsets(np.empty((0, 2)))
    return loss_line, loss_point, epoch_text

def animate(frame_idx):
    frame = frames_data[frame_idx]
    
    # Update decision boundary
    ax1.clear()
    ax1.contourf(frame['xx'], frame['yy'], frame['Z'], 
                levels=20, cmap='RdYlBu', alpha=0.8)
    ax1.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=200, 
               cmap='RdYlBu', edgecolor='black', linewidth=2, zorder=5)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_xlabel('Input 1', fontsize=12)
    ax1.set_ylabel('Input 2', fontsize=12)
    ax1.set_title('Decision Boundary Evolution', fontsize=14, fontweight='bold')
    ax1.text(0.02, 0.98, f"Epoch: {frame['epoch']}", 
            transform=ax1.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Update loss curve
    epochs_so_far = [f['epoch'] for f in frames_data[:frame_idx+1]]
    losses_so_far = [f['loss'] for f in frames_data[:frame_idx+1]]
    loss_line.set_data(epochs_so_far, losses_so_far)
    loss_point.set_offsets([[frame['epoch'], frame['loss']]])
    
    # Update weight heatmap
    ax3.clear()
    all_weights = np.concatenate([frame['w1'].flatten(), frame['w2'].flatten()])
    weight_matrix = all_weights.reshape(-1, 1)
    sns.heatmap(weight_matrix, ax=ax3, cmap='coolwarm', center=0,
                cbar_kws={'label': 'Weight Value'}, vmin=-2, vmax=2)
    ax3.set_title(f'Weight Magnitudes (Epoch {frame["epoch"]})', 
                 fontsize=14, fontweight='bold')
    ax3.set_ylabel('Weight Index')
    ax3.set_xlabel('')
    
    return loss_line, loss_point

# Create animation
print("Rendering animation frames...")
anim = FuncAnimation(fig, animate, init_func=init, frames=len(frames_data),
                    interval=100, blit=False, repeat=True)

plt.tight_layout()

# Save as GIF
print("Saving animation as GIF...")
writer = PillowWriter(fps=10)
anim.save('neural_network_training.gif', writer=writer, dpi=100)
print("✓ Saved: neural_network_training.gif")

# Also show the figure
plt.show()

# Create a summary plot showing key frames
print("\nCreating key frames summary...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Neural Network Training: Key Frames', fontsize=16, fontweight='bold')

key_frames = [0, len(frames_data)//5, len(frames_data)//2, 
              3*len(frames_data)//4, len(frames_data)-1]

for idx, frame_idx in enumerate(key_frames[:6]):
    row, col = idx // 3, idx % 3
    frame = frames_data[frame_idx]
    
    axes[row, col].contourf(frame['xx'], frame['yy'], frame['Z'], 
                           levels=20, cmap='RdYlBu', alpha=0.8)
    axes[row, col].scatter(X[:, 0], X[:, 1], c=y.ravel(), s=150, 
                          cmap='RdYlBu', edgecolor='black', linewidth=2)
    axes[row, col].set_title(f"Epoch {frame['epoch']} | Loss: {frame['loss']:.4f}",
                            fontweight='bold')
    axes[row, col].set_xlabel('Input 1')
    axes[row, col].set_ylabel('Input 2')

plt.tight_layout()
plt.savefig('training_key_frames.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_key_frames.png")
plt.show()

print("ANIMATION COMPLETE!")
print("\nGenerated files:")
print("  1. neural_network_training.gif - Animated training visualization")
print("  2. training_key_frames.png - Key training moments")
print("\nThe animation shows:")
print("  • Decision boundary evolution (left)")
print("  • Loss reduction over time (middle)")
print("  • Weight updates (right)")
