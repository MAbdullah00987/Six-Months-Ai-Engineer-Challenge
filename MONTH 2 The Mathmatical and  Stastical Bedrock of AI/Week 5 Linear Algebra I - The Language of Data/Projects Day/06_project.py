
#6: Linear Transformation Visualizer: Create a visualization that shows how a 2x2 matrix transforms the 2D plane.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd

class LinearTransformVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 7))
        
        # Create subplots for original and transformed spaces
        self.ax1 = plt.subplot(1, 2, 1)
        self.ax2 = plt.subplot(1, 2, 2)
        
        # Initial transformation matrix
        self.matrix = np.array([[1, 0], [0, 1]])
        
        # Create grid points
        self.create_grid()
        
        # Setup plots
        self.setup_plots()
        
        # Create sliders
        self.create_sliders()
        
        # Create preset buttons
        self.create_buttons()
        
        plt.subplots_adjust(left=0.1, bottom=0.35)
        
    def create_grid(self):
        """Create a grid of points and vectors"""
        # Grid lines
        self.grid_range = 5
        x = np.linspace(-self.grid_range, self.grid_range, 21)
        y = np.linspace(-self.grid_range, self.grid_range, 21)
        
        # Vertical lines
        self.vert_lines = []
        for xi in x:
            self.vert_lines.append(np.array([[xi, xi], [y[0], y[-1]]]))
        
        # Horizontal lines
        self.horiz_lines = []
        for yi in y:
            self.horiz_lines.append(np.array([[x[0], x[-1]], [yi, yi]]))
        
        # Unit vectors
        self.i_hat = np.array([[0, 1], [0, 0]])
        self.j_hat = np.array([[0, 0], [0, 1]])
        
        # Sample vectors
        self.sample_vectors = [
            np.array([[0, 2], [0, 1]]),
            np.array([[0, 1], [0, 2]]),
            np.array([[0, -1], [0, 1]])
        ]
        
    def setup_plots(self):
        """Setup the original and transformed plots"""
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(-self.grid_range, self.grid_range)
            ax.set_ylim(-self.grid_range, self.grid_range)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
        
        self.ax1.set_title('Original Space', fontsize=12, fontweight='bold')
        self.ax2.set_title('Transformed Space', fontsize=12, fontweight='bold')
        
        # Plot original space (static)
        self.plot_original()
        
        # Plot transformed space (will update)
        self.plot_transformed()
        
    def plot_original(self):
        """Plot the original coordinate system"""
        # Plot grid lines
        for line in self.vert_lines:
            self.ax1.plot(line[0], line[1], 'b-', alpha=0.3, linewidth=0.5)
        for line in self.horiz_lines:
            self.ax1.plot(line[0], line[1], 'b-', alpha=0.3, linewidth=0.5)
        
        # Plot basis vectors
        self.ax1.arrow(0, 0, 1, 0, head_width=0.2, head_length=0.2, 
                      fc='red', ec='red', linewidth=2, label='i-hat')
        self.ax1.arrow(0, 0, 0, 1, head_width=0.2, head_length=0.2, 
                      fc='green', ec='green', linewidth=2, label='j-hat')
        
        # Plot sample vectors
        colors = ['purple', 'orange', 'brown']
        for vec, color in zip(self.sample_vectors, colors):
            self.ax1.arrow(vec[0, 0], vec[1, 0], vec[0, 1]-vec[0, 0], 
                          vec[1, 1]-vec[1, 0], head_width=0.2, head_length=0.2,
                          fc=color, ec=color, linewidth=1.5, alpha=0.6)
        
        self.ax1.legend(loc='upper right')
        
    def plot_transformed(self):
        """Plot the transformed coordinate system"""
        self.ax2.clear()
        self.ax2.set_xlim(-self.grid_range, self.grid_range)
        self.ax2.set_ylim(-self.grid_range, self.grid_range)
        self.ax2.set_aspect('equal')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.axhline(y=0, color='k', linewidth=0.5)
        self.ax2.axvline(x=0, color='k', linewidth=0.5)
        self.ax2.set_title('Transformed Space', fontsize=12, fontweight='bold')
        
        # Transform and plot grid lines
        for line in self.vert_lines:
            transformed = self.matrix @ line
            self.ax2.plot(transformed[0], transformed[1], 'b-', alpha=0.3, linewidth=0.5)
        for line in self.horiz_lines:
            transformed = self.matrix @ line
            self.ax2.plot(transformed[0], transformed[1], 'b-', alpha=0.3, linewidth=0.5)
        
        # Transform and plot basis vectors
        i_transformed = self.matrix @ self.i_hat
        j_transformed = self.matrix @ self.j_hat
        
        self.ax2.arrow(0, 0, i_transformed[0, 1], i_transformed[1, 1],
                      head_width=0.2, head_length=0.2, fc='red', ec='red',
                      linewidth=2, label='i-hat transformed')
        self.ax2.arrow(0, 0, j_transformed[0, 1], j_transformed[1, 1],
                      head_width=0.2, head_length=0.2, fc='green', ec='green',
                      linewidth=2, label='j-hat transformed')
        
        # Transform and plot sample vectors
        colors = ['purple', 'orange', 'brown']
        for vec, color in zip(self.sample_vectors, colors):
            transformed = self.matrix @ vec
            self.ax2.arrow(transformed[0, 0], transformed[1, 0],
                          transformed[0, 1]-transformed[0, 0],
                          transformed[1, 1]-transformed[1, 0],
                          head_width=0.2, head_length=0.2, fc=color, ec=color,
                          linewidth=1.5, alpha=0.6)
        
        # Display matrix and determinant
        det = np.linalg.det(self.matrix)
        matrix_text = f'Matrix:\n[{self.matrix[0,0]:.2f}  {self.matrix[0,1]:.2f}]\n'
        matrix_text += f'[{self.matrix[1,0]:.2f}  {self.matrix[1,1]:.2f}]\n\n'
        matrix_text += f'Det: {det:.2f}'
        
        self.ax2.text(0.02, 0.98, matrix_text, transform=self.ax2.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax2.legend(loc='upper right')
        
    def create_sliders(self):
        """Create sliders for matrix elements"""
        slider_color = 'lightgoldenrodyellow'
        
        # Create axes for sliders
        ax_a = plt.axes([0.15, 0.25, 0.3, 0.03], facecolor=slider_color)
        ax_b = plt.axes([0.15, 0.20, 0.3, 0.03], facecolor=slider_color)
        ax_c = plt.axes([0.15, 0.15, 0.3, 0.03], facecolor=slider_color)
        ax_d = plt.axes([0.15, 0.10, 0.3, 0.03], facecolor=slider_color)
        
        # Create sliders
        self.slider_a = Slider(ax_a, 'a (top-left)', -3.0, 3.0, valinit=1.0)
        self.slider_b = Slider(ax_b, 'b (top-right)', -3.0, 3.0, valinit=0.0)
        self.slider_c = Slider(ax_c, 'c (bottom-left)', -3.0, 3.0, valinit=0.0)
        self.slider_d = Slider(ax_d, 'd (bottom-right)', -3.0, 3.0, valinit=1.0)
        
        # Connect update function
        self.slider_a.on_changed(self.update)
        self.slider_b.on_changed(self.update)
        self.slider_c.on_changed(self.update)
        self.slider_d.on_changed(self.update)
        
    def create_buttons(self):
        """Create preset transformation buttons"""
        button_width = 0.12
        button_height = 0.04
        button_y = 0.02
        
        # Identity
        ax_identity = plt.axes([0.15, button_y, button_width, button_height])
        self.btn_identity = Button(ax_identity, 'Identity')
        self.btn_identity.on_clicked(lambda x: self.set_preset([[1, 0], [0, 1]]))
        
        # Rotation 90°
        ax_rot90 = plt.axes([0.28, button_y, button_width, button_height])
        self.btn_rot90 = Button(ax_rot90, 'Rotate 90°')
        self.btn_rot90.on_clicked(lambda x: self.set_preset([[0, -1], [1, 0]]))
        
        # Reflection
        ax_reflect = plt.axes([0.41, button_y, button_width, button_height])
        self.btn_reflect = Button(ax_reflect, 'Reflect X')
        self.btn_reflect.on_clicked(lambda x: self.set_preset([[1, 0], [0, -1]]))
        
        # Shear
        ax_shear = plt.axes([0.54, button_y, button_width, button_height])
        self.btn_shear = Button(ax_shear, 'Shear')
        self.btn_shear.on_clicked(lambda x: self.set_preset([[1, 1], [0, 1]]))
        
        # Scale
        ax_scale = plt.axes([0.67, button_y, button_width, button_height])
        self.btn_scale = Button(ax_scale, 'Scale 2x')
        self.btn_scale.on_clicked(lambda x: self.set_preset([[2, 0], [0, 2]]))
        
    def set_preset(self, matrix):
        """Set a preset transformation matrix"""
        self.slider_a.set_val(matrix[0][0])
        self.slider_b.set_val(matrix[0][1])
        self.slider_c.set_val(matrix[1][0])
        self.slider_d.set_val(matrix[1][1])
        
    def update(self, val):
        """Update the transformation when sliders change"""
        self.matrix = np.array([
            [self.slider_a.val, self.slider_b.val],
            [self.slider_c.val, self.slider_d.val]
        ])
        self.plot_transformed()
        self.fig.canvas.draw_idle()
        
    def show(self):
        """Display the visualizer"""
        plt.show()

# Create and show the visualizer
visualizer = LinearTransformVisualizer()
visualizer.show()