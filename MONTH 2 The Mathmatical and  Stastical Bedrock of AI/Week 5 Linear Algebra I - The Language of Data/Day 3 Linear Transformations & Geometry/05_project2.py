
#Project 2: Linear Transformation Visualizer

#Build interactive visualization showing grid transformation
#Display eigenvectors if time permits
#Show effect of different 2×2 matrices

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches

class LinearTransformationVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 7))
        
        # Create subplots for original and transformed grids
        self.ax_orig = self.fig.add_subplot(121)
        self.ax_trans = self.fig.add_subplot(122)
        
        # Initial transformation matrix
        self.matrix = np.array([[1.5, 0.5], [0.5, 1.5]])
        
        # Setup the plot
        self.setup_plot()
        self.create_sliders()
        self.create_preset_buttons()
        self.update_visualization()
        
    def setup_plot(self):
        """Setup the plot axes and titles"""
        self.ax_orig.set_xlim(-3, 3)
        self.ax_orig.set_ylim(-3, 3)
        self.ax_orig.set_aspect('equal')
        self.ax_orig.grid(True, alpha=0.3)
        self.ax_orig.set_title('Original Grid', fontsize=14, fontweight='bold')
        self.ax_orig.axhline(0, color='black', linewidth=0.5)
        self.ax_orig.axvline(0, color='black', linewidth=0.5)
        
        self.ax_trans.set_xlim(-5, 5)
        self.ax_trans.set_ylim(-5, 5)
        self.ax_trans.set_aspect('equal')
        self.ax_trans.grid(True, alpha=0.3)
        self.ax_trans.set_title('Transformed Grid', fontsize=14, fontweight='bold')
        self.ax_trans.axhline(0, color='black', linewidth=0.5)
        self.ax_trans.axvline(0, color='black', linewidth=0.5)
        
    def create_sliders(self):
        """Create sliders for matrix elements"""
        plt.subplots_adjust(bottom=0.35)
        
        # Create slider axes
        ax_a11 = plt.axes([0.15, 0.25, 0.3, 0.02])
        ax_a12 = plt.axes([0.15, 0.20, 0.3, 0.02])
        ax_a21 = plt.axes([0.15, 0.15, 0.3, 0.02])
        ax_a22 = plt.axes([0.15, 0.10, 0.3, 0.02])
        
        # Create sliders
        self.slider_a11 = Slider(ax_a11, 'a₁₁', -3.0, 3.0, valinit=self.matrix[0,0], valstep=0.1)
        self.slider_a12 = Slider(ax_a12, 'a₁₂', -3.0, 3.0, valinit=self.matrix[0,1], valstep=0.1)
        self.slider_a21 = Slider(ax_a21, 'a₂₁', -3.0, 3.0, valinit=self.matrix[1,0], valstep=0.1)
        self.slider_a22 = Slider(ax_a22, 'a₂₂', -3.0, 3.0, valinit=self.matrix[1,1], valstep=0.1)
        
        # Connect sliders to update function
        self.slider_a11.on_changed(self.update_matrix)
        self.slider_a12.on_changed(self.update_matrix)
        self.slider_a21.on_changed(self.update_matrix)
        self.slider_a22.on_changed(self.update_matrix)
        
    def create_preset_buttons(self):
        """Create preset transformation buttons"""
        button_width = 0.10
        button_height = 0.03
        button_y = 0.02
        
        # Button positions
        buttons_data = [
            ('Rotation', [0.55, button_y, button_width, button_height], self.rotation_matrix),
            ('Scaling', [0.66, button_y, button_width, button_height], self.scaling_matrix),
            ('Shear', [0.77, button_y, button_width, button_height], self.shear_matrix),
            ('Reflection', [0.88, button_y, button_width, button_height], self.reflection_matrix),
        ]
        
        self.buttons = []
        for label, pos, func in buttons_data:
            ax_button = plt.axes(pos)
            button = Button(ax_button, label)
            button.on_clicked(func)
            self.buttons.append(button)
    
    def update_matrix(self, val):
        """Update the transformation matrix from sliders"""
        self.matrix[0,0] = self.slider_a11.val
        self.matrix[0,1] = self.slider_a12.val
        self.matrix[1,0] = self.slider_a21.val
        self.matrix[1,1] = self.slider_a22.val
        self.update_visualization()
    
    def update_visualization(self):
        """Update the visualization with current matrix"""
        self.ax_orig.clear()
        self.ax_trans.clear()
        self.setup_plot()
        
        # Draw original grid
        self.draw_grid(self.ax_orig, np.eye(2))
        
        # Draw transformed grid
        self.draw_grid(self.ax_trans, self.matrix)
        
        # Draw basis vectors
        self.draw_basis_vectors()
        
        # Calculate and display eigenvalues and eigenvectors
        self.display_eigeninfo()
        
        plt.draw()
    
    def draw_grid(self, ax, matrix):
        """Draw a grid of lines transformed by the matrix"""
        grid_range = np.arange(-2, 3, 0.5)
        
        # Vertical lines
        for x in grid_range:
            points = np.array([[x, x], [-2, 2]])
            transformed = matrix @ points
            ax.plot(transformed[0], transformed[1], 'b-', alpha=0.4, linewidth=1)
        
        # Horizontal lines
        for y in grid_range:
            points = np.array([[-2, 2], [y, y]])
            transformed = matrix @ points
            ax.plot(transformed[0], transformed[1], 'b-', alpha=0.4, linewidth=1)
    
    def draw_basis_vectors(self):
        """Draw basis vectors before and after transformation"""
        # Original basis vectors
        self.ax_orig.arrow(0, 0, 1, 0, head_width=0.15, head_length=0.15, 
                          fc='red', ec='red', linewidth=2, label='e₁')
        self.ax_orig.arrow(0, 0, 0, 1, head_width=0.15, head_length=0.15, 
                          fc='green', ec='green', linewidth=2, label='e₂')
        
        # Transformed basis vectors
        e1_trans = self.matrix @ np.array([1, 0])
        e2_trans = self.matrix @ np.array([0, 1])
        
        self.ax_trans.arrow(0, 0, e1_trans[0], e1_trans[1], 
                           head_width=0.15, head_length=0.15, 
                           fc='red', ec='red', linewidth=2, label='T(e₁)')
        self.ax_trans.arrow(0, 0, e2_trans[0], e2_trans[1], 
                           head_width=0.15, head_length=0.15, 
                           fc='green', ec='green', linewidth=2, label='T(e₂)')
        
        self.ax_orig.legend(loc='upper right')
        self.ax_trans.legend(loc='upper right')
    
    def display_eigeninfo(self):
        """Calculate and display eigenvalues and eigenvectors"""
        try:
            eigenvalues, eigenvectors = np.linalg.eig(self.matrix)
            
            # Display matrix info
            det = np.linalg.det(self.matrix)
            trace = np.trace(self.matrix)
            
            info_text = f'Matrix:\n[{self.matrix[0,0]:.2f}  {self.matrix[0,1]:.2f}]\n'
            info_text += f'[{self.matrix[1,0]:.2f}  {self.matrix[1,1]:.2f}]\n\n'
            info_text += f'Det: {det:.3f}\n'
            info_text += f'Trace: {trace:.3f}\n\n'
            
            # Check if eigenvalues are real
            if np.all(np.isreal(eigenvalues)):
                info_text += f'λ₁ = {eigenvalues[0].real:.3f}\n'
                info_text += f'λ₂ = {eigenvalues[1].real:.3f}'
                
                # Draw eigenvectors on transformed plot
                for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                    if np.isreal(eigval):
                        eigvec = eigvec.real
                        # Normalize for better visualization
                        eigvec_normalized = eigvec / np.linalg.norm(eigvec) * 1.5
                        
                        color = 'purple' if i == 0 else 'orange'
                        self.ax_trans.arrow(0, 0, eigvec_normalized[0], eigvec_normalized[1],
                                          head_width=0.2, head_length=0.2,
                                          fc=color, ec=color, linewidth=2.5,
                                          linestyle='--', alpha=0.7,
                                          label=f'v₁ (λ={eigval.real:.2f})' if i == 0 else f'v₂ (λ={eigval.real:.2f})')
                
                self.ax_trans.legend(loc='upper right', fontsize=9)
            else:
                info_text += 'Complex eigenvalues\n'
                info_text += f'λ = {eigenvalues[0].real:.2f} ± {abs(eigenvalues[0].imag):.2f}i'
            
            self.fig.text(0.02, 0.50, info_text, fontsize=10, 
                         verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        except np.linalg.LinAlgError:
            self.fig.text(0.02, 0.50, 'Singular Matrix', fontsize=10)
    
    # Preset transformation matrices
    def rotation_matrix(self, event):
        """Apply rotation matrix (45 degrees)"""
        angle = np.pi / 4
        self.matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
        self.update_sliders()
    
    def scaling_matrix(self, event):
        """Apply scaling matrix"""
        self.matrix = np.array([[2.0, 0.0],
                               [0.0, 0.5]])
        self.update_sliders()
    
    def shear_matrix(self, event):
        """Apply shear matrix"""
        self.matrix = np.array([[1.0, 1.0],
                               [0.0, 1.0]])
        self.update_sliders()
    
    def reflection_matrix(self, event):
        """Apply reflection matrix"""
        self.matrix = np.array([[1.0, 0.0],
                               [0.0, -1.0]])
        self.update_sliders()
    
    def update_sliders(self):
        """Update slider values to match current matrix"""
        self.slider_a11.set_val(self.matrix[0,0])
        self.slider_a12.set_val(self.matrix[0,1])
        self.slider_a21.set_val(self.matrix[1,0])
        self.slider_a22.set_val(self.matrix[1,1])
        self.update_visualization()
    
    def show(self):
        """Display the visualization"""
        plt.show()

# Create and show the visualizer
if __name__ == "__main__":
    visualizer = LinearTransformationVisualizer()
    print("Linear Transformation Visualizer")
    print("=" * 50)
    print("Use the sliders to adjust matrix elements")
    print("Click preset buttons for common transformations")
    print("Purple/Orange dashed arrows show eigenvectors (when real)")
    print("=" * 50)
    visualizer.show()