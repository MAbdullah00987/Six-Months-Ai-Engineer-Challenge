
#10: Affine Transformations: Combine linear transformations and vector addition to perform affine transformations on geometric shapes.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class AffineTransformation:
    """Class to perform affine transformations on geometric shapes"""
    
    def __init__(self):
        self.transformation_history = []
    
    @staticmethod
    def create_shape(shape_type='square', center=(0, 0), size=1):
        """Create basic geometric shapes"""
        if shape_type == 'square':
            return np.array([
                [center[0] - size/2, center[1] - size/2],
                [center[0] + size/2, center[1] - size/2],
                [center[0] + size/2, center[1] + size/2],
                [center[0] - size/2, center[1] + size/2]
            ])
        elif shape_type == 'triangle':
            return np.array([
                [center[0], center[1] + size],
                [center[0] - size, center[1] - size],
                [center[0] + size, center[1] - size]
            ])
        elif shape_type == 'hexagon':
            angles = np.linspace(0, 2*np.pi, 7)
            return np.array([[center[0] + size*np.cos(a), 
                            center[1] + size*np.sin(a)] for a in angles[:-1]])
        elif shape_type == 'arrow':
            return np.array([
                [center[0], center[1] + size],
                [center[0] + size/2, center[1] + size/2],
                [center[0] + size/4, center[1] + size/2],
                [center[0] + size/4, center[1] - size],
                [center[0] - size/4, center[1] - size],
                [center[0] - size/4, center[1] + size/2],
                [center[0] - size/2, center[1] + size/2]
            ])
    
    def apply_affine(self, points, matrix, translation):
        """
        Apply affine transformation: y = Ax + b
        where A is the linear transformation matrix and b is translation vector
        """
        # Convert points to homogeneous coordinates if needed
        points = np.array(points)
        
        # Apply linear transformation
        transformed = points @ matrix.T
        
        # Apply translation
        transformed = transformed + translation
        
        # Store transformation details
        self.transformation_history.append({
            'matrix': matrix.copy(),
            'translation': translation.copy(),
            'type': 'affine'
        })
        
        return transformed
    
    def rotation_matrix(self, angle_degrees):
        """Create 2D rotation matrix"""
        theta = np.radians(angle_degrees)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    
    def scaling_matrix(self, sx, sy):
        """Create 2D scaling matrix"""
        return np.array([
            [sx, 0],
            [0, sy]
        ])
    
    def shear_matrix(self, shx, shy):
        """Create 2D shear matrix"""
        return np.array([
            [1, shx],
            [shy, 1]
        ])
    
    def reflection_matrix(self, axis='x'):
        """Create 2D reflection matrix"""
        if axis == 'x':
            return np.array([[1, 0], [0, -1]])
        elif axis == 'y':
            return np.array([[-1, 0], [0, 1]])
        elif axis == 'origin':
            return np.array([[-1, 0], [0, -1]])
    
    def compose_transformations(self, *matrices):
        """Compose multiple linear transformations"""
        result = np.eye(2)
        for matrix in matrices:
            result = result @ matrix
        return result


def visualize_transformations():
    """Create comprehensive visualization of affine transformations"""
    
    at = AffineTransformation()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Affine Transformations on Geometric Shapes', fontsize=16, fontweight='bold')
    
    # Original shape
    original = at.create_shape('square', center=(0, 0), size=2)
    
    transformations = [
        {
            'title': '1. Original Shape',
            'matrix': np.eye(2),
            'translation': np.array([0, 0]),
            'color': 'blue'
        },
        {
            'title': '2. Rotation (45°) + Translation',
            'matrix': at.rotation_matrix(45),
            'translation': np.array([3, 1]),
            'color': 'red'
        },
        {
            'title': '3. Scaling (1.5x, 0.7y) + Translation',
            'matrix': at.scaling_matrix(1.5, 0.7),
            'translation': np.array([0, -2]),
            'color': 'green'
        },
        {
            'title': '4. Shear + Translation',
            'matrix': at.shear_matrix(0.5, 0.3),
            'translation': np.array([-3, 1]),
            'color': 'orange'
        },
        {
            'title': '5. Reflection (x-axis) + Translation',
            'matrix': at.reflection_matrix('x'),
            'translation': np.array([2, -3]),
            'color': 'purple'
        },
        {
            'title': '6. Combined: Rotate + Scale + Translate',
            'matrix': at.compose_transformations(
                at.rotation_matrix(30),
                at.scaling_matrix(1.2, 0.8)
            ),
            'translation': np.array([-2, -2]),
            'color': 'brown'
        }
    ]
    
    # Apply and plot transformations
    for idx, (ax, trans) in enumerate(zip(axes.flat, transformations)):
        # Transform the shape
        transformed = at.apply_affine(original, trans['matrix'], trans['translation'])
        
        # Plot original (light gray)
        original_patch = Polygon(original, fill=True, alpha=0.2, 
                                edgecolor='gray', facecolor='lightgray', 
                                linewidth=1, linestyle='--')
        ax.add_patch(original_patch)
        
        # Plot transformed shape
        transformed_patch = Polygon(transformed, fill=True, alpha=0.5, 
                                   edgecolor=trans['color'], 
                                   facecolor=trans['color'], linewidth=2)
        ax.add_patch(transformed_patch)
        
        # Add vertices labels
        for i, (x, y) in enumerate(transformed):
            ax.plot(x, y, 'ko', markersize=4)
            ax.text(x, y, f'  P{i}', fontsize=8)
        
        # Set axis properties
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(trans['title'], fontweight='bold')
        
        # Add transformation matrix info
        matrix_str = f"A = {trans['matrix']}\nb = {trans['translation']}"
        ax.text(0.02, 0.98, matrix_str, transform=ax.transAxes,
               verticalalignment='top', fontsize=7, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def demonstrate_multiple_shapes():
    """Demonstrate affine transformations on multiple shapes"""
    
    at = AffineTransformation()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Affine Transformations on Different Shapes', 
                 fontsize=16, fontweight='bold')
    
    shapes = [
        ('triangle', 'Triangle'),
        ('hexagon', 'Hexagon'),
        ('arrow', 'Arrow')
    ]
    
    # Define a complex transformation
    rotation = at.rotation_matrix(30)
    scaling = at.scaling_matrix(1.3, 0.9)
    combined_matrix = at.compose_transformations(rotation, scaling)
    translation = np.array([3, 2])
    
    for ax, (shape_type, shape_name) in zip(axes, shapes):
        # Create original shape
        original = at.create_shape(shape_type, center=(0, 0), size=1.5)
        
        # Apply transformation
        transformed = at.apply_affine(original, combined_matrix, translation)
        
        # Plot original
        original_patch = Polygon(original, fill=True, alpha=0.3, 
                                edgecolor='blue', facecolor='lightblue', 
                                linewidth=2, label='Original')
        ax.add_patch(original_patch)
        
        # Plot transformed
        transformed_patch = Polygon(transformed, fill=True, alpha=0.6, 
                                   edgecolor='red', facecolor='salmon', 
                                   linewidth=2, label='Transformed')
        ax.add_patch(transformed_patch)
        
        ax.set_xlim(-3, 6)
        ax.set_ylim(-3, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(f'{shape_name}', fontweight='bold')
        ax.legend()
    
    plt.tight_layout()
    return fig


def create_transformation_dataframe():
    """Create a pandas DataFrame with transformation details"""
    
    at = AffineTransformation()
    
    transformations_data = []
    
    # Define various transformations
    configs = [
        ('Rotation 45°', at.rotation_matrix(45), np.array([1, 2])),
        ('Scaling 2x, 0.5y', at.scaling_matrix(2, 0.5), np.array([0, 0])),
        ('Shear 0.5x', at.shear_matrix(0.5, 0), np.array([1, 1])),
        ('Reflection X', at.reflection_matrix('x'), np.array([0, 2])),
        ('Combined', at.compose_transformations(
            at.rotation_matrix(30),
            at.scaling_matrix(1.5, 1.5)
        ), np.array([2, 1]))
    ]
    
    # Test point
    test_point = np.array([[1, 1]])
    
    for name, matrix, translation in configs:
        transformed = at.apply_affine(test_point, matrix, translation)
        
        transformations_data.append({
            'Transformation': name,
            'Original_X': test_point[0, 0],
            'Original_Y': test_point[0, 1],
            'Transformed_X': transformed[0, 0],
            'Transformed_Y': transformed[0, 1],
            'Matrix_A11': matrix[0, 0],
            'Matrix_A12': matrix[0, 1],
            'Matrix_A21': matrix[1, 0],
            'Matrix_A22': matrix[1, 1],
            'Translation_X': translation[0],
            'Translation_Y': translation[1]
        })
    
    df = pd.DataFrame(transformations_data)
    
    print("Affine Transformation Results")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n")
    
    # Display transformation equations
    print("Affine Transformation Equation: y = Ax + b")
    print("where A is the linear transformation matrix and b is the translation vector")
    print("=" * 80)
    
    return df


# Run demonstrations
if __name__ == "__main__":
    print("\n" + "="*80)
    print("AFFINE TRANSFORMATIONS DEMONSTRATION")
    print("="*80 + "\n")
    
    # Create DataFrame with transformation details
    df = create_transformation_dataframe()
    
    # Visualize various transformations
    fig1 = visualize_transformations()
    
    # Demonstrate on different shapes
    fig2 = demonstrate_multiple_shapes()
    
    plt.show()
    
    print("\nKey Concepts:")
    print("1. Affine transformation combines linear transformation (Ax) and translation (+b)")
    print("2. Linear transformations include: rotation, scaling, shear, reflection")
    print("3. Multiple transformations can be composed by matrix multiplication")
    print("4. Order of composition matters: AB ≠ BA in general")
    print("5. Translation must be applied after linear transformation")