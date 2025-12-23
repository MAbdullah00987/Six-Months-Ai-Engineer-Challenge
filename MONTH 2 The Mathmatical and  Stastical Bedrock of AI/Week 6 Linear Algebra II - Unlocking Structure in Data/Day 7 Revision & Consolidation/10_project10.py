
# Project 10:
# Change of Basis: Write a script to transform the coordinates of a vector from one basis to another.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class BasisTransformer:
    """
    A class to perform change of basis transformations.
    """
    
    def __init__(self, old_basis, new_basis):
        """
        Initialize with old and new basis vectors.
        
        Parameters:
        - old_basis: matrix where columns are the old basis vectors
        - new_basis: matrix where columns are the new basis vectors
        """
        self.old_basis = np.array(old_basis)
        self.new_basis = np.array(new_basis)
        
        # Check if bases are valid (square matrices)
        if self.old_basis.shape[0] != self.old_basis.shape[1]:
            raise ValueError("Old basis must be a square matrix")
        if self.new_basis.shape[0] != self.new_basis.shape[1]:
            raise ValueError("New basis must be a square matrix")
        
        # Calculate change of basis matrix
        self.change_matrix = self._calculate_change_matrix()
    
    def _calculate_change_matrix(self):
        """
        Calculate the change of basis matrix from old basis to new basis.
        The formula is: P = B_new^(-1) @ B_old
        """
        return np.linalg.inv(self.new_basis) @ self.old_basis
    
    def transform(self, vector_old):
        """
        Transform a vector from old basis coordinates to new basis coordinates.
        
        Parameters:
        - vector_old: coordinates in the old basis
        
        Returns:
        - vector_new: coordinates in the new basis
        """
        vector_old = np.array(vector_old)
        return self.change_matrix @ vector_old
    
    def inverse_transform(self, vector_new):
        """
        Transform a vector from new basis coordinates to old basis coordinates.
        
        Parameters:
        - vector_new: coordinates in the new basis
        
        Returns:
        - vector_old: coordinates in the old basis
        """
        vector_new = np.array(vector_new)
        return np.linalg.inv(self.change_matrix) @ vector_new
    
    def get_standard_coordinates(self, vector_basis, basis='old'):
        """
        Get standard (Cartesian) coordinates from basis coordinates.
        
        Parameters:
        - vector_basis: coordinates in the specified basis
        - basis: 'old' or 'new'
        """
        vector_basis = np.array(vector_basis)
        if basis == 'old':
            return self.old_basis @ vector_basis
        elif basis == 'new':
            return self.new_basis @ vector_basis
        else:
            raise ValueError("basis must be 'old' or 'new'")
    
    def display_info(self):
        """Display information about the bases and transformation."""
        print("=" * 60)
        print("CHANGE OF BASIS TRANSFORMATION")
        print("=" * 60)
        print("\nOld Basis (columns are basis vectors):")
        print(self.old_basis)
        print("\nNew Basis (columns are basis vectors):")
        print(self.new_basis)
        print("\nChange of Basis Matrix (Old → New):")
        print(self.change_matrix)
        print("\nInverse Change Matrix (New → Old):")
        print(np.linalg.inv(self.change_matrix))
        print("=" * 60)


def plot_2d_transformation(transformer, vector_old, figsize=(14, 6)):
    """
    Visualize 2D change of basis transformation.
    """
    # Transform the vector
    vector_new = transformer.transform(vector_old)
    
    # Get standard coordinates for visualization
    v_standard = transformer.get_standard_coordinates(vector_old, 'old')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Standard coordinate system with both bases
    ax1.set_title('Standard Coordinate System', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Plot old basis vectors
    for i in range(transformer.old_basis.shape[1]):
        basis_vec = transformer.old_basis[:, i]
        ax1.quiver(0, 0, basis_vec[0], basis_vec[1], 
                   angles='xy', scale_units='xy', scale=1,
                   color='blue', width=0.008, alpha=0.7,
                   label=f'Old e{i+1}')
    
    # Plot new basis vectors
    for i in range(transformer.new_basis.shape[1]):
        basis_vec = transformer.new_basis[:, i]
        ax1.quiver(0, 0, basis_vec[0], basis_vec[1], 
                   angles='xy', scale_units='xy', scale=1,
                   color='red', width=0.008, alpha=0.7,
                   label=f'New f{i+1}')
    
    # Plot the actual vector
    ax1.quiver(0, 0, v_standard[0], v_standard[1], 
               angles='xy', scale_units='xy', scale=1,
               color='green', width=0.012,
               label=f'Vector v')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.legend(loc='best')
    ax1.axis('equal')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    
    # Plot 2: Coordinate comparison
    ax2.set_title('Coordinate Comparison', fontsize=14, fontweight='bold')
    
    bases_names = ['Old Basis', 'New Basis']
    x_pos = np.arange(len(bases_names))
    
    # Plot coordinates for each dimension
    width = 0.35
    for i in range(len(vector_old)):
        coords = [vector_old[i], vector_new[i]]
        ax2.bar(x_pos + i*width, coords, width, 
                label=f'Dimension {i+1}', alpha=0.8)
    
    ax2.set_ylabel('Coordinate Value', fontsize=12)
    ax2.set_xticks(x_pos + width/2)
    ax2.set_xticklabels(bases_names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Display coordinate information
    coord_data = {
        'Basis': ['Old Basis', 'New Basis', 'Standard (Cartesian)'],
        'Coordinates': [str(vector_old), str(vector_new), str(v_standard)]
    }
    df = pd.DataFrame(coord_data)
    print("\nCoordinate Representations:")
    print(df.to_string(index=False))


# Example 1: 2D Transformation
print("\n" + "="*60)
print("EXAMPLE 1: 2D Change of Basis")
print("="*60)

# Define standard basis
standard_basis = np.array([[1, 0],
                           [0, 1]])

# Define a new rotated and scaled basis
new_basis = np.array([[2, 1],
                      [1, 2]])

# Create transformer
transformer_2d = BasisTransformer(standard_basis, new_basis)
transformer_2d.display_info()

# Define a vector in the old basis
vector_old_2d = np.array([3, 2])
print(f"\nVector in old basis: {vector_old_2d}")

# Transform to new basis
vector_new_2d = transformer_2d.transform(vector_old_2d)
print(f"Vector in new basis: {vector_new_2d}")

# Verify by transforming back
vector_back_2d = transformer_2d.inverse_transform(vector_new_2d)
print(f"Transformed back to old basis: {vector_back_2d}")
print(f"Verification (should be close to zero): {np.linalg.norm(vector_old_2d - vector_back_2d)}")

# Visualize
plot_2d_transformation(transformer_2d, vector_old_2d)


# Example 2: 3D Transformation
print("\n" + "="*60)
print("EXAMPLE 2: 3D Change of Basis")
print("="*60)

# Standard 3D basis
standard_basis_3d = np.eye(3)

# New basis (rotation + scaling)
new_basis_3d = np.array([[1, 1, 0],
                         [0, 1, 1],
                         [1, 0, 1]])

transformer_3d = BasisTransformer(standard_basis_3d, new_basis_3d)
transformer_3d.display_info()

# Vector in old basis
vector_old_3d = np.array([2, 3, 1])
print(f"\nVector in old basis: {vector_old_3d}")

# Transform
vector_new_3d = transformer_3d.transform(vector_old_3d)
print(f"Vector in new basis: {vector_new_3d}")

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Dimension': ['x/1', 'y/2', 'z/3'],
    'Old Basis Coords': vector_old_3d,
    'New Basis Coords': vector_new_3d
})
print("\nCoordinate Comparison Table:")
print(comparison_df.to_string(index=False))

# Visualize 3D transformation
fig = plt.figure(figsize=(15, 5))

# Plot 1: Old basis
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_title('Old Basis (Standard)', fontweight='bold')
colors = ['r', 'g', 'b']
for i in range(3):
    basis_vec = standard_basis_3d[:, i]
    ax1.quiver(0, 0, 0, basis_vec[0], basis_vec[1], basis_vec[2],
               color=colors[i], arrow_length_ratio=0.1, linewidth=2,
               label=f'e{i+1}')
v_std = transformer_3d.get_standard_coordinates(vector_old_3d, 'old')
ax1.quiver(0, 0, 0, v_std[0], v_std[1], v_std[2],
           color='black', arrow_length_ratio=0.1, linewidth=3, label='Vector')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# Plot 2: New basis
ax2 = fig.add_subplot(132, projection='3d')
ax2.set_title('New Basis', fontweight='bold')
for i in range(3):
    basis_vec = new_basis_3d[:, i]
    ax2.quiver(0, 0, 0, basis_vec[0], basis_vec[1], basis_vec[2],
               color=colors[i], arrow_length_ratio=0.1, linewidth=2,
               label=f'f{i+1}')
ax2.quiver(0, 0, 0, v_std[0], v_std[1], v_std[2],
           color='black', arrow_length_ratio=0.1, linewidth=3, label='Vector')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()

# Plot 3: Coordinate comparison
ax3 = fig.add_subplot(133)
ax3.set_title('Coordinate Values', fontweight='bold')
x = np.arange(3)
width = 0.35
ax3.bar(x - width/2, vector_old_3d, width, label='Old Basis', alpha=0.8)
ax3.bar(x + width/2, vector_new_3d, width, label='New Basis', alpha=0.8)
ax3.set_ylabel('Coordinate Value')
ax3.set_xlabel('Dimension')
ax3.set_xticks(x)
ax3.set_xticklabels(['Dim 1', 'Dim 2', 'Dim 3'])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("TRANSFORMATION COMPLETE!")
print("="*60)