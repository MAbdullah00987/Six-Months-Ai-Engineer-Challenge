
#1: Vector Operations from Scratch: Implement functions for vector addition, scalar multiplication, and dot product in pure Python.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# STEP 1: Vector Operations using NumPy
# ============================================

class VectorOperations:
    """A class to perform vector operations using NumPy"""
    
    @staticmethod
    def vector_addition(v1, v2):
        """
        Add two vectors element-wise
        Args:
            v1, v2: numpy arrays or lists
        Returns:
            numpy array
        """
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
        
        return v1 + v2
    
    @staticmethod
    def scalar_multiplication(scalar, v):
        """
        Multiply vector by a scalar
        Args:
            scalar: number
            v: numpy array or list
        Returns:
            numpy array
        """
        v = np.array(v)
        return scalar * v
    
    @staticmethod
    def dot_product(v1, v2):
        """
        Calculate dot product of two vectors
        Args:
            v1, v2: numpy arrays or lists
        Returns:
            scalar value
        """
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
        
        return np.dot(v1, v2)
    
    @staticmethod
    def magnitude(v):
        """Calculate the magnitude (length) of a vector"""
        v = np.array(v)
        return np.linalg.norm(v)
    
    @staticmethod
    def angle_between(v1, v2):
        """Calculate angle between two vectors in degrees"""
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle_rad)


# ============================================
# STEP 2: Create and Store Results in Pandas
# ============================================

def create_results_dataframe():
    """Create a DataFrame to store multiple vector operation results"""
    
    # Define test vectors
    vectors = {
        'v1': np.array([2, 3, 1]),
        'v2': np.array([4, -1, 2]),
        'v3': np.array([1, 1, 1]),
        'v4': np.array([0, 5, -2])
    }
    
    ops = VectorOperations()
    
    # Store results
    results = []
    
    # Perform operations on different vector pairs
    pairs = [('v1', 'v2'), ('v1', 'v3'), ('v2', 'v4'), ('v3', 'v4')]
    
    for pair in pairs:
        v1_name, v2_name = pair
        v1 = vectors[v1_name]
        v2 = vectors[v2_name]
        
        addition = ops.vector_addition(v1, v2)
        dot = ops.dot_product(v1, v2)
        angle = ops.angle_between(v1, v2)
        
        results.append({
            'Vector 1': v1_name,
            'Vector 2': v2_name,
            'V1 Values': str(v1),
            'V2 Values': str(v2),
            'Addition': str(addition),
            'Dot Product': dot,
            'Angle (degrees)': round(angle, 2),
            'V1 Magnitude': round(ops.magnitude(v1), 2),
            'V2 Magnitude': round(ops.magnitude(v2), 2)
        })
    
    df = pd.DataFrame(results)
    return df, vectors


# ============================================
# STEP 3: Visualize Vectors using Matplotlib
# ============================================

def plot_2d_vectors(vectors_dict):
    """Plot 2D vectors and their operations"""
    
    # Extract first two components for 2D plotting
    v1 = vectors_dict['v1'][:2]
    v2 = vectors_dict['v2'][:2]
    
    ops = VectorOperations()
    v_add = ops.vector_addition(v1, v2)
    v_scalar = ops.scalar_multiplication(2, v1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Vector Addition
    ax1 = axes[0]
    ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.006, label='v1')
    ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.006, label='v2')
    ax1.quiver(0, 0, v_add[0], v_add[1], angles='xy', scale_units='xy', scale=1, 
               color='green', width=0.008, label='v1 + v2')
    ax1.set_xlim(-1, 7)
    ax1.set_ylim(-2, 5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Vector Addition', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 2: Scalar Multiplication
    ax2 = axes[1]
    ax2.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.006, label='v1')
    ax2.quiver(0, 0, v_scalar[0], v_scalar[1], angles='xy', scale_units='xy', scale=1, 
               color='purple', width=0.008, label='2 × v1')
    ax2.set_xlim(-1, 7)
    ax2.set_ylim(-2, 8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Scalar Multiplication', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 3: Dot Product Visualization (angle between vectors)
    ax3 = axes[2]
    ax3.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.006, label='v1')
    ax3.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.006, label='v2')
    
    dot = ops.dot_product(v1, v2)
    angle = ops.angle_between(v1, v2)
    
    ax3.text(2, 3, f'Dot Product: {dot:.2f}\nAngle: {angle:.2f}°', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.set_xlim(-1, 7)
    ax3.set_ylim(-2, 5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Dot Product & Angle', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('vector_operations_2d.png', dpi=300, bbox_inches='tight')
    print("✓ 2D visualization saved as 'vector_operations_2d.png'")
    plt.show()


def plot_3d_vectors(vectors_dict):
    """Plot 3D vectors"""
    
    v1 = vectors_dict['v1']
    v2 = vectors_dict['v2']
    
    ops = VectorOperations()
    v_add = ops.vector_addition(v1, v2)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D Vector Addition
    ax1 = fig.add_subplot(121, projection='3d')
    
    ax1.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', 
               arrow_length_ratio=0.1, linewidth=2, label='v1')
    ax1.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='red', 
               arrow_length_ratio=0.1, linewidth=2, label='v2')
    ax1.quiver(0, 0, 0, v_add[0], v_add[1], v_add[2], color='green', 
               arrow_length_ratio=0.1, linewidth=3, label='v1 + v2')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Vector Addition', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xlim([0, 7])
    ax1.set_ylim([-2, 4])
    ax1.set_zlim([0, 4])
    
    # Multiple 3D Vectors
    ax2 = fig.add_subplot(122, projection='3d')
    
    colors = ['blue', 'red', 'green', 'purple']
    for (name, vec), color in zip(vectors_dict.items(), colors):
        ax2.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
                   color=color, arrow_length_ratio=0.1, linewidth=2, label=name)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Multiple 3D Vectors', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('vector_operations_3d.png', dpi=300, bbox_inches='tight')
    print("✓ 3D visualization saved as 'vector_operations_3d.png'")
    plt.show()


# ============================================
# STEP 4: Main Execution
# ============================================

def main():
    print("=" * 60)
    print("VECTOR OPERATIONS WITH NUMPY, PANDAS & MATPLOTLIB")
    print("=" * 60)
    
    # Initialize
    ops = VectorOperations()
    
    # Example vectors
    print("\n📊 Example Vectors:")
    v1 = np.array([2, 3, 1])
    v2 = np.array([4, -1, 2])
    scalar = 3
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"scalar = {scalar}")
    
    # Perform operations
    print("\n🔧 Vector Operations:")
    print("-" * 60)
    
    addition = ops.vector_addition(v1, v2)
    print(f"✓ Vector Addition (v1 + v2) = {addition}")
    
    scalar_mult = ops.scalar_multiplication(scalar, v1)
    print(f"✓ Scalar Multiplication ({scalar} × v1) = {scalar_mult}")
    
    dot = ops.dot_product(v1, v2)
    print(f"✓ Dot Product (v1 · v2) = {dot}")
    
    mag_v1 = ops.magnitude(v1)
    mag_v2 = ops.magnitude(v2)
    print(f"✓ Magnitude of v1 = {mag_v1:.4f}")
    print(f"✓ Magnitude of v2 = {mag_v2:.4f}")
    
    angle = ops.angle_between(v1, v2)
    print(f"✓ Angle between v1 and v2 = {angle:.2f}°")
    
    # Create DataFrame
    print("\n📋 Creating Results DataFrame...")
    df, vectors = create_results_dataframe()
    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    # Save to CSV
    df.to_csv('vector_operations_results.csv', index=False)
    print("\n✓ Results saved to 'vector_operations_results.csv'")
    
    # Visualizations
    print("\n📈 Generating Visualizations...")
    plot_2d_vectors(vectors)
    plot_3d_vectors(vectors)
    
    print("\n✅ All operations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()