#Project 1: Eigenvalues & Eigenvectors
#Check for Orthogonality:
#  Write a function that checks if the columns of a matrix are orthogonal (using dot products). Visualize orthogonal vs non-orthogonal vectors.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def check_orthogonality(matrix, tolerance=1e-10):
    """
    Check if the columns of a matrix are orthogonal.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix to check
    tolerance : float
        Numerical tolerance for checking if dot product is zero
    
    Returns:
    --------
    dict : Dictionary containing orthogonality results
    """
    # Ensure matrix is numpy array
    matrix = np.array(matrix)
    n_cols = matrix.shape[1]
    
    # Calculate dot product matrix (Gram matrix)
    gram_matrix = matrix.T @ matrix
    
    # Check each pair of columns
    results = {
        'is_orthogonal': True,
        'gram_matrix': gram_matrix,
        'dot_products': [],
        'column_pairs': []
    }
    
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            dot_prod = np.dot(matrix[:, i], matrix[:, j])
            results['dot_products'].append(dot_prod)
            results['column_pairs'].append((i, j))
            
            if abs(dot_prod) > tolerance:
                results['is_orthogonal'] = False
    
    return results

def visualize_vectors_2d(vectors, labels=None, title="Vector Visualization"):
    """
    Visualize 2D vectors to show orthogonality.
    
    Parameters:
    -----------
    vectors : list of np.ndarray
        List of 2D vectors to visualize
    labels : list of str
        Labels for each vector
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = sns.color_palette("husl", len(vectors))
    
    if labels is None:
        labels = [f"v{i+1}" for i in range(len(vectors))]
    
    # Plot each vector as an arrow from origin
    for i, (vec, label, color) in enumerate(zip(vectors, labels, colors)):
        ax.quiver(0, 0, vec[0], vec[1], 
                 angles='xy', scale_units='xy', scale=1,
                 color=color, width=0.008, label=label, alpha=0.8)
        
        # Add vector label at the tip
        ax.text(vec[0]*1.1, vec[1]*1.1, label, 
               fontsize=12, fontweight='bold', color=color)
    
    # Calculate and display dot products
    dot_products_text = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot_prod = np.dot(vectors[i], vectors[j])
            dot_products_text.append(f"{labels[i]}·{labels[j]} = {dot_prod:.3f}")
    
    # Add dot products to plot
    if dot_products_text:
        textstr = '\n'.join(dot_products_text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', bbox=props)
    
    # Set equal aspect ratio and grid
    max_val = max([max(abs(v[0]), abs(v[1])) for v in vectors]) * 1.3
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def visualize_gram_matrix(gram_matrix, title="Gram Matrix (Dot Products)"):
    """
    Visualize the Gram matrix as a heatmap.
    
    Parameters:
    -----------
    gram_matrix : np.ndarray
        Gram matrix to visualize
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(gram_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=ax, cbar_kws={'label': 'Dot Product'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column Index', fontsize=12)
    ax.set_ylabel('Column Index', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_comparison_plot():
    """
    Create a comprehensive comparison of orthogonal vs non-orthogonal vectors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Example 1: Orthogonal vectors
    orth_vectors = [np.array([1, 0]), np.array([0, 1])]
    colors = ['#FF6B6B', '#4ECDC4']
    
    ax1 = axes[0]
    for vec, color, label in zip(orth_vectors, colors, ['v1', 'v2']):
        ax1.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', 
                  scale=1, color=color, width=0.01, label=label, alpha=0.8)
        ax1.text(vec[0]*1.15, vec[1]*1.15, label, fontsize=14, 
                fontweight='bold', color=color)
    
    dot_prod_orth = np.dot(orth_vectors[0], orth_vectors[1])
    ax1.text(0.5, -0.3, f'v1·v2 = {dot_prod_orth:.3f}\n(Orthogonal)', 
            ha='center', fontsize=12, bbox=dict(boxstyle='round', 
            facecolor='lightgreen', alpha=0.8))
    
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_title('Orthogonal Vectors (90°)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.legend()
    
    # Example 2: Non-orthogonal vectors
    non_orth_vectors = [np.array([1, 0]), np.array([0.8, 0.6])]
    
    ax2 = axes[1]
    for vec, color, label in zip(non_orth_vectors, colors, ['v1', 'v2']):
        ax2.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', 
                  scale=1, color=color, width=0.01, label=label, alpha=0.8)
        ax2.text(vec[0]*1.15, vec[1]*1.15, label, fontsize=14, 
                fontweight='bold', color=color)
    
    dot_prod_non = np.dot(non_orth_vectors[0], non_orth_vectors[1])
    ax2.text(0.5, -0.3, f'v1·v2 = {dot_prod_non:.3f}\n(Non-orthogonal)', 
            ha='center', fontsize=12, bbox=dict(boxstyle='round', 
            facecolor='lightcoral', alpha=0.8))
    
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.set_title('Non-Orthogonal Vectors (≠90°)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.legend()
    
    plt.tight_layout()
    return fig

# ============EXAMPLES ============

print("="*60)
print("MATRIX ORTHOGONALITY CHECKER")
print("="*60)

# Example 1: Orthogonal matrix (Identity matrix)
print("\n1. ORTHOGONAL MATRIX (Identity Matrix)")
print("-" * 40)
orthogonal_matrix = np.eye(3)
print("Matrix:")
print(orthogonal_matrix)

results1 = check_orthogonality(orthogonal_matrix)
print(f"\nIs Orthogonal: {results1['is_orthogonal']}")
print("\nGram Matrix (Column dot products):")
print(results1['gram_matrix'])

# Example 2: Non-orthogonal matrix
print("\n\n2. NON-ORTHOGONAL MATRIX")
print("-" * 40)
non_orthogonal_matrix = np.array([[1, 2, 1],
                                   [0, 1, 1],
                                   [0, 0, 1]])
print("Matrix:")
print(non_orthogonal_matrix)

results2 = check_orthogonality(non_orthogonal_matrix)
print(f"\nIs Orthogonal: {results2['is_orthogonal']}")
print("\nGram Matrix (Column dot products):")
print(results2['gram_matrix'])

# Create DataFrame for better visualization of dot products
print("\nPairwise Dot Products:")
df = pd.DataFrame({
    'Column Pair': [f"Col {i} · Col {j}" for i, j in results2['column_pairs']],
    'Dot Product': results2['dot_products']
})
print(df.to_string(index=False))

# Example 3: Orthonormal matrix (rotation matrix)
print("\n\n3. ORTHONORMAL MATRIX (Rotation Matrix)")
print("-" * 40)
angle = np.pi / 4  # 45 degrees
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
print("Matrix:")
print(rotation_matrix)

results3 = check_orthogonality(rotation_matrix)
print(f"\nIs Orthogonal: {results3['is_orthogonal']}")
print("\nGram Matrix:")
print(results3['gram_matrix'])

# ============ VISUALIZATIONS ============

# Visualization 1: Comparison plot
print("\n\nGenerating visualizations...")
fig1 = create_comparison_plot()
plt.savefig('orthogonal_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: orthogonal_comparison.png")

# Visualization 2: Orthogonal vectors example
orthogonal_vecs = [np.array([3, 0]), np.array([0, 2]), np.array([1, 0])]
fig2 = visualize_vectors_2d(orthogonal_vecs[:2], 
                            labels=['v1 (3,0)', 'v2 (0,2)'],
                            title="Orthogonal Vectors Example")
plt.savefig('orthogonal_vectors.png', dpi=300, bbox_inches='tight')
print("Saved: orthogonal_vectors.png")

# Visualization 3: Non-orthogonal vectors example
non_orthogonal_vecs = [np.array([2, 1]), np.array([1, 2])]
fig3 = visualize_vectors_2d(non_orthogonal_vecs, 
                            labels=['v1 (2,1)', 'v2 (1,2)'],
                            title="Non-Orthogonal Vectors Example")
plt.savefig('non_orthogonal_vectors.png', dpi=300, bbox_inches='tight')
print("Saved: non_orthogonal_vectors.png")

# Visualization 4: Gram matrix heatmaps
fig4, axes = plt.subplots(1, 2, figsize=(14, 5))

plt.sca(axes[0])
visualize_gram_matrix(results1['gram_matrix'], "Orthogonal Matrix - Gram Matrix")

plt.sca(axes[1])
visualize_gram_matrix(results2['gram_matrix'], "Non-Orthogonal Matrix - Gram Matrix")

plt.savefig('gram_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: gram_matrices.png")

plt.show()

