#Linear Independence Check: Use matrix rank to determine if a set of vectors is linearly independent.
#  Create test cases with dependent and independent vectors.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class LinearIndependenceChecker:
    """
    A class to check linear independence of vectors using matrix rank.
    """
    
    def __init__(self, vectors: List[np.ndarray]):
        """
        Initialize with a list of vectors.
        
        Parameters:
        vectors: List of numpy arrays representing vectors
        """
        self.vectors = vectors
        self.matrix = np.column_stack(vectors)
        self.num_vectors = len(vectors)
        self.vector_dimension = len(vectors[0])
        
    def check_independence(self, tolerance=1e-10) -> Tuple[bool, int, int]:
        """
        Check if vectors are linearly independent using matrix rank.
        
        Returns:
        Tuple of (is_independent, rank, num_vectors)
        """
        rank = np.linalg.matrix_rank(self.matrix, tol=tolerance)
        is_independent = (rank == self.num_vectors)
        
        return is_independent, rank, self.num_vectors
    
    def get_report(self) -> dict:
        """
        Generate a detailed report of the linear independence check.
        """
        is_independent, rank, num_vectors = self.check_independence()
        
        report = {
            'Matrix': self.matrix,
            'Number of Vectors': num_vectors,
            'Vector Dimension': self.vector_dimension,
            'Matrix Rank': rank,
            'Is Linearly Independent': is_independent,
            'Dependency': 'Independent' if is_independent else 'Dependent',
            'Null Space Dimension': num_vectors - rank
        }
        
        return report
    
    def visualize_2d(self, save_path=None):
        """
        Visualize vectors in 2D space.
        """
        if self.vector_dimension != 2:
            print("Visualization only available for 2D vectors")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_vectors))
        
        for i, (vec, color) in enumerate(zip(self.vectors, colors)):
            ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', 
                     scale=1, color=color, width=0.006, label=f'v{i+1}')
            ax.text(vec[0]*1.1, vec[1]*1.1, f'v{i+1}', fontsize=12, 
                   fontweight='bold', color=color)
        
        # Set axis properties
        max_val = np.max(np.abs(self.matrix)) * 1.5
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        is_independent, _, _ = self.check_independence()
        status = "Linearly Independent" if is_independent else "Linearly Dependent"
        ax.set_title(f'2D Vector Visualization\n{status}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X-axis', fontsize=12)
        ax.set_ylabel('Y-axis', fontsize=12)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_3d(self, save_path=None):
        """
        Visualize vectors in 3D space.
        """
        if self.vector_dimension != 3:
            print("3D visualization only available for 3D vectors")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_vectors))
        
        for i, (vec, color) in enumerate(zip(self.vectors, colors)):
            ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
                     color=color, arrow_length_ratio=0.1, linewidth=2.5,
                     label=f'v{i+1}')
            ax.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, f'v{i+1}', 
                   fontsize=12, fontweight='bold', color=color)
        
        # Set axis properties
        max_val = np.max(np.abs(self.matrix)) * 1.5
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)
        
        is_independent, _, _ = self.check_independence()
        status = "Linearly Independent" if is_independent else "Linearly Dependent"
        ax.set_title(f'3D Vector Visualization\n{status}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X-axis', fontsize=12)
        ax.set_ylabel('Y-axis', fontsize=12)
        ax.set_zlabel('Z-axis', fontsize=12)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_test_cases():
    """
    Create comprehensive test cases with both dependent and independent vectors.
    """
    test_cases = []
    
    # Test Case 1: 2D Independent vectors
    test_cases.append({
        'name': 'Test 1: 2D Independent Vectors',
        'vectors': [np.array([1, 0]), np.array([0, 1])],
        'description': 'Standard basis vectors in R²'
    })
    
    # Test Case 2: 2D Dependent vectors (parallel)
    test_cases.append({
        'name': 'Test 2: 2D Dependent Vectors (Parallel)',
        'vectors': [np.array([2, 4]), np.array([1, 2])],
        'description': 'Second vector is scalar multiple of first'
    })
    
    # Test Case 3: 3D Independent vectors
    test_cases.append({
        'name': 'Test 3: 3D Independent Vectors',
        'vectors': [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])],
        'description': 'Standard basis vectors in R³'
    })
    
    # Test Case 4: 3D Dependent vectors (coplanar)
    test_cases.append({
        'name': 'Test 4: 3D Dependent Vectors (Coplanar)',
        'vectors': [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1, 1, 0])],
        'description': 'Third vector is linear combination of first two'
    })
    
    # Test Case 5: 3D Independent non-standard vectors
    test_cases.append({
        'name': 'Test 5: 3D Independent Non-Standard Vectors',
        'vectors': [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 10])],
        'description': 'General independent vectors in R³'
    })
    
    # Test Case 6: 3D Dependent vectors (linear combination)
    test_cases.append({
        'name': 'Test 6: 3D Dependent Vectors (Linear Combination)',
        'vectors': [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([5, 7, 9])],
        'description': 'Third vector = first + second'
    })
    
    # Test Case 7: 2D Independent non-standard vectors
    test_cases.append({
        'name': 'Test 7: 2D Independent Non-Standard Vectors',
        'vectors': [np.array([3, 4]), np.array([-1, 2])],
        'description': 'Two non-parallel vectors in R²'
    })
    
    # Test Case 8: 4D vectors with 3 independent
    test_cases.append({
        'name': 'Test 8: 4D Space - 3 Independent Vectors',
        'vectors': [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0])],
        'description': 'Three vectors in R⁴ (can add more)'
    })
    
    return test_cases


def run_all_tests():
    """
    Run all test cases and generate comprehensive report.
    """
    test_cases = create_test_cases()
    results = []
    
    print("=" * 80)
    print("LINEAR INDEPENDENCE CHECKER - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"{test['name']}")
        print(f"Description: {test['description']}")
        print(f"{'='*80}")
        
        checker = LinearIndependenceChecker(test['vectors'])
        report = checker.get_report()
        
        print(f"\nMatrix:")
        print(report['Matrix'])
        print(f"\nNumber of Vectors: {report['Number of Vectors']}")
        print(f"Vector Dimension: {report['Vector Dimension']}")
        print(f"Matrix Rank: {report['Matrix Rank']}")
        print(f"Dependency Status: {report['Dependency']}")
        print(f"Null Space Dimension: {report['Null Space Dimension']}")
        
        # Store results for DataFrame
        results.append({
            'Test Case': test['name'],
            'Dimension': report['Vector Dimension'],
            'Num Vectors': report['Number of Vectors'],
            'Rank': report['Matrix Rank'],
            'Status': report['Dependency'],
            'Null Space Dim': report['Null Space Dimension']
        })
        
        # Visualize if 2D or 3D
        if report['Vector Dimension'] == 2:
            checker.visualize_2d()
        elif report['Vector Dimension'] == 3:
            checker.visualize_3d()
    
    # Create summary DataFrame
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Create summary visualization
    create_summary_visualization(df_results)
    
    return df_results


def create_summary_visualization(df_results):
    """
    Create a summary visualization of all test results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Status distribution
    status_counts = df_results['Status'].value_counts()
    axes[0, 0].bar(status_counts.index, status_counts.values, 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7)
    axes[0, 0].set_title('Linear Independence Status Distribution', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Count', fontsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Rank vs Number of Vectors
    x = np.arange(len(df_results))
    width = 0.35
    axes[0, 1].bar(x - width/2, df_results['Num Vectors'], width, 
                   label='Num Vectors', alpha=0.7)
    axes[0, 1].bar(x + width/2, df_results['Rank'], width, 
                   label='Rank', alpha=0.7)
    axes[0, 1].set_title('Number of Vectors vs Matrix Rank', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Test Case', fontsize=10)
    axes[0, 1].set_ylabel('Count', fontsize=10)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'T{i+1}' for i in range(len(df_results))], 
                                rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Dimension distribution
    dim_counts = df_results['Dimension'].value_counts().sort_index()
    axes[1, 0].bar(dim_counts.index, dim_counts.values, 
                   color=sns.color_palette("husl", len(dim_counts)), alpha=0.7)
    axes[1, 0].set_title('Vector Dimension Distribution', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Dimension', fontsize=10)
    axes[1, 0].set_ylabel('Count', fontsize=10)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Null space dimension
    colors = ['#2ecc71' if status == 'Independent' else '#e74c3c' 
              for status in df_results['Status']]
    axes[1, 1].bar(range(len(df_results)), df_results['Null Space Dim'], 
                   color=colors, alpha=0.7)
    axes[1, 1].set_title('Null Space Dimension by Test Case', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Test Case', fontsize=10)
    axes[1, 1].set_ylabel('Null Space Dimension', fontsize=10)
    axes[1, 1].set_xticks(range(len(df_results)))
    axes[1, 1].set_xticklabels([f'T{i+1}' for i in range(len(df_results))], 
                                rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_independence_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


# Run the complete test suite
if __name__ == "__main__":
    results_df = run_all_tests()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETED!")
    print("="*80)
    print(f"\nTotal tests run: {len(results_df)}")
    print(f"Independent vectors: {(results_df['Status'] == 'Independent').sum()}")
    print(f"Dependent vectors: {(results_df['Status'] == 'Dependent').sum()}")