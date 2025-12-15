
#Project 1: Matrix Multiplication from Scratch 
#* Implement matrix_multiply() with proper dimension checking
#* Test with identity matrix, zero matrix, and random matrices
#* Verify associativity: (AB)C = A(BC)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple
import time

class MatrixMultiplier:
    """
    A comprehensive matrix multiplication implementation with visualization
    and performance analysis capabilities.
    """
    
    def __init__(self):
        self.operation_history = []
        sns.set_style("whitegrid")
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multiply two matrices from scratch with dimension checking.
        
        Parameters:
        -----------
        A : np.ndarray
            First matrix of shape (m, n)
        B : np.ndarray
            Second matrix of shape (n, p)
            
        Returns:
        --------
        np.ndarray
            Result matrix of shape (m, p)
            
        Raises:
        -------
        ValueError: If dimensions are incompatible
        """
        # Ensure inputs are numpy arrays
        A = np.array(A)
        B = np.array(B)
        
        # Dimension checking
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"Both inputs must be 2D matrices. Got shapes: {A.shape}, {B.shape}")
        
        m, n = A.shape
        n_b, p = B.shape
        
        if n != n_b:
            raise ValueError(
                f"Incompatible dimensions: A has {n} columns but B has {n_b} rows. "
                f"For matrix multiplication A×B, number of columns in A must equal number of rows in B."
            )
        
        # Initialize result matrix
        result = np.zeros((m, p))
        
        # Perform multiplication using triple nested loop
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i, j] += A[i, k] * B[k, j]
        
        return result
    
    def verify_associativity(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                            tolerance: float = 1e-10) -> Tuple[bool, dict]:
        """
        Verify associativity property: (AB)C = A(BC)
        
        Returns:
        --------
        tuple: (is_associative, results_dict)
        """
        print("\n" + "="*60)
        print("VERIFYING ASSOCIATIVITY: (AB)C = A(BC)")
        print("="*60)
        
        # Calculate (AB)C
        start = time.time()
        AB = self.matrix_multiply(A, B)
        AB_C = self.matrix_multiply(AB, C)
        time_left = time.time() - start
        
        # Calculate A(BC)
        start = time.time()
        BC = self.matrix_multiply(B, C)
        A_BC = self.matrix_multiply(A, BC)
        time_right = time.time() - start
        
        # Check if results are equal within tolerance
        is_equal = np.allclose(AB_C, A_BC, atol=tolerance)
        max_diff = np.max(np.abs(AB_C - A_BC))
        
        results = {
            'AB_C': AB_C,
            'A_BC': A_BC,
            'is_associative': is_equal,
            'max_difference': max_diff,
            'time_left_assoc': time_left,
            'time_right_assoc': time_right
        }
        
        print(f"\n(AB)C shape: {AB_C.shape}")
        print(f"A(BC) shape: {A_BC.shape}")
        print(f"Maximum difference: {max_diff:.2e}")
        print(f"Time for (AB)C: {time_left:.4f}s")
        print(f"Time for A(BC): {time_right:.4f}s")
        print(f"✓ Associativity verified!" if is_equal else "✗ Associativity failed!")
        
        return is_equal, results
    
    def test_identity_matrix(self, size: int = 4) -> None:
        """Test multiplication with identity matrix: AI = IA = A"""
        print("\n" + "="*60)
        print("TEST 1: IDENTITY MATRIX")
        print("="*60)
        
        A = np.random.randint(1, 10, size=(size, size))
        I = np.eye(size)
        
        AI = self.matrix_multiply(A, I)
        IA = self.matrix_multiply(I, A)
        
        print(f"\nOriginal matrix A:\n{A}")
        print(f"\nIdentity matrix I:\n{I}")
        print(f"\nA × I:\n{AI}")
        print(f"\nI × A:\n{IA}")
        
        test_passed = np.allclose(A, AI) and np.allclose(A, IA)
        print(f"\n{'✓ Test PASSED' if test_passed else '✗ Test FAILED'}: AI = IA = A")
        
        self.operation_history.append({
            'test': 'Identity Matrix',
            'passed': test_passed,
            'matrix_size': size
        })
    
    def test_zero_matrix(self, size: int = 4) -> None:
        """Test multiplication with zero matrix: A0 = 0A = 0"""
        print("\n" + "="*60)
        print("TEST 2: ZERO MATRIX")
        print("="*60)
        
        A = np.random.randint(1, 10, size=(size, size))
        Z = np.zeros((size, size))
        
        AZ = self.matrix_multiply(A, Z)
        ZA = self.matrix_multiply(Z, A)
        
        print(f"\nOriginal matrix A:\n{A}")
        print(f"\nZero matrix Z:\n{Z}")
        print(f"\nA × Z:\n{AZ}")
        print(f"\nZ × A:\n{ZA}")
        
        test_passed = np.allclose(AZ, Z) and np.allclose(ZA, Z)
        print(f"\n{'✓ Test PASSED' if test_passed else '✗ Test FAILED'}: AZ = ZA = Z")
        
        self.operation_history.append({
            'test': 'Zero Matrix',
            'passed': test_passed,
            'matrix_size': size
        })
    
    def test_random_matrices(self, sizes: list = [(3, 4), (4, 5), (5, 3)]) -> None:
        """Test with random matrices and verify against NumPy"""
        print("\n" + "="*60)
        print("TEST 3: RANDOM MATRICES")
        print("="*60)
        
        A = np.random.randint(-10, 10, size=sizes[0])
        B = np.random.randint(-10, 10, size=sizes[1])
        C = np.random.randint(-10, 10, size=sizes[2])
        
        print(f"\nMatrix A shape: {A.shape}")
        print(f"Matrix B shape: {B.shape}")
        print(f"Matrix C shape: {C.shape}")
        
        # Our implementation
        start = time.time()
        result_custom = self.matrix_multiply(A, B)
        time_custom = time.time() - start
        
        # NumPy's implementation
        start = time.time()
        result_numpy = np.matmul(A, B)
        time_numpy = time.time() - start
        
        test_passed = np.allclose(result_custom, result_numpy)
        
        print(f"\nCustom implementation time: {time_custom:.6f}s")
        print(f"NumPy implementation time: {time_numpy:.6f}s")
        print(f"Speed ratio (Custom/NumPy): {time_custom/time_numpy:.2f}x")
        print(f"\n{'✓ Test PASSED' if test_passed else '✗ Test FAILED'}: Results match NumPy")
        
        self.operation_history.append({
            'test': 'Random Matrices',
            'passed': test_passed,
            'matrix_size': f"{A.shape}×{B.shape}",
            'time_custom': time_custom,
            'time_numpy': time_numpy
        })
        
        # Verify associativity
        self.verify_associativity(A, B, C)
    
    def performance_comparison(self, max_size: int = 200, step: int = 20) -> pd.DataFrame:
        """
        Compare performance of custom implementation vs NumPy
        across different matrix sizes.
        """
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        sizes = range(10, max_size + 1, step)
        results = []
        
        for size in sizes:
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            # Custom implementation
            start = time.time()
            _ = self.matrix_multiply(A, B)
            time_custom = time.time() - start
            
            # NumPy implementation
            start = time.time()
            _ = np.matmul(A, B)
            time_numpy = time.time() - start
            
            results.append({
                'Matrix Size': f"{size}×{size}",
                'Size': size,
                'Custom (s)': time_custom,
                'NumPy (s)': time_numpy,
                'Ratio': time_custom / time_numpy
            })
            
            print(f"Size {size}×{size}: Custom={time_custom:.4f}s, NumPy={time_numpy:.6f}s, Ratio={time_custom/time_numpy:.1f}x")
        
        return pd.DataFrame(results)
    
    def visualize_performance(self, df: pd.DataFrame) -> None:
        """Create comprehensive visualizations of performance data"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Execution Time Comparison
        ax1 = axes[0, 0]
        ax1.plot(df['Size'], df['Custom (s)'], marker='o', label='Custom Implementation', linewidth=2)
        ax1.plot(df['Size'], df['NumPy (s)'], marker='s', label='NumPy', linewidth=2)
        ax1.set_xlabel('Matrix Size', fontsize=12)
        ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax1.set_title('Execution Time: Custom vs NumPy', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speed Ratio
        ax2 = axes[0, 1]
        ax2.plot(df['Size'], df['Ratio'], marker='D', color='red', linewidth=2)
        ax2.set_xlabel('Matrix Size', fontsize=12)
        ax2.set_ylabel('Speed Ratio (Custom/NumPy)', fontsize=12)
        ax2.set_title('Performance Ratio: How Much Slower is Custom?', fontsize=14, fontweight='bold')
        ax2.axhline(y=1, color='green', linestyle='--', label='Equal Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Log Scale Comparison
        ax3 = axes[1, 0]
        ax3.semilogy(df['Size'], df['Custom (s)'], marker='o', label='Custom Implementation')
        ax3.semilogy(df['Size'], df['NumPy (s)'], marker='s', label='NumPy')
        ax3.set_xlabel('Matrix Size', fontsize=12)
        ax3.set_ylabel('Execution Time (seconds, log scale)', fontsize=12)
        ax3.set_title('Log Scale Performance Comparison', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Table Heatmap
        ax4 = axes[1, 1]
        table_data = df[['Size', 'Custom (s)', 'NumPy (s)', 'Ratio']].tail(10)
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=table_data.values,
                         colLabels=table_data.columns,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.3, 0.3, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color code the header
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Performance Summary (Last 10 Sizes)', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_matrix_multiplication(self, A: np.ndarray, B: np.ndarray) -> None:
        """Visualize matrix multiplication process with heatmaps"""
        result = self.matrix_multiply(A, B)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Matrix A
        sns.heatmap(A, annot=True, fmt='.2f', cmap='Blues', ax=axes[0], 
                    cbar_kws={'label': 'Value'}, linewidths=0.5)
        axes[0].set_title(f'Matrix A ({A.shape[0]}×{A.shape[1]})', fontsize=14, fontweight='bold')
        
        # Matrix B
        sns.heatmap(B, annot=True, fmt='.2f', cmap='Greens', ax=axes[1], 
                    cbar_kws={'label': 'Value'}, linewidths=0.5)
        axes[1].set_title(f'Matrix B ({B.shape[0]}×{B.shape[1]})', fontsize=14, fontweight='bold')
        
        # Multiplication symbol
        axes[2].text(0.5, 0.5, '×', fontsize=80, ha='center', va='center', fontweight='bold')
        axes[2].axis('off')
        axes[2].set_title('Operation', fontsize=14, fontweight='bold')
        
        # Result matrix
        sns.heatmap(result, annot=True, fmt='.2f', cmap='Reds', ax=axes[3], 
                    cbar_kws={'label': 'Value'}, linewidths=0.5)
        axes[3].set_title(f'Result ({result.shape[0]}×{result.shape[1]})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> pd.DataFrame:
        """Generate a summary report of all tests"""
        if not self.operation_history:
            print("No operations recorded yet.")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.operation_history)
        
        print("\n" + "="*60)
        print("TEST SUMMARY REPORT")
        print("="*60)
        print(df.to_string(index=False))
        print(f"\nTotal Tests: {len(df)}")
        print(f"Tests Passed: {df['passed'].sum()}")
        print(f"Tests Failed: {(~df['passed']).sum()}")
        print(f"Success Rate: {df['passed'].mean()*100:.1f}%")
        
        return df


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("MATRIX MULTIPLICATION PROJECT")
    print("="*60)
    
    # Initialize the multiplier
    multiplier = MatrixMultiplier()
    
    # Test 1: Identity Matrix
    multiplier.test_identity_matrix(size=4)
    
    # Test 2: Zero Matrix
    multiplier.test_zero_matrix(size=4)
    
    # Test 3: Random Matrices with Associativity Verification
    multiplier.test_random_matrices(sizes=[(3, 4), (4, 5), (5, 3)])
    
    # Visualize a small matrix multiplication
    print("\n" + "="*60)
    print("VISUALIZATION: MATRIX MULTIPLICATION")
    print("="*60)
    A_small = np.random.randint(1, 10, size=(3, 4))
    B_small = np.random.randint(1, 10, size=(4, 3))
    multiplier.visualize_matrix_multiplication(A_small, B_small)
    
    # Performance comparison
    print("\nRunning performance comparison (this may take a moment)...")
    perf_df = multiplier.performance_comparison(max_size=100, step=10)
    
    # Visualize performance
    multiplier.visualize_performance(perf_df)
    
    # Generate final report
    report = multiplier.generate_report()
    
    