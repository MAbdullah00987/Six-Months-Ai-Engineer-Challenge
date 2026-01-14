
#PROJECT: Complete Descriptive Statistics Report Generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DescriptiveStatsAnalyzer:
    """
    Comprehensive statistical analyzer that calculates everything from scratch
    and compares with library functions for verification.
    """
    
    def __init__(self, csv_file=None, data=None):
        """Load data from CSV or accept DataFrame directly"""
        if csv_file:
            self.df = pd.read_csv(csv_file)
        elif data is not None:
            self.df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        else:
            raise ValueError("Provide either csv_file or data")
        
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.results = {}
    
    # ============ MANUAL CALCULATIONS ============
    
    def calculate_mean_manual(self, column):
        """Calculate mean from scratch"""
        data = self.df[column].dropna()
        return sum(data) / len(data)
    
    def calculate_median_manual(self, column):
        """Calculate median from scratch"""
        data = sorted(self.df[column].dropna())
        n = len(data)
        if n % 2 == 0:
            return (data[n//2 - 1] + data[n//2]) / 2
        return data[n//2]
    
    def calculate_variance_manual(self, column, ddof=0):
        """
        Calculate variance from scratch
        ddof=0 for population variance, ddof=1 for sample variance
        """
        data = self.df[column].dropna()
        mean = sum(data) / len(data)
        squared_diffs = [(x - mean)**2 for x in data]
        return sum(squared_diffs) / (len(data) - ddof)
    
    def calculate_std_manual(self, column, ddof=0):
        """Calculate standard deviation from scratch"""
        return self.calculate_variance_manual(column, ddof) ** 0.5
    
    def calculate_quartiles_manual(self, column):
        """Calculate Q1, Q2 (median), Q3 manually"""
        data = sorted(self.df[column].dropna())
        n = len(data)
        
        q1 = data[int(n * 0.25)]
        q2 = data[int(n * 0.50)] if n % 2 == 1 else (data[n//2 - 1] + data[n//2]) / 2
        q3 = data[int(n * 0.75)]
        
        return q1, q2, q3
    
    def calculate_skewness_manual(self, column):
        """Calculate skewness from scratch"""
        data = self.df[column].dropna()
        mean = sum(data) / len(data)
        std = self.calculate_std_manual(column, ddof=1)
        n = len(data)
        
        skew_sum = sum(((x - mean) / std) ** 3 for x in data)
        return (n / ((n-1) * (n-2))) * skew_sum
    
    def calculate_kurtosis_manual(self, column):
        """Calculate kurtosis from scratch"""
        data = self.df[column].dropna()
        mean = sum(data) / len(data)
        std = self.calculate_std_manual(column, ddof=1)
        n = len(data)
        
        kurt_sum = sum(((x - mean) / std) ** 4 for x in data)
        return ((n * (n+1)) / ((n-1) * (n-2) * (n-3))) * kurt_sum - (3 * (n-1)**2) / ((n-2) * (n-3))
    
    # ============ COMPREHENSIVE ANALYSIS ============
    
    def analyze_column(self, column):
        """Perform complete statistical analysis on a column"""
        data = self.df[column].dropna()
        
        # Manual calculations
        mean_manual = self.calculate_mean_manual(column)
        median_manual = self.calculate_median_manual(column)
        var_manual = self.calculate_variance_manual(column, ddof=1)
        std_manual = self.calculate_std_manual(column, ddof=1)
        q1, q2, q3 = self.calculate_quartiles_manual(column)
        iqr = q3 - q1
        
        # Library calculations for verification
        mean_np = np.mean(data)
        median_np = np.median(data)
        var_np = np.var(data, ddof=1)
        std_np = np.std(data, ddof=1)
        q1_np = np.percentile(data, 25)
        q3_np = np.percentile(data, 75)
        
        # Additional stats
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        
        results = {
            'count': len(data),
            'mean_manual': mean_manual,
            'mean_numpy': mean_np,
            'mean_match': np.isclose(mean_manual, mean_np),
            'median_manual': median_manual,
            'median_numpy': median_np,
            'median_match': np.isclose(median_manual, median_np),
            'variance_manual': var_manual,
            'variance_numpy': var_np,
            'variance_match': np.isclose(var_manual, var_np),
            'std_manual': std_manual,
            'std_numpy': std_np,
            'std_match': np.isclose(std_manual, std_np),
            'min': float(data.min()),
            'max': float(data.max()),
            'range': float(data.max() - data.min()),
            'q1': q1,
            'q2': q2,
            'q3': q3,
            'iqr': iqr,
            'skewness': skew,
            'kurtosis': kurt
        }
        
        return results
    
    def generate_report(self):
        """Generate comprehensive report for all numeric columns"""
        print("=" * 80)
        print("DESCRIPTIVE STATISTICS REPORT".center(80))
        print("=" * 80)
        print(f"\nDataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Numeric Columns: {len(self.numeric_cols)}")
        print("\n" + "=" * 80)
        
        for col in self.numeric_cols:
            print(f"\n📊 COLUMN: {col}")
            print("-" * 80)
            
            results = self.analyze_column(col)
            self.results[col] = results
            
            print(f"Count: {results['count']}")
            print(f"\nCENTRAL TENDENCY:")
            print(f"  Mean (Manual):    {results['mean_manual']:.4f}")
            print(f"  Mean (NumPy):     {results['mean_numpy']:.4f}")
            print(f"  ✓ Match: {results['mean_match']}")
            print(f"  Median (Manual):  {results['median_manual']:.4f}")
            print(f"  Median (NumPy):   {results['median_numpy']:.4f}")
            print(f"  ✓ Match: {results['median_match']}")
            
            print(f"\nSPREAD:")
            print(f"  Variance (Manual): {results['variance_manual']:.4f}")
            print(f"  Variance (NumPy):  {results['variance_numpy']:.4f}")
            print(f"  ✓ Match: {results['variance_match']}")
            print(f"  Std Dev (Manual):  {results['std_manual']:.4f}")
            print(f"  Std Dev (NumPy):   {results['std_numpy']:.4f}")
            print(f"  ✓ Match: {results['std_match']}")
            print(f"  IQR:               {results['iqr']:.4f}")
            
            print(f"\nRANGE:")
            print(f"  Min:    {results['min']:.4f}")
            print(f"  Q1:     {results['q1']:.4f}")
            print(f"  Median: {results['q2']:.4f}")
            print(f"  Q3:     {results['q3']:.4f}")
            print(f"  Max:    {results['max']:.4f}")
            print(f"  Range:  {results['range']:.4f}")
            
            print(f"\nSHAPE:")
            print(f"  Skewness: {results['skewness']:.4f}", end="")
            if results['skewness'] > 0.5:
                print(" (Right-skewed)")
            elif results['skewness'] < -0.5:
                print(" (Left-skewed)")
            else:
                print(" (Roughly symmetric)")
            
            print(f"  Kurtosis: {results['kurtosis']:.4f}", end="")
            if results['kurtosis'] > 0:
                print(" (Heavy-tailed)")
            else:
                print(" (Light-tailed)")
            
            print("\n" + "=" * 80)
        
        return self.results
    
    def visualize_distributions(self):
        """Create comprehensive visualizations"""
        n_cols = len(self.numeric_cols)
        fig, axes = plt.subplots(n_cols, 3, figsize=(18, 5*n_cols))
        
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(self.numeric_cols):
            data = self.df[col].dropna()
            results = self.results[col]
            
            # Histogram with KDE
            axes[idx, 0].hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7, density=True)
            axes[idx, 0].axvline(results['mean_manual'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['mean_manual']:.2f}")
            axes[idx, 0].axvline(results['median_manual'], color='green', linestyle='--', linewidth=2, label=f"Median: {results['median_manual']:.2f}")
            
            # Add KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            axes[idx, 0].plot(x_range, kde(x_range), 'r-', linewidth=2, alpha=0.6)
            
            axes[idx, 0].set_title(f'{col} - Distribution', fontweight='bold')
            axes[idx, 0].set_xlabel('Value')
            axes[idx, 0].set_ylabel('Density')
            axes[idx, 0].legend()
            axes[idx, 0].grid(alpha=0.3)
            
            # Boxplot
            box = axes[idx, 1].boxplot(data, vert=True, patch_artist=True,
                                        boxprops=dict(facecolor='lightgreen', alpha=0.7),
                                        medianprops=dict(color='red', linewidth=2))
            axes[idx, 1].set_title(f'{col} - Boxplot (IQR={results["iqr"]:.2f})', fontweight='bold')
            axes[idx, 1].set_ylabel('Value')
            axes[idx, 1].grid(alpha=0.3)
            
            # Q-Q plot for normality
            stats.probplot(data, dist="norm", plot=axes[idx, 2])
            axes[idx, 2].set_title(f'{col} - Q-Q Plot (Normality Check)', fontweight='bold')
            axes[idx, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ============ DEMO WITH SAMPLE DATA ============

# Create sample dataset
np.random.seed(42)
sample_data = pd.DataFrame({
    'Normal_Distribution': np.random.normal(100, 15, 1000),
    'With_Outliers': np.concatenate([np.random.normal(50, 10, 950), 
                                      np.random.normal(200, 5, 50)]),
    'Skewed_Right': np.random.exponential(20, 1000)
})

# Run analysis
analyzer = DescriptiveStatsAnalyzer(data=sample_data)
results = analyzer.generate_report()
analyzer.visualize_distributions()