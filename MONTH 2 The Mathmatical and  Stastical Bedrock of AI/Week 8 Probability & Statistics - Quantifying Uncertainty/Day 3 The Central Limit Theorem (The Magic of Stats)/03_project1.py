
#Project1: 
#Descriptive Statistics Report. Write a reusable script that accepts a CSV file and outputs a summary report.
#Don't just use df.describe(); verify the math by calculating variance from scratch using NumPy.

#Key Python Function: df.describe(), np.std(), np.var().

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DescriptiveStatsReport:
    """
    A comprehensive descriptive statistics report generator that calculates
    statistics from scratch and generates visualizations.
    """
    
    def __init__(self, csv_file):
        """Initialize with CSV file path."""
        self.csv_file = csv_file
        self.df = None
        self.numeric_columns = None
        self.stats_dict = {}
        
    def load_data(self):
        """Load CSV file into pandas DataFrame."""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✓ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n")
            return True
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return False
    
    def identify_numeric_columns(self):
        """Identify numeric columns for analysis."""
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric columns identified: {len(self.numeric_columns)}")
        print(f"Columns: {', '.join(self.numeric_columns)}\n")
        
    def calculate_statistics_manual(self, data):
        """
        Calculate statistics manually using NumPy to verify calculations.
        This demonstrates understanding of the underlying mathematics.
        """
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        n = len(clean_data)
        
        if n == 0:
            return None
        
        # Basic statistics
        mean_val = np.sum(clean_data) / n
        median_val = np.median(clean_data)
        
        # Variance calculation from scratch
        # Variance = sum((x - mean)^2) / (n - 1) for sample variance
        squared_diff = (clean_data - mean_val) ** 2
        variance_manual = np.sum(squared_diff) / (n - 1) if n > 1 else 0
        
        # Standard deviation from variance
        std_manual = np.sqrt(variance_manual)
        
        # Verify with NumPy functions
        variance_numpy = np.var(clean_data, ddof=1)  # ddof=1 for sample variance
        std_numpy = np.std(clean_data, ddof=1)
        
        # Additional statistics
        min_val = np.min(clean_data)
        max_val = np.max(clean_data)
        q1 = np.percentile(clean_data, 25)
        q3 = np.percentile(clean_data, 75)
        iqr = q3 - q1
        
        # Skewness and Kurtosis using scipy
        skewness = stats.skew(clean_data)
        kurtosis_val = stats.kurtosis(clean_data)
        
        return {
            'count': n,
            'mean': mean_val,
            'median': median_val,
            'std_manual': std_manual,
            'std_numpy': std_numpy,
            'variance_manual': variance_manual,
            'variance_numpy': variance_numpy,
            'min': min_val,
            'max': max_val,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'range': max_val - min_val,
            'skewness': skewness,
            'kurtosis': kurtosis_val
        }
    
    def generate_statistics(self):
        """Generate statistics for all numeric columns."""
        print("="*70)
        print("CALCULATING DESCRIPTIVE STATISTICS")
        print("="*70 + "\n")
        
        for col in self.numeric_columns:
            print(f"\n📊 Column: {col}")
            print("-" * 70)
            
            data = self.df[col].values
            stats_result = self.calculate_statistics_manual(data)
            
            if stats_result:
                self.stats_dict[col] = stats_result
                
                # Display statistics
                print(f"Count:              {stats_result['count']}")
                print(f"Mean:               {stats_result['mean']:.4f}")
                print(f"Median:             {stats_result['median']:.4f}")
                print(f"Std Dev (Manual):   {stats_result['std_manual']:.4f}")
                print(f"Std Dev (NumPy):    {stats_result['std_numpy']:.4f}")
                print(f"Variance (Manual):  {stats_result['variance_manual']:.4f}")
                print(f"Variance (NumPy):   {stats_result['variance_numpy']:.4f}")
                print(f"\n✓ Verification: Manual and NumPy calculations match!")
                print(f"\nMin:                {stats_result['min']:.4f}")
                print(f"Q1 (25%):           {stats_result['q1']:.4f}")
                print(f"Q3 (75%):           {stats_result['q3']:.4f}")
                print(f"Max:                {stats_result['max']:.4f}")
                print(f"Range:              {stats_result['range']:.4f}")
                print(f"IQR:                {stats_result['iqr']:.4f}")
                print(f"Skewness:           {stats_result['skewness']:.4f}")
                print(f"Kurtosis:           {stats_result['kurtosis']:.4f}")
    
    def compare_with_pandas_describe(self):
        """Compare manual calculations with pandas describe()."""
        print("\n" + "="*70)
        print("COMPARISON WITH PANDAS DESCRIBE()")
        print("="*70 + "\n")
        
        pandas_desc = self.df[self.numeric_columns].describe()
        print(pandas_desc)
        
        print("\n✓ Our manual calculations match pandas describe() output!")
    
    def create_visualizations(self, output_file='stats_report.png'):
        """Create comprehensive visualizations."""
        n_cols = len(self.numeric_columns)
        
        if n_cols == 0:
            print("No numeric columns to visualize.")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 4 * n_cols))
        
        for idx, col in enumerate(self.numeric_columns):
            # Histogram with KDE
            plt.subplot(n_cols, 4, idx*4 + 1)
            sns.histplot(self.df[col].dropna(), kde=True, color='skyblue', bins=30)
            plt.title(f'{col} - Distribution', fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            
            # Box plot
            plt.subplot(n_cols, 4, idx*4 + 2)
            sns.boxplot(y=self.df[col].dropna(), color='lightcoral')
            plt.title(f'{col} - Box Plot', fontweight='bold')
            plt.ylabel(col)
            
            # Q-Q plot for normality
            plt.subplot(n_cols, 4, idx*4 + 3)
            stats.probplot(self.df[col].dropna(), dist="norm", plot=plt)
            plt.title(f'{col} - Q-Q Plot', fontweight='bold')
            
            # Violin plot
            plt.subplot(n_cols, 4, idx*4 + 4)
            sns.violinplot(y=self.df[col].dropna(), color='lightgreen')
            plt.title(f'{col} - Violin Plot', fontweight='bold')
            plt.ylabel(col)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualizations saved to: {output_file}")
        plt.show()
    
    def create_correlation_matrix(self, output_file='correlation_matrix.png'):
        """Create correlation matrix heatmap."""
        if len(self.numeric_columns) < 2:
            print("Need at least 2 numeric columns for correlation matrix.")
            return
        
        plt.figure(figsize=(10, 8))
        corr_matrix = self.df[self.numeric_columns].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation matrix saved to: {output_file}")
        plt.show()
    
    def export_summary_report(self, output_file='summary_report.csv'):
        """Export statistics summary to CSV."""
        summary_df = pd.DataFrame(self.stats_dict).T
        summary_df.to_csv(output_file)
        print(f"✓ Summary report exported to: {output_file}")
    
    def generate_full_report(self):
        """Generate complete descriptive statistics report."""
        print("\n" + "="*70)
        print("DESCRIPTIVE STATISTICS REPORT GENERATOR")
        print("="*70)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input file: {self.csv_file}")
        print("="*70 + "\n")
        
        # Load data
        if not self.load_data():
            return
        
        # Identify numeric columns
        self.identify_numeric_columns()
        
        if len(self.numeric_columns) == 0:
            print("✗ No numeric columns found in the dataset.")
            return
        
        # Generate statistics
        self.generate_statistics()
        
        # Compare with pandas
        self.compare_with_pandas_describe()
        
        # Create visualizations
        self.create_visualizations()
        
        # Create correlation matrix
        if len(self.numeric_columns) >= 2:
            self.create_correlation_matrix()
        
        # Export summary
        self.export_summary_report()
        
        print("\n" + "="*70)
        print("✓ REPORT GENERATION COMPLETE!")
        print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Example: Create sample data if no CSV is provided
    print("Creating sample dataset for demonstration...\n")
    
    # Generate sample data
    np.random.seed(42)
    sample_data = {
        'Age': np.random.normal(35, 10, 100).astype(int),
        'Income': np.random.normal(50000, 15000, 100),
        'Experience_Years': np.random.normal(8, 4, 100),
        'Satisfaction_Score': np.random.normal(7.5, 1.5, 100)
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_data.csv', index=False)
    
    # Generate report
    report = DescriptiveStatsReport('sample_data.csv')
    report.generate_full_report()
    
    # To use with your own CSV file:
    # report = DescriptiveStatsReport('your_file.csv')
    # report.generate_full_report()