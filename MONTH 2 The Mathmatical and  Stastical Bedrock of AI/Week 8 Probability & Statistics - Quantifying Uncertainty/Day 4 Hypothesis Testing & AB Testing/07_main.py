

#Project 1: Descriptive Statistics Report - Create a comprehensive script that generates statistical summaries for datasets.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, skew, kurtosis
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

class StatisticalReportGenerator:
    """
    A comprehensive statistical analysis and reporting tool for datasets.
    Generates descriptive statistics, visualizations, and statistical tests.
    """
    
    def __init__(self, data, output_prefix='statistical_report'):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        data : pandas.DataFrame or str
            DataFrame or path to CSV file
        output_prefix : str
            Prefix for output files
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
            print(f"✓ Data loaded from {data}")
        else:
            self.df = data.copy()
        
        self.output_prefix = output_prefix
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"✓ Dataset shape: {self.df.shape}")
        print(f"✓ Numeric columns: {len(self.numeric_cols)}")
        print(f"✓ Categorical columns: {len(self.categorical_cols)}")
    
    def basic_info(self):
        """Generate basic dataset information."""
        print("\n" + "="*70)
        print("BASIC DATASET INFORMATION")
        print("="*70)
        
        print(f"\nDataset Dimensions: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n--- Column Data Types ---")
        print(self.df.dtypes.value_counts())
        
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        if missing.sum() == 0:
            print("No missing values found!")
        
        print("\n--- Duplicate Rows ---")
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
    
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics."""
        print("\n" + "="*70)
        print("DESCRIPTIVE STATISTICS - NUMERIC VARIABLES")
        print("="*70)
        
        if not self.numeric_cols:
            print("No numeric columns found!")
            return None
        
        # Basic descriptive stats
        desc_stats = self.df[self.numeric_cols].describe().T
        
        # Additional statistics
        desc_stats['variance'] = self.df[self.numeric_cols].var()
        desc_stats['skewness'] = self.df[self.numeric_cols].skew()
        desc_stats['kurtosis'] = self.df[self.numeric_cols].kurtosis()
        desc_stats['range'] = desc_stats['max'] - desc_stats['min']
        desc_stats['iqr'] = desc_stats['75%'] - desc_stats['25%']
        desc_stats['cv'] = (desc_stats['std'] / desc_stats['mean']) * 100  # Coefficient of variation
        
        print("\n", desc_stats.round(3))
        
        # Save to CSV
        desc_stats.to_csv(f'{self.output_prefix}_descriptive_stats.csv')
        print(f"\n✓ Saved to '{self.output_prefix}_descriptive_stats.csv'")
        
        return desc_stats
    
    def categorical_summary(self):
        """Generate summary for categorical variables."""
        print("\n" + "="*70)
        print("CATEGORICAL VARIABLES SUMMARY")
        print("="*70)
        
        if not self.categorical_cols:
            print("No categorical columns found!")
            return
        
        for col in self.categorical_cols:
            print(f"\n--- {col} ---")
            value_counts = self.df[col].value_counts()
            value_pcts = self.df[col].value_counts(normalize=True) * 100
            
            summary = pd.DataFrame({
                'Count': value_counts,
                'Percentage': value_pcts.round(2)
            })
            
            print(f"Unique values: {self.df[col].nunique()}")
            print(f"Mode: {self.df[col].mode().values[0] if len(self.df[col].mode()) > 0 else 'N/A'}")
            print("\nTop 10 categories:")
            print(summary.head(10))
    
    def normality_tests(self):
        """Test normality of numeric variables."""
        print("\n" + "="*70)
        print("NORMALITY TESTS (D'Agostino-Pearson)")
        print("="*70)
        
        if not self.numeric_cols:
            return
        
        normality_results = []
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            if len(data) >= 8:  # Minimum sample size for test
                stat, p_value = normaltest(data)
                is_normal = "Yes" if p_value > 0.05 else "No"
                normality_results.append({
                    'Variable': col,
                    'Statistic': stat,
                    'P-value': p_value,
                    'Normal (α=0.05)': is_normal
                })
        
        norm_df = pd.DataFrame(normality_results)
        print("\n", norm_df.to_string(index=False))
        print("\nNote: P-value > 0.05 suggests data is normally distributed")
    
    def outlier_detection(self):
        """Detect outliers using IQR method."""
        print("\n" + "="*70)
        print("OUTLIER DETECTION (IQR Method)")
        print("="*70)
        
        if not self.numeric_cols:
            return None
        
        outlier_summary = []
        
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(self.df)) * 100
            
            outlier_summary.append({
                'Variable': col,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound,
                'Outlier Count': outlier_count,
                'Outlier %': outlier_pct
            })
        
        outlier_df = pd.DataFrame(outlier_summary)
        print("\n", outlier_df.round(3).to_string(index=False))
        
        return outlier_df
    
    def correlation_analysis(self):
        """Analyze correlations between numeric variables."""
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)
        
        if len(self.numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation analysis!")
            return None
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))
        
        # Find strong correlations (excluding diagonal)
        print("\n--- Strong Correlations (|r| > 0.7) ---")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    strong_corr.append({
                        'Var1': corr_matrix.columns[i],
                        'Var2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if strong_corr:
            print(pd.DataFrame(strong_corr).to_string(index=False))
        else:
            print("No strong correlations found.")
        
        return corr_matrix
    
    def create_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Distribution plots for numeric variables
        if self.numeric_cols:
            n_cols = min(3, len(self.numeric_cols))
            n_rows = (len(self.numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            
            for idx, col in enumerate(self.numeric_cols):
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[idx])
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            
            # Hide extra subplots
            for idx in range(len(self.numeric_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_prefix}_distributions.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved distributions plot")
            plt.show()  # Display the plot
            plt.close()
        
        # 2. Box plots for outlier visualization
        if self.numeric_cols:
            n_cols = min(3, len(self.numeric_cols))
            n_rows = (len(self.numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            
            for idx, col in enumerate(self.numeric_cols):
                sns.boxplot(data=self.df, y=col, ax=axes[idx])
                axes[idx].set_title(f'Box Plot of {col}')
            
            for idx in range(len(self.numeric_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_prefix}_boxplots.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved box plots")
            plt.show()  # Display the plot
            plt.close()
        
        # 3. Correlation heatmap
        if len(self.numeric_cols) >= 2:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.df[self.numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1)
            plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.output_prefix}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved correlation heatmap")
            plt.show()  # Display the plot
            plt.close()
        
        # 4. Categorical bar plots (top 10 categories)
        if self.categorical_cols:
            for col in self.categorical_cols[:5]:  # Limit to first 5 categorical columns
                plt.figure(figsize=(12, 6))
                top_categories = self.df[col].value_counts().head(10)
                sns.barplot(x=top_categories.values, y=top_categories.index)
                plt.title(f'Top 10 Categories in {col}', fontsize=14, fontweight='bold')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.tight_layout()
                plt.savefig(f'{self.output_prefix}_categorical_{col}.png', dpi=300, bbox_inches='tight')
                plt.show()  # Display the plot
                plt.close()
            
            print(f"✓ Saved categorical plots")
        
        # 5. Q-Q plots for normality assessment
        if self.numeric_cols:
            n_cols = min(3, len(self.numeric_cols))
            n_rows = (len(self.numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            
            for idx, col in enumerate(self.numeric_cols):
                sm.qqplot(self.df[col].dropna(), line='s', ax=axes[idx])
                axes[idx].set_title(f'Q-Q Plot: {col}')
            
            for idx in range(len(self.numeric_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_prefix}_qq_plots.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved Q-Q plots")
            plt.show()  # Display the plot
            plt.close()
        
        print(f"\n✓ All visualizations saved with prefix '{self.output_prefix}'")
    
    def generate_full_report(self):
        """Generate complete statistical report."""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE STATISTICAL REPORT")
        print("="*70)
        
        self.basic_info()
        self.descriptive_statistics()
        self.categorical_summary()
        self.normality_tests()
        self.outlier_detection()
        self.correlation_analysis()
        self.create_visualizations()
        
        print("\n" + "="*70)
        print("REPORT GENERATION COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved with prefix: '{self.output_prefix}'")


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("STATISTICAL REPORT GENERATOR")
    print("="*70)
    
    # Example 1: Create sample dataset
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Age': np.random.normal(35, 10, 1000).clip(18, 80),
        'Income': np.random.lognormal(10.5, 0.5, 1000),
        'Experience': np.random.normal(8, 4, 1000).clip(0, 40),
        'Satisfaction': np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.1, 0.15, 0.3, 0.3, 0.15]),
        'Department': np.random.choice(['Sales', 'IT', 'HR', 'Finance', 'Marketing'], 1000),
        'Performance': np.random.normal(75, 15, 1000).clip(0, 100)
    })
    
    print("\nUsing sample dataset for demonstration...")
    
    # Create report generator
    report = StatisticalReportGenerator(sample_data, output_prefix='demo_report')
    
    # Generate full report
    report.generate_full_report()
    
    print("\n" + "="*70)
    print("TO USE WITH YOUR OWN DATA:")
    print("="*70)
    print("""
    # Load your CSV file:
    report = StatisticalReportGenerator('your_data.csv', output_prefix='my_analysis')
    report.generate_full_report()
    
    # Or use a DataFrame:
    df = pd.read_csv('your_data.csv')
    report = StatisticalReportGenerator(df, output_prefix='my_analysis')
    report.generate_full_report()
    
    # Generate specific sections:
    report.basic_info()
    report.descriptive_statistics()
    report.correlation_analysis()
    report.create_visualizations()
    """)