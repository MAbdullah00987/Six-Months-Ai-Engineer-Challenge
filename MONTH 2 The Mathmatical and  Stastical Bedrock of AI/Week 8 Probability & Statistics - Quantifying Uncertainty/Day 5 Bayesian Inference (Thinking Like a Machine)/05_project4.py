
#Confidence Interval Calculator - Build a tool to calculate confidence intervals for sample means

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t, norm
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ConfidenceIntervalCalculator:
    """
    A comprehensive tool to calculate confidence intervals for sample means
    """
    
    def __init__(self):
        self.data = None
        self.mean = None
        self.std = None
        self.n = None
        self.confidence_level = 0.95
        
    def load_data_from_array(self, data):
        """Load data from numpy array or list"""
        self.data = np.array(data)
        self.n = len(self.data)
        self.mean = np.mean(self.data)
        self.std = np.std(self.data, ddof=1)  # Sample standard deviation
        print(f"✓ Data loaded: {self.n} observations")
        
    def load_data_from_file(self, filename):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filename)
            # Assume first column contains the data
            self.data = df.iloc[:, 0].values
            self.n = len(self.data)
            self.mean = np.mean(self.data)
            self.std = np.std(self.data, ddof=1)
            print(f"✓ Data loaded from {filename}: {self.n} observations")
        except Exception as e:
            print(f"Error loading file: {e}")
            
    def set_manual_stats(self, mean, std, n):
        """Set statistics manually without raw data"""
        self.mean = mean
        self.std = std
        self.n = n
        self.data = None
        print(f"✓ Manual statistics set: mean={mean}, std={std}, n={n}")
        
    def calculate_ci(self, confidence_level=None):
        """
        Calculate confidence interval for the mean
        
        Parameters:
        -----------
        confidence_level : float (0 to 1)
            The confidence level (e.g., 0.95 for 95%)
        
        Returns:
        --------
        dict : Dictionary containing CI results
        """
        if confidence_level:
            self.confidence_level = confidence_level
            
        if self.mean is None or self.std is None or self.n is None:
            raise ValueError("Data not loaded. Use load_data or set_manual_stats first.")
        
        # Calculate standard error
        se = self.std / np.sqrt(self.n)
        
        # Degrees of freedom
        df = self.n - 1
        
        # Calculate alpha
        alpha = 1 - self.confidence_level
        
        # Get t-critical value for two-tailed test
        t_critical = t.ppf(1 - alpha/2, df)
        
        # Calculate margin of error
        margin_of_error = t_critical * se
        
        # Calculate confidence interval
        ci_lower = self.mean - margin_of_error
        ci_upper = self.mean + margin_of_error
        
        results = {
            'mean': self.mean,
            'std': self.std,
            'n': self.n,
            'se': se,
            'df': df,
            'confidence_level': self.confidence_level,
            't_critical': t_critical,
            'margin_of_error': margin_of_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        return results
    
    def print_results(self, results):
        """Pretty print the confidence interval results"""
        print("\n" + "="*60)
        print("CONFIDENCE INTERVAL CALCULATION RESULTS")
        print("="*60)
        print(f"\nSample Statistics:")
        print(f"  Sample Mean (x̄):           {results['mean']:.4f}")
        print(f"  Sample Std Dev (s):         {results['std']:.4f}")
        print(f"  Sample Size (n):            {results['n']}")
        print(f"  Standard Error (SE):        {results['se']:.4f}")
        print(f"  Degrees of Freedom (df):    {results['df']}")
        
        print(f"\nConfidence Interval ({results['confidence_level']*100:.0f}%):")
        print(f"  t-critical value:           {results['t_critical']:.4f}")
        print(f"  Margin of Error (MOE):      {results['margin_of_error']:.4f}")
        print(f"  Lower Bound:                {results['ci_lower']:.4f}")
        print(f"  Upper Bound:                {results['ci_upper']:.4f}")
        
        print(f"\nInterpretation:")
        print(f"  We are {results['confidence_level']*100:.0f}% confident that the true")
        print(f"  population mean lies between {results['ci_lower']:.4f}")
        print(f"  and {results['ci_upper']:.4f}")
        print("="*60 + "\n")
        
    def visualize_ci(self, results):
        """Create visualizations for the confidence interval"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Confidence Interval Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distribution with CI
        ax1 = axes[0, 0]
        x_range = np.linspace(results['ci_lower'] - 3*results['se'], 
                             results['ci_upper'] + 3*results['se'], 1000)
        y = t.pdf((x_range - results['mean'])/results['se'], results['df'])
        y = y / results['se']  # Adjust for scale
        
        ax1.plot(x_range, y, 'b-', linewidth=2, label='t-distribution')
        ax1.axvline(results['mean'], color='green', linestyle='--', 
                   linewidth=2, label=f"Mean = {results['mean']:.2f}")
        ax1.axvline(results['ci_lower'], color='red', linestyle='--', 
                   linewidth=2, label=f"Lower = {results['ci_lower']:.2f}")
        ax1.axvline(results['ci_upper'], color='red', linestyle='--', 
                   linewidth=2, label=f"Upper = {results['ci_upper']:.2f}")
        
        # Shade the confidence interval
        x_fill = x_range[(x_range >= results['ci_lower']) & (x_range <= results['ci_upper'])]
        y_fill = t.pdf((x_fill - results['mean'])/results['se'], results['df']) / results['se']
        ax1.fill_between(x_fill, y_fill, alpha=0.3, color='green')
        
        ax1.set_xlabel('Value', fontsize=11)
        ax1.set_ylabel('Probability Density', fontsize=11)
        ax1.set_title('Sampling Distribution with CI', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Data histogram (if raw data available)
        ax2 = axes[0, 1]
        if self.data is not None:
            ax2.hist(self.data, bins=30, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black')
            ax2.axvline(results['mean'], color='red', linestyle='--', 
                       linewidth=2, label=f"Mean = {results['mean']:.2f}")
            ax2.set_xlabel('Value', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Data Distribution', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No raw data available\n(Manual stats used)', 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Data Distribution', fontsize=12, fontweight='bold')
        
        # 3. Confidence Interval Bar
        ax3 = axes[1, 0]
        ax3.barh([0], [results['ci_upper'] - results['ci_lower']], 
                left=results['ci_lower'], height=0.5, color='lightgreen', 
                edgecolor='black', linewidth=2)
        ax3.plot(results['mean'], 0, 'ro', markersize=15, label='Sample Mean')
        ax3.axvline(results['ci_lower'], color='red', linestyle='--', alpha=0.7)
        ax3.axvline(results['ci_upper'], color='red', linestyle='--', alpha=0.7)
        
        # Add text annotations
        ax3.text(results['mean'], -0.3, f"{results['mean']:.2f}", 
                ha='center', fontsize=10, fontweight='bold')
        ax3.text(results['ci_lower'], 0.3, f"{results['ci_lower']:.2f}", 
                ha='center', fontsize=9)
        ax3.text(results['ci_upper'], 0.3, f"{results['ci_upper']:.2f}", 
                ha='center', fontsize=9)
        
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_xlabel('Value', fontsize=11)
        ax3.set_title(f'{results["confidence_level"]*100:.0f}% Confidence Interval', 
                     fontsize=12, fontweight='bold')
        ax3.set_yticks([])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Comparison of different confidence levels
        ax4 = axes[1, 1]
        conf_levels = [0.90, 0.95, 0.99]
        ci_widths = []
        labels = []
        
        for conf in conf_levels:
            alpha = 1 - conf
            t_crit = t.ppf(1 - alpha/2, results['df'])
            moe = t_crit * results['se']
            ci_widths.append(moe * 2)
            labels.append(f"{conf*100:.0f}%")
        
        colors = ['lightblue', 'lightgreen', 'lightsalmon']
        bars = ax4.bar(labels, ci_widths, color=colors, edgecolor='black', linewidth=2)
        
        # Highlight current confidence level
        current_idx = conf_levels.index(results['confidence_level'])
        bars[current_idx].set_edgecolor('red')
        bars[current_idx].set_linewidth(3)
        
        ax4.set_ylabel('CI Width', fontsize=11)
        ax4.set_xlabel('Confidence Level', fontsize=11)
        ax4.set_title('CI Width vs Confidence Level', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, width) in enumerate(zip(bars, ci_widths)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{width:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    def compare_confidence_levels(self):
        """Compare different confidence levels"""
        if self.mean is None:
            raise ValueError("Data not loaded.")
        
        conf_levels = [0.80, 0.85, 0.90, 0.95, 0.99]
        results_list = []
        
        print("\n" + "="*80)
        print("COMPARISON OF DIFFERENT CONFIDENCE LEVELS")
        print("="*80)
        print(f"{'Conf Level':<15} {'t-critical':<15} {'MOE':<15} {'Lower':<15} {'Upper':<15} {'Width':<15}")
        print("-"*80)
        
        for conf in conf_levels:
            result = self.calculate_ci(conf)
            width = result['ci_upper'] - result['ci_lower']
            print(f"{conf*100:>6.0f}%{'':<8} {result['t_critical']:<15.4f} "
                  f"{result['margin_of_error']:<15.4f} {result['ci_lower']:<15.4f} "
                  f"{result['ci_upper']:<15.4f} {width:<15.4f}")
            results_list.append(result)
        
        print("="*80 + "\n")
        return results_list


# Example usage and demo
def demo():
    """Demonstration of the Confidence Interval Calculator"""
    
    print("\n" + "="*60)
    print("CONFIDENCE INTERVAL CALCULATOR - DEMO")
    print("="*60 + "\n")
    
    # Create calculator instance
    calc = ConfidenceIntervalCalculator()
    
    # Example 1: Using generated sample data
    print("Example 1: Random Sample Data")
    print("-" * 40)
    np.random.seed(42)
    sample_data = np.random.normal(loc=75, scale=10, size=30)
    calc.load_data_from_array(sample_data)
    
    results = calc.calculate_ci(confidence_level=0.95)
    calc.print_results(results)
    calc.visualize_ci(results)
    
    # Example 2: Manual statistics
    print("\nExample 2: Manual Statistics Input")
    print("-" * 40)
    calc2 = ConfidenceIntervalCalculator()
    calc2.set_manual_stats(mean=100, std=15, n=50)
    
    results2 = calc2.calculate_ci(confidence_level=0.99)
    calc2.print_results(results2)
    
    # Example 3: Compare confidence levels
    print("\nExample 3: Comparing Confidence Levels")
    print("-" * 40)
    calc.compare_confidence_levels()
    

# Interactive mode
def interactive_mode():
    """Interactive command-line interface"""
    print("\n" + "="*60)
    print("CONFIDENCE INTERVAL CALCULATOR - INTERACTIVE MODE")
    print("="*60 + "\n")
    
    calc = ConfidenceIntervalCalculator()
    
    print("Choose input method:")
    print("1. Enter manual statistics (mean, std, n)")
    print("2. Generate random sample data")
    print("3. Enter data manually")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        mean = float(input("Enter sample mean: "))
        std = float(input("Enter sample standard deviation: "))
        n = int(input("Enter sample size: "))
        calc.set_manual_stats(mean, std, n)
        
    elif choice == '2':
        n = int(input("Enter sample size: "))
        mean = float(input("Enter population mean (for simulation): "))
        std = float(input("Enter population std dev (for simulation): "))
        sample_data = np.random.normal(loc=mean, scale=std, size=n)
        calc.load_data_from_array(sample_data)
        
    elif choice == '3':
        data_str = input("Enter data values (comma-separated): ")
        data = [float(x.strip()) for x in data_str.split(',')]
        calc.load_data_from_array(data)
    
    conf_level = float(input("\nEnter confidence level (e.g., 0.95 for 95%): "))
    
    results = calc.calculate_ci(confidence_level=conf_level)
    calc.print_results(results)
    
    visualize = input("\nShow visualizations? (y/n): ").strip().lower()
    if visualize == 'y':
        calc.visualize_ci(results)
    
    compare = input("\nCompare different confidence levels? (y/n): ").strip().lower()
    if compare == 'y':
        calc.compare_confidence_levels()


if __name__ == "__main__":
    # Run demo
    demo()
    
    # Uncomment below to run interactive mode
    # interactive_mode()