
#Project 1:
#A/B Test Analyzer - Create a script to analyze A/B test results and determine statistical significance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

class ABTestAnalyzer:
    """
    A comprehensive A/B Test analyzer for statistical significance testing
    """
    
    def __init__(self, alpha=0.05):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        alpha : float
            Significance level (default: 0.05 for 95% confidence)
        """
        self.alpha = alpha
        self.results = {}
        
    def analyze_conversion_rate(self, control_conversions, control_total, 
                                variant_conversions, variant_total, 
                                test_name="Conversion Rate Test"):
        """
        Analyze conversion rates between control and variant groups
        
        Parameters:
        -----------
        control_conversions : int
            Number of conversions in control group
        control_total : int
            Total size of control group
        variant_conversions : int
            Number of conversions in variant group
        variant_total : int
            Total size of variant group
        test_name : str
            Name of the test
        """
        print(f"\n{'='*70}")
        print(f"A/B TEST ANALYSIS: {test_name}")
        print(f"{'='*70}\n")
        
        # Calculate conversion rates
        control_rate = control_conversions / control_total
        variant_rate = variant_conversions / variant_total
        
        # Relative improvement
        relative_improvement = ((variant_rate - control_rate) / control_rate) * 100
        absolute_improvement = (variant_rate - control_rate) * 100
        
        print("CONVERSION RATES:")
        print(f"  Control:  {control_conversions}/{control_total} = {control_rate:.4f} ({control_rate*100:.2f}%)")
        print(f"  Variant:  {variant_conversions}/{variant_total} = {variant_rate:.4f} ({variant_rate*100:.2f}%)")
        print(f"  Absolute Improvement: {absolute_improvement:+.2f} percentage points")
        print(f"  Relative Improvement: {relative_improvement:+.2f}%\n")
        
        # Two-proportion z-test
        count = np.array([variant_conversions, control_conversions])
        nobs = np.array([variant_total, control_total])
        z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
        
        # Calculate confidence interval for the difference
        se_control = np.sqrt(control_rate * (1 - control_rate) / control_total)
        se_variant = np.sqrt(variant_rate * (1 - variant_rate) / variant_total)
        se_diff = np.sqrt(se_control**2 + se_variant**2)
        
        ci_lower = (variant_rate - control_rate) - 1.96 * se_diff
        ci_upper = (variant_rate - control_rate) + 1.96 * se_diff
        
        # Statistical power calculation
        effect_size = sms.proportion_effectsize(control_rate, variant_rate)
        power_analysis = sms.zt_ind_solve_power(effect_size=effect_size, 
                                                nobs1=control_total,
                                                alpha=self.alpha, 
                                                ratio=variant_total/control_total)
        
        print("STATISTICAL SIGNIFICANCE:")
        print(f"  Z-statistic: {z_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Significance level (α): {self.alpha}")
        print(f"  95% CI for difference: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
        print(f"  Statistical Power: {power_analysis:.4f} ({power_analysis*100:.1f}%)\n")
        
        # Interpretation
        print("INTERPRETATION:")
        if p_value < self.alpha:
            print(f"  ✓ STATISTICALLY SIGNIFICANT (p = {p_value:.6f} < {self.alpha})")
            if variant_rate > control_rate:
                print(f"  → The variant performs BETTER than control")
            else:
                print(f"  → The variant performs WORSE than control")
        else:
            print(f"  ✗ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.6f} >= {self.alpha})")
            print(f"  → Cannot conclude there's a difference between groups")
        
        if power_analysis < 0.8:
            print(f"\n  ⚠ WARNING: Low statistical power ({power_analysis:.2f})")
            print(f"    Consider increasing sample size for more reliable results")
        
        # Store results
        self.results[test_name] = {
            'control_rate': control_rate,
            'variant_rate': variant_rate,
            'p_value': p_value,
            'z_stat': z_stat,
            'significant': p_value < self.alpha,
            'relative_improvement': relative_improvement,
            'power': power_analysis,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        return self.results[test_name]
    
    def analyze_continuous_metric(self, control_data, variant_data, 
                                  metric_name="Continuous Metric Test"):
        """
        Analyze continuous metrics (e.g., revenue, time on site) between groups
        
        Parameters:
        -----------
        control_data : array-like
            Control group data
        variant_data : array-like
            Variant group data
        metric_name : str
            Name of the metric
        """
        print(f"\n{'='*70}")
        print(f"A/B TEST ANALYSIS: {metric_name}")
        print(f"{'='*70}\n")
        
        control_data = np.array(control_data)
        variant_data = np.array(variant_data)
        
        # Descriptive statistics
        control_mean = np.mean(control_data)
        variant_mean = np.mean(variant_data)
        control_std = np.std(control_data, ddof=1)
        variant_std = np.std(variant_data, ddof=1)
        control_median = np.median(control_data)
        variant_median = np.median(variant_data)
        
        relative_improvement = ((variant_mean - control_mean) / control_mean) * 100
        
        print("DESCRIPTIVE STATISTICS:")
        print(f"  Control:  Mean = {control_mean:.2f}, Median = {control_median:.2f}, SD = {control_std:.2f}, N = {len(control_data)}")
        print(f"  Variant:  Mean = {variant_mean:.2f}, Median = {variant_median:.2f}, SD = {variant_std:.2f}, N = {len(variant_data)}")
        print(f"  Mean Difference: {variant_mean - control_mean:+.2f} ({relative_improvement:+.2f}%)\n")
        
        # Normality tests
        _, p_normal_control = stats.shapiro(control_data[:5000])  # Shapiro limited to 5000 samples
        _, p_normal_variant = stats.shapiro(variant_data[:5000])
        
        print("NORMALITY TESTS (Shapiro-Wilk):")
        print(f"  Control p-value: {p_normal_control:.4f}")
        print(f"  Variant p-value: {p_normal_variant:.4f}")
        
        # Choose appropriate test
        if p_normal_control > 0.05 and p_normal_variant > 0.05:
            print("  → Data appears normally distributed, using t-test\n")
            
            # Independent t-test
            t_stat, p_value = ttest_ind(variant_data, control_data)
            test_used = "Independent t-test"
            
            # Cohen's d effect size
            pooled_std = np.sqrt(((len(control_data)-1)*control_std**2 + 
                                 (len(variant_data)-1)*variant_std**2) / 
                                (len(control_data) + len(variant_data) - 2))
            cohens_d = (variant_mean - control_mean) / pooled_std
            
            print("STATISTICAL SIGNIFICANCE (t-test):")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  Cohen's d (effect size): {cohens_d:.4f}")
            
        else:
            print("  → Data not normally distributed, using Mann-Whitney U test\n")
            
            # Mann-Whitney U test
            u_stat, p_value = mannwhitneyu(variant_data, control_data, alternative='two-sided')
            test_used = "Mann-Whitney U test"
            
            print("STATISTICAL SIGNIFICANCE (Mann-Whitney U):")
            print(f"  U-statistic: {u_stat:.4f}")
        
        # Confidence interval for mean difference (t-test based)
        se = np.sqrt(control_std**2/len(control_data) + variant_std**2/len(variant_data))
        ci_lower = (variant_mean - control_mean) - 1.96 * se
        ci_upper = (variant_mean - control_mean) + 1.96 * se
        
        print(f"  P-value: {p_value:.6f}")
        print(f"  Significance level (α): {self.alpha}")
        print(f"  95% CI for difference: [{ci_lower:.2f}, {ci_upper:.2f}]\n")
        
        # Interpretation
        print("INTERPRETATION:")
        if p_value < self.alpha:
            print(f"  ✓ STATISTICALLY SIGNIFICANT (p = {p_value:.6f} < {self.alpha})")
            if variant_mean > control_mean:
                print(f"  → The variant performs BETTER than control")
            else:
                print(f"  → The variant performs WORSE than control")
        else:
            print(f"  ✗ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.6f} >= {self.alpha})")
            print(f"  → Cannot conclude there's a difference between groups")
        
        return {
            'control_mean': control_mean,
            'variant_mean': variant_mean,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'test_used': test_used,
            'relative_improvement': relative_improvement
        }
    
    def calculate_sample_size(self, baseline_rate, mde, alpha=None, power=0.8):
        """
        Calculate required sample size for A/B test
        
        Parameters:
        -----------
        baseline_rate : float
            Expected baseline conversion rate
        mde : float
            Minimum detectable effect (e.g., 0.02 for 2% absolute increase)
        alpha : float
            Significance level (uses self.alpha if None)
        power : float
            Statistical power (default: 0.8)
        """
        if alpha is None:
            alpha = self.alpha
        
        effect_size = sms.proportion_effectsize(baseline_rate, baseline_rate + mde)
        sample_size = sms.zt_ind_solve_power(effect_size=effect_size,
                                             alpha=alpha,
                                             power=power,
                                             ratio=1.0)
        
        print(f"\n{'='*70}")
        print("SAMPLE SIZE CALCULATION")
        print(f"{'='*70}\n")
        print(f"Parameters:")
        print(f"  Baseline conversion rate: {baseline_rate*100:.2f}%")
        print(f"  Minimum detectable effect: {mde*100:.2f} percentage points")
        print(f"  Significance level (α): {alpha}")
        print(f"  Statistical power: {power}\n")
        print(f"Required sample size per group: {int(np.ceil(sample_size)):,}")
        print(f"Total required sample size: {int(np.ceil(sample_size * 2)):,}")
        
        return int(np.ceil(sample_size))
    
    def visualize_results(self, control_data, variant_data, 
                         data_type='conversion', labels=None):
        """
        Create visualizations for A/B test results
        
        Parameters:
        -----------
        control_data : array-like
            Control group data
        variant_data : array-like
            Variant group data
        data_type : str
            'conversion' for binary data, 'continuous' for continuous metrics
        labels : dict
            Dictionary with 'title', 'xlabel', 'ylabel' keys
        """
        if labels is None:
            labels = {'title': 'A/B Test Results', 'xlabel': 'Group', 'ylabel': 'Metric'}
        
        sns.set_style("whitegrid")
        
        if data_type == 'conversion':
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Bar chart of conversion rates
            groups = ['Control', 'Variant']
            rates = [np.mean(control_data), np.mean(variant_data)]
            colors = ['#3498db', '#e74c3c']
            
            axes[0].bar(groups, rates, color=colors, alpha=0.7, edgecolor='black')
            axes[0].set_ylabel('Conversion Rate')
            axes[0].set_title('Conversion Rate Comparison')
            axes[0].set_ylim([0, max(rates) * 1.2])
            
            for i, (group, rate) in enumerate(zip(groups, rates)):
                axes[0].text(i, rate + 0.01, f'{rate*100:.2f}%', 
                           ha='center', va='bottom', fontweight='bold')
            
            # Sample size comparison
            sizes = [len(control_data), len(variant_data)]
            axes[1].bar(groups, sizes, color=colors, alpha=0.7, edgecolor='black')
            axes[1].set_ylabel('Sample Size')
            axes[1].set_title('Sample Size Comparison')
            
            for i, (group, size) in enumerate(zip(groups, sizes)):
                axes[1].text(i, size + max(sizes)*0.02, f'{size:,}', 
                           ha='center', va='bottom', fontweight='bold')
            
        else:  # continuous data
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Box plot
            data_plot = [control_data, variant_data]
            bp = axes[0, 0].boxplot(data_plot, labels=['Control', 'Variant'],
                                    patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[0, 0].set_ylabel(labels['ylabel'])
            axes[0, 0].set_title('Distribution Comparison (Box Plot)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Violin plot
            parts = axes[0, 1].violinplot(data_plot, positions=[1, 2], 
                                         showmeans=True, showmedians=True)
            axes[0, 1].set_xticks([1, 2])
            axes[0, 1].set_xticklabels(['Control', 'Variant'])
            axes[0, 1].set_ylabel(labels['ylabel'])
            axes[0, 1].set_title('Distribution Comparison (Violin Plot)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Histograms
            axes[1, 0].hist(control_data, bins=30, alpha=0.7, color='#3498db', 
                          label='Control', edgecolor='black')
            axes[1, 0].hist(variant_data, bins=30, alpha=0.7, color='#e74c3c', 
                          label='Variant', edgecolor='black')
            axes[1, 0].set_xlabel(labels['ylabel'])
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution Overlay')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Summary statistics
            stats_text = f"Control:\n  Mean: {np.mean(control_data):.2f}\n  Median: {np.median(control_data):.2f}\n  SD: {np.std(control_data):.2f}\n\n"
            stats_text += f"Variant:\n  Mean: {np.mean(variant_data):.2f}\n  Median: {np.median(variant_data):.2f}\n  SD: {np.std(variant_data):.2f}"
            
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                          verticalalignment='center', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.show()


# Example Usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*20 + "A/B TEST ANALYZER")
    print("="*70)
    
    # Initialize analyzer
    analyzer = ABTestAnalyzer(alpha=0.05)
    
    # Example 1: Conversion Rate Test
    print("\n\nEXAMPLE 1: Testing a new checkout button design")
    print("-" * 70)
    
    control_conversions = 180
    control_total = 2000
    variant_conversions = 225
    variant_total = 2000
    
    result1 = analyzer.analyze_conversion_rate(
        control_conversions, control_total,
        variant_conversions, variant_total,
        test_name="Checkout Button Design"
    )
    
    # Visualize conversion test
    control_binary = np.array([1]*control_conversions + [0]*(control_total-control_conversions))
    variant_binary = np.array([1]*variant_conversions + [0]*(variant_total-variant_conversions))
    analyzer.visualize_results(control_binary, variant_binary, data_type='conversion')
    
    # Example 2: Continuous Metric Test (Revenue per user)
    print("\n\nEXAMPLE 2: Testing impact on average order value")
    print("-" * 70)
    
    np.random.seed(42)
    control_revenue = np.random.gamma(shape=2, scale=25, size=1500)
    variant_revenue = np.random.gamma(shape=2, scale=28, size=1500)
    
    result2 = analyzer.analyze_continuous_metric(
        control_revenue, variant_revenue,
        metric_name="Average Order Value"
    )
    
    # Visualize continuous metric test
    analyzer.visualize_results(
        control_revenue, variant_revenue, 
        data_type='continuous',
        labels={'title': 'Revenue Comparison', 'xlabel': 'Group', 'ylabel': 'Revenue ($)'}
    )
    
    # Example 3: Sample Size Calculation
    print("\n\nEXAMPLE 3: Planning a new A/B test")
    print("-" * 70)
    
    required_n = analyzer.calculate_sample_size(
        baseline_rate=0.10,  # 10% baseline conversion
        mde=0.02,            # Want to detect 2% absolute increase
        power=0.8
    )
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)