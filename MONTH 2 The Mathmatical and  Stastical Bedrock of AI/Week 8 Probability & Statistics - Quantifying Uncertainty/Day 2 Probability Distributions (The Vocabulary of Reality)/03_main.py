
#Probability Distribution Plotter - Build an interactive tool to visualize different distributions with adjustable parameters

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy import stats
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class DistributionPlotter:
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle('Interactive Probability Distribution Plotter', fontsize=16, fontweight='bold')
        
        # Create main plot
        self.ax = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)
        
        # Create radio buttons for distribution selection
        self.rax = plt.subplot2grid((4, 3), (0, 2), rowspan=2)
        self.radio = RadioButtons(
            self.rax,
            ('Normal', 'Exponential', 'Uniform', 'Poisson', 'Binomial', 
             'Gamma', 'Beta', 'Chi-Square', 'Student-t'),
            active=0
        )
        self.radio.on_clicked(self.update_distribution)
        
        # Stats display area
        self.stats_ax = plt.subplot2grid((4, 3), (2, 2), rowspan=2)
        self.stats_ax.axis('off')
        
        # Create slider axes
        self.slider_ax1 = plt.subplot2grid((4, 3), (3, 0))
        self.slider_ax2 = plt.subplot2grid((4, 3), (3, 1))
        
        # Initialize with Normal distribution
        self.current_dist = 'Normal'
        self.setup_sliders()
        self.plot()
        
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    def setup_sliders(self):
        """Setup sliders based on selected distribution"""
        self.slider_ax1.clear()
        self.slider_ax2.clear()
        
        if self.current_dist == 'Normal':
            self.slider1 = Slider(self.slider_ax1, 'Mean (μ)', -10, 10, valinit=0, valstep=0.1)
            self.slider2 = Slider(self.slider_ax2, 'Std Dev (σ)', 0.1, 5, valinit=1, valstep=0.1)
        
        elif self.current_dist == 'Exponential':
            self.slider1 = Slider(self.slider_ax1, 'Lambda (λ)', 0.1, 5, valinit=1, valstep=0.1)
            self.slider2 = None
        
        elif self.current_dist == 'Uniform':
            self.slider1 = Slider(self.slider_ax1, 'Lower (a)', -10, 10, valinit=0, valstep=0.5)
            self.slider2 = Slider(self.slider_ax2, 'Upper (b)', -10, 20, valinit=10, valstep=0.5)
        
        elif self.current_dist == 'Poisson':
            self.slider1 = Slider(self.slider_ax1, 'Lambda (λ)', 0.1, 20, valinit=5, valstep=0.5)
            self.slider2 = None
        
        elif self.current_dist == 'Binomial':
            self.slider1 = Slider(self.slider_ax1, 'n (trials)', 1, 100, valinit=20, valstep=1)
            self.slider2 = Slider(self.slider_ax2, 'p (prob)', 0, 1, valinit=0.5, valstep=0.01)
        
        elif self.current_dist == 'Gamma':
            self.slider1 = Slider(self.slider_ax1, 'Shape (α)', 0.1, 10, valinit=2, valstep=0.1)
            self.slider2 = Slider(self.slider_ax2, 'Scale (θ)', 0.1, 5, valinit=2, valstep=0.1)
        
        elif self.current_dist == 'Beta':
            self.slider1 = Slider(self.slider_ax1, 'Alpha (α)', 0.1, 10, valinit=2, valstep=0.1)
            self.slider2 = Slider(self.slider_ax2, 'Beta (β)', 0.1, 10, valinit=5, valstep=0.1)
        
        elif self.current_dist == 'Chi-Square':
            self.slider1 = Slider(self.slider_ax1, 'df (k)', 1, 30, valinit=5, valstep=1)
            self.slider2 = None
        
        elif self.current_dist == 'Student-t':
            self.slider1 = Slider(self.slider_ax1, 'df (ν)', 1, 30, valinit=10, valstep=1)
            self.slider2 = None
        
        # Connect sliders to update function
        self.slider1.on_changed(self.update)
        if self.slider2:
            self.slider2.on_changed(self.update)
    
    def update_distribution(self, label):
        """Update distribution type"""
        self.current_dist = label
        self.setup_sliders()
        self.plot()
    
    def update(self, val):
        """Update plot when slider changes"""
        self.plot()
    
    def plot(self):
        """Plot the selected distribution"""
        self.ax.clear()
        
        if self.current_dist == 'Normal':
            mean = self.slider1.val
            std = self.slider2.val
            x = np.linspace(mean - 4*std, mean + 4*std, 1000)
            y = stats.norm.pdf(x, mean, std)
            self.ax.plot(x, y, linewidth=2, color='#667eea')
            self.ax.fill_between(x, y, alpha=0.3, color='#667eea')
            
            stats_text = f"Mean: {mean:.3f}\nVariance: {std**2:.3f}\nStd Dev: {std:.3f}\nSkewness: 0.000"
        
        elif self.current_dist == 'Exponential':
            lam = self.slider1.val
            x = np.linspace(0, 10/lam, 1000)
            y = stats.expon.pdf(x, scale=1/lam)
            self.ax.plot(x, y, linewidth=2, color='#667eea')
            self.ax.fill_between(x, y, alpha=0.3, color='#667eea')
            
            stats_text = f"Mean: {1/lam:.3f}\nVariance: {1/lam**2:.3f}\nStd Dev: {1/lam:.3f}\nSkewness: 2.000"
        
        elif self.current_dist == 'Uniform':
            a = self.slider1.val
            b = self.slider2.val
            if b <= a:
                b = a + 1
            x = np.linspace(a-1, b+1, 1000)
            y = stats.uniform.pdf(x, a, b-a)
            self.ax.plot(x, y, linewidth=2, color='#667eea')
            self.ax.fill_between(x, y, alpha=0.3, color='#667eea')
            
            mean = (a + b) / 2
            variance = (b - a)**2 / 12
            stats_text = f"Mean: {mean:.3f}\nVariance: {variance:.3f}\nStd Dev: {np.sqrt(variance):.3f}\nSkewness: 0.000"
        
        elif self.current_dist == 'Poisson':
            lam = self.slider1.val
            x = np.arange(0, int(lam + 5*np.sqrt(lam)))
            y = stats.poisson.pmf(x, lam)
            self.ax.bar(x, y, color='#667eea', alpha=0.7, edgecolor='black')
            
            stats_text = f"Mean: {lam:.3f}\nVariance: {lam:.3f}\nStd Dev: {np.sqrt(lam):.3f}\nSkewness: {1/np.sqrt(lam):.3f}"
        
        elif self.current_dist == 'Binomial':
            n = int(self.slider1.val)
            p = self.slider2.val
            x = np.arange(0, n+1)
            y = stats.binom.pmf(x, n, p)
            self.ax.bar(x, y, color='#667eea', alpha=0.7, edgecolor='black')
            
            mean = n * p
            variance = n * p * (1 - p)
            skew = (1 - 2*p) / np.sqrt(n * p * (1 - p)) if variance > 0 else 0
            stats_text = f"Mean: {mean:.3f}\nVariance: {variance:.3f}\nStd Dev: {np.sqrt(variance):.3f}\nSkewness: {skew:.3f}"
        
        elif self.current_dist == 'Gamma':
            alpha = self.slider1.val
            theta = self.slider2.val
            x = np.linspace(0, alpha*theta + 5*np.sqrt(alpha)*theta, 1000)
            y = stats.gamma.pdf(x, alpha, scale=theta)
            self.ax.plot(x, y, linewidth=2, color='#667eea')
            self.ax.fill_between(x, y, alpha=0.3, color='#667eea')
            
            mean = alpha * theta
            variance = alpha * theta**2
            skew = 2 / np.sqrt(alpha)
            stats_text = f"Mean: {mean:.3f}\nVariance: {variance:.3f}\nStd Dev: {np.sqrt(variance):.3f}\nSkewness: {skew:.3f}"
        
        elif self.current_dist == 'Beta':
            alpha = self.slider1.val
            beta = self.slider2.val
            x = np.linspace(0, 1, 1000)
            y = stats.beta.pdf(x, alpha, beta)
            self.ax.plot(x, y, linewidth=2, color='#667eea')
            self.ax.fill_between(x, y, alpha=0.3, color='#667eea')
            
            mean = alpha / (alpha + beta)
            variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            skew = (2 * (beta - alpha) * np.sqrt(alpha + beta + 1)) / ((alpha + beta + 2) * np.sqrt(alpha * beta))
            stats_text = f"Mean: {mean:.3f}\nVariance: {variance:.3f}\nStd Dev: {np.sqrt(variance):.3f}\nSkewness: {skew:.3f}"
        
        elif self.current_dist == 'Chi-Square':
            k = int(self.slider1.val)
            x = np.linspace(0, k + 5*np.sqrt(2*k), 1000)
            y = stats.chi2.pdf(x, k)
            self.ax.plot(x, y, linewidth=2, color='#667eea')
            self.ax.fill_between(x, y, alpha=0.3, color='#667eea')
            
            stats_text = f"Mean: {k:.3f}\nVariance: {2*k:.3f}\nStd Dev: {np.sqrt(2*k):.3f}\nSkewness: {np.sqrt(8/k):.3f}"
        
        elif self.current_dist == 'Student-t':
            nu = int(self.slider1.val)
            x = np.linspace(-5, 5, 1000)
            y = stats.t.pdf(x, nu)
            self.ax.plot(x, y, linewidth=2, color='#667eea')
            self.ax.fill_between(x, y, alpha=0.3, color='#667eea')
            
            mean = 0 if nu > 1 else "Undefined"
            variance = nu / (nu - 2) if nu > 2 else "Undefined"
            std = np.sqrt(variance) if isinstance(variance, (int, float)) else "Undefined"
            skew = 0 if nu > 3 else "Undefined"
            stats_text = f"Mean: {mean}\nVariance: {variance}\nStd Dev: {std}\nSkewness: {skew}"
        
        # Update plot labels
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('Probability Density/Mass', fontsize=12)
        self.ax.set_title(f'{self.current_dist} Distribution', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Update stats display
        self.stats_ax.clear()
        self.stats_ax.axis('off')
        self.stats_ax.text(0.1, 0.8, 'Statistics:', fontsize=12, fontweight='bold', 
                          transform=self.stats_ax.transAxes)
        self.stats_ax.text(0.1, 0.5, stats_text, fontsize=10, 
                          transform=self.stats_ax.transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.draw()
    
    def show(self):
        """Display the plot"""
        plt.show()

# Create and show the plotter
if __name__ == "__main__":
    plotter = DistributionPlotter()
    plotter.show()