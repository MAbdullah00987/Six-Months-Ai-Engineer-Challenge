"""
Interactive Sine Wave Plotter with Sliders
-------------------------------------------
Use this in Jupyter Notebook to interactively explore sine wave parameters.

Installation required:
    pip install ipywidgets
    jupyter nbextension enable --py widgetsnbextension
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider


def interactive_sine_wave(amplitude=1.0, frequency=1.0, phase=0.0, sampling_rate=1000):
    """
    Interactive sine wave plotter with adjustable parameters.
    
    Parameters controlled by sliders:
    - amplitude: Height of the wave (0.1 to 5.0)
    - frequency: Cycles per second (0.1 to 10.0 Hz)
    - phase: Horizontal shift (0 to 2π radians)
    - sampling_rate: Samples per second (100 to 5000)
    """
    # Time configuration
    duration = 2.0
    num_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, num_samples)
    
    # Generate sine wave
    y = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    # Create plot
    plt.figure(figsize=(14, 8))
    plt.plot(t, y, linewidth=2.5, color='#E63946', alpha=0.8)
    
    # Customize plot
    plt.xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=13, fontweight='bold')
    plt.title(f'Interactive Sine Wave\ny = {amplitude:.2f} × sin(2π × {frequency:.2f}t + {phase:.2f})', 
             fontsize=15, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    
    # Set consistent y-axis limits
    plt.ylim(-5.5, 5.5)
    plt.xlim(0, duration)
    
    # Add information text
    info_text = f'Samples: {num_samples}\nMax: {y.max():.3f}\nMin: {y.min():.3f}\nPeriod: {1/frequency:.3f}s'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
            fontsize=11, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()


def create_interactive_plot():
    """
    Create interactive plot with sliders for all parameters.
    
    Usage in Jupyter Notebook:
        from interactive_sine_wave import create_interactive_plot
        create_interactive_plot()
    """
    # Define sliders
    amplitude_slider = FloatSlider(
        value=1.0,
        min=0.1,
        max=5.0,
        step=0.1,
        description='Amplitude:',
        continuous_update=True,
        style={'description_width': '120px'}
    )
    
    frequency_slider = FloatSlider(
        value=2.0,
        min=0.1,
        max=10.0,
        step=0.1,
        description='Frequency (Hz):',
        continuous_update=True,
        style={'description_width': '120px'}
    )
    
    phase_slider = FloatSlider(
        value=0.0,
        min=0.0,
        max=2*np.pi,
        step=0.1,
        description='Phase (rad):',
        continuous_update=True,
        style={'description_width': '120px'}
    )
    
    sampling_slider = IntSlider(
        value=1000,
        min=100,
        max=5000,
        step=100,
        description='Sampling Rate:',
        continuous_update=True,
        style={'description_width': '120px'}
    )
    
    # Create interactive widget
    interact(
        interactive_sine_wave,
        amplitude=amplitude_slider,
        frequency=frequency_slider,
        phase=phase_slider,
        sampling_rate=sampling_slider
    )


def compare_sampling_rates():
    """
    Create a comparison showing the effect of different sampling rates.
    Useful for understanding the importance of adequate sampling.
    """
    # Time configuration
    duration = 1.0
    frequency = 5.0
    amplitude = 1.0
    
    # Different sampling rates
    sampling_rates = [50, 100, 500, 2000]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, sr in enumerate(sampling_rates):
        # Generate data
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples)
        y = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Plot
        axes[idx].plot(t, y, 'o-', linewidth=2, markersize=4, color='#E63946', alpha=0.8)
        axes[idx].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'Sampling Rate: {sr} Hz ({num_samples} samples)', 
                          fontsize=13, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, linestyle='--')
        axes[idx].axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
        
        # Add quality indicator
        quality = "Poor" if sr < 100 else "Adequate" if sr < 500 else "Good" if sr < 1000 else "Excellent"
        color = "#E63946" if sr < 100 else "#F4A261" if sr < 500 else "#2A9D8F" if sr < 1000 else "#457B9D"
        
        props = dict(boxstyle='round', facecolor=color, alpha=0.3)
        axes[idx].text(0.02, 0.98, f'Quality: {quality}', transform=axes[idx].transAxes,
                      fontsize=11, verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.suptitle(f'Sampling Rate Comparison (Frequency = {frequency} Hz)', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_complex_waveforms():
    """
    Create complex waveforms by combining multiple sine waves.
    Demonstrates the principle of Fourier synthesis.
    """
    # Time configuration
    duration = 2.0
    sampling_rate = 2000
    t = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Define component waves (fundamental + harmonics)
    fundamental_freq = 2.0
    
    y1 = 1.0 * np.sin(2 * np.pi * fundamental_freq * t)  # Fundamental
    y2 = 0.5 * np.sin(2 * np.pi * 2 * fundamental_freq * t)  # 2nd harmonic
    y3 = 0.25 * np.sin(2 * np.pi * 3 * fundamental_freq * t)  # 3rd harmonic
    
    # Combined wave
    y_combined = y1 + y2 + y3
    
    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    
    # Plot individual components
    axes[0].plot(t, y1, linewidth=2, color='#E63946', label='Fundamental (f)')
    axes[0].set_title('Component 1: Fundamental Frequency', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    axes[0].axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    
    axes[1].plot(t, y2, linewidth=2, color='#F4A261', label='2nd Harmonic (2f)')
    axes[1].set_title('Component 2: Second Harmonic (2× Fundamental)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    axes[1].axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    
    axes[2].plot(t, y3, linewidth=2, color='#2A9D8F', label='3rd Harmonic (3f)')
    axes[2].set_title('Component 3: Third Harmonic (3× Fundamental)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)
    axes[2].axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    
    axes[3].plot(t, y_combined, linewidth=2, color='#457B9D', label='Combined Wave')
    axes[3].set_title('Combined Waveform (Sum of All Components)', fontsize=13, fontweight='bold')
    axes[3].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    axes[3].set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=10)
    axes[3].axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    
    plt.suptitle('Fourier Synthesis: Building Complex Waveforms from Sine Waves', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function for non-interactive execution.
    For Jupyter notebooks, call create_interactive_plot() directly.
    """
    print("=" * 70)
    print("INTERACTIVE SINE WAVE DEMONSTRATIONS")
    print("=" * 70)
    
    print("\nNote: For full interactivity, run this in Jupyter Notebook!")
    print("      Use: create_interactive_plot()")
    
    print("\n1. Comparing different sampling rates...")
    compare_sampling_rates()
    
    print("\n2. Demonstrating complex waveforms...")
    plot_complex_waveforms()
    
    print("\nFor interactive sliders, use this code in Jupyter Notebook:")
    print("    from interactive_sine_wave import create_interactive_plot")
    print("    create_interactive_plot()")


if __name__ == "__main__":
    main()