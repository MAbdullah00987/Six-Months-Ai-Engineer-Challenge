"""
Project Setup and Testing Script
Run this file to verify your installation and test the implementations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_directory_structure():
    """Create the project directory structure"""
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'tests',
        'figures'
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ {directory}/")
    
    # Create __init__.py files
    init_files = ['src/__init__.py', 'tests/__init__.py']
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization\n')
            print(f"   ✓ {init_file}")
    
    print("✅ Directory structure created!\n")


def test_imports():
    """Test if all required libraries are installed"""
    print("📦 Testing imports...")
    
    libraries = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'Scikit-Learn'),
        ('scipy', 'SciPy'),
        ('statsmodels', 'Statsmodels')
    ]
    
    failed = []
    for module, name in libraries:
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - NOT INSTALLED")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️  Missing libraries: {', '.join(failed)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All libraries installed!\n")
        return True


def test_custom_models():
    """Test custom linear regression implementations"""
    print("🧪 Testing custom implementations...")
    
    try:
        # Import custom modules
        sys.path.append('src')
        from models import LinearRegressionScratch, LinearRegressionNormalEquation
        from preprocessing import FeatureScaler, train_test_split_custom
        
        # Generate test data
        np.random.seed(42)
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.1
        
        # Test 1: Gradient Descent
        print("   Testing Gradient Descent...")
        model_gd = LinearRegressionScratch(learning_rate=0.01, n_iterations=500)
        model_gd.fit(X, y, verbose=False)
        r2 = model_gd.score(X, y)
        assert r2 > 0.95, f"R² too low: {r2}"
        print(f"   ✓ Gradient Descent (R² = {r2:.4f})")
        
        # Test 2: Normal Equation
        print("   Testing Normal Equation...")
        model_ne = LinearRegressionNormalEquation()
        model_ne.fit(X, y)
        r2_ne = model_ne.score(X, y)
        assert r2_ne > 0.95, f"R² too low: {r2_ne}"
        print(f"   ✓ Normal Equation (R² = {r2_ne:.4f})")
        
        # Test 3: Feature Scaler
        print("   Testing Feature Scaler...")
        scaler = FeatureScaler(method='standard')
        X_scaled = scaler.fit_transform(X)
        assert abs(np.mean(X_scaled)) < 0.01, "Mean not close to 0"
        assert abs(np.std(X_scaled) - 1) < 0.01, "Std not close to 1"
        print("   ✓ Feature Scaler")
        
        # Test 4: Train-Test Split
        print("   Testing Train-Test Split...")
        X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)
        assert len(X_train) == 80, "Train size incorrect"
        assert len(X_test) == 20, "Test size incorrect"
        print("   ✓ Train-Test Split")
        
        print("\n✅ All custom implementations working!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing custom implementations: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_sklearn_compatibility():
    """Test compatibility with Scikit-Learn"""
    print("🔗 Testing Scikit-Learn compatibility...")
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        
        sys.path.append('src')
        from models import LinearRegressionScratch
        
        # Generate test data
        np.random.seed(42)
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train both models
        model_custom = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
        model_custom.fit(X_train, y_train, verbose=False)
        
        model_sklearn = LinearRegression()
        model_sklearn.fit(X_train, y_train)
        
        # Compare results
        r2_custom = model_custom.score(X_test, y_test)
        r2_sklearn = model_sklearn.score(X_test, y_test)
        
        print(f"   Custom Model R²: {r2_custom:.4f}")
        print(f"   Sklearn Model R²: {r2_sklearn:.4f}")
        print(f"   Difference: {abs(r2_custom - r2_sklearn):.6f}")
        
        assert abs(r2_custom - r2_sklearn) < 0.01, "Results differ too much"
        print("\n✅ Compatible with Scikit-Learn!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing Scikit-Learn compatibility: {e}\n")
        return False


def run_quick_demo():
    """Run a quick demonstration"""
    print("🎬 Running quick demonstration...")
    
    try:
        sys.path.append('src')
        from models import LinearRegressionScratch
        import matplotlib.pyplot as plt
        
        # Generate data
        np.random.seed(42)
        X = 2 * np.random.rand(50, 1)
        y = 4 + 3 * X.flatten() + np.random.randn(50) * 0.5
        
        # Train model
        model = LinearRegressionScratch(learning_rate=0.01, n_iterations=500)
        model.fit(X, y, verbose=False)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Data and predictions
        axes[0].scatter(X, y, alpha=0.6, s=50, label='Data')
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_line)
        axes[0].plot(X_line, y_pred, 'r-', linewidth=2, label='Model')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('y')
        axes[0].set_title('Linear Regression Demo')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Convergence
        axes[1].plot(model.cost_history, linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Cost')
        axes[1].set_title('Cost Function Convergence')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/quick_demo.png', dpi=150, bbox_inches='tight')
        print("   ✓ Figure saved to 'figures/quick_demo.png'")
        
        # plt.show()  # Uncomment to display
        plt.close()
        
        print(f"   ✓ Model R²: {model.score(X, y):.4f}")
        print(f"   ✓ Final Cost: {model.cost_history[-1]:.4f}")
        
        print("\n✅ Demo completed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error running demo: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """Print project summary and next steps"""
    print("="*70)
    print("PROJECT SETUP SUMMARY")
    print("="*70)
    print("""
✅ Project structure created
✅ All libraries installed
✅ Custom implementations tested
✅ Scikit-Learn compatibility verified
✅ Quick demo completed

📚 Next Steps:

1. Open Jupyter Notebook:
   jupyter notebook

2. Navigate to notebooks/ and open:
   - 01_linear_regression_complete.ipynb (Main project)
   - 02_salary_prediction.ipynb (Real-world example)

3. Explore the code in src/:
   - models.py (Linear regression implementations)
   - preprocessing.py (Data preprocessing utilities)

4. Check out the figures/ directory for visualizations

5. Read README.md for detailed documentation

💡 Tips:
- Run all cells in the notebooks to see full analysis
- Experiment with different learning rates and iterations
- Try with your own datasets
- Check the convergence plots to understand optimization

📖 Documentation:
- Models: See docstrings in src/models.py
- Functions: See docstrings in src/preprocessing.py
- Examples: Check notebooks/ directory

🎓 Learning Resources:
- Andrew Ng's Machine Learning Course
- Scikit-Learn documentation
- "Hands-On Machine Learning" book

Happy Learning! 🚀
""")
    print("="*70)


def main():
    """Main setup and testing function"""
    print("\n" + "="*70)
    print("LINEAR REGRESSION PROJECT - SETUP & TESTING")
    print("="*70 + "\n")
    
    # Run all tests
    all_passed = True
    
    create_directory_structure()
    
    if not test_imports():
        all_passed = False
        print("\n⚠️  Please install missing libraries before continuing.\n")
        return
    
    if not test_custom_models():
        all_passed = False
    
    if not test_sklearn_compatibility():
        all_passed = False
    
    if not run_quick_demo():
        all_passed = False
    
    if all_passed:
        print_summary()
        print("✅ Setup completed successfully! You're ready to start.\n")
    else:
        print("❌ Some tests failed. Please check the errors above.\n")


if __name__ == "__main__":
    main()