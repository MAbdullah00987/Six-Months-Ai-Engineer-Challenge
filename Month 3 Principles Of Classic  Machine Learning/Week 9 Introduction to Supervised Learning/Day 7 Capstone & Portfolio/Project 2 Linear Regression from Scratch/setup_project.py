"""
setup_project.py
Quick setup script for Week 9 Linear Regression Project

This script:
1. Creates the project directory structure
2. Generates sample datasets
3. Runs basic tests to verify installation
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure."""
    
    print("="*70)
    print("CREATING PROJECT DIRECTORY STRUCTURE")
    print("="*70)
    
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'results',
        'results/figures',
        'results/models'
    ]
    
    base_dir = Path('week-9-supervised-learning')
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    print(f"\n✓ Directory structure created successfully!")
    return base_dir


def generate_datasets(base_dir):
    """Generate sample datasets."""
    
    print("\n" + "="*70)
    print("GENERATING SAMPLE DATASETS")
    print("="*70)
    
    try:
        import numpy as np
        import pandas as pd
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate Salary Dataset
        print("\n1. Generating salary dataset...")
        n_samples = 500
        
        years_experience = np.random.uniform(0, 15, n_samples)
        education_level = np.random.choice([1, 2, 3, 4], n_samples)
        age = 22 + years_experience + np.random.uniform(-2, 5, n_samples)
        hours_per_week = np.random.uniform(35, 60, n_samples)
        
        salary = (
            30000 +
            3000 * years_experience +
            8000 * education_level +
            500 * age +
            200 * hours_per_week +
            np.random.normal(0, 5000, n_samples)
        )
        
        salary_df = pd.DataFrame({
            'YearsExperience': years_experience,
            'EducationLevel': education_level,
            'Age': age,
            'HoursPerWeek': hours_per_week,
            'Salary': salary
        })
        
        salary_path = base_dir / 'data' / 'raw' / 'salary_data.csv'
        salary_df.to_csv(salary_path, index=False)
        print(f"   ✓ Saved to: {salary_path}")
        print(f"   ✓ Shape: {salary_df.shape}")
        
        # Generate Housing Dataset
        print("\n2. Generating housing dataset...")
        n_samples = 1000
        
        square_feet = np.random.uniform(800, 4000, n_samples)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples)
        bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], n_samples)
        age = np.random.uniform(0, 50, n_samples)
        location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
        
        location_multiplier = {'Urban': 1.5, 'Suburban': 1.0, 'Rural': 0.7}
        location_effect = np.array([location_multiplier[loc] for loc in location])
        
        price = (
            100000 +
            150 * square_feet +
            20000 * bedrooms +
            15000 * bathrooms -
            1000 * age +
            50000 * location_effect +
            np.random.normal(0, 20000, n_samples)
        )
        
        housing_df = pd.DataFrame({
            'SquareFeet': square_feet,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Age': age,
            'Location': location,
            'Price': price
        })
        
        housing_path = base_dir / 'data' / 'raw' / 'housing_data.csv'
        housing_df.to_csv(housing_path, index=False)
        print(f"   ✓ Saved to: {housing_path}")
        print(f"   ✓ Shape: {housing_df.shape}")
        
        print("\n✓ Datasets generated successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error generating datasets: {e}")
        return False


def verify_dependencies():
    """Verify that all required packages are installed."""
    
    print("\n" + "="*70)
    print("VERIFYING DEPENDENCIES")
    print("="*70)
    
    required_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'scipy': 'SciPy',
        'statsmodels': 'Statsmodels'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:15} - Installed")
        except ImportError:
            print(f"✗ {name:15} - NOT FOUND")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n✗ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def run_basic_test():
    """Run a basic test of the linear regression implementation."""
    
    print("\n" + "="*70)
    print("RUNNING BASIC FUNCTIONALITY TEST")
    print("="*70)
    
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        # Generate simple test data
        print("\nGenerating test data...")
        np.random.seed(42)
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X.squeeze() + np.random.randn(100)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Test custom implementation (simplified inline version)
        print("Testing gradient descent...")
        
        # Simple gradient descent
        learning_rate = 0.1
        n_iterations = 1000
        m, n = X_train.shape
        weights = np.zeros(n)
        bias = 0
        
        for i in range(n_iterations):
            predictions = X_train.dot(weights) + bias
            errors = predictions - y_train
            
            dw = (1/m) * X_train.T.dot(errors)
            db = (1/m) * np.sum(errors)
            
            weights -= learning_rate * dw
            bias -= learning_rate * db
        
        # Make predictions
        y_pred = X_test.dot(weights) + bias
        
        # Calculate R²
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"\n✓ Test completed successfully!")
        print(f"  Final R² Score: {r2:.4f}")
        print(f"  Weight: {weights[0]:.4f} (expected ~3.0)")
        print(f"  Bias: {bias:.4f} (expected ~4.0)")
        
        if r2 > 0.8:
            print("\n✓ Model performance is good!")
            return True
        else:
            print("\n⚠ Model performance is lower than expected")
            return False
            
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_next_steps():
    """Print next steps for the user."""
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    
    print("""
Your project is ready! Here's what to do next:

1. EXPLORE THE DATA
   cd week-9-supervised-learning/data/raw
   # Check out salary_data.csv and housing_data.csv

2. START JUPYTER NOTEBOOK
   jupyter notebook
   # Open notebooks in the 'notebooks/' folder

3. RECOMMENDED ORDER
   a) notebooks/01_salary_prediction.ipynb
   b) notebooks/02_housing_prices.ipynb
   c) notebooks/03_advanced_model_comparison.ipynb

4. KEY FILES TO REVIEW
   - src/models.py              (Linear regression implementation)
   - src/preprocessing.py       (Data preprocessing utilities)
   - README.md                  (Complete documentation)

5. LEARN MORE
   - Check README.md for detailed documentation
   - Review mathematical foundations in the notebooks
   - Experiment with different hyperparameters

Happy Learning! 🚀
""")


def main():
    """Main setup function."""
    
    print("\n" + "="*70)
    print("WEEK 9: LINEAR REGRESSION PROJECT SETUP")
    print("="*70)
    
    # Step 1: Verify dependencies
    if not verify_dependencies():
        print("\n⚠ Please install missing dependencies first!")
        print("Run: pip install -r requirements.txt")
        return
    
    # Step 2: Create directory structure
    base_dir = create_directory_structure()
    
    # Step 3: Generate datasets
    if not generate_datasets(base_dir):
        print("\n⚠ Warning: Dataset generation failed")
    
    # Step 4: Run basic test
    if not run_basic_test():
        print("\n⚠ Warning: Basic test failed")
    
    # Step 5: Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()