
"""
COMPLETE USAGE EXAMPLES
Week 9: Linear Regression from Scratch

This file contains practical examples of how to use all components
of the linear regression project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLR
import sys
sys.path.append('src')

# ============================================================================
# EXAMPLE 1: BASIC LINEAR REGRESSION
# ============================================================================

def example_1_basic_regression():
    """Simple linear regression example."""
    
    print("="*70)
    print("EXAMPLE 1: BASIC LINEAR REGRESSION")
    print("="*70)
    
    from models import LinearRegressionScratch
    
    # Generate simple data: y = 3x + 5 + noise
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 3 * X.squeeze() + 5 + np.random.randn(100)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create and train model
    model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train, verbose=False)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    r2_score = model.score(X_test, y_test)
    
    print(f"\nTrue relationship: y = 3x + 5")
    print(f"Learned: y = {model.weights[0]:.4f}x + {model.bias:.4f}")
    print(f"R² Score: {r2_score:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, alpha=0.6, label='Actual', s=50, edgecolors='k')
    plt.plot(X_test, y_pred, 'r-', linewidth=2, label='Predicted')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Basic Linear Regression', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE 2: MULTIVARIATE REGRESSION WITH PREPROCESSING
# ============================================================================

def example_2_multivariate_regression():
    """Multivariate regression with data preprocessing."""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: MULTIVARIATE REGRESSION WITH PREPROCESSING")
    print("="*70)
    
    from models import LinearRegressionScratch, calculate_metrics
    from preprocessing import DataPreprocessor, generate_salary_dataset
    
    # Generate dataset
    df = generate_salary_dataset(n_samples=300)
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # Preprocess data
    preprocessor = DataPreprocessor(scaling_method='standard')
    X, y = preprocessor.prepare_features(df, target_column='Salary')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=2000)
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:,.4f}")
    
    print("\nFeature Importance:")
    for i, feature in enumerate(preprocessor.feature_names):
        print(f"  {feature}: {model.weights[i]:.4f}")


# ============================================================================
# EXAMPLE 3: COMPARISON WITH SCIKIT-LEARN
# ============================================================================

def example_3_sklearn_comparison():
    """Compare custom implementation with scikit-learn."""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: COMPARISON WITH SCIKIT-LEARN")
    print("="*70)
    
    from models import LinearRegressionScratch, ModelVisualizer
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = 2 + 3*X[:, 0] - 1.5*X[:, 1] + 0.5*X[:, 2] + np.random.randn(200)*0.5
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train both models
    model_custom = LinearRegressionScratch(learning_rate=0.01, n_iterations=1500)
    model_custom.fit(X_train, y_train, verbose=False)
    
    model_sklearn = SklearnLR()
    model_sklearn.fit(X_train, y_train)
    
    # Compare
    r2_custom = model_custom.score(X_test, y_test)
    r2_sklearn = model_sklearn.score(X_test, y_test)
    
    print(f"\nR² Scores:")
    print(f"  Custom Implementation: {r2_custom:.6f}")
    print(f"  Scikit-Learn:          {r2_sklearn:.6f}")
    print(f"  Difference:            {abs(r2_custom - r2_sklearn):.8f}")
    
    print(f"\nCoefficients Comparison:")
    print(f"  Feature  {'Custom':>12} {'Sklearn':>12} {'Diff':>12}")
    print("  " + "-"*50)
    for i in range(3):
        diff = abs(model_custom.weights[i] - model_sklearn.coef_[i])
        print(f"  {i}        {model_custom.weights[i]:>12.6f} {model_sklearn.coef_[i]:>12.6f} {diff:>12.8f}")
    
    # Visualize comparison
    visualizer = ModelVisualizer()
    visualizer.compare_models(
        {'Custom': model_custom, 'Scikit-Learn': model_sklearn},
        X_test, y_test
    )
    plt.show()


# ============================================================================
# EXAMPLE 4: GRADIENT DESCENT VISUALIZATION
# ============================================================================

def example_4_gradient_descent_visualization():
    """Visualize gradient descent convergence."""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: GRADIENT DESCENT VISUALIZATION")
    print("="*70)
    
    from models import LinearRegressionScratch, ModelVisualizer
    
    # Generate data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100)
    
    # Train with different learning rates
    learning_rates = [0.001, 0.01, 0.1]
    models = {}
    
    for lr in learning_rates:
        model = LinearRegressionScratch(learning_rate=lr, n_iterations=1000)
        model.fit(X, y, verbose=False)
        models[f'LR={lr}'] = model
    
    # Plot convergence
    plt.figure(figsize=(14, 6))
    
    for name, model in models.items():
        plt.plot(model.cost_history, linewidth=2, label=name)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (MSE)', fontsize=12)
    plt.title('Gradient Descent Convergence - Different Learning Rates', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    print("\nConvergence Summary:")
    for name, model in models.items():
        print(f"  {name}:")
        print(f"    Iterations: {len(model.cost_history)}")
        print(f"    Final Cost: {model.cost_history[-1]:.6f}")


# ============================================================================
# EXAMPLE 5: REGULARIZATION (L1/L2)
# ============================================================================

def example_5_regularization():
    """Demonstrate regularization effects."""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: REGULARIZATION (L1/L2)")
    print("="*70)
    
    from models import LinearRegressionScratch
    
    # Generate data with noise
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # Only first 2 features are relevant
    y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(100)*2
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train models with different regularization
    models = {
        'No Regularization': LinearRegressionScratch(learning_rate=0.01, n_iterations=1000),
        'L2 (Ridge)': LinearRegressionScratch(learning_rate=0.01, n_iterations=1000, 
                                               regularization='l2', lambda_reg=0.1),
        'L1 (Lasso)': LinearRegressionScratch(learning_rate=0.01, n_iterations=1000,
                                               regularization='l1', lambda_reg=0.1)
    }
    
    print("\nCoefficients Comparison:")
    print(f"{'Feature':<10}", end="")
    for name in models.keys():
        print(f"{name:>20}", end="")
    print()
    print("-" * 70)
    
    for name, model in models.items():
        model.fit(X_train, y_train, verbose=False)
    
    for i in range(5):
        print(f"Feature {i:<3}", end="")
        for model in models.values():
            print(f"{model.weights[i]:>20.6f}", end="")
        print()
    
    print("\nTest Set Performance:")
    for name, model in models.items():
        r2 = model.score(X_test, y_test)
        print(f"  {name}: R² = {r2:.4f}")


# ============================================================================
# EXAMPLE 6: POLYNOMIAL REGRESSION
# ============================================================================

def example_6_polynomial_regression():
    """Demonstrate polynomial regression."""
    
    print("\n" + "="*70)
    print("EXAMPLE 6: POLYNOMIAL REGRESSION")
    print("="*70)
    
    from models import LinearRegressionScratch
    from preprocessing import DataPreprocessor
    
    # Generate non-linear data
    np.random.seed(42)
    X = np.linspace(0, 3, 100).reshape(-1, 1)
    y = 0.5 * X.squeeze()**2 + X.squeeze() + 2 + np.random.randn(100) * 0.3
    
    # Create polynomial features
    preprocessor = DataPreprocessor(scaling_method='none')
    X_poly = preprocessor.create_polynomial_features(X, degree=2)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2)
    
    # Train model
    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=2000)
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    
    print(f"\nPolynomial Regression (Degree 2)")
    print(f"R² Score: {r2:.4f}")
    print(f"Coefficients: {model.weights}")
    
    # Plot
    X_plot = np.linspace(0, 3, 300).reshape(-1, 1)
    X_plot_poly = preprocessor.create_polynomial_features(X_plot, degree=2)
    y_plot = model.predict(X_plot_poly)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, s=50, edgecolors='k', label='Data')
    plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Polynomial Fit')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Polynomial Regression (Degree 2)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE 7: RESIDUAL ANALYSIS
# ============================================================================

def example_7_residual_analysis():
    """Comprehensive residual analysis."""
    
    print("\n" + "="*70)
    print("EXAMPLE 7: RESIDUAL ANALYSIS")
    print("="*70)
    
    from models import LinearRegressionScratch, ModelVisualizer
    from scipy import stats
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = 2 + 3*X[:, 0] - 1.5*X[:, 1] + 0.5*X[:, 2] + np.random.randn(200)*0.8
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1500)
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions and residuals
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    # Statistical tests
    print("\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std Dev: {residuals.std():.6f}")
    print(f"  Min: {residuals.min():.6f}")
    print(f"  Max: {residuals.max():.6f}")
    
    # Normality test
    _, p_value = stats.normaltest(residuals)
    print(f"\nNormality Test (p-value): {p_value:.6f}")
    print(f"Residuals are {'normally distributed' if p_value > 0.05 else 'NOT normally distributed'} (α=0.05)")
    
    # Visualize residuals
    visualizer = ModelVisualizer()
    visualizer.plot_residuals(y_test, y_pred)
    plt.show()


# ============================================================================
# EXAMPLE 8: BATCH PREDICTION
# ============================================================================

def example_8_batch_prediction():
    """Make predictions on new data."""
    
    print("\n" + "="*70)
    print("EXAMPLE 8: BATCH PREDICTION")
    print("="*70)
    
    from models import LinearRegressionScratch
    from preprocessing import DataPreprocessor, generate_salary_dataset
    
    # Load and train on full dataset
    df = generate_salary_dataset(n_samples=500)
    
    preprocessor = DataPreprocessor(scaling_method='standard')
    X, y = preprocessor.prepare_features(df, target_column='Salary')
    
    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=2000)
    model.fit(X, y, verbose=False)
    
    # Create new data for prediction
    new_employees = pd.DataFrame({
        'YearsExperience': [1, 3, 5, 10, 15],
        'EducationLevel': [2, 2, 3, 4, 4],
        'Age': [23, 27, 32, 38, 42],
        'HoursPerWeek': [40, 42, 45, 50, 55]
    })
    
    # Prepare new data (use same preprocessor, but don't fit)
    X_new = new_employees.values
    X_new_scaled = preprocessor.scaler.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_new_scaled)
    
    print("\nSalary Predictions for New Employees:")
    print("-" * 70)
    print(f"{'Exp':>4} {'Edu':>4} {'Age':>4} {'Hours':>6} {'Predicted Salary':>20}")
    print("-" * 70)
    
    for i, row in new_employees.iterrows():
        print(f"{row['YearsExperience']:>4.0f} {row['EducationLevel']:>4.0f} "
              f"{row['Age']:>4.0f} {row['HoursPerWeek']:>6.0f} "
              f"${predictions[i]:>19,.2f}")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

def run_all_examples():
    """Run all examples sequentially."""
    
    print("\n" + "="*70)
    print("RUNNING ALL EXAMPLES")
    print("="*70)
    
    examples = [
        ("Basic Regression", example_1_basic_regression),
        ("Multivariate Regression", example_2_multivariate_regression),
        ("Sklearn Comparison", example_3_sklearn_comparison),
        ("Gradient Descent Visualization", example_4_gradient_descent_visualization),
        ("Regularization", example_5_regularization),
        ("Polynomial Regression", example_6_polynomial_regression),
        ("Residual Analysis", example_7_residual_analysis),
        ("Batch Prediction", example_8_batch_prediction)
    ]
    
    for name, func in examples:
        try:
            func()
            print(f"\n✓ {name} completed successfully!")
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "."*70 + "\n")


if __name__ == "__main__":
    # Run individual examples or all at once
    
    # Option 1: Run specific example
    # example_1_basic_regression()
    
    # Option 2: Run all examples
    run_all_examples()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)