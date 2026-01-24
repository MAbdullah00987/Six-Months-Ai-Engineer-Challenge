"""
Data Preprocessing Utilities
src/preprocessing.py
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """
    Comprehensive data preprocessing for machine learning.
    """
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        scaling_method : str
            Method for feature scaling ('standard', 'minmax', or 'none')
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_names = None
        self.target_name = None
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        strategy : str
            Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
            
        Returns:
        --------
        pd.DataFrame : Dataframe with handled missing values
        """
        df_copy = df.copy()
        
        if strategy == 'drop':
            return df_copy.dropna()
        
        for col in df_copy.columns:
            if df_copy[col].isnull().any():
                if df_copy[col].dtype in ['float64', 'int64']:
                    if strategy == 'mean':
                        df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df_copy[col].fillna(df_copy[col].median(), inplace=True)
                else:
                    # For categorical, use mode
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        
        return df_copy
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str], optional
            List of columns to encode. If None, encode all object columns
            
        Returns:
        --------
        pd.DataFrame : Dataframe with encoded categorical variables
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=['object']).columns.tolist()
        
        if columns:
            df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)
        
        return df_copy
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from specified columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to check for outliers
        method : str
            Method for outlier detection ('iqr' or 'zscore')
        threshold : float
            Threshold for outlier detection
            
        Returns:
        --------
        pd.DataFrame : Dataframe with outliers removed
        """
        df_copy = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                df_copy = df_copy[z_scores < threshold]
        
        return df_copy
    
    def prepare_features(self, df: pd.DataFrame, target_column: str,
                        feature_columns: Optional[List[str]] = None,
                        fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for model training.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_column : str
            Name of target column
        feature_columns : List[str], optional
            List of feature columns. If None, use all columns except target
        fit : bool
            Whether to fit the scaler (True for training, False for test)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (X, y) arrays
        """
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        self.feature_names = feature_columns
        self.target_name = target_column
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Scale features
        if self.scaler is not None:
            if fit:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        return X, y
    
    def create_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Create polynomial features.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        degree : int
            Degree of polynomial features
            
        Returns:
        --------
        np.ndarray : Polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X)


def load_and_prepare_data(filepath: str, target_column: str,
                          test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Load and prepare data for training.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    target_column : str
        Name of target column
    test_size : float
        Proportion of test set
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Dictionary containing train/test splits and metadata
    """
    from sklearn.model_selection import train_test_split
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    
    # Prepare features
    preprocessor = DataPreprocessor(scaling_method='standard')
    df_clean = preprocessor.handle_missing_values(df, strategy='mean')
    df_encoded = preprocessor.encode_categorical(df_clean)
    
    X, y = preprocessor.prepare_features(df_encoded, target_column)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': preprocessor.feature_names,
        'target_name': preprocessor.target_name,
        'preprocessor': preprocessor,
        'original_df': df
    }


def generate_salary_dataset(n_samples: int = 500, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate synthetic salary prediction dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    save_path : str, optional
        Path to save CSV file
        
    Returns:
    --------
    pd.DataFrame : Generated dataset
    """
    np.random.seed(42)
    
    # Generate features
    years_experience = np.random.uniform(0, 15, n_samples)
    education_level = np.random.choice([1, 2, 3, 4], n_samples)  # 1=HS, 2=Bachelor, 3=Master, 4=PhD
    age = 22 + years_experience + np.random.uniform(-2, 5, n_samples)
    hours_per_week = np.random.uniform(35, 60, n_samples)
    
    # Generate target (salary)
    base_salary = 30000
    salary = (
        base_salary +
        3000 * years_experience +
        8000 * education_level +
        500 * age +
        200 * hours_per_week +
        np.random.normal(0, 5000, n_samples)
    )
    
    # Create dataframe
    df = pd.DataFrame({
        'YearsExperience': years_experience,
        'EducationLevel': education_level,
        'Age': age,
        'HoursPerWeek': hours_per_week,
        'Salary': salary
    })
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
    
    return df


def generate_housing_dataset(n_samples: int = 1000, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate synthetic housing price dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    save_path : str, optional
        Path to save CSV file
        
    Returns:
    --------
    pd.DataFrame : Generated dataset
    """
    np.random.seed(42)
    
    # Generate features
    square_feet = np.random.uniform(800, 4000, n_samples)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples)
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], n_samples)
    age = np.random.uniform(0, 50, n_samples)
    location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
    
    # Generate target (price)
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
    
    # Create dataframe
    df = pd.DataFrame({
        'SquareFeet': square_feet,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age': age,
        'Location': location,
        'Price': price
    })
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
    
    return df


if __name__ == "__main__":
    # Generate sample datasets
    print("Generating sample datasets...")
    
    salary_df = generate_salary_dataset(n_samples=500)
    print("\nSalary Dataset Preview:")
    print(salary_df.head())
    print(f"\nShape: {salary_df.shape}")
    print(f"\nStatistics:\n{salary_df.describe()}")
    
    housing_df = generate_housing_dataset(n_samples=1000)
    print("\n" + "="*60)
    print("Housing Dataset Preview:")
    print(housing_df.head())
    print(f"\nShape: {housing_df.shape}")
    print(f"\nStatistics:\n{housing_df.describe()}")