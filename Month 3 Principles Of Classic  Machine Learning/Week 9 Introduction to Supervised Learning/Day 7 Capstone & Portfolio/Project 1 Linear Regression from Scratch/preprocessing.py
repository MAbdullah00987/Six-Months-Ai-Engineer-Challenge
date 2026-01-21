

"""
Data Preprocessing Utilities
preprocessing.py - Contains functions for data preprocessing and feature engineering
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureScaler:
    """
    Custom feature scaling implementation
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Parameters:
        -----------
        method : str, default='standard'
            Scaling method: 'standard', 'minmax', or 'normalize'
        """
        self.method = method
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        
    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        """Compute scaling parameters"""
        X = np.array(X)
        
        if self.method == 'standard':
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
        elif self.method == 'minmax':
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply scaling transformation"""
        X = np.array(X)
        
        if self.method == 'standard':
            # Z-score normalization: (x - mean) / std
            return (X - self.mean_) / (self.std_ + 1e-8)
        elif self.method == 'minmax':
            # Min-Max scaling: (x - min) / (max - min)
            return (X - self.min_) / (self.max_ - self.min_ + 1e-8)
        elif self.method == 'normalize':
            # L2 normalization
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            return X / (norms + 1e-8)
        
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaling transformation"""
        X = np.array(X)
        
        if self.method == 'standard':
            return X * self.std_ + self.mean_
        elif self.method == 'minmax':
            return X * (self.max_ - self.min_) + self.min_
        
        return X


def train_test_split_custom(X: np.ndarray, y: np.ndarray, 
                            test_size: float = 0.2, 
                            random_state: Optional[int] = None) -> Tuple:
    """
    Split data into train and test sets
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Features
    y : array-like, shape (n_samples,)
        Target
    test_size : float, default=0.2
        Proportion of test set
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Random permutation
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str, default='mean'
        Strategy for imputation: 'mean', 'median', 'mode', 'drop'
    
    Returns:
    --------
    pd.DataFrame with missing values handled
    """
    df = df.copy()
    
    if strategy == 'drop':
        return df.dropna()
    
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                # For categorical columns, use mode
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


def encode_categorical(df: pd.DataFrame, columns: List[str], 
                       method: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to encode
    method : str, default='onehot'
        Encoding method: 'onehot' or 'label'
    
    Returns:
    --------
    pd.DataFrame with encoded variables
    """
    df = df.copy()
    
    if method == 'onehot':
        df = pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        for col in columns:
            df[col] = pd.Categorical(df[col]).codes
    
    return df


def remove_outliers(X: np.ndarray, y: np.ndarray, 
                    method: str = 'iqr', 
                    threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from dataset
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    method : str, default='iqr'
        Method for outlier detection: 'iqr' or 'zscore'
    threshold : float
        Threshold for outlier detection
    
    Returns:
    --------
    X_clean, y_clean without outliers
    """
    if method == 'iqr':
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((y - np.mean(y)) / np.std(y))
        mask = z_scores < threshold
    
    return X[mask], y[mask]


def polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Generate polynomial features
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
    degree : int, default=2
        Polynomial degree
    
    Returns:
    --------
    X_poly with polynomial features
    """
    X = np.array(X)
    n_samples, n_features = X.shape
    
    # Start with original features
    X_poly = X.copy()
    
    # Add polynomial terms
    for d in range(2, degree + 1):
        X_poly = np.c_[X_poly, X ** d]
    
    return X_poly


def create_interaction_features(X: np.ndarray) -> np.ndarray:
    """
    Create interaction features between all pairs of features
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
    
    Returns:
    --------
    X with interaction features added
    """
    X = np.array(X)
    n_samples, n_features = X.shape
    
    X_with_interactions = X.copy()
    
    # Add all pairwise interactions
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
            X_with_interactions = np.c_[X_with_interactions, interaction]
    
    return X_with_interactions


def feature_correlation_analysis(df: pd.DataFrame, target_col: str, 
                                 threshold: float = 0.7) -> pd.DataFrame:
    """
    Analyze feature correlations and identify highly correlated features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    threshold : float, default=0.7
        Correlation threshold for flagging
    
    Returns:
    --------
    pd.DataFrame with correlation analysis
    """
    # Correlation with target
    correlations = df.corr()[target_col].sort_values(ascending=False)
    
    # Find highly correlated feature pairs
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = [
        (column, row, corr_matrix.loc[row, column])
        for column in upper_triangle.columns
        for row in upper_triangle.index
        if upper_triangle.loc[row, column] > threshold
    ]
    
    print(f"\n=== Feature Correlation with Target '{target_col}' ===")
    print(correlations)
    
    if high_corr_pairs:
        print(f"\n=== Highly Correlated Feature Pairs (>{threshold}) ===")
        for col1, col2, corr in high_corr_pairs:
            print(f"{col1} <-> {col2}: {corr:.3f}")
    
    return correlations


def prepare_data_for_modeling(df: pd.DataFrame, 
                              target_col: str,
                              test_size: float = 0.2,
                              scale: bool = True,
                              random_state: Optional[int] = 42) -> dict:
    """
    Complete data preparation pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    test_size : float, default=0.2
        Proportion of test set
    scale : bool, default=True
        Whether to scale features
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict containing X_train, X_test, y_train, y_test, scaler, and feature_names
    """
    # Separate features and target
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_custom(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = None
    if scale:
        scaler = FeatureScaler(method='standard')
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names
    }


def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, 
                            model, k: int = 5, 
                            random_state: Optional[int] = None) -> dict:
    """
    Perform k-fold cross-validation
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    model : object
        Model instance with fit and score methods
    k : int, default=5
        Number of folds
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict with cross-validation results
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // k
    
    scores = []
    
    for i in range(k):
        # Define test indices for this fold
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else n_samples
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        # Split data
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_test_fold = X[test_indices]
        y_test_fold = y[test_indices]
        
        # Train and evaluate
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_test_fold, y_test_fold)
        scores.append(score)
        
        print(f"Fold {i+1}/{k}: R² = {score:.4f}")
    
    return {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores)
    }