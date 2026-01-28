
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# XGBoost
from xgboost import XGBClassifier

# Statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Utilities
import pickle

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_explore_data(filepath):
    """Load the dataset and perform initial exploration."""
    print("="*70)
    print("STEP 1: LOADING AND EXPLORING DATA")
    print("="*70)
    
    df = pd.read_csv(filepath)
    
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"\n📋 Columns:\n{df.columns.tolist()}")
    print(f"\n🔍 First few rows:")
    print(df.head())
    
    print(f"\n📈 Dataset Info:")
    df.info()
    
    print(f"\n📊 Basic Statistics:")
    print(df.describe())
    
    print(f"\n❌ Missing Values:")
    print(df.isnull().sum())
    
    print(f"\n🎯 Target Variable Distribution:")
    print(df['Churn'].value_counts())
    print(f"\nChurn Rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
    
    return df


def visualize_eda(df):
    """Perform Exploratory Data Analysis with visualizations."""
    print("\n" + "="*70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Churn distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.countplot(data=df, x='Churn', ax=axes[0], palette='Set2')
    axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
    
    churn_counts = df['Churn'].value_counts()
    axes[1].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=['#66b3ff', '#ff9999'])
    axes[1].set_title('Churn Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: churn_distribution.png")
    plt.show()
    
    # Churn rate by key features
    key_features = ['Contract', 'InternetService', 'PaymentMethod']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, feature in enumerate(key_features):
        if feature in df.columns:
            churn_rate = df.groupby(feature)['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).sort_values(ascending=False)
            
            churn_rate.plot(kind='bar', ax=axes[idx], color='coral')
            axes[idx].set_title(f'Churn Rate by {feature}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Churn Rate (%)')
            axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('churn_by_features.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: churn_by_features.png")
    plt.show()


def preprocess_data(df):
    """Preprocess the data: handle missing values, encode variables."""
    print("\n" + "="*70)
    print("STEP 3: DATA PREPROCESSING")
    print("="*70)
    
    df_processed = df.copy()
    
    # Drop customerID
    if 'customerID' in df_processed.columns:
        df_processed = df_processed.drop('customerID', axis=1)
        print("\n✅ Dropped customerID column")
    
    # Convert TotalCharges to numeric
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    
    # Fill missing values
    df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
    print(f"✅ Handled missing values. Total missing: {df_processed.isnull().sum().sum()}")
    
    # Encode target variable
    df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
    print("✅ Encoded target variable (Yes=1, No=0)")
    
    # Identify categorical columns
    binary_cols = []
    multi_cols = []
    
    for col in df_processed.select_dtypes(include=['object']).columns:
        if df_processed[col].nunique() == 2:
            binary_cols.append(col)
        else:
            multi_cols.append(col)
    
    print(f"\n📊 Binary columns: {binary_cols}")
    print(f"📊 Multi-class columns: {multi_cols}")
    
    # Label encode binary columns
    le = LabelEncoder()
    for col in binary_cols:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    # One-hot encode multi-class columns
    df_processed = pd.get_dummies(df_processed, columns=multi_cols, drop_first=True)
    
    print(f"\n✅ Encoding complete. New shape: {df_processed.shape}")
    
    return df_processed


def statistical_analysis(df_processed):
    """Perform statistical analysis using statsmodels."""
    print("\n" + "="*70)
    print("STEP 4: STATISTICAL ANALYSIS")
    print("="*70)
    
    # Correlation matrix
    correlation_matrix = df_processed.corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: correlation_heatmap.png")
    plt.show()
    
    print("\n📊 Top 10 correlations with Churn:")
    print(correlation_matrix['Churn'].sort_values(ascending=False).head(10))
    
    # VIF for numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_temp = df_processed[numerical_features]
    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numerical_features
    vif_data['VIF'] = [variance_inflation_factor(X_temp.values, i) 
                       for i in range(len(numerical_features))]
    
    print("\n📊 Variance Inflation Factor (VIF):")
    print(vif_data)
    print("Note: VIF > 10 indicates high multicollinearity")


def prepare_train_test_split(df_processed):
    """Split data into train, validation, and test sets."""
    print("\n" + "="*70)
    print("STEP 5: TRAIN-VALIDATION-TEST SPLIT")
    print("="*70)
    
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    print(f"\n📊 Features shape: {X.shape}")
    print(f"📊 Target shape: {y.shape}")
    print(f"📊 Churn rate: {(y.sum() / len(y)) * 100:.2f}%")
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Second split: 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n✅ Training set: {X_train.shape[0]} samples ({(X_train.shape[0]/len(X))*100:.1f}%)")
    print(f"✅ Validation set: {X_val.shape[0]} samples ({(X_val.shape[0]/len(X))*100:.1f}%)")
    print(f"✅ Test set: {X_test.shape[0]} samples ({(X_test.shape[0]/len(X))*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n✅ Feature scaling completed")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def train_xgboost_with_early_stopping(X_train, X_val, y_train, y_val):
    """Train XGBoost model with early stopping."""
    print("\n" + "="*70)
    print("STEP 6: TRAINING XGBOOST WITH EARLY STOPPING")
    print("="*70)
    
    # Initialize model
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        eval_metric='auc'
    )
    
    print("\n🚀 Training model with early stopping...")
    print(f"   Max iterations: {xgb_model.n_estimators}")
    print(f"   Early stopping rounds: 50")
    print(f"   Evaluation metric: AUC")
    
    # Train with early stopping
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50
    )
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"🎯 Best iteration: {xgb_model.best_iteration}")
    print(f"🎯 Best score: {xgb_model.best_score:.4f}")
    print(f"💾 Iterations saved: {1000 - xgb_model.best_iteration}")
    
    # Plot training progress
    results = xgb_model.evals_result()
    
    plt.figure(figsize=(14, 6))
    plt.plot(results['validation_0']['auc'], label='Training AUC', linewidth=2)
    plt.plot(results['validation_1']['auc'], label='Validation AUC', linewidth=2)
    plt.axvline(x=xgb_model.best_iteration, color='red', linestyle='--', 
                label=f'Best Iteration ({xgb_model.best_iteration})', linewidth=2)
    plt.xlabel('Number of Boosting Rounds', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('XGBoost Training Progress with Early Stopping', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: training_progress.png")
    plt.show()
    
    return xgb_model


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate model performance on all datasets."""
    print("\n" + "="*70)
    print("STEP 7: MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Print metrics for each set
    for name, y_true, y_pred, y_proba in [
        ("Training", y_train, y_train_pred, y_train_proba),
        ("Validation", y_val, y_val_pred, y_val_proba),
        ("Test", y_test, y_test_pred, y_test_proba)
    ]:
        print(f"\n{'='*60}")
        print(f"{name} Set Performance")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
        print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
        print(f"AUC-ROC:   {roc_auc_score(y_true, y_proba):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: confusion_matrix.png")
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: roc_curve.png")
    plt.show()
    
    return y_test_pred, y_test_proba


def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance."""
    print("\n" + "="*70)
    print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Top 20 Most Important Features:")
    print(feature_importance.head(20))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
    plt.title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: feature_importance.png")
    plt.show()
    
    return feature_importance


def save_model(model, scaler, filepath_model='xgb_churn_model.pkl', 
               filepath_scaler='scaler.pkl'):
    """Save the trained model and scaler."""
    print("\n" + "="*70)
    print("STEP 9: SAVING MODEL")
    print("="*70)
    
    with open(filepath_model, 'wb') as f:
        pickle.dump(model, f)
    
    with open(filepath_scaler, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n✅ Model saved: {filepath_model}")
    print(f"✅ Scaler saved: {filepath_scaler}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("CUSTOMER CHURN PREDICTION WITH XGBOOST")
    print("Using Early Stopping to Prevent Overfitting")
    print("="*70)
    
    # Define file path (update this to your dataset location)
    DATASET_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    try:
        # Step 1: Load and explore data
        df = load_and_explore_data(DATASET_PATH)
        
        # Step 2: EDA with visualizations
        visualize_eda(df)
        
        # Step 3: Preprocess data
        df_processed = preprocess_data(df)
        
        # Step 4: Statistical analysis
        statistical_analysis(df_processed)
        
        # Step 5: Prepare train-test split
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
            prepare_train_test_split(df_processed)
        
        # Step 6: Train XGBoost with early stopping
        xgb_model = train_xgboost_with_early_stopping(
            X_train, X_val, y_train, y_val
        )
        
        # Step 7: Evaluate model
        y_test_pred, y_test_proba = evaluate_model(
            xgb_model, X_train, X_val, X_test, 
            y_train, y_val, y_test
        )
        
        # Step 8: Feature importance
        feature_importance = analyze_feature_importance(
            xgb_model, X_train.columns
        )
        
        # Step 9: Save model
        save_model(xgb_model, scaler)
        
        # Final summary
        print("\n" + "="*70)
        print("PROJECT COMPLETED SUCCESSFULLY! 🎉")
        print("="*70)
        print(f"\n📊 Final Results:")
        print(f"   • Test AUC-ROC: {roc_auc_score(y_test, y_test_proba):.4f}")
        print(f"   • Test F1-Score: {f1_score(y_test, y_test_pred):.4f}")
        print(f"   • Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print(f"   • Optimal Trees: {xgb_model.best_iteration}")
        print(f"\n💡 Key Takeaway:")
        print(f"   Early stopping prevented overfitting by stopping at")
        print(f"   iteration {xgb_model.best_iteration} instead of 1000!")
        
    except FileNotFoundError:
        print(f"\n Error: Dataset not found at '{DATASET_PATH}'")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/blastchar/telco-customer-churn")
        print("And update the DATASET_PATH variable in the script.")
    
    except Exception as e:
        print(f"\n An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()