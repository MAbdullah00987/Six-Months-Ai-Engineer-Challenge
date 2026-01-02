
#Part 4: Complete ML Pipeline - Real Estate Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(42)
sns.set_style("whitegrid")

print("="*70)
print("COMPLETE ML PIPELINE: REAL ESTATE PRICE PREDICTION")
print("="*70)

# ===== PHASE 1: DATA GENERATION AND EXPLORATION =====
print("\n" + "="*70)
print("PHASE 1: DATA GENERATION AND EXPLORATION")
print("="*70)

n_samples = 500

# Realistic feature ranges
data = {
    'sqft': np.random.uniform(800, 4000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 5, n_samples),
    'age_years': np.random.uniform(0, 100, n_samples),
    'lot_size': np.random.uniform(2000, 10000, n_samples),
    'garage_spaces': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.5, 0.1]),
    'stories': np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.5, 0.1]),
}

df = pd.DataFrame(data)

# Add location (categorical)
df['location'] = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.3, 0.5, 0.2])

# Generate realistic price with complex relationships
base_price = 50000
sqft_price = 150
bedroom_value = 20000
bathroom_value = 15000
age_depreciation = -500
lot_value = 5
garage_value = 10000

location_premium = {'Urban': 100000, 'Suburban': 50000, 'Rural': 0}

df['price'] = (
    base_price +
    sqft_price * df['sqft'] +
    bedroom_value * df['bedrooms'] +
    bathroom_value * df['bathrooms'] +
    age_depreciation * df['age_years'] +
    lot_value * df['lot_size'] +
    garage_value * df['garage_spaces'] +
    df['location'].map(location_premium) +
    np.random.randn(n_samples) * 50000  # Noise
)

# Add some outliers
outlier_indices = np.random.choice(n_samples, 10, replace=False)
df.loc[outlier_indices, 'price'] *= np.random.uniform(1.5, 2.0, 10)

print("\n1.1 Dataset Overview")
print("-" * 70)
print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

print("\n1.2 Statistical Summary")
print("-" * 70)
print(df.describe())

# ===== PHASE 2: EXPLORATORY DATA ANALYSIS =====
print("\n" + "="*70)
print("PHASE 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

# Create comprehensive EDA visualizations
fig = plt.figure(figsize=(18, 12))

# Plot 1: Price distribution
ax1 = plt.subplot(3, 4, 1)
sns.histplot(df['price'], bins=40, kde=True, ax=ax1, color='skyblue')
ax1.axvline(df['price'].median(), color='red', linestyle='--', linewidth=2)
ax1.set_title('Price Distribution', fontweight='bold')
ax1.set_xlabel('Price ($)')

# Plot 2: Price by location
ax2 = plt.subplot(3, 4, 2)
sns.boxplot(data=df, x='location', y='price', ax=ax2, palette='Set2')
ax2.set_title('Price by Location', fontweight='bold')
ax2.set_ylabel('Price ($)')

# Plot 3: Sqft vs Price
ax3 = plt.subplot(3, 4, 3)
sns.scatterplot(data=df, x='sqft', y='price', hue='location', ax=ax3, alpha=0.6, s=30)
ax3.set_title('Square Feet vs Price', fontweight='bold')

# Plot 4: Age vs Price
ax4 = plt.subplot(3, 4, 4)
sns.scatterplot(data=df, x='age_years', y='price', alpha=0.6, ax=ax4, s=30, color='coral')
ax4.set_title('Age vs Price', fontweight='bold')

# Plot 5: Correlation heatmap
ax5 = plt.subplot(3, 4, 5)
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax5, 
            cbar_kws={'shrink': 0.8}, square=True)
ax5.set_title('Correlation Matrix', fontweight='bold')

# Plot 6: Bedrooms distribution
ax6 = plt.subplot(3, 4, 6)
bedroom_counts = df['bedrooms'].value_counts().sort_index()
ax6.bar(bedroom_counts.index, bedroom_counts.values, color='lightgreen', edgecolor='black')
ax6.set_xlabel('Bedrooms')
ax6.set_ylabel('Count')
ax6.set_title('Bedroom Distribution', fontweight='bold')

# Plot 7: Price vs Garage
ax7 = plt.subplot(3, 4, 7)
sns.boxplot(data=df, x='garage_spaces', y='price', ax=ax7, palette='pastel')
ax7.set_title('Price vs Garage Spaces', fontweight='bold')

# Plot 8: Stories distribution
ax8 = plt.subplot(3, 4, 8)
story_counts = df['stories'].value_counts().sort_index()
ax8.bar(story_counts.index, story_counts.values, color='lightcoral', edgecolor='black')
ax8.set_xlabel('Stories')
ax8.set_ylabel('Count')
ax8.set_title('Stories Distribution', fontweight='bold')

# Plot 9: Lot size vs Price
ax9 = plt.subplot(3, 4, 9)
sns.scatterplot(data=df, x='lot_size', y='price', alpha=0.6, ax=ax9, s=30, color='purple')
ax9.set_title('Lot Size vs Price', fontweight='bold')

# Plot 10: Bathrooms vs Price
ax10 = plt.subplot(3, 4, 10)
sns.boxplot(data=df, x='bathrooms', y='price', ax=ax10, palette='muted')
ax10.set_title('Price vs Bathrooms', fontweight='bold')

# Plot 11: Price per sqft
ax11 = plt.subplot(3, 4, 11)
df['price_per_sqft'] = df['price'] / df['sqft']
sns.histplot(df['price_per_sqft'], bins=40, kde=True, ax=ax11, color='orange')
ax11.set_title('Price per Square Foot', fontweight='bold')

# Plot 12: Location distribution
ax12 = plt.subplot(3, 4, 12)
location_counts = df['location'].value_counts()
ax12.pie(location_counts.values, labels=location_counts.index, autopct='%1.1f%%', 
         colors=['#ff9999', '#66b3ff', '#99ff99'], startangle=90)
ax12.set_title('Location Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_comprehensive.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n2.1 Key Insights from EDA:")
print("-" * 70)
print(f"• Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"• Median price: ${df['price'].median():,.0f}")
print(f"• Average price per sqft: ${df['price_per_sqft'].mean():.2f}")
print(f"• Strongest correlation with price: {corr['price'].abs().sort_values(ascending=False).index[1]}")

# ===== PHASE 3: FEATURE ENGINEERING =====
print("\n" + "="*70)
print("PHASE 3: FEATURE ENGINEERING")
print("="*70)

# One-hot encode location
df_encoded = pd.get_dummies(df, columns=['location'], drop_first=True)

# Create interaction features
df_encoded['sqft_x_bedrooms'] = df_encoded['sqft'] * df_encoded['bedrooms']
df_encoded['total_rooms'] = df_encoded['bedrooms'] + df_encoded['bathrooms']
df_encoded['age_squared'] = df_encoded['age_years'] ** 2

print("\n3.1 Feature Engineering:")
print("-" * 70)
print(f"Original features: {df.shape[1]}")
print(f"After encoding and engineering: {df_encoded.shape[1]}")
print(f"\nNew features: {list(df_encoded.columns[-5:])}")

# ===== PHASE 4: DATA PREPROCESSING =====
print("\n" + "="*70)
print("PHASE 4: DATA PREPROCESSING")
print("="*70)

# Separate features and target
feature_cols = [col for col in df_encoded.columns if col not in ['price', 'price_per_sqft']]
X = df_encoded[feature_cols].values
y = df_encoded['price'].values

print(f"\n4.1 Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n4.2 Split sizes:")
print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n4.3 Scaling completed")
print(f"Mean of scaled features (should be ~0): {X_train_scaled.mean():.6f}")
print(f"Std of scaled features (should be ~1): {X_train_scaled.std():.6f}")

# ===== PHASE 5: MODEL TRAINING =====
print("\n" + "="*70)
print("PHASE 5: MODEL TRAINING WITH ADAM")
print("="*70)

class LinearRegressionAdam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, n_iterations=1000):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = {'loss': [], 'train_r2': [], 'val_r2': []}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Initialize Adam parameters
        m_w, m_b = np.zeros(n_features), 0.0
        v_w, v_b = np.zeros(n_features), 0.0
        
        for t in range(1, self.n_iterations + 1):
            # Forward pass
            y_pred = X_train @ self.weights + self.bias
            
            # Compute loss
            loss = np.mean((y_train - y_pred) ** 2)
            self.history['loss'].append(loss)
            
            # Compute gradients
            error = y_pred - y_train
            grad_w = (2/n_samples) * (X_train.T @ error)
            grad_b = (2/n_samples) * np.sum(error)
            
            # Update first moment
            m_w = self.beta1 * m_w + (1 - self.beta1) * grad_w
            m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b
            
            # Update second moment
            v_w = self.beta2 * v_w + (1 - self.beta2) * grad_w**2
            v_b = self.beta2 * v_b + (1 - self.beta2) * grad_b**2
            
            # Bias correction
            m_w_hat = m_w / (1 - self.beta1**t)
            m_b_hat = m_b / (1 - self.beta1**t)
            v_w_hat = v_w / (1 - self.beta2**t)
            v_b_hat = v_b / (1 - self.beta2**t)
            
            # Update parameters
            self.weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            self.bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            
            # Track R² scores
            if X_val is not None and y_val is not None:
                train_r2 = r2_score(y_train, y_pred)
                val_pred = self.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)
                self.history['train_r2'].append(train_r2)
                self.history['val_r2'].append(val_r2)
            
            if t % 100 == 0:
                print(f"Iteration {t}/{self.n_iterations}, Loss: {loss:.2f}")
    
    def predict(self, X):
        return X @ self.weights + self.bias

# Train model
model = LinearRegressionAdam(learning_rate=0.01, n_iterations=1000)
model.fit(X_train_scaled, y_train, X_test_scaled, y_test)

# ===== PHASE 6: MODEL EVALUATION =====
print("\n" + "="*70)
print("PHASE 6: MODEL EVALUATION")
print("="*70)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Metrics
metrics = {
    'Train': {
        'R²': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    },
    'Test': {
        'R²': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
}

print("\n6.1 Performance Metrics:")
print("-" * 70)
print(f"{'Metric':<10} {'Training':<20} {'Testing':<20}")
print("-" * 70)
for metric in ['R²', 'RMSE', 'MAE']:
    print(f"{metric:<10} {metrics['Train'][metric]:<20.4f} {metrics['Test'][metric]:<20.4f}")

# Feature importance
print("\n6.2 Top 10 Most Important Features:")
print("-" * 70)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.weights,
    'abs_coefficient': np.abs(model.weights)
}).sort_values('abs_coefficient', ascending=False).head(10)

for idx, row in feature_importance.iterrows():
    print(f"{row['feature']:<30s}: {row['coefficient']:>10.4f}")

# ===== PHASE 7: VISUALIZATION OF RESULTS =====
print("\n" + "="*70)
print("PHASE 7: RESULTS VISUALIZATION")
print("="*70)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Training history
ax1 = plt.subplot(2, 3, 1)
ax1.plot(model.history['loss'], linewidth=2, color='blue')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training Loss Curve', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: R² evolution
ax2 = plt.subplot(2, 3, 2)
ax2.plot(model.history['train_r2'], linewidth=2, label='Train R²', color='blue')
ax2.plot(model.history['val_r2'], linewidth=2, label='Test R²', color='red')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('R² Score')
ax2.set_title('R² Score Evolution', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Predicted vs Actual (Test)
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_test, y_test_pred, alpha=0.5, s=30)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax3.set_xlabel('Actual Price ($)')
ax3.set_ylabel('Predicted Price ($)')
ax3.set_title(f'Predictions (Test R²={metrics["Test"]["R²"]:.3f})', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Residual plot
ax4 = plt.subplot(2, 3, 4)
residuals = y_test - y_test_pred
ax4.scatter(y_test_pred, residuals, alpha=0.5, s=30)
ax4.axhline(0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Price ($)')
ax4.set_ylabel('Residuals ($)')
ax4.set_title('Residual Plot', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Residual distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
ax5.axvline(0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Residual ($)')
ax5.set_ylabel('Frequency')
ax5.set_title('Residual Distribution', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Feature importance
ax6 = plt.subplot(2, 3, 6)
top_features = feature_importance.head(10)
colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]
ax6.barh(range(len(top_features)), top_features['abs_coefficient'], color=colors, alpha=0.7)
ax6.set_yticks(range(len(top_features)))
ax6.set_yticklabels([f[:20] for f in top_features['feature']], fontsize=9)
ax6.set_xlabel('|Coefficient|')
ax6.set_title('Top 10 Feature Importance', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('complete_pipeline_results.png', dpi=150, bbox_inches='tight')
plt.show()

