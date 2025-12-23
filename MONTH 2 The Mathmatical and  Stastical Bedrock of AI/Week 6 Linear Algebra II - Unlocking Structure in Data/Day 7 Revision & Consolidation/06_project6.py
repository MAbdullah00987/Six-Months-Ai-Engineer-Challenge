
# Poject 6:
# Recommender System with SVD: Implement a basic movie recommender system using SVD on a user-item rating matrix.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

# ============================================================================
# Create Synthetic Movie Rating Dataset
# ============================================================================

def create_movie_dataset():
    """Create a synthetic movie rating dataset"""
    
    # Movies with genres
    movies = {
        'Movie ID': range(1, 21),
        'Title': [
            'The Matrix', 'Inception', 'Interstellar', 'The Dark Knight',
            'Avengers', 'Iron Man', 'Thor', 'Black Panther',
            'The Notebook', 'Titanic', 'La La Land', 'Pride & Prejudice',
            'The Hangover', 'Superbad', 'Step Brothers', 'Bridesmaids',
            'The Conjuring', 'Get Out', 'A Quiet Place', 'Hereditary'
        ],
        'Genre': [
            'Sci-Fi', 'Sci-Fi', 'Sci-Fi', 'Action',
            'Action', 'Action', 'Action', 'Action',
            'Romance', 'Romance', 'Romance', 'Romance',
            'Comedy', 'Comedy', 'Comedy', 'Comedy',
            'Horror', 'Horror', 'Horror', 'Horror'
        ]
    }
    
    movies_df = pd.DataFrame(movies)
    
    # Create user-movie rating matrix (users x movies)
    # Ratings from 1-5, with NaN for unrated movies
    np.random.seed(42)
    n_users = 15
    n_movies = 20
    
    # Create sparse ratings (not all users rate all movies)
    ratings = np.full((n_users, n_movies), np.nan)
    
    # User preferences (to create realistic patterns)
    user_preferences = {
        'Sci-Fi Lovers': [0, 1, 2, 3],  # Users who love sci-fi
        'Action Fans': [4, 5, 6, 7],    # Users who love action
        'Romance Lovers': [8, 9, 10],   # Users who love romance
        'Comedy Fans': [11, 12],        # Users who love comedy
        'Horror Fans': [13, 14]         # Users who love horror
    }
    
    # Assign ratings based on preferences
    for user in range(n_users):
        # Determine user type
        if user < 4:  # Sci-Fi lovers
            # Rate sci-fi movies highly
            ratings[user, 0:4] = np.random.randint(4, 6, 4)
            ratings[user, 4:8] = np.random.randint(3, 5, 4)  # Like action too
            ratings[user, 8:12] = np.random.randint(1, 3, 4)  # Don't like romance
        elif user < 8:  # Action fans
            ratings[user, 4:8] = np.random.randint(4, 6, 4)
            ratings[user, 0:4] = np.random.randint(3, 5, 4)
            ratings[user, 8:12] = np.random.randint(2, 4, 4)
        elif user < 11:  # Romance lovers
            ratings[user, 8:12] = np.random.randint(4, 6, 4)
            ratings[user, 12:16] = np.random.randint(3, 5, 4)  # Like comedy
            ratings[user, 0:4] = np.random.randint(1, 3, 4)
        elif user < 13:  # Comedy fans
            ratings[user, 12:16] = np.random.randint(4, 6, 4)
            ratings[user, 8:12] = np.random.randint(3, 5, 4)
        else:  # Horror fans
            ratings[user, 16:20] = np.random.randint(4, 6, 4)
            ratings[user, 0:8] = np.random.randint(2, 4, 8)
    
    # Add some random ratings for sparsity
    for user in range(n_users):
        n_random = np.random.randint(2, 5)
        random_movies = np.random.choice(n_movies, n_random, replace=False)
        for movie in random_movies:
            if np.isnan(ratings[user, movie]):
                ratings[user, movie] = np.random.randint(1, 6)
    
    return ratings, movies_df

# ============================================================================
# SVD-based Recommender System
# ============================================================================

class SVDRecommender:
    """Movie Recommender System using SVD"""
    
    def __init__(self, n_factors=5):
        self.n_factors = n_factors
        self.user_factors = None
        self.movie_factors = None
        self.global_mean = None
        self.predicted_ratings = None
        self.U = None
        self.S = None
        self.Vt = None
        
    def fit(self, ratings_matrix):
        """
        Fit the SVD model
        
        Parameters:
        - ratings_matrix: user-movie rating matrix (with NaN for missing)
        """
        # Step 1: Calculate global mean (only from observed ratings)
        self.global_mean = np.nanmean(ratings_matrix)
        
        # Step 2: Fill missing values with global mean for SVD
        filled_matrix = ratings_matrix.copy()
        filled_matrix[np.isnan(filled_matrix)] = self.global_mean
        
        # Step 3: Center the data (subtract mean)
        centered_matrix = filled_matrix - self.global_mean
        
        # Step 4: Perform SVD
        self.U, self.S, self.Vt = np.linalg.svd(centered_matrix, full_matrices=False)
        
        # Step 5: Reduce to k factors
        self.user_factors = self.U[:, :self.n_factors]
        self.movie_factors = self.Vt[:self.n_factors, :].T
        S_reduced = np.diag(self.S[:self.n_factors])
        
        # Step 6: Reconstruct the rating matrix
        reconstructed = self.user_factors @ S_reduced @ self.movie_factors.T
        self.predicted_ratings = reconstructed + self.global_mean
        
        # Clip ratings to valid range [1, 5]
        self.predicted_ratings = np.clip(self.predicted_ratings, 1, 5)
        
        print(f"SVD Decomposition completed:")
        print(f"  Original matrix: {ratings_matrix.shape}")
        print(f"  User factors (U): {self.user_factors.shape}")
        print(f"  Singular values (S): {self.n_factors} factors")
        print(f"  Movie factors (V^T): {self.movie_factors.shape}")
        print(f"  Global mean rating: {self.global_mean:.2f}")
        
    def predict(self, user_id, movie_id):
        """Predict rating for a specific user-movie pair"""
        return self.predicted_ratings[user_id, movie_id]
    
    def recommend_movies(self, user_id, movies_df, n_recommendations=5, 
                        rated_movies=None):
        """
        Recommend top N movies for a user
        
        Parameters:
        - user_id: user index
        - movies_df: DataFrame with movie information
        - n_recommendations: number of movies to recommend
        - rated_movies: list of movie indices already rated by user
        """
        # Get predicted ratings for this user
        user_predictions = self.predicted_ratings[user_id, :]
        
        # Filter out already rated movies
        if rated_movies is not None:
            user_predictions = user_predictions.copy()
            user_predictions[rated_movies] = -1
        
        # Get top N movie indices
        top_indices = np.argsort(user_predictions)[::-1][:n_recommendations]
        
        # Create recommendations DataFrame
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'Movie ID': movies_df.iloc[idx]['Movie ID'],
                'Title': movies_df.iloc[idx]['Title'],
                'Genre': movies_df.iloc[idx]['Genre'],
                'Predicted Rating': user_predictions[idx]
            })
        
        return pd.DataFrame(recommendations)
    
    def evaluate(self, actual_ratings):
        """
        Evaluate the model on observed ratings
        
        Parameters:
        - actual_ratings: original rating matrix
        """
        # Get mask of observed ratings
        mask = ~np.isnan(actual_ratings)
        
        # Calculate metrics only on observed ratings
        actual_observed = actual_ratings[mask]
        predicted_observed = self.predicted_ratings[mask]
        
        rmse = np.sqrt(mean_squared_error(actual_observed, predicted_observed))
        mae = mean_absolute_error(actual_observed, predicted_observed)
        
        return rmse, mae

# ============================================================================
# Load Data and Train Model
# ============================================================================

print("="*80)
print("MOVIE RECOMMENDER SYSTEM WITH SVD")
print("="*80)

# Create dataset
ratings_matrix, movies_df = create_movie_dataset()
n_users, n_movies = ratings_matrix.shape

print(f"\nDataset Statistics:")
print(f"  Number of users: {n_users}")
print(f"  Number of movies: {n_movies}")
print(f"  Total possible ratings: {n_users * n_movies}")
print(f"  Actual ratings: {np.sum(~np.isnan(ratings_matrix))}")
print(f"  Sparsity: {np.sum(np.isnan(ratings_matrix)) / (n_users * n_movies) * 100:.1f}%")

# Create DataFrame for better visualization
print("\n" + "="*80)
print("RATING MATRIX (NaN = Not Rated)")
print("="*80)

ratings_df = pd.DataFrame(
    ratings_matrix,
    columns=[f"M{i+1}" for i in range(n_movies)],
    index=[f"User{i+1}" for i in range(n_users)]
)
print(ratings_df.to_string())

# ============================================================================
# Train SVD Recommender
# ============================================================================

print("\n" + "="*80)
print("TRAINING SVD RECOMMENDER")
print("="*80)

# Try different numbers of latent factors
best_rmse = float('inf')
best_k = None
results = []

for k in [2, 3, 5, 8, 10]:
    recommender = SVDRecommender(n_factors=k)
    recommender.fit(ratings_matrix)
    rmse, mae = recommender.evaluate(ratings_matrix)
    results.append({'k': k, 'RMSE': rmse, 'MAE': mae})
    
    print(f"\nk={k}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_k = k
        best_recommender = recommender

print(f"\n✓ Best model: k={best_k} with RMSE={best_rmse:.4f}")

# ============================================================================
# Generate Recommendations
# ============================================================================

print("\n" + "="*80)
print("GENERATING RECOMMENDATIONS")
print("="*80)

# Select a user to get recommendations for
test_user = 0  # User 1 (index 0)

# Get movies this user has already rated
rated_movies = np.where(~np.isnan(ratings_matrix[test_user]))[0]
print(f"\nUser {test_user+1}'s Rated Movies:")
user_ratings = []
for movie_idx in rated_movies:
    user_ratings.append({
        'Title': movies_df.iloc[movie_idx]['Title'],
        'Genre': movies_df.iloc[movie_idx]['Genre'],
        'Actual Rating': ratings_matrix[test_user, movie_idx]
    })
print(pd.DataFrame(user_ratings).to_string(index=False))

# Get recommendations
print(f"\n{'='*80}")
print(f"TOP 5 RECOMMENDATIONS FOR USER {test_user+1}")
print("="*80)
recommendations = best_recommender.recommend_movies(
    test_user, movies_df, n_recommendations=5, rated_movies=rated_movies
)
print(recommendations.to_string(index=False))

# ============================================================================
# Visualizations
# ============================================================================

fig = plt.figure(figsize=(20, 14))

# Plot 1: Original Rating Matrix Heatmap
ax1 = plt.subplot(3, 4, 1)
sns.heatmap(ratings_matrix, annot=False, cmap='YlOrRd', 
            cbar_kws={'label': 'Rating'}, ax=ax1, vmin=1, vmax=5)
ax1.set_xlabel('Movies')
ax1.set_ylabel('Users')
ax1.set_title('Original Rating Matrix\n(white = missing)', 
              fontsize=12, fontweight='bold')

# Plot 2: Predicted Rating Matrix Heatmap
ax2 = plt.subplot(3, 4, 2)
sns.heatmap(best_recommender.predicted_ratings, annot=False, cmap='YlOrRd',
            cbar_kws={'label': 'Rating'}, ax=ax2, vmin=1, vmax=5)
ax2.set_xlabel('Movies')
ax2.set_ylabel('Users')
ax2.set_title('Predicted Ratings (SVD)', fontsize=12, fontweight='bold')

# Plot 3: Singular Values
ax3 = plt.subplot(3, 4, 3)
ax3.bar(range(1, len(best_recommender.S[:15])+1), best_recommender.S[:15], 
        color='steelblue', alpha=0.7)
ax3.axvline(x=best_k, color='red', linestyle='--', linewidth=2, 
            label=f'k={best_k} (selected)')
ax3.set_xlabel('Factor Index')
ax3.set_ylabel('Singular Value')
ax3.set_title('Singular Values', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend()

# Plot 4: RMSE vs Number of Factors
ax4 = plt.subplot(3, 4, 4)
ks = [r['k'] for r in results]
rmses = [r['RMSE'] for r in results]
maes = [r['MAE'] for r in results]

ax4.plot(ks, rmses, 'bo-', linewidth=2, markersize=8, label='RMSE')
ax4.plot(ks, maes, 'ro-', linewidth=2, markersize=8, label='MAE')
ax4.set_xlabel('Number of Latent Factors (k)')
ax4.set_ylabel('Error')
ax4.set_title('Model Performance vs k', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

# Plot 5: User Factors Heatmap
ax5 = plt.subplot(3, 4, 5)
sns.heatmap(best_recommender.user_factors, annot=False, cmap='coolwarm',
            center=0, cbar_kws={'label': 'Factor Value'}, ax=ax5)
ax5.set_xlabel('Latent Factors')
ax5.set_ylabel('Users')
ax5.set_title('User Latent Factors', fontsize=12, fontweight='bold')

# Plot 6: Movie Factors Heatmap
ax6 = plt.subplot(3, 4, 6)
sns.heatmap(best_recommender.movie_factors.T, annot=False, cmap='coolwarm',
            center=0, cbar_kws={'label': 'Factor Value'}, ax=ax6)
ax6.set_xlabel('Movies')
ax6.set_ylabel('Latent Factors')
ax6.set_title('Movie Latent Factors', fontsize=12, fontweight='bold')

# Plot 7: Rating Distribution (Original vs Predicted)
ax7 = plt.subplot(3, 4, 7)
observed_ratings = ratings_matrix[~np.isnan(ratings_matrix)]
predicted_ratings_observed = best_recommender.predicted_ratings[~np.isnan(ratings_matrix)]

ax7.hist(observed_ratings, bins=20, alpha=0.5, label='Actual', color='blue', 
         edgecolor='black')
ax7.hist(predicted_ratings_observed, bins=20, alpha=0.5, label='Predicted', 
         color='red', edgecolor='black')
ax7.set_xlabel('Rating')
ax7.set_ylabel('Frequency')
ax7.set_title('Rating Distribution', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Actual vs Predicted Ratings Scatter
ax8 = plt.subplot(3, 4, 8)
ax8.scatter(observed_ratings, predicted_ratings_observed, alpha=0.5, s=30)
ax8.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
ax8.set_xlabel('Actual Rating')
ax8.set_ylabel('Predicted Rating')
ax8.set_title('Actual vs Predicted Ratings', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.legend()
ax8.set_xlim(0.5, 5.5)
ax8.set_ylim(0.5, 5.5)

# Plot 9: Average Rating by Genre
ax9 = plt.subplot(3, 4, 9)
genre_ratings = []
for genre in movies_df['Genre'].unique():
    genre_movies = movies_df[movies_df['Genre'] == genre].index
    genre_data = ratings_matrix[:, genre_movies]
    avg_rating = np.nanmean(genre_data)
    genre_ratings.append({'Genre': genre, 'Avg Rating': avg_rating})

genre_df = pd.DataFrame(genre_ratings).sort_values('Avg Rating', ascending=False)
ax9.barh(genre_df['Genre'], genre_df['Avg Rating'], color='teal', alpha=0.7)
ax9.set_xlabel('Average Rating')
ax9.set_title('Average Rating by Genre', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='x')

# Plot 10: User Rating Activity
ax10 = plt.subplot(3, 4, 10)
ratings_per_user = [np.sum(~np.isnan(ratings_matrix[i])) for i in range(n_users)]
ax10.bar(range(1, n_users+1), ratings_per_user, color='orange', alpha=0.7)
ax10.set_xlabel('User ID')
ax10.set_ylabel('Number of Ratings')
ax10.set_title('User Rating Activity', fontsize=12, fontweight='bold')
ax10.grid(True, alpha=0.3, axis='y')

# Plot 11: Movie Popularity
ax11 = plt.subplot(3, 4, 11)
ratings_per_movie = [np.sum(~np.isnan(ratings_matrix[:, i])) for i in range(n_movies)]
ax11.bar(range(1, n_movies+1), ratings_per_movie, color='green', alpha=0.7)
ax11.set_xlabel('Movie ID')
ax11.set_ylabel('Number of Ratings')
ax11.set_title('Movie Popularity', fontsize=12, fontweight='bold')
ax11.grid(True, alpha=0.3, axis='y')

# Plot 12: Error Distribution
ax12 = plt.subplot(3, 4, 12)
errors = observed_ratings - predicted_ratings_observed
ax12.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax12.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax12.set_xlabel('Prediction Error (Actual - Predicted)')
ax12.set_ylabel('Frequency')
ax12.set_title(f'Error Distribution (Mean: {np.mean(errors):.3f})', 
              fontsize=12, fontweight='bold')
ax12.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SYSTEM SUMMARY")
print("="*80)

summary_df = pd.DataFrame(results)
print("\nModel Performance:")
print(summary_df.to_string(index=False))

print(f"\n✓ Best Configuration: k={best_k} latent factors")
print(f"✓ RMSE: {best_rmse:.4f}")
print(f"✓ Predictions made for {np.sum(np.isnan(ratings_matrix))} missing ratings")

print("\n" + "="*80)
print("HOW SVD RECOMMENDER WORKS")
print("="*80)
print("""
1. MATRIX FACTORIZATION:
   R ≈ U @ S @ V^T
   - R: User-Movie rating matrix (n_users × n_movies)
   - U: User factors (n_users × k)
   - S: Singular values (k × k)
   - V^T: Movie factors (k × n_movies)

2. DIMENSIONALITY REDUCTION:
   - Keep only top k factors (latent features)
   - k < min(n_users, n_movies)
   - Captures main patterns in data

3. PREDICTION:
   - Reconstruct full rating matrix
   - Predicted_Rating[i,j] = U[i,:] @ S @ V[j,:].T
   - Fill in missing ratings

4. RECOMMENDATION:
   - For each user, rank unpredicted movies
   - Recommend top N highest predicted ratings

5. BENEFITS:
   - Discovers latent features (genres, themes)
   - Handles sparse data
   - Collaborative filtering
   - Scalable to large datasets
""")

print("="*80)