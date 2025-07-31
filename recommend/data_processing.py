import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create necessary directories
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load ratings
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Load movies (specify relevant columns for now)
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', 
                    names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 
                        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

print("Missing values in ratings:")
print(ratings.isna().sum())
print("\nMissing values in movies (key columns):")
print(movies[['movie_id', 'title']].isna().sum())

# Check for invalid ratings (should be 1-5)
print("\nUnique ratings:", ratings['rating'].unique())
invalid_ratings = ratings[~ratings['rating'].isin([1, 2, 3, 4, 5])]
print("Invalid ratings (if any):")
print(invalid_ratings)

ratings = ratings[ratings['rating'].isin([1, 2, 3, 4, 5])]
print("Users in ratings but not in movies:", 
    set(ratings['user_id']) - set(ratings['user_id'].unique()))
print("Movies in ratings but not in movies:", 
    set(ratings['movie_id']) - set(movies['movie_id']))

ratings = ratings[ratings['movie_id'].isin(movies['movie_id'])]
ratings.to_csv('data/cleaned_ratings.csv', index=False)
print("Cleaned ratings saved to data/cleaned_ratings.csv")

# Create user-item matrix
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')

# Display matrix info
print("User-Item Matrix Shape:", user_item_matrix.shape)
print("Number of non-NaN entries:", user_item_matrix.notna().sum().sum())
print("Sparsity (% of missing entries):", 
      (1 - user_item_matrix.notna().sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100)

# Save the matrix for later use
user_item_matrix.to_csv('data/user_item_matrix.csv')

# Option 1: Fill NaNs with 0
user_item_matrix_zero = user_item_matrix.fillna(0)

# Option 2: Fill NaNs with user's average rating
user_means = user_item_matrix.mean(axis=1)
user_item_matrix_mean = user_item_matrix.copy()
for user_id in user_item_matrix.index:
    user_item_matrix_mean.loc[user_id] = user_item_matrix.loc[user_id].fillna(user_means[user_id])

# Save both matrices
user_item_matrix_zero.to_csv('data/user_item_matrix_zero.csv')
user_item_matrix_mean.to_csv('data/user_item_matrix_mean.csv')

print("User-Item Matrices (zero-filled and mean-filled) saved.")
# Plot ratings distribution
plt.figure(figsize=(8, 6))
sns.histplot(ratings['rating'], bins=5, kde=False, color='blue', discrete=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('plots/ratings_distribution.png')
plt.close()

# Visualize a small subset (e.g., first 20 users and 20 movies)
plt.figure(figsize=(10, 8))
sns.heatmap(user_item_matrix.iloc[:20, :20], cmap='viridis', cbar_kws={'label': 'Rating'})
plt.title('User-Item Matrix (Subset, NaNs as Blank)')
plt.xlabel('Movie ID')
plt.ylabel('User ID')
plt.savefig('data/user_item_matrix_heatmap.png')
plt.close()

def calculate_rmse(predicted, targets):
    """
    Calculate the Root Mean Squared Error (RMSE) between predictions and targets.
    
    :param predicted: Predicted ratings
    :param targets: Actual ratings
    :return: RMSE value
    """
    return np.sqrt(mean_squared_error(targets, predicted))

def precision_recall_at_k(user_id, recommendations, test_data, k=5, threshold=4):
    """
    Compute precision and recall at k for a given user.
    
    :param user_id: ID of the user
    :param recommendations: List of recommended movie_ids
    :param test_data: Test dataset
    :param k: Number of top recommendations to consider
    :param threshold: Rating threshold to consider item as relevant
    :return: precision, recall
    """
    user_test = test_data[test_data['user_id'] == user_id]
    relevant_items = user_test[user_test['rating'] >= threshold]['movie_id'].tolist()
    if not relevant_items:
        return 0, 0
    recommended_items = recommendations[:k]
    hits = len(set(recommended_items) & set(relevant_items))
    precision = hits / k
    recall = hits / len(relevant_items)
    return precision, recall

# Split ratings into train and test set
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Verify the split
print(f"Training set size: {len(train_data)} ratings")
print(f"Test set size: {len(test_data)} ratings")

# Save split to CSV for reproducibility
train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

# Display unique users and movies
print("Unique users in train:", train_data['user_id'].nunique())
print("Unique movies in train:", train_data['movie_id'].nunique())
print("Unique users in test:", test_data['user_id'].nunique())
print("Unique movies in test:", test_data['movie_id'].nunique())

# Calculate number of ratings per user
ratings_per_user = ratings.groupby('user_id').size()

# Plot distribution of ratings per user
plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_user, bins=50)
plt.title('Distribution of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.savefig('plots/ratings_per_user.png')
plt.close()

# Summary statistics for ratings per user
print("Ratings per user - Mean:", ratings_per_user.mean())
print("Ratings per user - Median:", ratings_per_user.median())
print("Ratings per user - Min:", ratings_per_user.min())
print("Ratings per user - Max:", ratings_per_user.max())

# Calculate number of ratings per movie
ratings_per_movie = ratings.groupby('movie_id').size()

# Plot distribution of ratings per movie
plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_movie, bins=50)
plt.title('Distribution of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.savefig('plots/ratings_per_movie.png')
plt.close()

# Summary statistics for ratings per movie
print("Ratings per movie - Mean:", ratings_per_movie.mean())
print("Ratings per movie - Median:", ratings_per_movie.median())
print("Ratings per movie - Min:", ratings_per_movie.min())
print("Ratings per movie - Max:", ratings_per_movie.max())
