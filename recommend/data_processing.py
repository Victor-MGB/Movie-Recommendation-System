import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# --- Day 1-2: Data Loading and Exploration ---

# Load ratings
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Load movies
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', 
                     names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 
                            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

# Data sanity checks
print("Missing values in ratings:")
print(ratings.isna().sum())
print("\nMissing values in movies (key columns):")
print(movies[['movie_id', 'title']].isna().sum())
print("\nUnique ratings:", ratings['rating'].unique())
invalid_ratings = ratings[~ratings['rating'].isin([1, 2, 3, 4, 5])]
print("Invalid ratings (if any):", invalid_ratings)

# Clean ratings
ratings = ratings[ratings['rating'].isin([1, 2, 3, 4, 5])]
ratings = ratings[ratings['movie_id'].isin(movies['movie_id'])]
ratings.to_csv('data/cleaned_ratings.csv', index=False)
print("Cleaned ratings saved to data/cleaned_ratings.csv")

# Visualize rating distribution (Day 1)
plt.figure(figsize=(8, 6))
sns.histplot(ratings['rating'], bins=5)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('plots/rating_distribution.png')
plt.show()

# Visualize ratings per user and movie (Day 2)
ratings_per_user = ratings.groupby('user_id').size()
plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_user, bins=50)
plt.title('Distribution of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.savefig('plots/ratings_per_user.png')
plt.show()

ratings_per_movie = ratings.groupby('movie_id').size()
plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_movie, bins=50)
plt.title('Distribution of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.savefig('plots/ratings_per_movie.png')
plt.show()

# Split train/test (Day 2)
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

# --- Day 3-4: Data Preprocessing ---

# Create user-item matrix
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
user_item_matrix.to_csv('data/user_item_matrix.csv')
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Visualize sparsity (Day 3)
plt.figure(figsize=(10, 8))
sns.heatmap(user_item_matrix.iloc[:20, :20], cmap='viridis', cbar_kws={'label': 'Rating'})
plt.title('User-Item Matrix (Subset)')
plt.xlabel('Movie ID')
plt.ylabel('User ID')
plt.savefig('plots/user_item_matrix_heatmap.png')
plt.show()

# Handle sparsity
user_means = user_item_matrix.mean(axis=1)
user_item_matrix_zero = user_item_matrix.fillna(0)
user_item_matrix_mean = user_item_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
user_item_matrix_zero.to_csv('data/user_item_matrix_zero.csv')
user_item_matrix_mean.to_csv('data/user_item_matrix_mean.csv')

# Normalize ratings
normalized_matrix = user_item_matrix.sub(user_means, axis=0)
normalized_matrix.to_csv('data/normalized_user_item_matrix.csv')
print("Mean of normalized ratings per user (should be ~0):")
print(normalized_matrix.mean(axis=1).describe())

# Create training user-item matrix
train_user_item_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating')
train_user_item_matrix.to_csv('data/train_user_item_matrix.csv')

# --- Day 5-6: Collaborative Filtering Model ---

# Compute user similarity
normalized_matrix_zero = normalized_matrix.fillna(0)
similarity_matrix = cosine_similarity(normalized_matrix_zero)
similarity_df = pd.DataFrame(similarity_matrix, index=normalized_matrix_zero.index, 
                            columns=normalized_matrix_zero.index)
similarity_df.to_csv('data/user_similarity_matrix.csv')

# Visualize user similarity (Day 5)
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df.iloc[:20, :20], cmap='coolwarm', center=0)
plt.title('User Similarity Matrix (Subset)')
plt.xlabel('User ID')
plt.ylabel('User ID')
plt.savefig('plots/user_similarity_heatmap.png')
plt.show()

# Define prediction functions
def predict_ratings(user_id, movie_id, similarity_df, user_item_matrix, user_means, k=10):
    """
    Predict a rating for a user-movie pair using top-K similar users.
    
    Parameters:
    -----------
    user_id : int
        ID of the target user.
    movie_id : int
        ID of the movie to predict.
    similarity_df : pd.DataFrame
        DataFrame of user similarities.
    user_item_matrix : pd.DataFrame
        User-item matrix with ratings.
    user_means : pd.Series
        Series of user mean ratings.
    k : int, optional
        Number of similar users (default=10).
    
    Returns:
    --------
    float or None
        Predicted rating (1-5) or None if no valid ratings.
    """
    try:
        similar_users = similarity_df.loc[:, user_id].nlargest(k+1)[1:]
        similar_users_ratings = user_item_matrix.loc[similar_users.index, movie_id]
        if similar_users_ratings.isna().all():
            return None
        valid_ratings = similar_users_ratings.dropna()
        valid_similarities = similar_users[valid_ratings.index]
        if valid_similarities.sum() == 0:
            return None
        weighted_sum = (valid_ratings * valid_similarities).sum()
        sim_sum = valid_similarities.sum()
        normalized_prediction = weighted_sum / sim_sum
        prediction = normalized_prediction + user_means.loc[user_id]
        return np.clip(prediction, 1, 5)
    except KeyError:
        return None

def recommend_movies(user_id, similarity_df, user_item_matrix, user_means, movies, k=10, n=5):
    """
    Generate top-N movie recommendations for a user.
    
    Parameters:
    -----------
    user_id : int
        ID of the target user.
    similarity_df : pd.DataFrame
        DataFrame of user similarities.
    user_item_matrix : pd.DataFrame
        User-item matrix with ratings.
    user_means : pd.Series
        Series of user mean ratings.
    movies : pd.DataFrame
        DataFrame with movie metadata.
    k : int, optional
        Number of similar users (default=10).
    n : int, optional
        Number of recommendations (default=5).
    
    Returns:
    --------
    pd.DataFrame or str
        Top-N movies with predicted ratings or error message.
    """
    if user_id not in user_item_matrix.index:
        return f"Error: User ID {user_id} not found."
    try:
        unrated_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index
        predictions = []
        for movie_id in unrated_movies:
            pred = predict_ratings(user_id, movie_id, similarity_df, user_item_matrix, user_means, k)
            if pred is not None:
                predictions.append((movie_id, pred))
        if not predictions:
            return f"No recommendations for User {user_id} due to insufficient data."
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]
        top_movie_ids = [x[0] for x in top_n]
        top_ratings = [x[1] for x in top_n]
        recommendations = movies[movies['movie_id'].isin(top_movie_ids)][['movie_id', 'title']].copy()
        recommendations['predicted_rating'] = top_ratings
        return recommendations.sort_values('predicted_rating', ascending=False)
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

# --- Day 7-8: Evaluation and Optimization ---

# Evaluate user-based collaborative filtering
test_predictions = []
actual_ratings = []
for _, row in test_data.iterrows():
    pred = predict_ratings(int(row['user_id']), int(row['movie_id']), similarity_df, 
                          user_item_matrix, user_means, k=10)
    if pred is not None:
        test_predictions.append(pred)
        actual_ratings.append(row['rating'])

rmse = np.sqrt(mean_squared_error(actual_ratings, test_predictions))
print(f"User-Based RMSE: {rmse:.4f}")

# Save predictions
predictions_df = pd.DataFrame({
    'user_id': test_data.loc[:len(test_predictions)-1, 'user_id'],
    'movie_id': test_data.loc[:len(test_predictions)-1, 'movie_id'],
    'actual_rating': actual_ratings,
    'predicted_rating': test_predictions
})
predictions_df.to_csv('data/test_predictions.csv', index=False)

# Visualize predicted vs. actual ratings (Day 8)
plt.figure(figsize=(8, 6))
plt.scatter(actual_ratings, test_predictions, alpha=0.5)
plt.plot([1, 5], [1, 5], 'r--')  # Reference line
plt.title('Predicted vs. Actual Ratings')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.savefig('plots/predicted_vs_actual.png')
plt.show()

# Test different K values
k_values = [5, 10, 20, 50]
rmse_results = []
for k in k_values:
    test_predictions = []
    actual_ratings = []
    for _, row in test_data.iterrows():
        pred = predict_ratings(row['user_id'], row['movie_id'], similarity_df, 
                              user_item_matrix, user_means, k=k)
        if pred is not None:
            test_predictions.append(pred)
            actual_ratings.append(row['rating'])
    rmse = np.sqrt(mean_squared_error(actual_ratings, test_predictions))
    rmse_results.append((k, rmse))
    print(f"RMSE for K={k}: {rmse:.4f}")

rmse_df = pd.DataFrame(rmse_results, columns=['K', 'RMSE'])
rmse_df.to_csv('data/rmse_by_k.csv', index=False)

# Item-based collaborative filtering
item_similarity_matrix = cosine_similarity(user_item_matrix.fillna(0).T)
item_similarity_df = pd.DataFrame(item_similarity_matrix, index=user_item_matrix.columns, 
                                 columns=user_item_matrix.columns)
item_similarity_df.to_csv('data/item_similarity_matrix.csv')

def predict_ratings_item_based(user_id, movie_id, item_similarity_df, user_item_matrix, k=10):
    """
    Predict rating using item-based collaborative filtering.
    
    Parameters:
    -----------
    user_id : int
        ID of the target user.
    movie_id : int
        ID of the movie to predict.
    item_similarity_df : pd.DataFrame
        DataFrame of movie similarities.
    user_item_matrix : pd.DataFrame
        User-item matrix with ratings.
    k : int, optional
        Number of similar movies (default=10).
    
    Returns:
    --------
    float or None
        Predicted rating or None if no valid ratings.
    """
    try:
        similar_movies = item_similarity_df[movie_id].nlargest(k+1)[1:]
        user_ratings = user_item_matrix.loc[user_id, similar_movies.index]
        if user_ratings.isna().all():
            return None
        valid_ratings = user_ratings.dropna()
        valid_similarities = similar_movies[valid_ratings.index]
        if valid_similarities.sum() == 0:
            return None
        weighted_sum = (valid_ratings * valid_similarities).sum()
        sim_sum = valid_similarities.sum()
        return weighted_sum / sim_sum
    except KeyError:
        return None

# Evaluate item-based filtering
test_predictions_item = []
actual_ratings_item = []
for _, row in test_data.iterrows():
    pred = predict_ratings_item_based(row['user_id'], row['movie_id'], item_similarity_df, 
                                     user_item_matrix, k=10)
    if pred is not None:
        test_predictions_item.append(pred)
        actual_ratings_item.append(row['rating'])

rmse_item = np.sqrt(mean_squared_error(actual_ratings_item, test_predictions_item))
print(f"Item-Based RMSE: {rmse_item:.4f}")

# SVD
svd = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd.fit_transform(user_item_matrix.fillna(0))
item_latent_matrix = svd.components_.T

def predict_ratings_svd(user_id, movie_id, latent_matrix, item_latent_matrix):
    """
    Predict rating using SVD matrix factorization.
    
    Parameters:
    -----------
    user_id : int
        ID of the target user.
    movie_id : int
        ID of the movie to predict.
    latent_matrix : np.ndarray
        User latent factors.
    item_latent_matrix : np.ndarray
        Item latent factors.
    
    Returns:
    --------
    float
        Predicted rating (clipped to 1-5).
    """
    return np.clip(np.dot(latent_matrix[user_id-1], item_latent_matrix[movie_id-1]), 1, 5)

# Evaluate SVD
svd_predictions = []
svd_actuals = []
for _, row in test_data.iterrows():
    pred = predict_ratings_svd(row['user_id'], row['movie_id'], latent_matrix, item_latent_matrix)
    svd_predictions.append(pred)
    svd_actuals.append(row['rating'])

rmse_svd = np.sqrt(mean_squared_error(svd_actuals, svd_predictions))
print(f"SVD RMSE: {rmse_svd:.4f}")

# Filter sparse data
min_user_ratings = 20
min_movie_ratings = 10
user_counts = ratings.groupby('user_id').size()
movie_counts = ratings.groupby('movie_id').size()
filtered_ratings = ratings[
    (ratings['user_id'].isin(user_counts[user_counts >= min_user_ratings].index)) &
    (ratings['movie_id'].isin(movie_counts[movie_counts >= min_movie_ratings].index))
]
filtered_user_item_matrix = filtered_ratings.pivot(index='user_id', columns='movie_id', values='rating')
filtered_user_item_matrix.to_csv('data/filtered_user_item_matrix.csv')

# Recompute for filtered matrix
filtered_user_means = filtered_user_item_matrix.mean(axis=1)
filtered_normalized_matrix = filtered_user_item_matrix.sub(filtered_user_means, axis=0)
filtered_normalized_matrix_zero = filtered_normalized_matrix.fillna(0)
filtered_similarity_matrix = cosine_similarity(filtered_normalized_matrix_zero)
filtered_similarity_df = pd.DataFrame(filtered_similarity_matrix, 
                                    index=filtered_normalized_matrix_zero.index, 
                                    columns=filtered_normalized_matrix_zero.index)
filtered_similarity_df.to_csv('data/filtered_user_similarity_matrix.csv')

# Evaluate filtered user-based model
filtered_test_data = test_data[
    (test_data['user_id'].isin(filtered_user_item_matrix.index)) &
    (test_data['movie_id'].isin(filtered_user_item_matrix.columns))
]
test_predictions_filtered = []
actual_ratings_filtered = []
for _, row in filtered_test_data.iterrows():
    pred = predict_ratings(row['user_id'], row['movie_id'], filtered_similarity_df, 
                          filtered_user_item_matrix, filtered_user_means, k=10)
    if pred is not None:
        test_predictions_filtered.append(pred)
        actual_ratings_filtered.append(row['rating'])

rmse_filtered = np.sqrt(mean_squared_error(actual_ratings_filtered, test_predictions_filtered))
print(f"Filtered User-Based RMSE: {rmse_filtered:.4f}")

# Precision/Recall for top-N
def precision_recall_at_k(user_id, recommendations, test_data, k=5, threshold=4):
    """
    Calculate Precision@K and Recall@K for recommendations.
    
    Parameters:
    -----------
    user_id : int
        ID of the target user.
    recommendations : pd.DataFrame
        Recommended movies with movie_id and predicted_rating.
    test_data : pd.DataFrame
        Test set with user_id, movie_id, rating.
    k : int, optional
        Number of recommendations to evaluate (default=5).
    threshold : float, optional
        Minimum rating for relevance (default=4).
    
    Returns:
    --------
    tuple
        Precision@K, Recall@K.
    """
    user_test = test_data[test_data['user_id'] == user_id]
    relevant_items = user_test[user_test['rating'] >= threshold]['movie_id'].tolist()
    if not relevant_items:
        return 0, 0
    recommended_items = recommendations['movie_id'][:k].tolist()
    hits = len(set(recommended_items) & set(relevant_items))
    precision = hits / k
    recall = hits / len(relevant_items)
    return precision, recall

sample_users = test_data['user_id'].drop_duplicates().sample(10, random_state=42).tolist()
precisions, recalls = [], []
for user_id in sample_users:
    recs = recommend_movies(user_id, similarity_df, user_item_matrix, user_means, movies, k=10, n=5)
    precision, recall = precision_recall_at_k(user_id, recs, test_data, k=5)
    precisions.append(precision)
    recalls.append(recall)

avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
print(f"Average Precision@5: {avg_precision:.4f}")
print(f"Average Recall@5: {avg_recall:.4f}")

eval_results = pd.DataFrame({
    'user_id': sample_users,
    'precision@5': precisions,
    'recall@5': recalls
})
eval_results.to_csv('data/evaluation_results.csv', index=False)

# Visualize top-N recommendations (Day 8)
sample_user = 1
recs = recommend_movies(sample_user, similarity_df, user_item_matrix, user_means, movies, k=10, n=5)
plt.figure(figsize=(10, 6))
sns.barplot(data=recs, x='predicted_rating', y='title')
plt.title(f'Top-5 Recommendations for User {sample_user}')
plt.xlabel('Predicted Rating')
plt.ylabel('Movie Title')
plt.savefig('plots/top_n_recommendations.png')
plt.show()

# --- Day 9-10: Demo and Summary ---

# Demo
print(f"Top-5 Recommendations for User {sample_user}:")
print(recs)

# Pipeline test
print("\nTesting pipeline...")
try:
    # Reload data and run key steps
    ratings = pd.read_csv('data/cleaned_ratings.csv')
    user_item_matrix = pd.read_csv('data/user_item_matrix.csv', index_col=0)
    similarity_df = pd.read_csv('data/user_similarity_matrix.csv', index_col=0)
    recs = recommend_movies(1, similarity_df, user_item_matrix, user_means, movies, k=10, n=5)
    print("Pipeline test successful!")
except Exception as e:
    print(f"Pipeline test failed: {str(e)}")