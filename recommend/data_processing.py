import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create necessary directories
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

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

# Check for invalid ratings (should be 1-5)
print("\nUnique ratings:", ratings['rating'].unique())
invalid_ratings = ratings[~ratings['rating'].isin([1, 2, 3, 4, 5])]
print("Invalid ratings (if any):")
print(invalid_ratings)

# Clean ratings
ratings = ratings[ratings['rating'].isin([1, 2, 3, 4, 5])]
ratings = ratings[ratings['movie_id'].isin(movies['movie_id'])]
ratings.to_csv('data/cleaned_ratings.csv', index=False)
print("Cleaned ratings saved to data/cleaned_ratings.csv")

# Create user-item matrix
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
user_item_matrix.to_csv('data/user_item_matrix.csv')
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Fill NaNs
user_means = user_item_matrix.mean(axis=1)
user_item_matrix_zero = user_item_matrix.fillna(0)
user_item_matrix_mean = user_item_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
user_item_matrix_zero.to_csv('data/user_item_matrix_zero.csv')
user_item_matrix_mean.to_csv('data/user_item_matrix_mean.csv')

# Normalize ratings
normalized_matrix = user_item_matrix.sub(user_means, axis=0)
normalized_matrix.to_csv('data/normalized_user_item_matrix.csv')

# Cosine similarity
normalized_matrix_zero = normalized_matrix.fillna(0)
similarity_matrix = cosine_similarity(normalized_matrix_zero)
similarity_df = pd.DataFrame(similarity_matrix, index=normalized_matrix_zero.index, columns=normalized_matrix_zero.index)
similarity_df.to_csv('data/user_similarity_matrix.csv')

# Split train/test
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)
train_user_item_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating')
train_user_item_matrix.to_csv('data/train_user_item_matrix.csv')

# Apply SVD
svd = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd.fit_transform(user_item_matrix.fillna(0))
item_latent_matrix = svd.components_.T

# Predict ratings
def predict_ratings_svd(user_id, movie_id, latent_matrix, item_latent_matrix):
    return np.dot(latent_matrix[user_id-1], item_latent_matrix[movie_id-1])

# Evaluate SVD
svd_predictions = []
svd_actuals = []
for _, row in test_data.iterrows():
    pred = predict_ratings_svd(row['user_id'], row['movie_id'], latent_matrix, item_latent_matrix)
    svd_predictions.append(np.clip(pred, 1, 5))
    svd_actuals.append(row['rating'])

rmse_svd = np.sqrt(mean_squared_error(svd_actuals, svd_predictions))
print(f"SVD RMSE: {rmse_svd:.4f}")

# Define predict_ratings early
def predict_ratings(user_id, movie_id, similarity_df, user_item_matrix, user_means, k=10):
    similar_users = similarity_df.loc[:, user_id].nlargest(k+1)[1:]
    try:
        similar_users_ratings = user_item_matrix.loc[similar_users.index, movie_id]
    except KeyError:
        return None
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

# Reload to ensure proper types
user_item_matrix = pd.read_csv('data/user_item_matrix.csv', index_col=0)
user_item_matrix.columns = user_item_matrix.columns.astype(int)
user_item_matrix.index = user_item_matrix.index.astype(int)
user_means = user_item_matrix.mean(axis=1)
similarity_df = pd.read_csv('data/user_similarity_matrix.csv', index_col=0)
similarity_df.columns = similarity_df.columns.astype(int)
similarity_df.index = similarity_df.index.astype(int)
test_data = pd.read_csv('data/test_data.csv')

# Predict test set
test_predictions = []
actual_ratings = []
for _, row in test_data.iterrows():
    pred = predict_ratings(int(row['user_id']), int(row['movie_id']), similarity_df, user_item_matrix, user_means, k=10)
    if pred is not None:
        test_predictions.append(pred)
        actual_ratings.append(row['rating'])

# Evaluate
rmse = np.sqrt(mean_squared_error(actual_ratings, test_predictions))
print(f"RMSE on test set: {rmse:.4f}")

# Save predictions
predictions_df = pd.DataFrame({
    'user_id': test_data.loc[:len(test_predictions)-1, 'user_id'],
    'movie_id': test_data.loc[:len(test_predictions)-1, 'movie_id'],
    'actual_rating': actual_ratings,
    'predicted_rating': test_predictions
})
predictions_df.to_csv('data/test_predictions.csv', index=False)


# Test different K values for RMSE
k_values = [5, 10, 20, 50]
rmse_results = []
for k in k_values:
    test_predictions = []
    actual_ratings = []
    for _, row in test_data.iterrows():
        pred = predict_ratings(row['user_id'], row['movie_id'], similarity_df, user_item_matrix, user_means, k=k)
        if pred is not None:
            test_predictions.append(pred)
            actual_ratings.append(row['rating'])
    rmse = np.sqrt(mean_squared_error(actual_ratings, test_predictions))
    rmse_results.append((k, rmse))
    print(f"RMSE for K={k}: {rmse:.4f}")

# Save RMSE results
rmse_df = pd.DataFrame(rmse_results, columns=['K', 'RMSE'])
rmse_df.to_csv('data/rmse_by_k.csv', index=False)
print("RMSE results for different K saved to data/rmse_by_k.csv")

# Compute item similarity matrix
item_similarity_matrix = cosine_similarity(user_item_matrix.fillna(0).T)
item_similarity_df = pd.DataFrame(
    item_similarity_matrix,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# Save item similarity matrix
item_similarity_df.to_csv('data/item_similarity_matrix.csv')
print("Item similarity matrix saved to data/item_similarity_matrix.csv")

# Predict ratings using item-based filtering
def predict_ratings_item_based(user_id, movie_id, item_similarity_df, user_item_matrix, k=10):
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

# Evaluate item-based filtering
test_predictions_item = []
actual_ratings_item = []
for _, row in test_data.iterrows():
    pred = predict_ratings_item_based(row['user_id'], row['movie_id'], item_similarity_df, user_item_matrix, k=10)
    if pred is not None:
        test_predictions_item.append(pred)
        actual_ratings_item.append(row['rating'])

rmse_item = np.sqrt(mean_squared_error(actual_ratings_item, test_predictions_item))
print(f"Item-Based RMSE: {rmse_item:.4f}")

# Filter users and movies
min_user_ratings = 20
min_movie_ratings = 10
user_counts = ratings.groupby('user_id').size()
movie_counts = ratings.groupby('movie_id').size()
filtered_ratings = ratings[
    (ratings['user_id'].isin(user_counts[user_counts >= min_user_ratings].index)) &
    (ratings['movie_id'].isin(movie_counts[movie_counts >= min_movie_ratings].index))
]

# Recreate user-item matrix
filtered_user_item_matrix = filtered_ratings.pivot(index='user_id', columns='movie_id', values='rating')
filtered_user_item_matrix.to_csv('data/filtered_user_item_matrix.csv')

# Recalculate similarity and RMSE (repeat steps above with filtered matrix)

# Define recommend_movies
def recommend_movies(user_id, similarity_df, user_item_matrix, user_means, movies, k=10, n=5):
    unrated_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index
    predictions = []
    for movie_id in unrated_movies:
        pred = predict_ratings(user_id, movie_id, similarity_df, user_item_matrix, user_means, k)
        if pred is not None:
            predictions.append((movie_id, pred))
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    top_movie_ids = [x[0] for x in top_n]
    top_ratings = [x[1] for x in top_n]
    recommendations = movies[movies['movie_id'].isin(top_movie_ids)][['movie_id', 'title']].copy()
    recommendations['predicted_rating'] = top_ratings
    return recommendations

# Define precision/recall
def precision_recall_at_k(user_id, recommendations, test_data, k=5, threshold=4):
    user_test = test_data[test_data['user_id'] == user_id]
    relevant_items = user_test[user_test['rating'] >= threshold]['movie_id'].tolist()
    if not relevant_items:
        return 0, 0
    recommended_items = recommendations['movie_id'][:k].tolist()
    hits = len(set(recommended_items) & set(relevant_items))
    precision = hits / k
    recall = hits / len(relevant_items)
    return precision, recall

# Evaluate top-N recommendations
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
eval_results.to_csv('ml-100k/evaluation_results.csv', index=False)

# Sample recommendation
sample_user = 1
recs = recommend_movies(sample_user, similarity_df, user_item_matrix, user_means, movies, k=10, n=5)
print(f"Top-5 Recommendations for User {sample_user}:")
print(recs)
