import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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
print("User-Item Matrix Shape:", user_item_matrix.shape)
print("Number of non-NaN entries:", user_item_matrix.notna().sum().sum())
print("Sparsity (% missing):", 
    (1 - user_item_matrix.notna().sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100)

user_item_matrix.to_csv('data/user_item_matrix.csv')

# Fill NaNs
user_item_matrix_zero = user_item_matrix.fillna(0)
user_means = user_item_matrix.mean(axis=1)
user_item_matrix_mean = user_item_matrix.copy()
for user_id in user_item_matrix.index:
    user_item_matrix_mean.loc[user_id] = user_item_matrix.loc[user_id].fillna(user_means[user_id])

user_item_matrix_zero.to_csv('data/user_item_matrix_zero.csv')
user_item_matrix_mean.to_csv('data/user_item_matrix_mean.csv')
print("User-Item Matrices (zero-filled and mean-filled) saved.")

#load normalized matrix
normalized_matrix = pd.read_csv('data/normalized_user_item_matrix.csv', index_col=0)

#fill NANS with 0 for cosine similarity
normalized_matrix_zero = normalized_matrix.fillna(0)

#Display matrix info
print("Normalized User-Item Matrix Shape:", normalized_matrix.shape)
print("Number of non-NaN entries in normalized matrix:", normalized_matrix_zero.iloc[:5,:5])

similarity_matrix = cosine_similarity(normalized_matrix_zero)

# Convert to DataFrame for easier lookup
similarity_df = pd.DataFrame(
    similarity_matrix,
    index = normalized_matrix_zero.index,
    columns = normalized_matrix_zero.index
)

#Display sample
print("User Similarity Matrix Shape:", similarity_df.shape)
print("Sample of Similarity Matrix:")
print("Similarity Matrix Statistics:")
print(similarity_df.describe())
print(similarity_df.iloc[:5, :5])

# Save similarity matrix
similarity_df.to_csv('data/user_similarity_matrix.csv')
print("User similarity matrix saved to data/user_similarity_matrix.csv")

# Normalize ratings by user mean
normalized_matrix = user_item_matrix.sub(user_means, axis=0)
print("Sample of Normalized User-Item Matrix:")
print(normalized_matrix.iloc[:5, :5])
normalized_matrix.to_csv('data/normalized_user_item_matrix.csv')
print("Normalized matrix saved.")

print("Mean of normalized ratings per user:")
print(normalized_matrix.mean(axis=1).describe())

# Split ratings into train/test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

print(f"Training set size: {len(train_data)} ratings")
print(f"Test set size: {len(test_data)} ratings")
print("Unique users in train:", train_data['user_id'].nunique())
print("Unique movies in train:", train_data['movie_id'].nunique())
print("Unique users in test:", test_data['user_id'].nunique())
print("Unique movies in test:", test_data['movie_id'].nunique())

# Training matrix
train_user_item_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating')
train_user_item_matrix.to_csv('data/train_user_item_matrix.csv')
print("Training user-item matrix saved.")

# Overlap check
overlap = train_data.merge(test_data, on=['user_id', 'movie_id'], how='inner')
print("Overlap check (should be empty):")
print(overlap)

# Distribution plots
plt.figure(figsize=(8, 6))
sns.histplot(ratings['rating'], bins=5, kde=False, color='blue', discrete=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('plots/ratings_distribution.png')
plt.close()

ratings_per_user = ratings.groupby('user_id').size()
plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_user, bins=50)
plt.title('Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Users')
plt.savefig('plots/ratings_per_user.png')
plt.close()

ratings_per_movie = ratings.groupby('movie_id').size()
plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_movie, bins=50)
plt.title('Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Movies')
plt.savefig('plots/ratings_per_movie.png')
plt.close()

# Summary
print("Ratings per user - Mean:", ratings_per_user.mean())
print("Ratings per user - Median:", ratings_per_user.median())
print("Ratings per user - Min:", ratings_per_user.min())
print("Ratings per user - Max:", ratings_per_user.max())

print("Ratings per movie - Mean:", ratings_per_movie.mean())
print("Ratings per movie - Median:", ratings_per_movie.median())
print("Ratings per movie - Min:", ratings_per_movie.min())
print("Ratings per movie - Max:", ratings_per_movie.max())

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(user_item_matrix.iloc[:20, :20], cmap='viridis', cbar_kws={'label': 'Rating'})
plt.title('User-Item Matrix (Subset)')
plt.xlabel('Movie ID')
plt.ylabel('User ID')
plt.savefig('plots/user_item_matrix_heatmap.png')
plt.close()

# Visualize a subset (first 20 users)
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df.iloc[:20, :20], cmap='coolwarm', center=0)
plt.title('User Similarity Matrix (Subset)')
plt.xlabel('User ID')
plt.ylabel('User ID')
plt.savefig('plots/user_similarity_heatmap.png')
plt.close()

# Evaluation helpers
def calculate_rmse(predicted, targets):
    return np.sqrt(mean_squared_error(targets, predicted))

def precision_recall_at_k(user_id, recommendations, test_data, k=5, threshold=4):
    user_test = test_data[test_data['user_id'] == user_id]
    relevant_items = user_test[user_test['rating'] >= threshold]['movie_id'].tolist()
    if not relevant_items:
        return 0, 0
    recommended_items = recommendations[:k]
    hits = len(set(recommended_items) & set(relevant_items))
    precision = hits / k
    recall = hits / len(relevant_items)
    return precision, recall
