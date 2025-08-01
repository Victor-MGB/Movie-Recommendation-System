import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import seaborn as sns


# --- Streamlit App Configuration ---
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎥", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    """Load precomputed data files."""
    ratings = pd.read_csv('data/cleaned_ratings.csv')
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', 
                        names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 
                                'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    user_item_matrix = pd.read_csv('data/user_item_matrix.csv', index_col=0)
    user_item_matrix.columns = user_item_matrix.columns.astype(int)
    user_item_matrix.index = user_item_matrix.index.astype(int)
    user_means = user_item_matrix.mean(axis=1)
    similarity_df = pd.read_csv('data/user_similarity_matrix.csv', index_col=0)
    similarity_df.columns = similarity_df.columns.astype(int)
    similarity_df.index = similarity_df.index.astype(int)
    item_similarity_df = pd.read_csv('data/item_similarity_matrix.csv', index_col=0)
    item_similarity_df.columns = item_similarity_df.columns.astype(int)
    item_similarity_df.index = item_similarity_df.index.astype(int)
    
    # Load SVD data
    svd = TruncatedSVD(n_components=20, random_state=42)
    latent_matrix = svd.fit_transform(user_item_matrix.fillna(0))
    item_latent_matrix = svd.components_.T
    
    return ratings, movies, user_item_matrix, user_means, similarity_df, item_similarity_df, latent_matrix, item_latent_matrix

# --- Prediction Functions ---
def predict_ratings(user_id, movie_id, similarity_df, user_item_matrix, user_means, k=10):
    """Predict rating for a user-movie pair using user-based collaborative filtering."""
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

def predict_ratings_item_based(user_id, movie_id, item_similarity_df, user_item_matrix, k=10):
    """Predict rating using item-based collaborative filtering."""
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
        return np.clip(weighted_sum / sim_sum, 1, 5)
    except KeyError:
        return None

def predict_ratings_svd(user_id, movie_id, latent_matrix, item_latent_matrix):
    """Predict rating using SVD matrix factorization."""
    return np.clip(np.dot(latent_matrix[user_id-1], item_latent_matrix[movie_id-1]), 1, 5)

def recommend_movies(user_id, method, similarity_df, item_similarity_df, user_item_matrix, user_means, 
                    movies, latent_matrix, item_latent_matrix, k=10, n=5):
    """Generate top-N movie recommendations using the specified method."""
    if user_id not in user_item_matrix.index:
        return f"Error: User ID {user_id} not found."
    
    try:
        unrated_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index
        predictions = []
        for movie_id in unrated_movies:
            if method == "User-Based":
                pred = predict_ratings(user_id, movie_id, similarity_df, user_item_matrix, user_means, k)
            elif method == "Item-Based":
                pred = predict_ratings_item_based(user_id, movie_id, item_similarity_df, user_item_matrix, k)
            else:  # SVD
                pred = predict_ratings_svd(user_id, movie_id, latent_matrix, item_latent_matrix)
            if pred is not None:
                predictions.append((movie_id, pred))
        
        if not predictions:
            return f"No recommendations available for User {user_id}."
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]
        top_movie_ids = [x[0] for x in top_n]
        top_ratings = [x[1] for x in top_n]
        recommendations = movies[movies['movie_id'].isin(top_movie_ids)][['movie_id', 'title']].copy()
        recommendations['predicted_rating'] = top_ratings
        return recommendations.sort_values('predicted_rating', ascending=False)
    
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

# --- Streamlit UI ---
st.title("🎥 Movie Recommendation System")
st.markdown("""
Welcome to the Movie Recommendation System built with the MovieLens 100K dataset! 
Enter a user ID and select a recommendation method to get personalized movie suggestions.
""")

# Load data
ratings, movies, user_item_matrix, user_means, similarity_df, item_similarity_df, latent_matrix, item_latent_matrix = load_data()

# Sidebar inputs
st.sidebar.header("Recommendation Settings")
user_id = st.sidebar.number_input("Enter User ID (1-943):", min_value=1, max_value=943, value=1)
n_recommendations = st.sidebar.slider("Number of Recommendations:", 1, 10, 5)
k_similar_users = st.sidebar.slider("Number of Similar Users/Movies (K):", 5, 50, 10, 5)
method = st.sidebar.selectbox("Recommendation Method:", ["User-Based", "Item-Based", "SVD"])

# Generate recommendations
if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        recommendations = recommend_movies(
            user_id, method, similarity_df, item_similarity_df, user_item_matrix, user_means, 
            movies, latent_matrix, item_latent_matrix, k=k_similar_users, n=n_recommendations
        )
        
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.subheader(f"Top-{n_recommendations} Recommendations for User {user_id} ({method})")
            st.dataframe(recommendations[['title', 'predicted_rating']].style.format({'predicted_rating': '{:.2f}'}))
            
            # Visualize recommendations
            st.subheader("Recommendation Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=recommendations, x='predicted_rating', y='title', ax=ax)
            ax.set_title(f'Top-{n_recommendations} Recommendations for User {user_id}')
            ax.set_xlabel('Predicted Rating')
            ax.set_ylabel('Movie Title')
            st.pyplot(fig)

# Project info
st.markdown("""
---
### About This Project
- **Dataset**: MovieLens 100K (~100,000 ratings, 943 users, 1,682 movies).
- **Methods**: User-based and item-based collaborative filtering, SVD matrix factorization.
- **Evaluation**: RMSE (~0.89-1.01), Precision@5 (~0.25), Recall@5 (~0.20).
- **GitHub**: [Movie Recommendation System](https://github.com/your-username/Movie-Recommendation-System)
- Built to showcase data science skills for xAI.
""")