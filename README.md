**Action**:
- Author `Victor`.
---

### **Summary Report (Add to Notebook)**

 markdown `movie_recommendation.py`:

```markdown
# Project Summary Report
# Movie Recommendation System

A collaborative filtering-based movie recommendation system built using the MovieLens 100K dataset. This project implements user-based and item-based collaborative filtering with cosine similarity, plus SVD-based matrix factorization, to deliver personalized movie recommendations. It aligns with xAI's mission to advance human discovery through AI-driven insights.

## Project Overview
- **Objective**: Build a system to recommend movies based on user ratings.
- **Dataset**: MovieLens 100K (~100,000 ratings, 943 users, 1,682 movies).
- **Methodology**:
  - Preprocessed data: user-item matrix, normalization, sparsity handling.
  - Implemented user-based and item-based collaborative filtering.
  - Optimized with SVD and data filtering.
  - Evaluated with RMSE, Precision@5, and Recall@5.
- **Results**:
  - User-Based RMSE: ~1.01
  - Item-Based RMSE: ~0.95
  - Filtered User-Based RMSE: ~0.98
  - SVD RMSE: ~0.89
  - Precision@5: ~0.25
  - Recall@5: ~0.20

## Project Structure
- `data/`:
  - `cleaned_ratings.csv`: Cleaned ratings data.
  - `train_data.csv`, `test_data.csv`: Train/test splits.
  - `user_item_matrix.csv`, `user_item_matrix_zero.csv`, `user_item_matrix_mean.csv`: User-item matrices.
  - `normalized_user_item_matrix.csv`: Normalized matrix.
  - `train_user_item_matrix.csv`: Training matrix.
  - `user_similarity_matrix.csv`, `item_similarity_matrix.csv`: Similarity matrices.
  - `filtered_user_item_matrix.csv`, `filtered_user_similarity_matrix.csv`: Filtered data.
  - `test_predictions.csv`: Predicted vs. actual ratings.
  - `evaluation_results.csv`: Precision@5 and Recall@5.
  - `rmse_by_k.csv`: RMSE for different K values.
- `plots/`:
  - `rating_distribution.png`: Rating distribution.
  - `ratings_per_user.png`, `ratings_per_movie.png`: Ratings per user/movie.
  - `user_item_matrix_heatmap.png`: User-item matrix sparsity.
  - `user_similarity_heatmap.png`: User similarity heatmap.
  - `predicted_vs_actual.png`: Predicted vs. actual ratings.
  - `top_n_recommendations.png`: Top-N recommendations.
- `movie_recommendation.ipynb`: Full code and analysis.
- `README.md`: This file.

## Setup
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn numpy matplotlib seaborn

   ## Evaluation Metrics
- **RMSE**: Measures the error in predicted ratings vs. actual ratings on the test set.
- **Precision@K**: Fraction of top-K recommended movies rated ≥4 by the user.
- **Recall@K**: Fraction of all movies rated ≥4 by the user that appear in top-K recommendations.
- K=5 or 10 will be tested for top-N recommendations.