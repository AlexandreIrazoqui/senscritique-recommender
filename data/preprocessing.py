# src/data/preprocessing.py
import pickle
from pathlib import Path
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.preprocessing import LabelEncoder


def load_raw_data(films_path, ratings_path):
    films = pd.read_csv(films_path).rename(columns={'product_id': 'film_id'})
    ratings = pd.read_csv(ratings_path).rename(columns={'product_id': 'film_id'})
    return films, ratings


def filter_ratings(ratings, min_user_ratings=20, min_film_ratings=50,
                   exclude_seed_user=True):
    if exclude_seed_user:
        seed_user = ratings['user_id'].value_counts().index[0]
        ratings = ratings[ratings['user_id'] != seed_user]
    
    user_counts = ratings['user_id'].value_counts()
    ratings = ratings[ratings['user_id'].isin(user_counts[user_counts >= min_user_ratings].index)]
    
    film_counts = ratings['film_id'].value_counts()
    ratings = ratings[ratings['film_id'].isin(film_counts[film_counts >= min_film_ratings].index)]
    
    return ratings


def encode_ids(ratings):
    user_encoder = LabelEncoder()
    film_encoder = LabelEncoder()
    ratings = ratings.copy()
    ratings['user_id'] = user_encoder.fit_transform(ratings['user_id'])
    ratings['film_id'] = film_encoder.fit_transform(ratings['film_id'])
    return ratings, user_encoder, film_encoder


def build_sparse_matrix(ratings, shape_order='user_item'):
    """shape_order='user_item' pour implicit, 'item_user' pour l'ancien format."""
    n_users = ratings['user_id'].nunique()
    n_films = ratings['film_id'].nunique()
    
    if shape_order == 'user_item':
        return csr_matrix(
            (ratings['rating'].values,
             (ratings['user_id'].values, ratings['film_id'].values)),
            shape=(n_users, n_films)
        )
    else:
        return csr_matrix(
            (ratings['rating'].values,
             (ratings['film_id'].values, ratings['user_id'].values)),
            shape=(n_films, n_users)
        )


def save_artifacts(output_dir, sparse_matrix, user_encoder, film_encoder, films_df):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_npz(output_dir / 'sparse_matrix.npz', sparse_matrix)
    
    with open(output_dir / 'user_encoder.pkl', 'wb') as f:
        pickle.dump(user_encoder, f)
    with open(output_dir / 'film_encoder.pkl', 'wb') as f:
        pickle.dump(film_encoder, f)
    
    films_df.to_csv(output_dir / 'films.csv', index=False)


def load_artifacts(input_dir):
    input_dir = Path(input_dir)
    sparse_matrix = load_npz(input_dir / 'sparse_matrix.npz')
    
    with open(input_dir / 'user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)
    with open(input_dir / 'film_encoder.pkl', 'rb') as f:
        film_encoder = pickle.load(f)
    
    films = pd.read_csv(input_dir / 'films.csv')
    return sparse_matrix, user_encoder, film_encoder, films
