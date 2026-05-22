from data.preprocessing import (
    load_raw_data, filter_ratings, encode_ids,
    build_sparse_matrix, save_artifacts
)

films, ratings = load_raw_data(
    'data/processed/films.csv',
    'data/processed/all_ratings.csv'
)
ratings = filter_ratings(ratings)
ratings, user_enc, film_enc = encode_ids(ratings)
sparse_matrix = build_sparse_matrix(ratings, shape_order='user_item')

save_artifacts('data/prepared/', sparse_matrix, user_enc, film_enc, films)
print(f"Done. Shape: {sparse_matrix.shape}")
