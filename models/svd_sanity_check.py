import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

from data.preprocessing import load_raw_data, filter_ratings

# Chargement et filtrage
films, ratings = load_raw_data('data/processed/films.csv', 'data/processed/all_ratings.csv')
ratings = filter_ratings(ratings)

# 500K ratings suffisent pour valider le pipeline et avoir un RMSE de référence.
ratings = ratings.sample(n=500_000, random_state=42)
print(f"{len(ratings):,} ratings | {ratings['user_id'].nunique():,} users | {ratings['film_id'].nunique():,} films")

# Dataset Surprise
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['user_id', 'film_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=67)

# n_factors=50 : ~5K films dans le dataset, 50 dimensions capturent bien la diversité sans overfitter
# n_epochs=20  : convergence standard pour SGD sur cette échelle
# lr_all=0.005 : learning rate par défaut Surprise, stable en pratique
# reg_all=0.02 : régularisation légère — dataset dense (sparsité 92.5%), peu de risque d'overfitting
model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, verbose=True)
model.fit(trainset)

predictions = model.test(testset)
print(f"RMSE : {accuracy.rmse(predictions):.4f}")
print(f"MAE  : {accuracy.mae(predictions):.4f}")
