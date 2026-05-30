"""
3 protocoles d'évaluation pour le recommandeur ALS.

Lancer depuis la racine du projet :
    .venv/bin/python -m models.evaluate

Premier lancement : entraîne le modèle (~5-10 min) et met en cache dans data/models/.
Lancements suivants : charge le cache en quelques secondes.
"""
import pickle
from pathlib import Path

import numpy as np

from data.preprocessing import (
    load_raw_data, filter_ratings, encode_ids, build_sparse_matrix
)
from models.ALS import ALSExplicit

CACHE_FILE = Path("data/models/als_eval_cache.pkl")
RANDOM_STATE = 67
MY_USER_ID = 1695743  # Mon ID SensCritique


# utilitaire

def film_label(idx, films_df, film_encoder):
    film_id = film_encoder.inverse_transform([idx])[0]
    row = films_df[films_df["film_id"] == film_id]
    return row["title"].values[0] if len(row) else "id=" + str(film_id)


def search_film(query, films_df, film_encoder):
    """Retourne (idx_encodé, titre) — priorité à l'exact, puis au plus court contenant la query."""
    q = query.lower().strip()
    titles = films_df["title"].str.lower()

    exact = films_df[titles == q]
    if not exact.empty:
        film_id = exact.iloc[0]["film_id"]
    else:
        matches = films_df[titles.str.contains(q, na=False, regex=False)]
        if matches.empty:
            return None, None
        # titre le plus court = le moins ambigu (ex : "Stalker" avant "Stalker 2")
        matches = matches.iloc[matches["title"].str.len().argsort()]
        film_id = matches.iloc[0]["film_id"]

    row = films_df[films_df["film_id"] == film_id].iloc[0]
    idx = int(film_encoder.transform([film_id])[0])
    return idx, row["title"]


def popularity(R):
    """Nombre de notes par film (tableau dense, longueur = n_items)."""
    return np.diff(R.T.tocsr().indptr)


# chargement ou entraînement

def load_or_train():
    if CACHE_FILE.exists():
        print("Chargement du cache :", CACHE_FILE)
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Entraînement du modèle (première fois, ~5-10 min)...")
    films_df, ratings = load_raw_data(
        "data/processed/films.csv", "data/processed/all_ratings.csv"
    )
    ratings = filter_ratings(ratings)
    ratings, user_encoder, film_encoder = encode_ids(ratings)
    R = build_sparse_matrix(ratings)
    print("shape :", R.shape, "| nnz :", R.nnz)

    model = ALSExplicit(n_factors=20, n_iterations=15, reg=20.0, random_state=RANDOM_STATE)
    model.fit(R)

    cache = dict(
        model=model,
        films_df=films_df,
        user_encoder=user_encoder,
        film_encoder=film_encoder,
        R=R,
    )
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print("Cache sauvegardé →", CACHE_FILE)
    return cache


#  Test 1 — plus proches voisins de films

def test_nearest_films(model, films_df, film_encoder, k=10):
    print("=" * 60)
    print("TEST 1 — Plus proches voisins dans l'espace latent Q")
    print("=" * 60)

    queries = [
        "Pulp Fiction",
        "Chihiro",
        "Dark Knight - Le",
        "Shining",
        "Stalker",
        "Orange mécanique",
    ]

    norms = np.linalg.norm(model.Q, axis=1)

    for query in queries:
        idx, title = search_film(query, films_df, film_encoder)
        if idx is None:
            print("  introuvable :", query)
            continue
        sims = (model.Q @ model.Q[idx]) / (norms * norms[idx] + 1e-9)
        sims[idx] = -np.inf
        top_k = np.argsort(sims)[::-1][:k]
        print("\n  >", title)
        for rank, i in enumerate(top_k, 1):
            label = film_label(i, films_df, film_encoder)
            print("   ", rank, label, ":", round(sims[i], 3))


# Test 2 — mon profil

def test_mon_profil(model, films_df, film_encoder, user_encoder, R):
    print("\n" + "=" * 60)
    print("TEST 2 — Mon profil (user_id=" + str(MY_USER_ID) + ")")
    print("=" * 60)

    if MY_USER_ID not in user_encoder.classes_:
        print("User absent du dataset (filtré ou non scrapé)")
        return

    u_enc = int(user_encoder.transform([MY_USER_ID])[0])
    R_csr = R.tocsr()
    pop = popularity(R)

    # Films que j'ai notés, triés par note décroissante
    start, end = R_csr.indptr[u_enc], R_csr.indptr[u_enc + 1]
    items = R_csr.indices[start:end]
    ratings = R_csr.data[start:end]
    order = np.argsort(ratings)[::-1]

    print("\n  Mes top 10 films notés :")
    for i in order[:10]:
        label = film_label(items[i], films_df, film_encoder)
        print("   ", int(ratings[i]), " ", label)

    # Recommandations
    recos = model.recommend(u_enc, n=10)
    print("\n  Mes 10 recommandations :")
    for rank, i in enumerate(recos, 1):
        score = model.predict(u_enc, i)
        label = film_label(i, films_df, film_encoder)
        print("   ", rank, label, "| score :", round(score, 2), "| pop :", pop[i])


# Test 3 — biais de popularité

def test_popularity_bias(model, R, n_sample=2000, top_n=10):
    print("\n" + "=" * 60)
    print("TEST 3 — Biais de popularité")
    print("=" * 60)

    pop = popularity(R)
    mean_pop_global = pop.mean()

    rng = np.random.default_rng(RANDOM_STATE)
    R_csr = R.tocsr()
    users_with_ratings = np.where(np.diff(R_csr.indptr) > 0)[0]
    sample_users = rng.choice(users_with_ratings, size=min(n_sample, len(users_with_ratings)), replace=False)

    top_pops = []
    for u in sample_users:
        recos = model.recommend(u, n=top_n)
        top_pops.append(pop[recos].mean())

    mean_pop_top = np.mean(top_pops)
    ratio = mean_pop_top / mean_pop_global

    print("\n  Popularité moyenne globale :", round(mean_pop_global), "notes")
    print("  Popularité moyenne top-" + str(top_n) + "  :", round(mean_pop_top), "notes")
    print("  Ratio                      :", round(ratio, 2), "x")

    if ratio < 1.5:
        print("  → Biais faible. IPS apporterait peu.")
    elif ratio < 3.0:
        print("  → Biais modéré. IPS vaut le coup.")
    else:
        print("  → Biais fort (>3x). IPS est pertinent et le gain sera mesurable.")

    quantiles = [0.25, 0.5, 0.75, 0.9]
    q_global = np.quantile(pop, quantiles)
    q_top = np.quantile(top_pops, quantiles)
    print("\n  Quantile  |  Global  |  Top-recos")
    for q, vg, vt in zip(quantiles, q_global, q_top):
        print("  ", str(int(q * 100)) + "%", "      ", round(vg), "      ", round(vt))



if __name__ == "__main__":
    cache = load_or_train()
    model        = cache["model"]
    films_df     = cache["films_df"]
    user_encoder = cache["user_encoder"]
    film_encoder = cache["film_encoder"]
    R            = cache["R"]

    test_nearest_films(model, films_df, film_encoder)
    test_mon_profil(model, films_df, film_encoder, user_encoder, R)
    test_popularity_bias(model, R)
