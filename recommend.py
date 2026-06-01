"""
CLI de recommandation de films — modèle ALS entraîné sur SensCritique.

    .venv/bin/python recommend.py
"""
import difflib
import pickle
import string
import time
import unicodedata
from pathlib import Path

import numpy as np
import requests

CACHE_FILE  = Path("data/models/als_eval_cache.pkl")
SC_API      = "https://apollo.senscritique.com/"
SC_HEADERS  = {
    "content-type": "application/json",
    "Origin":       "https://www.senscritique.com",
    "Referer":      "https://www.senscritique.com/",
    "User-Agent":   "Mozilla/5.0 (X11; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0",
}
N_RECO      = 10


def load_model():
    if CACHE_FILE.exists():
        print("Chargement du modèle depuis le cache...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Cache absent — entraînement du modèle (~5-10 min)...")
    from data.preprocessing import (
        load_raw_data, filter_ratings, encode_ids, build_sparse_matrix
    )
    from models.ALS import ALSExplicit

    films_df, ratings = load_raw_data(
        "data/processed/films.csv", "data/processed/all_ratings.csv"
    )
    ratings = filter_ratings(ratings)
    ratings, user_encoder, film_encoder = encode_ids(ratings)
    R = build_sparse_matrix(ratings)

    model = ALSExplicit(n_factors=20, n_iterations=15, reg=20.0,
                        use_ips=True, ips_alpha=0.3, random_state=67)
    model.fit(R)

    cache = dict(model=model, films_df=films_df,
                 user_encoder=user_encoder, film_encoder=film_encoder, R=R)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print("Modèle sauvegardé →", str(CACHE_FILE))
    return cache


def _normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def search_film(query, films_df, film_encoder):
    q           = _normalize(query)
    norm_titles = films_df["title"].apply(_normalize).tolist()
    film_ids    = films_df["film_id"].tolist()
    valid_ids   = set(film_encoder.classes_)

    for i, t in enumerate(norm_titles):
        if t == q and film_ids[i] in valid_ids:
            return film_ids[i], films_df.iloc[i]["title"]

    containing = [
        (len(t), i)
        for i, t in enumerate(norm_titles)
        if q in t and film_ids[i] in valid_ids
    ]
    if containing:
        _, i = min(containing)
        return film_ids[i], films_df.iloc[i]["title"]

    matches = difflib.get_close_matches(q, norm_titles, n=5, cutoff=0.75)
    for m in matches:
        i = norm_titles.index(m)
        if film_ids[i] in valid_ids:
            return film_ids[i], films_df.iloc[i]["title"]

    return None, None


def get_user_id_from_username(username):
    payload = {
        "operationName": "User",
        "variables":     {"username": username},
        "query":         "query User($username: String!) { user(username: $username) { id } }",
    }
    resp = requests.post(SC_API, headers=SC_HEADERS, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]["user"]["id"]


_COLLECTION_QUERY = """
query UserCollectionFilms($userId: Int!, $universe: String, $limit: Int, $offset: Int) {
    user(id: $userId) {
        collection(universe: $universe, limit: $limit, offset: $offset) {
            products {
                id
                title
                otherUserInfos {
                    rating
                }
            }
        }
    }
}
"""

def get_user_seen_films(user_id, limit=100):
    seen, offset = [], 0
    while True:
        payload = {
            "operationName": "UserCollectionFilms",
            "variables": {
                "userId":   user_id,
                "universe": "movie",
                "limit":    limit,
                "offset":   offset,
            },
            "query": _COLLECTION_QUERY,
        }
        resp = requests.post(SC_API, headers=SC_HEADERS, json=payload, timeout=15)
        resp.raise_for_status()
        products = resp.json()["data"]["user"]["collection"]["products"]
        if not products:
            break
        for p in products:
            info = p.get("otherUserInfos") or {}
            seen.append({"id": p["id"], "title": p["title"], "rating": info.get("rating")})
        offset += limit
        time.sleep(0.4)
    return seen


def fold_in(model, rated_enc, rated_values):
    """Reproduit _update_users : sur les notes d'entraînement d'un user connu,
    redonne P[u] exactement. Pour un nouveau, meilleure approx à partir de ses notes."""
    if len(rated_enc) == 0:
        return np.zeros(model.n_factors)

    items = np.array(rated_enc)
    r     = np.array(rated_values, dtype=float)
    bu    = float(r.mean() - model.mu)
    r_adj = r - model.mu - bu - model.bi[items]
    Y     = model.Q[items]
    w     = model.w_i[items] if hasattr(model, "w_i") else np.ones(len(items))
    Yw    = Y * w[:, None]
    reg_I = model.reg * np.eye(model.n_factors)
    return np.linalg.solve(Yw.T @ Y + reg_I, Yw.T @ r_adj)


def print_recos(scores, exclude_enc, film_encoder, films_df):
    s = scores.copy()
    if exclude_enc:
        s[list(exclude_enc)] = -np.inf
    top = np.argsort(s)[::-1][:N_RECO]
    print("")
    for rank, idx in enumerate(top, 1):
        film_id = film_encoder.inverse_transform([idx])[0]
        row     = films_df[films_df["film_id"] == film_id]
        title   = row["title"].values[0] if len(row) else str(film_id)
        print(rank, ".", title)


def mode_manuel(model, films_df, film_encoder):
    print("")
    print("Entrez des films avec une note /10.")
    print("")

    rated_enc, rated_values, seen_enc = [], [], set()

    while True:
        titre = input("Film (vide pour terminer) : ").strip()
        if not titre:
            break
        film_id, found = search_film(titre, films_df, film_encoder)
        if film_id is None:
            print("  Film introuvable")
            continue
        print("  Trouvé :", found)
        note_str = input("  Note /10 : ").strip()
        try:
            note = float(note_str)
        except ValueError:
            continue
        enc = int(film_encoder.transform([film_id])[0])
        rated_enc.append(enc)
        rated_values.append(note)
        seen_enc.add(enc)

    if not rated_enc:
        return

    p      = fold_in(model, rated_enc, rated_values)
    bu     = float(np.array(rated_values).mean() - model.mu)
    scores = model.Q @ p + model.mu + bu + model.bi
    print_recos(scores, seen_enc, film_encoder, films_df)


def mode_pseudo(model, films_df, film_encoder):
    print("")
    username = input("Pseudo SensCritique : ").strip()
    if not username:
        return

    try:
        user_id = get_user_id_from_username(username)
    except Exception as e:
        print("Erreur API :", e)
        return

    try:
        seen = get_user_seen_films(user_id)
    except Exception as e:
        print("Erreur collection :", e)
        seen = []
    print("Films vus :", len(seen))

    valid_ids = set(film_encoder.classes_)
    rated_enc, rated_values, seen_enc = [], [], set()
    for f in seen:
        if f["id"] in valid_ids:
            enc = int(film_encoder.transform([f["id"]])[0])
            seen_enc.add(enc)
            if f["rating"]:
                rated_enc.append(enc)
                rated_values.append(float(f["rating"]))

    if not rated_enc:
        print("Aucune note exploitable.")
        return
    print("Notes utilisables :", len(rated_enc))

    p      = fold_in(model, rated_enc, rated_values)
    bu     = float(np.array(rated_values).mean() - model.mu)
    scores = model.Q @ p + model.mu + bu + model.bi

    print("")
    print("Recommandations pour", username, ":")
    print_recos(scores, seen_enc, film_encoder, films_df)


if __name__ == "__main__":
    cache        = load_model()
    model        = cache["model"]
    films_df     = cache["films_df"]
    film_encoder = cache["film_encoder"]
    user_encoder = cache["user_encoder"]

    print("")
    print("=== Recommandeur SensCritique ===")
    print("")
    print("1. Entrer des films manuellement")
    print("2. Utiliser un pseudo SensCritique")
    print("")
    choix = input("Choix (1 ou 2) : ").strip()

    if choix == "1":
        mode_manuel(model, films_df, film_encoder)
    elif choix == "2":
        mode_pseudo(model, films_df, film_encoder)
    else:
        print("Choix invalide.")
