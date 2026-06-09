"""
CLI de recommandation de films — modèle ALS entraîné sur SensCritique.

    .venv/bin/python recommend.py
"""
import difflib
import pickle
import string
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_FILE  = Path("data/models/als_eval_cache.pkl")
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


COLD_START_THRESHOLD = 20


def fold_in(model, rated_enc, rated_values):
    """Solve ALS pour le vecteur latent d'un nouvel utilisateur."""
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


def score_new_user(model, rated_enc, rated_values):
    """Score tous les items pour un nouvel utilisateur.
    < COLD_START_THRESHOLD notes : similarité cosinus au centroïde pondéré par
    les notes (le fold-in ALS n'a pas assez de signal avec le modèle IPS).
    >= COLD_START_THRESHOLD notes : fold-in ALS classique."""
    items = np.array(rated_enc)
    r     = np.array(rated_values, dtype=float)

    if len(items) >= COLD_START_THRESHOLD:
        p  = fold_in(model, rated_enc, rated_values)
        bu = float(r.mean() - model.mu)
        return model.Q @ p + model.mu + bu + model.bi

    weights  = np.maximum(r - model.mu, 0.1)
    centroid = (model.Q[items] * weights[:, None]).sum(axis=0)
    c_norm   = np.linalg.norm(centroid)
    if c_norm == 0:
        return model.bi.copy()
    centroid /= c_norm
    q_norms = np.linalg.norm(model.Q, axis=1)
    q_norms[q_norms == 0] = 1.0
    return (model.Q @ centroid) / q_norms


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

    scores = score_new_user(model, rated_enc, rated_values)
    print_recos(scores, seen_enc, film_encoder, films_df)


MAPPING_FILE = Path("data/processed/title_mapping.csv")


def load_title_mapping():
    """Charge la table de correspondance Letterboxd → SensCritique."""
    if not MAPPING_FILE.exists():
        return {}
    df = pd.read_csv(MAPPING_FILE)
    mapping = {}
    for _, r in df.iterrows():
        key = _normalize(str(r["lb_title"]))
        mapping[key] = int(r["product_id"])
    return mapping


def mode_csv(model, films_df, film_encoder):
    print("")
    csv_path = input("Chemin vers le fichier CSV : ").strip()
    if not csv_path:
        return

    path = Path(csv_path)
    if not path.exists():
        print("Fichier introuvable :", csv_path)
        return

    try:
        user_df = pd.read_csv(path)
    except Exception as e:
        print("Erreur lecture CSV :", e)
        return

    if "Title" not in user_df.columns or "Rating10" not in user_df.columns:
        print("Colonnes manquantes (attendu : Title, Rating10)")
        return

    lb_to_sc = load_title_mapping()
    valid_ids = set(film_encoder.classes_)

    rated_enc, rated_values, seen_enc = [], [], set()
    not_found, not_in_model = [], []

    for _, row in user_df.iterrows():
        title = str(row["Title"]).strip()
        norm = _normalize(title)

        film_id = lb_to_sc.get(norm)
        if film_id is None:
            film_id, _ = search_film(title, films_df, film_encoder)
        if film_id is None:
            not_found.append(title)
            continue
        if film_id not in valid_ids:
            not_in_model.append(title)
            continue

        enc = int(film_encoder.transform([film_id])[0])
        seen_enc.add(enc)
        rating = row.get("Rating10")
        if pd.notna(rating):
            try:
                rating = float(rating)
            except (ValueError, TypeError):
                continue
            rated_enc.append(enc)
            rated_values.append(rating)

    print(f"Films matchés : {len(seen_enc)} / {len(user_df)}")
    if not_found:
        print(f"Non trouvés ({len(not_found)}) :")
        for t in not_found:
            print(f"  - {t}")

    if not rated_enc:
        print("Aucune note exploitable.")
        return
    print(f"Notes utilisables : {len(rated_enc)}")

    scores = score_new_user(model, rated_enc, rated_values)
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
    print("2. Importer un fichier CSV")
    print("")
    choix = input("Choix (1 ou 2) : ").strip()

    if choix == "1":
        mode_manuel(model, films_df, film_encoder)
    elif choix == "2":
        mode_csv(model, films_df, film_encoder)
    else:
        print("Choix invalide.")
