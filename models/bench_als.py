import time
import numpy as np
from data.preprocessing import load_raw_data, filter_ratings, encode_ids, build_sparse_matrix, train_test_split_ratings
from models.ALS import ALSExplicit

#N_USERS = 20_000
RANDOM_STATE = 67

print("Chargement...")
_, ratings = load_raw_data('data/processed/films.csv', 'data/processed/all_ratings.csv')
ratings = filter_ratings(ratings)

rng = np.random.default_rng(RANDOM_STATE)
#sampled_users = rng.choice(ratings['user_id'].unique(), size=N_USERS, replace=False)
#ratings = ratings[ratings['user_id'].isin(sampled_users)]
ratings, _, _ = encode_ids(ratings)

R = build_sparse_matrix(ratings)
R_train, R_test = train_test_split_ratings(R, random_state=RANDOM_STATE)
print("shape :", R.shape)
print("ratings :", R.nnz, "| train :", R_train.nnz, "| test :", R_test.nnz)
print()


# Modèle (n_factors=20, reg=20, 15 iter)
print("--- Modèle (20 factors, reg=20, 15 iter) ---")
model = ALSExplicit(n_factors=20, n_iterations=15, reg=20.0, verbose=False, random_state=0)
t0 = time.time()
model.fit(R_train)
t = time.time() - t0
print("temps :", round(t, 1), "s | train RMSE :", round(model.score(R_train), 4), "| test RMSE :", round(model.score(R_test), 4))
print()


# Sanity check distributions
print("--- Sanity check ---")
rows, cols = R_test.nonzero()
preds_test = np.clip(
    np.sum(model.P[rows] * model.Q[cols], axis=1)
    + model.mu + model.bu[rows] + model.bi[cols],
    1, 10
)
print("Std prédictions :", round(preds_test.std(), 4))
print("Std vraies notes :", round(R_test.data.std(), 4))
print("Quantiles préd :", np.quantile(preds_test, [0.05, 0.5, 0.95]))
print("Quantiles vrai :", np.quantile(R_test.data, [0.05, 0.5, 0.95]))
print()


# Setup commun aux deux protocoles
R_test_csr = R_test.tocsr()
R_train_csr = R_train.tocsr()
n_items = R.shape[1]
eval_users = np.where(np.diff(R_test_csr.indptr) > 0)[0]
log2_ranks = np.log2(np.arange(2, 12))
rated_mask = np.zeros(n_items, dtype=bool)
rng_eval = np.random.default_rng(RANDOM_STATE)


# Protocole A — tous les positifs test + 100 négatifs
# "est-ce que mon modèle ranke bien l'ensemble des goûts de l'user ?"
print("--- Protocole A : positifs test + 100 négatifs ---")
ndcg_scores, map_scores = [], []

for u in eval_users:
    ts, te = R_test_csr.indptr[u], R_test_csr.indptr[u + 1]
    test_items = R_test_csr.indices[ts:te]
    test_ratings = R_test_csr.data[ts:te].astype(float)

    trs, tre = R_train_csr.indptr[u], R_train_csr.indptr[u + 1]
    train_items = R_train_csr.indices[trs:tre]

    rated_mask[train_items] = True
    rated_mask[test_items] = True
    unrated = np.where(~rated_mask)[0]
    neg_items = rng_eval.choice(unrated, size=min(100, len(unrated)), replace=False)
    rated_mask[train_items] = False
    rated_mask[test_items] = False

    candidates = np.concatenate([test_items, neg_items])
    scores_cand = model.Q[candidates] @ model.P[u] + model.mu + model.bu[u] + model.bi[candidates]
    top10 = candidates[np.argsort(scores_cand)[::-1]][:10]

    # NDCG@10 — relevance = rating brut
    test_rating_dict = dict(zip(test_items, test_ratings))
    gains = np.array([test_rating_dict.get(i, 0.0) for i in top10])
    ideal = np.sort(test_ratings)[::-1][:10]
    dcg  = np.sum(gains / log2_ranks[:len(gains)])
    idcg = np.sum(ideal / log2_ranks[:len(ideal)])
    ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    # MAP@10 — rating >= 7 = pertinent
    relevant = set(test_items[test_ratings >= 7])
    if relevant:
        hits, prec_sum = 0, 0.0
        for k, item in enumerate(top10):
            if item in relevant:
                hits += 1
                prec_sum += hits / (k + 1)
        map_scores.append(prec_sum / min(len(relevant), 10))
    else:
        map_scores.append(0.0)

print("NDCG@10 (relevance = rating)    :", round(np.mean(ndcg_scores), 4))
print("MAP@10  (relevance = rating>=7) :", round(np.mean(map_scores), 4))
print()


# Protocole B — 1 best positif + 99 négatifs (NeuMF strict)
# "est-ce que je sors le chef-d'œuvre de chaque user dans mon top-10 ?"
print("--- Protocole B : 1 best positif + 99 négatifs (NeuMF) ---")
hr_neumf, ndcg_neumf = [], []

for u in eval_users:
    ts, te = R_test_csr.indptr[u], R_test_csr.indptr[u + 1]
    test_items = R_test_csr.indices[ts:te]
    test_ratings = R_test_csr.data[ts:te].astype(float)

    trs, tre = R_train_csr.indptr[u], R_train_csr.indptr[u + 1]
    train_items = R_train_csr.indices[trs:tre]

    best_item = test_items[np.argmax(test_ratings)]

    rated_mask[train_items] = True
    rated_mask[test_items] = True
    unrated = np.where(~rated_mask)[0]
    neg_items = rng_eval.choice(unrated, size=min(99, len(unrated)), replace=False)
    rated_mask[train_items] = False
    rated_mask[test_items] = False

    candidates = np.concatenate([[best_item], neg_items])
    scores_cand = model.Q[candidates] @ model.P[u] + model.mu + model.bu[u] + model.bi[candidates]
    top10 = candidates[np.argsort(scores_cand)[::-1]][:10]

    hit = best_item in top10
    hr_neumf.append(float(hit))
    if hit:
        rank = int(np.where(top10 == best_item)[0][0])
        ndcg_neumf.append(1.0 / log2_ranks[rank])
    else:
        ndcg_neumf.append(0.0)

print("HR@10   (meilleur item user)    :", round(np.mean(hr_neumf), 4))
print("NDCG@10 (meilleur item user)    :", round(np.mean(ndcg_neumf), 4))
