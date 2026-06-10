"""
Comparaison ALS baseline vs ALS+IPS.
- RMSE par tertile de popularité (+ baseline biais-seuls)
- Ranking protocole B avec négatifs appariés en popularité (+ baselines popularité-pure et biais-seuls)

    .venv/bin/python -m models.bench_ips

Signal attendu si IPS fonctionne :
  RMSE : amélioration sur niche, légère dégradation sur populaire
  Ranking : gain sur HR/NDCG au-dessus de la baseline popularité-pure
"""
import time
import numpy as np

from data.preprocessing import (
    load_raw_data, filter_ratings, encode_ids,
    build_sparse_matrix, train_test_split_ratings,
)
from models.ALS import ALSExplicit

RANDOM_STATE  = 67
N_EVAL_USERS  = 2000
ALPHAS        = [1.0, 0.5, 0.3]
CONFIG        = dict(n_factors=20, n_iterations=15, reg=20.0,
                     random_state=RANDOM_STATE, verbose=False)

# Données

print("Chargement...")
_, ratings = load_raw_data("data/processed/films.csv", "data/processed/all_ratings.csv")
ratings = filter_ratings(ratings)
ratings, _, _ = encode_ids(ratings)
R = build_sparse_matrix(ratings)
R_train, R_test = train_test_split_ratings(R, random_state=RANDOM_STATE)
n_items = R.shape[1]
print(f"shape : {R.shape} | train : {R_train.nnz} | test : {R_test.nnz}\n")

n_i = np.diff(R_train.T.tocsr().indptr)   # nombre de notes par item sur le train

# tertiles pour RMSE (distribution des notes test)
rows_test, cols_test = R_test.nonzero()
r_test    = R_test.data.astype(float)
pop_test  = n_i[cols_test]
t33, t67  = np.percentile(pop_test, [33, 67])
rmse_masks = {
    f"niche     (≤{int(t33)})":          pop_test <= t33,
    f"moyen     ({int(t33)}–{int(t67)})": (pop_test > t33) & (pop_test <= t67),
    f"populaire (>{int(t67)})":           pop_test > t67,
}

# pré-tri par popularité pour le sampling apparié en log-bande
_sort_idx    = np.argsort(n_i)
_sorted_n_i  = n_i[_sort_idx]


#  RMSE

def rmse_tertiles(preds):
    preds = np.clip(preds, 1, 10)
    errs  = r_test - preds
    out   = {lbl: float(np.sqrt(np.mean(errs[mask] ** 2)))
             for lbl, mask in rmse_masks.items()}
    out["global"] = float(np.sqrt(np.mean(errs ** 2)))
    return out


#  Rankin

log2_ranks = np.log2(np.arange(2, 12))

def _sample_neg_log_band(best_item, rated_mask, rng, n_neg=99, factor=1.2):
    """
    Tire n_neg négatifs dans une bande log-popularité ±factor autour du positif.
    Fallback progressif (factor^2, factor^4, tous items) si la bande est trop petite.
    Si pop-pure ≈ 0.10 dans le tableau ranking, l'appariement est propre.
    """
    p = int(n_i[best_item])
    for f in [factor, factor ** 2, factor ** 4, float("inf")]:
        lo   = p / f if f < float("inf") else 0
        hi   = p * f if f < float("inf") else float("inf")
        l    = int(np.searchsorted(_sorted_n_i, lo,  side="left"))
        r    = int(np.searchsorted(_sorted_n_i, hi,  side="right"))
        pool = _sort_idx[l:r]
        pool = pool[~rated_mask[pool]]
        if len(pool) >= n_neg:
            break
    return rng.choice(pool, size=min(n_neg, len(pool)), replace=False)


def eval_ranking(score_fn, eval_users, R_test_csr, R_train_csr):
    """Protocole B : 1 best positif + 99 négatifs appariés en log-popularité (±20%)."""
    rng        = np.random.default_rng(RANDOM_STATE)   # seed fixe → mêmes négatifs pour tous les modèles
    hr_list, ndcg_list = [], []
    rated_mask = np.zeros(n_items, dtype=bool)

    for u in eval_users:
        ts, te = R_test_csr.indptr[u], R_test_csr.indptr[u + 1]
        if ts == te:
            continue
        test_items   = R_test_csr.indices[ts:te]
        test_ratings = R_test_csr.data[ts:te].astype(float)
        trs, tre     = R_train_csr.indptr[u], R_train_csr.indptr[u + 1]
        train_items  = R_train_csr.indices[trs:tre]

        best_item = test_items[np.argmax(test_ratings)]

        rated_mask[train_items] = True
        rated_mask[test_items]  = True

        neg_items = _sample_neg_log_band(best_item, rated_mask, rng)

        rated_mask[train_items] = False
        rated_mask[test_items]  = False

        candidates = np.concatenate([[best_item], neg_items])
        scores     = score_fn(u, candidates)
        top10      = candidates[np.argsort(scores)[::-1]][:10]

        hit = int(best_item) in top10.tolist()
        hr_list.append(float(hit))
        if hit:
            rank = int(np.where(top10 == best_item)[0][0])
            ndcg_list.append(1.0 / log2_ranks[rank])
        else:
            ndcg_list.append(0.0)

    return float(np.mean(hr_list)), float(np.mean(ndcg_list))


#  Entraînement

print("--- ALS standard (baseline) ---")
t0 = time.time()
m_base = ALSExplicit(**CONFIG, use_ips=False)
m_base.fit(R_train)
print(f"Entraîné en {time.time() - t0:.1f}s\n")

models_ips = {}
for alpha in ALPHAS:
    lbl = f"IPS α={alpha}"
    print(f"--- {lbl} ---")
    t0 = time.time()
    models_ips[lbl] = ALSExplicit(**CONFIG, use_ips=True, ips_alpha=alpha)
    models_ips[lbl].fit(R_train)
    print(f"Entraîné en {time.time() - t0:.1f}s\n")


# Résultats RMSE

rmse_res = {
    "biais seuls": rmse_tertiles(m_base.mu + m_base.bu[rows_test] + m_base.bi[cols_test]),
    "baseline":    rmse_tertiles(np.sum(m_base.P[rows_test] * m_base.Q[cols_test], axis=1)
                                 + m_base.mu + m_base.bu[rows_test] + m_base.bi[cols_test]),
}
for lbl, m in models_ips.items():
    rmse_res[lbl] = rmse_tertiles(np.sum(m.P[rows_test] * m.Q[cols_test], axis=1)
                                  + m.mu + m.bu[rows_test] + m.bi[cols_test])

tlabels = list(rmse_masks.keys()) + ["global"]
col_w   = 13

print("=" * (30 + col_w * len(rmse_res)))
print("RMSE PAR TERTILE  (* = meilleur)")
print("-" * (30 + col_w * len(rmse_res)))
print(f"{'':30}" + "".join(f"{k:>{col_w}}" for k in rmse_res))
print("-" * (30 + col_w * len(rmse_res)))
for tl in tlabels:
    vals = [rmse_res[k][tl] for k in rmse_res]
    best = min(vals)
    row  = f"{tl:<30}" + "".join(
        f"{v:>{col_w-1}.4f}{'*' if v == best else ' '}" for v in vals
    )
    print(row)
print("=" * (30 + col_w * len(rmse_res)))
print()


#  Résultats Ranking

R_test_csr  = R_test.tocsr()
R_train_csr = R_train.tocsr()

rng_sample  = np.random.default_rng(RANDOM_STATE)
all_users   = np.where(np.diff(R_test_csr.indptr) > 0)[0]
eval_users  = rng_sample.choice(all_users, size=min(N_EVAL_USERS, len(all_users)), replace=False)

ranking_fns = {
    "popularité pure": lambda u, cands: n_i[cands].astype(float),
    "biais seuls":     lambda u, cands: m_base.bi[cands],
    "baseline":        lambda u, cands: m_base.Q[cands] @ m_base.P[u] + m_base.bu[u] + m_base.bi[cands],
}
for lbl, m in models_ips.items():
    ranking_fns[lbl] = (lambda _m: lambda u, cands: _m.Q[cands] @ _m.P[u] + _m.bu[u] + _m.bi[cands])(m)

print(f"RANKING — Protocole B, négatifs appariés en popularité  (N={N_EVAL_USERS} users)")
print("=" * 55)
print(f"{'Modèle':<22} {'HR@10':>10} {'NDCG@10':>10}")
print("-" * 55)
for name, fn in ranking_fns.items():
    hr, ndcg = eval_ranking(fn, eval_users, R_test_csr, R_train_csr)
    print(f"{name:<22} {hr:>10.4f} {ndcg:>10.4f}")
print("=" * 55)
print("Repère hasard : HR@10 ≈ 0.10")
print("MF utile si baseline > popularité pure sur les deux métriques.")
