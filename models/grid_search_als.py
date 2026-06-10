"""
Grid search ALS avec jeu de validation dédié.

Split : train 64% / val 16% / test 20%. Le test (identique à celui des
benchmarks, seed 67) n'est utilisé qu'une seule fois, sur la config finale —
toute la sélection (grille (n_factors, reg) puis ips_alpha) se fait sur le val.

    .venv/bin/python -m models.grid_search_als
"""
import time

from data.preprocessing import (
    load_raw_data, filter_ratings, encode_ids,
    build_sparse_matrix, train_test_split_ratings,
)
from models.ALS import ALSExplicit

RANDOM_STATE = 67
VAL_SEED     = 68

GRID_FACTORS = [10, 20, 50]
GRID_REG     = [5.0, 20.0, 50.0]
GRID_ALPHA   = [1.0, 0.5, 0.3]

print("Chargement...")
_, ratings = load_raw_data("data/processed/films.csv", "data/processed/all_ratings.csv")
ratings = filter_ratings(ratings)
ratings, _, _ = encode_ids(ratings)
R = build_sparse_matrix(ratings)

# test mis de côté avec le même seed que les benchmarks, puis le train
# restant est re-découpé pour obtenir le jeu de validation (16% du total)
R_trainfull, R_test = train_test_split_ratings(R, random_state=RANDOM_STATE)
R_train, R_val = train_test_split_ratings(R_trainfull, random_state=VAL_SEED)
print(f"train : {R_train.nnz:,} | val : {R_val.nnz:,} | test : {R_test.nnz:,}\n")


def fit_and_score(**kw):
    t0 = time.time()
    m = ALSExplicit(n_iterations=15, random_state=RANDOM_STATE, verbose=False, **kw)
    m.fit(R_train)
    rmse_val = m.score(R_val)
    print(f"  {kw} -> val RMSE {rmse_val:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return rmse_val


# ── Étape 1 : grille (n_factors, reg), ALS sans IPS, sélection sur le val ────
print("--- Grille (n_factors, reg) — ALS baseline ---")
results = {}
for k in GRID_FACTORS:
    for reg in GRID_REG:
        results[(k, reg)] = fit_and_score(n_factors=k, reg=reg, use_ips=False)

best_k, best_reg = min(results, key=results.get)
print(f"\nMeilleure config : k={best_k}, reg={best_reg} "
      f"(val RMSE {results[(best_k, best_reg)]:.4f})\n")

# ── Étape 2 : ips_alpha à (k, reg) fixés, sélection sur le val ───────────────
print("--- IPS alpha à (k, reg) fixés ---")
res_alpha = {}
for a in GRID_ALPHA:
    res_alpha[a] = fit_and_score(n_factors=best_k, reg=best_reg,
                                 use_ips=True, ips_alpha=a)
best_alpha = min(res_alpha, key=res_alpha.get)
print(f"\nMeilleur alpha (val) : {best_alpha}\n")

# ── Étape 3 : ré-entraînement sur train+val, évaluation unique sur le test ───
print("--- Modèle final (train 80%) → test ---")
final = ALSExplicit(n_factors=best_k, reg=best_reg,
                    use_ips=True, ips_alpha=best_alpha,
                    n_iterations=15, random_state=RANDOM_STATE, verbose=False)
t0 = time.time()
final.fit(R_trainfull)
print(f"Entraîné en {time.time() - t0:.0f}s")
print(f"RMSE train         : {final.score(R_trainfull):.4f}")
print(f"RMSE test (unique) : {final.score(R_test):.4f}")
