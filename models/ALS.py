import numpy as np
from threadpoolctl import threadpool_limits


class ALSExplicit:
    def __init__(self, n_factors=50, n_iterations=15, reg=0.1,
                 use_bias=True, random_state=None, verbose=True):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self.use_bias = use_bias
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, R):
        # R_user[u] -> films notés par u et R_item[i] -> users ayant noté i
        self.R_user = R.tocsr()
        self.R_item = R.T.tocsr()
        self.n_users, self.n_items = R.shape

        rng = np.random.default_rng(self.random_state)
        self.P = rng.normal(0, 0.1, (self.n_users, self.n_factors))  # user factors
        self.Q = rng.normal(0, 0.1, (self.n_items, self.n_factors))  # item factors

        if self.use_bias:
            self._fit_biases()

        self._init_rmse_sample(rng)
        reg_I = self.reg * np.eye(self.n_factors)

        # Les solves sont petits (50x50) : BLAS multi-thread n'apporte rien, on le force en mono-thread.
        with threadpool_limits(limits=1, user_api="blas"):
            for it in range(self.n_iterations):
                self._update_users(reg_I)
                self._update_items(reg_I)
                if self.verbose:
                    print(f"Iteration {it + 1}/{self.n_iterations} - RMSE: {self._compute_rmse():.4f}")

        return self

    def _fit_biases(self):
        # Les biais sont précomputés une fois (bias baseline + MF).
        # L'ALS biaisé classique les ré-apprend à chaque itération, mais cette
        # variante est plus simple et donne des résultats proches en pratique.

        # mu = moyenne globale des ratings
        self.mu = self.R_user.data.mean()

        # b_u[u] = moyenne des ratings de u - mu
        row_sums = np.asarray(self.R_user.sum(axis=1)).flatten()
        row_counts = np.diff(self.R_user.indptr)
        self.bu = np.where(row_counts > 0, row_sums / row_counts - self.mu, 0.0)

        # b_i[i] = moyenne(r_ui - mu - b_u) sur les users ayant noté i
        # On utilise indptr pour reconstruire rows (plus robuste que nonzero())
        rows = np.repeat(np.arange(self.n_users), row_counts)
        cols = self.R_user.indices
        adjusted = self.R_user.data - self.mu - self.bu[rows]
        item_sums = np.bincount(cols, weights=adjusted, minlength=self.n_items)
        item_counts = np.bincount(cols, minlength=self.n_items)
        self.bi = np.where(item_counts > 0, item_sums / item_counts, 0.0)

    def _update_users(self, reg_I):
        # Thread-safe : chaque itération écrit sur une ligne distincte de self.P (indexée par u)
        for u in range(self.n_users):
            row = self.R_user[u]
            if row.nnz == 0:
                continue
            items, r = row.indices, row.data.astype(float)
            if self.use_bias:
                r -= self.mu + self.bu[u] + self.bi[items]
            Y = self.Q[items]
            #On résout les moindres carrés pour l'user
            self.P[u] = np.linalg.solve(Y.T @ Y + reg_I, Y.T @ r)

    def _update_items(self, reg_I):
        for i in range(self.n_items):
            col = self.R_item[i]
            if col.nnz == 0:
                continue
            users, r = col.indices, col.data.astype(float)
            if self.use_bias:
                r -= self.mu + self.bu[users] + self.bi[i]
            X = self.P[users]
            #On resout les moindres carrés pour l'item
            self.Q[i] = np.linalg.solve(X.T @ X + reg_I, X.T @ r)

    def _init_rmse_sample(self, rng, n=200_000):
        # Échantillon fixe pour monitorer le RMSE sans parcourir les 46M ratings
        rows, cols = self.R_user.nonzero()
        idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
        self._s_rows = rows[idx]
        self._s_cols = cols[idx]
        self._s_ratings = self.R_user.data[idx]

    def _compute_rmse(self):
        preds = np.sum(self.P[self._s_rows] * self.Q[self._s_cols], axis=1)
        if self.use_bias:
            preds += self.mu + self.bu[self._s_rows] + self.bi[self._s_cols]
        return float(np.sqrt(np.mean((self._s_ratings - preds) ** 2)))

    def predict(self, u, i):
        score = float(self.P[u] @ self.Q[i])
        if self.use_bias:
            score += self.mu + self.bu[u] + self.bi[i]
        return float(np.clip(score, 1, 10))

    def recommend(self, u, n=10, exclude_rated=True):
        scores = self.Q @ self.P[u]
        if self.use_bias:
            scores += self.mu + self.bu[u] + self.bi
        if exclude_rated:
            scores[self.R_user[u].indices] = -np.inf
        return np.argsort(scores)[::-1][:n]

    def score(self, R_test):
        R_test = R_test.tocsr()
        rows, cols = R_test.nonzero()
        preds = np.clip(
            np.sum(self.P[rows] * self.Q[cols], axis=1) + (
                self.mu + self.bu[rows] + self.bi[cols] if self.use_bias else 0
            ), 1, 10
        )
        return float(np.sqrt(np.mean((R_test.data - preds) ** 2)))
