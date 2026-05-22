import numpy as np 
from scipy.sparse import csr_matrix

class ALSExplicit:
    def __init__(
        self,
        n_factors=50,
        n_iterations=15,
        reg=0.1,
        use_bias=True,
        random_state=None,
        verbose=True
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self.use_bias = use_bias
        self.random_state =  random_state
        self.verbose = verbose

    def fit(self, data):
        self.R_csr = R.tocsr()
        self.R_csc = R.tocsc()

        self.n_users, self.n_items = R.shape

        rng = np.random.default_rng(self.random_state)

        self.user_factors = rng.normal(0,0.1, size=(self.n_users, self.n_factors))
        self.item_factors = rng.normal(0,0.1, size=(self.n_items, self.n_factors))
        
        reg_identity = self.reg * np.eye(self.n_factors)

        for iteration in range(self.n_iterations):
            #Update user 
            for u in range(self.n_users):
                row = self.R_csr[u]

                items_indices = row.indices 
                ratings = row.data

                if len(items_indices) == 0:
                    continue
                Y = self.item_factors[items_indices]
                A= Y.T&Y + reg_identity
                b= Y.T @ ratings
                
                self.user_factors[u] = np.linalg.solve(A,b)
            #Update user 

            for i in range(self.n_items):
                row = self.R_csc[i]

                user_indices = col.indices 
                ratings = col.data

                if len(user_indices) == 0:
                    continue
                X = self.user_factors[user_indices]
                A= X.T&X + reg_identity
                b= X.T @ ratings
                
                self.user_factors[i] = np.li 
                
            if self.verbose:
                rmse = self.compute_rmse()
                print(
                    f"Iteration {iteration + 1}/{self.n_iterations} "
                    f"- RMSE: {rmse:.4f}"
                )

        return selfnalg.solve(A,b)



    def _update_users(self):
        ...

    def _update_items(self):
        ...

    def predict(self, user_id, item_id):
        ...

    def recommend(self, user_id, n=10):
        ...

    def score(self, X_test):
        ...
