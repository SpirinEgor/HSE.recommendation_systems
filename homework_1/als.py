from typing import Optional

from numpy import ndarray, eye
from numpy.linalg import solve
from scipy.sparse import csr_matrix
from tqdm.auto import trange

from homework_1.matrix_factorization_base import MatrixFactorizationBase


class ALS(MatrixFactorizationBase):
    def __init__(
        self,
        factors: int,
        steps: int,
        reg_lambda: float = 0.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(factors, random_state)
        self.__steps = steps
        self.__lambda = reg_lambda

    def _als_step(self, fixed_emb: ndarray, target_emb: ndarray, ratings: csr_matrix):
        yty = fixed_emb.T.dot(fixed_emb)
        reg_term = eye(yty.shape[0]) * self.__lambda
        left_side = yty + reg_term
        for i in range(target_emb.shape[0]):
            right_side = ratings[i].dot(fixed_emb).T
            target_emb[i] = solve(left_side, right_side).reshape(1, -1)
        return target_emb

    def fit(self, user_item_csr: csr_matrix, item_user_csr: csr_matrix):
        n_users, n_items = user_item_csr.shape
        self._init_matrices(n_users, n_items)
        epoch_bar = trange(self.__steps, desc="Epoch")
        for _ in epoch_bar:
            # fix users
            self._V = self._als_step(self._U, self._V, item_user_csr)
            # fix items
            self._U = self._als_step(self._V, self._U, user_item_csr)

            epoch_bar.set_postfix({"mse": round(self.calculate_mse_loss(user_item_csr), 3)})
        epoch_bar.close()
