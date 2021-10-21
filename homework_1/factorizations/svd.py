from typing import Optional

import numpy
from numpy import ndarray
from tqdm.auto import tqdm, trange

from .matrix_factorization_base import MatrixFactorizationBase


class SVD(MatrixFactorizationBase):
    def __init__(
        self,
        factors: int,
        lr: float,
        steps: int,
        gamma: float = 0.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(factors, random_state)
        self.__lr = lr
        self.__steps = steps
        self.__gamma = gamma

        # [n users]
        self.__user_biases = None
        # [n items]
        self.__item_biases = None

        self.__avg_bias = None

    def _init_matrices(self, n_users: int, n_items: int):
        super()._init_matrices(n_users, n_items)
        self.__user_biases = numpy.zeros(n_users)
        self.__item_biases = numpy.zeros(n_items)

    def fit(self, user_ids: ndarray, item_ids: ndarray, ratings: ndarray):
        if not (len(user_ids) == len(item_ids) == len(ratings)):
            raise ValueError("Unequal sizes of passed data")

        n_users, n_items = numpy.max(user_ids) + 1, numpy.max(item_ids) + 1
        self._init_matrices(n_users, n_items)

        self.__avg_bias = numpy.mean(ratings)

        epoch_range = trange(self.__steps, desc="Epoch")
        for _ in epoch_range:
            mse = 0.0
            order = numpy.random.permutation(len(ratings))
            for idx in tqdm(order):
                user_id, item_id = user_ids[idx], item_ids[idx]

                user_emb = self._U[user_id]
                user_bias = self.__user_biases[user_id]
                item_emb = self._V[item_id]
                item_bias = self.__item_biases[item_id]

                error = (user_emb.dot(item_emb) + user_bias + item_bias + self.__avg_bias) - ratings[idx]

                du = error * item_emb + self.__gamma * user_emb
                dbu = error + self.__gamma * user_bias
                dv = error * user_emb + self.__gamma * item_emb
                dvu = error + self.__gamma * item_bias

                self._U[user_id] -= self.__lr * du
                self.__user_biases[user_id] -= self.__lr * dbu
                self._V[item_id] -= self.__lr * dv
                self.__item_biases[item_id] -= self.__lr * dvu

                mse += error ** 2
            epoch_range.set_postfix({"mse": round(mse / len(ratings), 3)})
        epoch_range.close()

    def _get_user_distances(self, user_id: int) -> ndarray:
        distances = (self._U[[user_id]].dot(self._V.T))[0]
        distances += self.__item_biases
        distances += self.__user_biases[user_id] + self.__avg_bias
        return distances
