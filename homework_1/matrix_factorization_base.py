from abc import abstractmethod
from typing import List, Tuple, Optional

import numpy
from implicit.evaluation import csr_matrix
from numpy import ndarray, argsort
from sklearn.metrics.pairwise import cosine_similarity


class MatrixFactorizationBase:
    def __init__(self, factors: int, random_state: Optional[int] = None):
        self._hidden_dim = factors
        numpy.random.seed(random_state)

        # [n users; d]
        self._U = None
        # [n items; d]
        self._V = None

    def _init_matrices(self, n_users: int, n_items: int):
        self._U = numpy.random.uniform(0, 1.0 / self._hidden_dim, (n_users, self._hidden_dim))
        self._V = numpy.random.uniform(0, 1.0 / self._hidden_dim, (n_items, self._hidden_dim))

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _get_user_distances(self, user_id: int) -> ndarray:
        raise NotImplementedError()

    def similar_items(self, item_id: int, n_samples: int = 10) -> List[Tuple[int, float]]:
        if self._V is None:
            raise RuntimeError("Fit model before requesting data")
        if item_id < 0 or item_id >= self._V.shape[0]:
            raise ValueError(f"item_id is out of range (0 ≤ item id < {self._V.shape[0]})")

        distances = 1 - cosine_similarity(self._V[[item_id]], self._V)[0]
        closest = argsort(distances)
        return [(i, d) for i, d in zip(closest[:n_samples], distances[:n_samples])]

    def recommend(self, user_id: int, user_item_csr: csr_matrix, n_samples: int = 10) -> List[Tuple[int, float]]:
        if self._U is None:
            raise RuntimeError("Fit model before requesting data")
        if user_id < 0 or user_id >= self._U.shape[0]:
            raise ValueError(f"user_id is out of range (0 ≤ user id < {self._U.shape[0]}")

        # [n items]
        distances = self._get_user_distances(user_id)
        closest = argsort(distances)[::-1]

        user_ratings = user_item_csr[user_id].todense()
        result = [(i, d) for i, d in zip(closest, distances) if user_ratings[0, i] == 0]
        return result[:n_samples]
