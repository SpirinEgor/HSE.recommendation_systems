from typing import Optional

import numpy
from scipy.sparse import csr_matrix
from scipy.special import expit
from tqdm.auto import trange, tqdm

from .matrix_factorization_base import MatrixFactorizationBase


class BPR(MatrixFactorizationBase):
    def __init__(
        self,
        factors: int,
        lr: float,
        steps: int,
        reg_lambda: float = 0.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(factors, random_state)
        self.__lr = lr
        self.__steps = steps
        self.__lambda = reg_lambda

    @staticmethod
    def _sample_negative(user_id: int, user_item_csr: csr_matrix) -> int:
        neg_sample = numpy.random.choice(user_item_csr.shape[1])
        while user_item_csr[user_id, neg_sample] != 0:
            neg_sample = numpy.random.choice(user_item_csr.shape[1])
        return neg_sample

    def _gradient_step(self, loss: float, user_id: int, pos_sample: int, neg_sample: int):
        # [ d ]
        du = loss * (self._V[pos_sample] - self._V[neg_sample]) + self.__lambda * self._U[user_id]
        dpi = loss * self._U[user_id] + self.__lambda * self._V[pos_sample]
        dni = loss * -self._U[user_id] + self.__lambda * self._V[neg_sample]

        self._U[user_id] -= self.__lr * du
        self._V[pos_sample] -= self.__lr * dpi
        self._V[neg_sample] -= self.__lr * dni

    def _step(self, user_id: int, pos_sample: int, user_item_csr: csr_matrix) -> float:
        neg_sample = self._sample_negative(user_id, user_item_csr)

        # [ 1 ]
        r_uij = numpy.dot(self._U[user_id], self._V[pos_sample] - self._V[neg_sample])
        sigmoid = expit(r_uij)

        self._gradient_step(sigmoid, user_id, pos_sample, neg_sample)

        return numpy.log(sigmoid)

    def fit(self, user_item_csr: csr_matrix):
        n_users, n_items = user_item_csr.shape
        self._init_matrices(n_users, n_items)

        user_item_coo = user_item_csr.tocoo()
        n_samples = user_item_csr.count_nonzero()
        assert len(user_item_coo.row) == len(user_item_coo.col) == n_samples

        epoch_range = trange(self.__steps, desc="Epoch")
        for _ in epoch_range:
            order = numpy.random.permutation(n_samples)
            log_loss = 0
            for user_id, pos_sample in tqdm(zip(user_item_coo.row[order], user_item_coo.col[order]), total=n_samples):
                log_loss += self._step(user_id, pos_sample, user_item_csr)

            epoch_range.set_postfix({"log loss": round(log_loss / n_samples, 3)})
        epoch_range.close()
