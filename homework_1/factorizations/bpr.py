from typing import Optional

import numpy
from scipy.sparse import csr_matrix
from scipy.special import expit
from tqdm.auto import trange, tqdm

from matrix_factorization_base import MatrixFactorizationBase


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

    def _step(self, user_id: int, pos_sample: int, user_item_csr: csr_matrix) -> float:
        neg_sample = numpy.random.choice(user_item_csr.shape[1])
        while user_item_csr[user_id, neg_sample] != 0:
            neg_sample = numpy.random.choice(user_item_csr.shape[1])

        # [ d ]
        user_emb = self._U[user_id]
        pos_item_emb = self._V[pos_sample]
        neg_item_emb = self._V[neg_sample]

        diff = pos_item_emb - neg_item_emb

        # [ 1 ]
        r_uij = numpy.dot(user_emb, diff)
        sigmoid = expit(r_uij)

        # [ d ]
        du = sigmoid * diff + self.__lambda * user_emb
        dpi = sigmoid * user_emb + self.__lambda * pos_item_emb
        dni = sigmoid * -user_emb + self.__lambda * neg_item_emb

        self._U[user_id] -= self.__lr * du
        self._V[pos_sample] -= self.__lr * dpi
        self._V[neg_sample] -= self.__lr * dni

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
