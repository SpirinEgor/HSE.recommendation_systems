from numpy import log
from scipy.sparse import csr_matrix

from .bpr import BPR


class WARP(BPR):
    def __init__(self, *args, n_negatives: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.__n_negatives = n_negatives

    def _step(self, user_id: int, pos_sample: int, user_item_csr: csr_matrix) -> float:
        pos_score = self._U[user_id] @ self._V[pos_sample]

        for i in range(1, self.__n_negatives + 1):
            neg_sample = self._sample_negative(user_id, user_item_csr)
            neg_score = self._U[user_id] @ self._V[neg_sample]

            violation = 1.0 + neg_score - pos_score
            if violation <= 0:
                continue

            rank_approx = self.__n_negatives // i
            loss = log(rank_approx) * violation

            self._gradient_step(-loss, user_id, pos_sample, neg_sample)

            return loss

        return 0
