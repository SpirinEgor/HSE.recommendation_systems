from typing import Tuple, Iterator, List

from catboost import Pool
from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.model_selection import GroupKFold


def yield_folds(group_ids: Series, n_folds: int) -> Iterator[Tuple[ndarray, ndarray]]:
    n_samples = len(group_ids)
    group_k_fold = GroupKFold(n_folds)
    for train_idx, test_idx in group_k_fold.split(range(n_samples), groups=group_ids):
        yield train_idx, test_idx


def create_pool(dataset: DataFrame, cat_features: List[str]) -> Pool:
    dataset = dataset.sort_values("msno")
    data = dataset.drop(["msno", "target"], axis=1)
    group_id = dataset["msno"]
    target = dataset["target"]
    return Pool(data, label=target, group_id=group_id, cat_features=cat_features)
