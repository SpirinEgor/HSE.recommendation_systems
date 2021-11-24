from typing import Iterable

from gensim.models import Word2Vec
from numpy import ndarray
from pandas import DataFrame


class Embedding:
    def __init__(self, embeddings_dim: int, random_seed: int):
        self.__embedding_dim = embeddings_dim
        self.__random_seed = random_seed
        self.__word_vectors = None

    def fit(self, dataset: DataFrame, n_epochs: int):
        sessions = [it for it in dataset.groupby("msno")["song_id"].apply(list) if len(it) > 0]
        word2vec = Word2Vec(vector_size=self.__embedding_dim, window=5, min_count=5, seed=self.__random_seed)
        word2vec.build_vocab(sessions)
        word2vec.train(sessions, total_examples=len(sessions), epochs=n_epochs)
        self.__word_vectors = word2vec.wv

    def has_embedding(self, word: str) -> bool:
        if self.__word_vectors is None:
            raise RuntimeError("Fit embeddings before using them")
        return word in self.__word_vectors

    def item_embedding(self, items: Iterable[str]) -> ndarray:
        if self.__word_vectors is None:
            raise RuntimeError("Fit embeddings before using them")
        return self.__word_vectors[items]
