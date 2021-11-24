import re
from dataclasses import dataclass
from os.path import join
from typing import Optional, Dict, List, Iterator, Tuple, Any

import numpy as np
from catboost import Pool
from pandas import read_csv, DataFrame
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


@dataclass
class Dataset:
    features: DataFrame
    user_id: np.ndarray
    labels: np.ndarray
    category_features: List[str]

    def to_pool(self) -> Pool:
        return Pool(data=self.features, label=self.labels, group_id=self.user_id, cat_features=self.category_features)

    def fold_split(self, n_folds: int) -> Iterator[Tuple["Dataset", "Dataset"]]:
        group_k_fold = GroupKFold(n_folds)
        for train_idx, test_idx in group_k_fold.split(self.features, self.labels, self.user_id):
            train_split = Dataset(
                self.features.iloc[train_idx], self.user_id[train_idx], self.labels[train_idx], self.category_features
            )
            test_split = Dataset(
                self.features.iloc[test_idx], self.user_id[test_idx], self.labels[test_idx], self.category_features
            )
            yield train_split, test_split


class DatasetBuilder:
    TRAIN_FILE = "train.csv"
    MEMBERS_FILE = "members.csv"
    SONG_FILE = "songs.csv"
    SONG_EXTRA_INFO_FILE = "song_extra_info.csv"

    def __init__(self, data_dir: str, svd_components: int):
        self.__data_dir = data_dir
        self.__svd_components = svd_components

    @staticmethod
    def create_age_converter(bottom_age: Optional[int] = None, upper_age: Optional[int] = None):
        def convert_age(value: str) -> Optional[np.int32]:
            """Replace outliers in age with None."""
            value = np.int32(value)
            if bottom_age is not None and value < bottom_age:
                return None
            if upper_age is not None and value > upper_age:
                return None
            return value

        return convert_age

    @staticmethod
    def create_categorical_converter():
        static_dict_values: Dict[str, np.int32] = {}

        def convert_categorical(value: str) -> np.int32:
            if value not in static_dict_values:
                static_dict_values[value] = np.int32(len(static_dict_values))
            return static_dict_values[value]

        return convert_categorical

    def _tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        print("\tBuilding count vectors...")
        vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer="char")
        tfidf = vectorizer.fit_transform(texts)

        print("\tDecomposing them with SVD...")
        svd = TruncatedSVD(self.__svd_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        embeddings = lsa.fit_transform(tfidf)
        explained_variance = svd.explained_variance_ratio_.sum()
        print(f"\tExplained variance of the SVD: {int(explained_variance * 100)}%")

        return embeddings

    _split_regexp = re.compile(r"[|/]")
    _sub_regexp = re.compile(r"[()]")

    def parse_str_value(self, value: Any) -> List[str]:
        if not isinstance(value, str):
            return []
        values = [it.strip().lower() for it in re.split(self._split_regexp, value)]
        return [re.sub(self._sub_regexp, "", it) for it in values]

    def read_train_data(self) -> DataFrame:
        return read_csv(
            join(self.__data_dir, self.TRAIN_FILE),
            dtype={"target": np.byte},
            converters={
                "source_system_tab": self.create_categorical_converter(),
                "source_screen_name": self.create_categorical_converter(),
                "source_type": self.create_categorical_converter(),
            },
        )

    def read_members_data(self, bottom_age: Optional[int] = None, upper_age: Optional[int] = None) -> DataFrame:
        return read_csv(
            join(self.__data_dir, self.MEMBERS_FILE),
            index_col="msno",
            dtype={"registration_init_time": np.int32, "expiration_date": np.int32},
            converters={
                "gender": self.create_categorical_converter(),
                "bd": self.create_age_converter(bottom_age, upper_age),
                "city": self.create_categorical_converter(),
                "registered_via": self.create_categorical_converter(),
            },
        )

    def read_song_data(self, use_extra_info: bool = False) -> DataFrame:
        song = read_csv(
            join(self.__data_dir, self.SONG_FILE),
            index_col="song_id",
            dtype={"song_length": np.int32},
            converters={
                "language": self.create_categorical_converter(),
                "genre_ids": self.create_categorical_converter(),
            },
        )
        if use_extra_info:
            song_extra_info = read_csv(join(self.__data_dir, self.SONG_EXTRA_INFO_FILE), index_col="song_id")
            song = song.merge(song_extra_info, how="left", left_index=True, right_index=True)
        return song

    def build_dataset(self) -> Tuple[DataFrame, List[str]]:
        print("Reading data...")
        train = self.read_train_data()
        members = self.read_members_data(10, 70)
        members["bd"] = members["bd"].fillna(members["bd"].mean()).astype(np.int32)
        song = self.read_song_data(use_extra_info=True)

        # Building embeddings for artist, composer, and lyricist
        print("Building embeddings for song...")
        combination = (
            song[["artist_name", "composer", "lyricist", "name"]]
            .apply(lambda x: " ".join(sum([self.parse_str_value(it) for it in x], [])), axis=1)
            .to_list()
        )
        embeddings = self._tfidf_embeddings(combination)
        emb_col_names = [f"song_emb_{i + 1}" for i in range(self.__svd_components)]
        song[emb_col_names] = embeddings
        song.drop(["artist_name", "composer", "lyricist"], axis=1, inplace=True)

        print("Joining all data into one single table...")
        full_data = train.merge(members, on="msno", how="left")
        full_data = full_data.merge(song, on="song_id", how="left")

        na_mask = full_data.isna().any(axis=1)
        full_data = full_data[~na_mask]

        # Order data by user id using stable sort (CatBoost requirements)
        full_data.sort_values("msno", inplace=True)

        all_columns = full_data.columns.tolist()
        not_cat_columns = ["msno", "target", "song_length", "registration_init_time", "expiration_date", "bd"]
        cat_features = [it for it in all_columns if it not in not_cat_columns and it not in emb_col_names]

        # Optimize categorical variables, convert into range 0..n
        print("Processing categorical features...")
        for cat_feature in cat_features:
            if cat_feature in ["msno", "song_id"]:
                continue
            full_data[cat_feature] = full_data[cat_feature].apply(self.create_categorical_converter())

        return full_data, cat_features
