from dataclasses import dataclass
from typing import List

from pandas import DataFrame, Series
from scipy.sparse import csr_matrix


@dataclass
class Movie:
    id: int
    name: str
    category: List[str]

    @classmethod
    def from_pandas_series(cls, movie: Series) -> "Movie":
        _id = int(movie.name)
        name = movie["name"]
        categories = movie["category"].split("|")
        return cls(_id, name, categories)

    def __repr__(self) -> str:
        return f"{self.id}: {self.name} ({', '.join(self.category)})"


def get_similar_items(item_id: int, movie_info: DataFrame, model) -> List[Movie]:
    return [Movie.from_pandas_series(movie_info.loc[x[0]]) for x in model.similar_items(item_id)]


def get_user_history(user_id: int, movie_info: DataFrame, ratings: DataFrame) -> List[Movie]:
    user_films = ratings[ratings["user_id"] == user_id]["movie_id"]
    return [Movie.from_pandas_series(movie_info.loc[x]) for x in user_films]


def get_recommendations(user_id: int, movie_info: DataFrame, user_item_csr: csr_matrix, model) -> List[Movie]:
    recommendations = model.recommend(user_id, user_item_csr)
    return [Movie.from_pandas_series(movie_info.loc[x[0]]) for x in recommendations]
