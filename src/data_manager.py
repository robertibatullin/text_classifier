from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz, csr_matrix

from .tag_preprocessor import TagPreprocessor
from .text_preprocessor import TextPreprocessor


class DataManager:

    def __init__(self,
                 tag_preprocessor: TagPreprocessor,
                 text_preprocessor: TextPreprocessor):
        self.__tag_preprocessor = tag_preprocessor
        self.__text_preprocessor = text_preprocessor

    @property
    def tags(self):
        return self.__tag_preprocessor.mapper.tags

    @property
    def features_to_ids(self) -> Dict[str, int]:
        return self.__text_preprocessor.vectorizer.vocabulary_

    @property
    def ids_to_features(self) -> List[str]:
        sorted_items = sorted(self.features_to_ids.items(),
                              key=lambda item: item[1])
        return [item[0] for item in sorted_items]

    @classmethod
    def load(cls):
        tag_preprocessor = TagPreprocessor.load()
        text_preprocessor = TextPreprocessor.load()
        return cls(tag_preprocessor, text_preprocessor)

    def save(self):
        self.__tag_preprocessor.save()
        self.__text_preprocessor.save()

    def get_data(self, csv_path) -> Tuple[csr_matrix, pd.DataFrame]:
        """
        Transforms text data to vectorized
        :param csv_path: path to DataFrame with ";"-separated columns "filename" and "tags",
        where each line of "filename" contains filenames of text files in "texts" directory
        and each line of "tags" contains comma-separated tags.
        :return: (x, Y), where x is a sparse matrix of vectorized texts and Y is dataframe
        of vectorized tags.
        """
        data = pd.read_csv(csv_path, sep=';')
        x = self.__text_preprocessor.preprocess(data)
        y = self.__tag_preprocessor.preprocess(data)
        return x, y

    @staticmethod
    def load_x(path: str) -> csr_matrix:
        return load_npz(path)

    @staticmethod
    def save_x(x: csr_matrix,
               path: str) -> None:
        save_npz(path, x)

    @staticmethod
    def load_y(path: str) -> pd.DataFrame:
        return pd.read_csv(path, dtype=int)

    @staticmethod
    def save_y(y: pd.DataFrame,
               path: str) -> None:
        y.to_csv(path, index=False)

    def transform(self, texts: List[str]) -> csr_matrix:
        return self.__text_preprocessor.vectorizer.transform(texts)

    def get_tags(self,
                 tag_ids: np.array) -> List[List[str]]:
        return list(map(self.__tag_preprocessor.mapper.ids_to_taglists, tag_ids))
