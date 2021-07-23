import os.path
from typing import Tuple, List
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz, csr_matrix

from .tag_mapper import TagMapper
from .tag_preprocessor import TagPreprocessor
from .text_preprocessor import TextPreprocessor

class DataManager:

    def __init__(self,
                 tag_preprocessor: TagPreprocessor,
                 text_preprocessor: TextPreprocessor):
        self.__tag_preprocessor = tag_preprocessor
        self.__text_preprocessor = text_preprocessor

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
        :return: (X, Y), where X is a sparse matrix of vectorized texts and Y is dataframe
        of vectorized tags.
        """
        data = pd.read_csv(csv_path, sep=';')
        X = self.__text_preprocessor.preprocess(data)
        Y = self.__tag_preprocessor.preprocess(data)
        return X, Y

    def load_X(self, path: str) -> csr_matrix:
        return load_npz(path)

    def save_X(self, X: csr_matrix,
               path: str) -> None:
        save_npz(path, X)

    def load_Y(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, dtype=int)

    def save_Y(self, Y: pd.DataFrame,
               path: str) -> None:
        Y.to_csv(path, index=False)

    def transform(self, texts: List[str]) -> csr_matrix:
        return self.__text_preprocessor.vectorizer.transform(texts)

    def get_tags(self,
                 tag_ids: np.array) -> List[List[str]]:
        return list(map(self.__tag_preprocessor.mapper.ids_to_taglists, tag_ids))
