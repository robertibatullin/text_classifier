import os
import pickle

import pandas as pd
from scipy.sparse.csr import csr_matrix

from .text_vectorizer import TextVectorizer


class TextPreprocessor:

    def __init__(self,
                 text_vectorizer: TextVectorizer):
        self.__text_vectorizer = text_vectorizer

    @property
    def vectorizer(self):
        return self.__text_vectorizer

    @classmethod
    def load(cls, path: str):
        vectorizer = pickle.load(open(path,'rb'))
        return cls(vectorizer)

    def save(self, path: str):
        pickle.dump(self.__text_vectorizer, open(path, 'wb'))
        
    def fit(self, data: pd.DataFrame, root: str):
        paths = [os.path.join(root, 'texts', filename)
                 for filename in data['filename']]
        texts = [open(path, 'r').read()
                 for path in paths]
        self.__text_vectorizer.fit(texts)

    def transform(self, data: pd.DataFrame, root: str) -> csr_matrix:
        """
        Transforms text data to vectorized
        :param data: DataFrame with column "filename"
        where each line contains a filename of a text file in "texts" directory
        :return: sparse matrix of vectorized texts
        """
        paths = [os.path.join(root, 'texts', filename)
                 for filename in data['filename']]
        texts = [open(path, 'r').read()
                 for path in paths]
        return self.__text_vectorizer.transform(texts)
