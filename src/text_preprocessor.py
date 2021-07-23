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
    def load(cls):
        vectorizer_path = os.path.join('model', 'vectorizer.pkl')
        vectorizer = pickle.load(open(vectorizer_path,'rb'))
        return cls(vectorizer)

    def save(self):
        vectorizer_path = os.path.join('model', 'vectorizer.pkl')
        pickle.dump(self.__text_vectorizer, open(vectorizer_path, 'wb'))

    def preprocess(self, data: pd.DataFrame) -> csr_matrix:
        """
        Transforms text data to vectorized
        :param data: DataFrame with column "filename"
        where each line contains a filename of a text file in "texts" directory
        :return: sparse matrix of vectorized texts
        """
        paths = [os.path.join('texts', filename)
                 for filename in data['filename']]
        texts = [open(path, 'r').read()
                 for path in paths]
        return self.__text_vectorizer.fit_transform(texts)
