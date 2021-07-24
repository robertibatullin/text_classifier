import os
import pickle

import pandas as pd
from scipy.sparse.csr import csr_matrix

from .proper_names_finder import ProperNamesFinder
from .text_vectorizer import TextVectorizer


class TextPreprocessor:
    
    PROPER_NAME_TOKEN = '[NAME]'

    def __init__(self,
                 proper_names_finder: ProperNamesFinder,
                 text_vectorizer: TextVectorizer):
        self.__proper_names_finder = proper_names_finder
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
        
    def __proper_names_to_token(self, text: str) -> str:
        for proper_name in self.__proper_names_finder.proper_names:
            text = text.replace(proper_name, self.PROPER_NAME_TOKEN)
        return text
        
    def fit(self, data: pd.DataFrame, root: str):
        paths = [os.path.join(root, 'texts', filename)
                 for filename in data['filename']]
        texts = [open(path, 'r').read()
                 for path in paths]
        for text in texts:
            self.__proper_names_finder.fit(text)
        texts = list(map(self.__proper_names_to_token, texts))
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
