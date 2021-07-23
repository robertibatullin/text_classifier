from typing import List
import pickle
import os
import glob

import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix

from .classifier import Classifier

PREFIX = 'model/classifier'
SUFFIX = '.pkl'
classifier_paths = sorted(glob.glob(PREFIX + '*' + SUFFIX),
                          key=lambda s: int(s[len(PREFIX):-len(SUFFIX)]))

class Model:

    def __init__(self, classifiers: List[Classifier]):
        self.__classifiers = classifiers

    @classmethod
    def load(cls):
        classifiers = [pickle.load(open(path, 'rb'))
                       for path in classifier_paths]
        return cls(classifiers)

    def save(self):
        for path in classifier_paths:
            if os.path.exists(path):
                os.remove(path)
        for i, classifier in enumerate(self.__classifiers):
            path = os.path.join('model', f'classifier{i}.pkl')
            pickle.dump(classifier, open(path, 'wb'))

    def fit(self, X: csr_matrix, Y: pd.DataFrame) -> None:
        fitted = []
        for tag_id, classifier in zip(Y.columns, self.__classifiers):
            try:
                classifier = classifier.fit(X, Y[tag_id])
            except TypeError:
                X = X.toarray()
                classifier = classifier.fit(X, Y[tag_id])
            fitted.append(classifier)
            print(f'Accuracy with tag #{tag_id}: {classifier.score(X, Y[tag_id])}')
        self.__classifiers = fitted

    def predict_proba(self, X: csr_matrix) -> np.array:
        """
        :param X: vectorized array of texts
        :return: list of tag probabilities
        """
        probabilities = []
        for classifier in self.__classifiers:
            try:
                predict_proba = classifier.predict_proba(X)
            except TypeError:
                X = X.toarray()
                predict_proba = classifier.predict_proba(X)
            probabilities.append(predict_proba[:, 1])
        probabilities = np.transpose(np.array(probabilities))
        return probabilities

    def predict(self, X: csr_matrix,
                threshold: float = 0.5) -> List[str]:
        probabilities = self.predict_proba(X)
        tag_ids = [np.argwhere(probabilities[idx] >= threshold).reshape(-1)
                   for idx in range(len(probabilities))]
        return tag_ids
