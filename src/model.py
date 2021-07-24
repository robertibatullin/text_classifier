from typing import List
import pickle
import os
import glob

import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix

from .classifier import Classifier, classifier_fit_kwargs, classifier_init_kwargs

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

    def fit(self, x: csr_matrix, y: pd.DataFrame) -> None:
        fitted = []
        for tag_id in y.columns:
            classifier = Classifier(**classifier_init_kwargs)
            try:
                classifier.fit(x, y[tag_id], **classifier_fit_kwargs)
            except TypeError:
                x = x.toarray()
                classifier.fit(x, y[tag_id], **classifier_fit_kwargs)
            fitted.append(classifier)
            print(f'Accuracy with tag #{tag_id}: {classifier.score(x, y[tag_id])}')
        self.__classifiers = fitted

    def predict_proba(self, x: csr_matrix) -> np.array:
        """
        :param x: vectorized array of texts
        :return: list of tag probabilities
        """
        probabilities = []
        for classifier in self.__classifiers:
            try:
                predict_proba = classifier.predict_proba(x)
            except TypeError:
                x = x.toarray()
                predict_proba = classifier.predict_proba(x)
            probabilities.append(predict_proba[:, 1])
        probabilities = np.transpose(np.array(probabilities))
        return probabilities

    def predict(self, x: csr_matrix,
                threshold: float = 0.5) -> List[str]:
        probabilities = self.predict_proba(x)
        tag_ids = [np.argwhere(probabilities[idx] >= threshold).reshape(-1)
                   for idx in range(len(probabilities))]
        return tag_ids

    def get_top_k_features(self, tag_id: int, k: int) -> List[int]:
        coefs = self.__classifiers[tag_id].coef_
        coefs = pd.Series(coefs[0])
        coefs.sort_values(ascending=False, inplace=True)
        return coefs.head(k).index.tolist()
