import os

import pandas as pd

from .tag_mapper import TagMapper
from .tag_vectorizer import TagVectorizer


class TagPreprocessor:

    def __init__(self,
                 tag_mapper: TagMapper,
                 tag_vectorizer: TagVectorizer):
        self.__tag_mapper = tag_mapper
        self.__tag_vectorizer = tag_vectorizer

    @classmethod
    def load(cls, path: str):
        tags = [tag.strip() for tag in open(path, 'r').readlines()]
        tag_mapper = TagMapper(tags)
        tag_vectorizer = TagVectorizer()
        return cls(tag_mapper, tag_vectorizer)
    
    def save(self, path: str):
        open(path, 'w').write('\n'.join(self.__tag_mapper.tags))

    @property
    def mapper(self):
        return self.__tag_mapper

    def fit(self, data: pd.DataFrame):
        self.__tag_mapper = TagMapper.from_taglines(data['tags'].tolist())

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms text data to vectorized
        :param data: DataFrame with column "tags"
        where each line contains comma-separated tags.
        :return: dataframe of vectorized tags.
        """
        taglines = data['tags'].tolist()
        tag_id_lines = list(map(self.__tag_mapper.tagline_to_ids, taglines))
        return self.__tag_vectorizer.vectorize(tag_id_lines)
