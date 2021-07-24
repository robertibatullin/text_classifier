import pandas as pd

from .tag_mapper import TagMapper
from .tag_vectorizer import TagVectorizer
from .text_vectorizer import TextVectorizer, text_vectorizer_kwargs
from .tag_preprocessor import TagPreprocessor
from .text_preprocessor import TextPreprocessor
from .data_manager import DataManager
from .classifier import Classifier, classifier_init_kwargs
from .model import Model

tags_df = pd.read_csv('tags/tags.csv', sep=';')
taglines = tags_df['tags'].tolist()
tag_mapper = TagMapper(taglines)
tag_vectorizer = TagVectorizer()
text_vectorizer = TextVectorizer(**text_vectorizer_kwargs)
tag_preprocessor = TagPreprocessor(tag_mapper, tag_vectorizer)
text_preprocessor = TextPreprocessor(text_vectorizer)
data_manager = DataManager(tag_preprocessor, text_preprocessor)
classifiers = [Classifier(**classifier_init_kwargs) for tag in tag_mapper.tags]
model = Model(classifiers)
