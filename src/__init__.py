import pandas as pd

from .tag_mapper import TagMapper
from .tag_vectorizer import TagVectorizer
from .text_vectorizer import TextVectorizer, text_vectorizer_kwargs
from .tag_preprocessor import TagPreprocessor
from .text_preprocessor import TextPreprocessor
from .data_manager import DataManager
from .model import Model

tag_mapper = TagMapper([])
tag_vectorizer = TagVectorizer()
text_vectorizer = TextVectorizer(**text_vectorizer_kwargs)
tag_preprocessor = TagPreprocessor(tag_mapper, tag_vectorizer)
text_preprocessor = TextPreprocessor(text_vectorizer)
data_manager = DataManager(tag_preprocessor, text_preprocessor)
model = Model([])
