#from sklearn.feature_extraction.text import CountVectorizer as TextVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as TextVectorizer

text_vectorizer_kwargs = {#"binary": True,
                         "max_features": 5000,
                         "min_df": 2,
                         "max_df": 0.75,
                         #"token_pattern": "(?u)\b\w\w\w+\b"
                         }
