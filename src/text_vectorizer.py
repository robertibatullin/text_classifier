from sklearn.feature_extraction.text import CountVectorizer as TextVectorizer

text_vectorizer_kwargs = {"binary": True,
                         "max_features": 2000,
                         "min_df": 2,
                         "max_df": 0.9,
                         #"token_pattern": "(?u)\b\w\w\w+\b"
                         }
