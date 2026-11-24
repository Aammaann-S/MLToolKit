"""TF-IDF vectorizer fit on training documents only."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_vectorize(train_text, test_text=None, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    train_matrix = vectorizer.fit_transform(train_text)
    if test_text is None:
        return train_matrix, vectorizer
    return train_matrix, vectorizer.transform(test_text), vectorizer


def tfidf_frame(train_text, test_text=None, max_features: int = 100) -> tuple:
    train_matrix, *rest = tfidf_vectorize(train_text, test_text, max_features=max_features)
    vectorizer = rest[-1]
    columns = vectorizer.get_feature_names_out()
    train_frame = pd.DataFrame(train_matrix.toarray(), columns=columns)
    if test_text is None:
        return train_frame, vectorizer
    test_frame = pd.DataFrame(rest[0].toarray(), columns=columns)
    return train_frame, test_frame, vectorizer
