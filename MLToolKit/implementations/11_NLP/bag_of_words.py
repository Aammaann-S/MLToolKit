"""Bag-of-words counts fit on training documents only."""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(train_text, test_text=None, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 1)):
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    train_matrix = vectorizer.fit_transform(train_text)
    if test_text is None:
        return train_matrix, vectorizer
    return train_matrix, vectorizer.transform(test_text), vectorizer
