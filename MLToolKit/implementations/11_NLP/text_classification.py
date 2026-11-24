"""Text classification pipeline. The vectorizer is fit inside the pipeline on training text only."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_text_classifier(texts, labels, test_size: float = 0.2, random_state: int = 42, max_features: int = 5000):
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": predictions,
    }
