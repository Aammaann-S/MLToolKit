"""A small Keras feed-forward classifier. TensorFlow is optional."""

from __future__ import annotations

import numpy as np


def train_ann(X, y, epochs: int = 20, hidden: int = 16, dropout: float = 0.2, random_state: int = 42):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Install TensorFlow with: pip install tensorflow") from exc

    tf.keras.utils.set_random_seed(random_state)
    features = np.asarray(X, dtype="float32")
    target = np.asarray(y)
    classes = int(target.max()) + 1
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(features.shape[1],)),
            tf.keras.layers.Dense(hidden, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(features, target, epochs=epochs, verbose=0)
    return {"model": model, "history": history.history}
