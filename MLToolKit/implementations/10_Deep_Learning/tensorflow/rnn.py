"""A tiny Keras recurrent classifier. TensorFlow is optional."""

from __future__ import annotations

import numpy as np


def train_rnn(sequences, labels, epochs: int = 10, hidden: int = 8, random_state: int = 42):
    """Train on sequences shaped (n, time, features)."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Install TensorFlow with: pip install tensorflow") from exc

    tf.keras.utils.set_random_seed(random_state)
    features = np.asarray(sequences, dtype="float32")
    target = np.asarray(labels)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=features.shape[1:]),
            tf.keras.layers.SimpleRNN(hidden),
            tf.keras.layers.Dense(int(target.max()) + 1, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(features, target, epochs=epochs, verbose=0)
    return {"model": model, "history": history.history}
