"""A tiny Keras CNN. TensorFlow is optional."""

from __future__ import annotations

import numpy as np


def train_cnn(images, labels, epochs: int = 3, random_state: int = 42):
    """Train on images shaped (n, height, width, channels)."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Install TensorFlow with: pip install tensorflow") from exc

    tf.keras.utils.set_random_seed(random_state)
    features = np.asarray(images, dtype="float32")
    target = np.asarray(labels)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=features.shape[1:]),
            tf.keras.layers.Conv2D(4, 3, activation="relu", padding="same"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(int(target.max()) + 1, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(features, target, epochs=epochs, verbose=0)
    return {"model": model, "history": history.history}
