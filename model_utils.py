"""
model_utils.py
----------------
마스크에서 뽑은 15개 특징(feature)을 입력으로 받아, 3가지 결함(pinhole/peeling/crack)의
존재 여부를 각각 0~1 확률로 예측하는 멀티라벨(multi-label) TensorFlow 모델을 정의합니다.

- 출력 3개 모두 sigmoid : 한 이미지에 여러 결함이 동시에 존재할 수 있으므로 softmax가 아닌
  독립적인 sigmoid + binary_crossentropy를 사용합니다.
- "정상(normal)"은 별도 클래스가 아니라, 3개 출력이 모두 임계값(기본 0.5) 미만인 경우로 정의합니다.
  (dataset_utils.multihot_to_label_string 참고)
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim: int = 15, num_classes: int = 3, dropout: float = 0.2) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="mask_features")
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="defect_probs")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="defect_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc", multi_label=True)],
    )
    return model


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    verbose: int = 1
) -> keras.callbacks.History:
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )
    return history
