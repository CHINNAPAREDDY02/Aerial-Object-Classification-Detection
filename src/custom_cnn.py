"""
custom_cnn.py — Custom CNN Architecture for Aerial Bird vs Drone Classification.

Architecture:
  4 × Conv Blocks (Conv2D → BatchNorm → MaxPool → Dropout)
  Global Average Pooling
  2 × Dense Layers with L2 regularization
  Sigmoid output (binary classification)
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

from src.config import (
    INPUT_SHAPE, LEARNING_RATE, DROPOUT_RATE, L2_REG,
    EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN,
    MONITOR_METRIC, CUSTOM_CNN_PATH, CHECKPOINTS_DIR,
    RESULTS_DIR, PLOTS_DIR
)
from src.utils import plot_training_history, ensure_dirs


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────

def build_custom_cnn(input_shape: tuple = INPUT_SHAPE,
                     dropout_rate: float = DROPOUT_RATE,
                     l2_reg: float = L2_REG) -> tf.keras.Model:
    """
    Build a custom CNN for binary image classification.

    Architecture Summary
    --------------------
    Block 1 : Conv2D(32)  → BN → ReLU → MaxPool → Dropout(0.25)
    Block 2 : Conv2D(64)  → BN → ReLU → MaxPool → Dropout(0.25)
    Block 3 : Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.35)
    Block 4 : Conv2D(256) → BN → ReLU → MaxPool → Dropout(0.4)
    GAP     : GlobalAveragePooling2D
    Dense   : 512 → BN → ReLU → Dropout(0.5)
    Dense   : 256 → BN → ReLU → Dropout(0.5)
    Output  : Dense(1) → Sigmoid

    Parameters
    ----------
    input_shape  : tuple  — (H, W, C) e.g. (224, 224, 3)
    dropout_rate : float  — Base dropout probability
    l2_reg       : float  — L2 regularization factor

    Returns
    -------
    tf.keras.Model (uncompiled)
    """
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape, name="input")

    # ── Block 1 ──────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop1")(x)

    # ── Block 2 ──────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop2")(x)

    # ── Block 3 ──────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(dropout_rate * 0.7, name="drop3")(x)

    # ── Block 4 ──────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg, name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu", name="relu4_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Dropout(dropout_rate * 0.8, name="drop4")(x)

    # ── Top ──────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(512, kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Activation("relu", name="relu_fc1")(x)
    x = layers.Dropout(dropout_rate, name="drop_fc1")(x)

    x = layers.Dense(256, kernel_regularizer=reg, name="fc2")(x)
    x = layers.BatchNormalization(name="bn_fc2")(x)
    x = layers.Activation("relu", name="relu_fc2")(x)
    x = layers.Dropout(dropout_rate, name="drop_fc2")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CustomCNN_BirdDrone")
    return model


# ─────────────────────────────────────────────
# Compile Helper
# ─────────────────────────────────────────────

def compile_model(model: tf.keras.Model,
                  learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Compile the model with Adam optimizer and binary crossentropy loss.
    Metrics: accuracy, precision, recall, AUC.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

def get_callbacks(model_save_path: str = CUSTOM_CNN_PATH,
                  log_dir: str = None) -> list:
    """
    Standard callbacks for training:
    - EarlyStopping
    - ModelCheckpoint (save best)
    - ReduceLROnPlateau
    - TensorBoard (optional)
    """
    ensure_dirs(CHECKPOINTS_DIR, PLOTS_DIR)
    callbacks = [
        EarlyStopping(
            monitor=MONITOR_METRIC,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor=MONITOR_METRIC,
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1,
        ),
    ]
    if log_dir:
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
    return callbacks


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

def train_custom_cnn(train_gen, valid_gen,
                     epochs: int = EPOCHS,
                     save_path: str = CUSTOM_CNN_PATH,
                     plot_save_dir: str = PLOTS_DIR):
    """
    Build, compile, and train the Custom CNN.

    Parameters
    ----------
    train_gen    : Keras generator for training data
    valid_gen    : Keras generator for validation data
    epochs       : int — max training epochs
    save_path    : str — path to save best model .h5
    plot_save_dir: str — directory for training curve plots

    Returns
    -------
    model   : trained tf.keras.Model
    history : Keras History object
    """
    ensure_dirs(os.path.dirname(save_path), plot_save_dir)

    print("\n" + "="*60)
    print("  TRAINING: Custom CNN")
    print("="*60)

    model = build_custom_cnn()
    model = compile_model(model)
    model.summary()

    callbacks = get_callbacks(model_save_path=save_path)

    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    plot_path = os.path.join(plot_save_dir, "custom_cnn_history.png")
    plot_training_history(history, model_name="Custom CNN", save_path=plot_path)

    print(f"\n[custom_cnn] Training complete. Best model saved: {save_path}")
    return model, history


if __name__ == "__main__":
    from src.data_preprocessing import get_all_generators
    from src.utils import set_seed

    set_seed()
    train_gen, valid_gen, test_gen = get_all_generators()
    model, history = train_custom_cnn(train_gen, valid_gen)
    print(model.summary())
