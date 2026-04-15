"""Transfer learning models used for bird-vs-drone classification."""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.config import (
    INPUT_SHAPE, LEARNING_RATE, FINE_TUNE_LR, DROPOUT_RATE, L2_REG,
    EPOCHS, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR, REDUCE_LR_MIN, MONITOR_METRIC,
    RESNET50_PATH, MOBILENET_PATH, EFFICIENTNET_PATH,
    PLOTS_DIR, CHECKPOINTS_DIR
)
from src.utils import plot_training_history, ensure_dirs


# ─────────────────────────────────────────────
# Common classifier head builder
# ─────────────────────────────────────────────

def _build_head(base_output, dropout_rate: float, l2_reg: float,
                units: int = 256, name_prefix: str = "tl"):
    """Add a compact dense head on top of a pretrained backbone."""
    reg = regularizers.l2(l2_reg)
    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(base_output)
    x = layers.Dense(units, kernel_regularizer=reg, name=f"{name_prefix}_fc1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop1")(x)
    x = layers.Dense(128, kernel_regularizer=reg, name=f"{name_prefix}_fc2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
    x = layers.Dropout(dropout_rate * 0.5, name=f"{name_prefix}_drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name=f"{name_prefix}_output")(x)
    return outputs


def _compile(model: tf.keras.Model, lr: float) -> tf.keras.Model:
    """Compile with Adam and the metrics tracked in this project."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _get_callbacks(save_path: str, phase: str = "phase1") -> list:
    base = os.path.splitext(save_path)[0]
    return [
        EarlyStopping(monitor=MONITOR_METRIC, patience=EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=f"{base}_{phase}.h5",
                        monitor=MONITOR_METRIC, save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR,
                          patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN, verbose=1),
    ]


# ─────────────────────────────────────────────
# ResNet50
# ─────────────────────────────────────────────

def build_resnet50(input_shape: tuple = INPUT_SHAPE,
                   dropout_rate: float = DROPOUT_RATE,
                   l2_reg: float = L2_REG) -> tf.keras.Model:
    """Build a ResNet50 classifier with the backbone frozen at the start."""
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    outputs = _build_head(base.output, dropout_rate, l2_reg, units=512, name_prefix="resnet")
    return models.Model(inputs=base.input, outputs=outputs, name="ResNet50_BirdDrone")


def build_mobilenet(input_shape: tuple = INPUT_SHAPE,
                    dropout_rate: float = DROPOUT_RATE,
                    l2_reg: float = L2_REG) -> tf.keras.Model:
    """Build a lighter MobileNetV2 classifier for faster inference."""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    outputs = _build_head(base.output, dropout_rate, l2_reg, units=256, name_prefix="mobilenet")
    return models.Model(inputs=base.input, outputs=outputs, name="MobileNetV2_BirdDrone")


def build_efficientnet(input_shape: tuple = INPUT_SHAPE,
                       dropout_rate: float = DROPOUT_RATE,
                       l2_reg: float = L2_REG) -> tf.keras.Model:
    """Build an EfficientNetB0 classifier with the same project head."""
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    outputs = _build_head(base.output, dropout_rate, l2_reg, units=256, name_prefix="effnet")
    return models.Model(inputs=base.input, outputs=outputs, name="EfficientNetB0_BirdDrone")


# ─────────────────────────────────────────────
# Fine-Tuning Helper
# ─────────────────────────────────────────────

def unfreeze_top_layers(model: tf.keras.Model, n_unfreeze: int = 30):
    """Open up the top part of the backbone while keeping BN layers frozen."""
    model.trainable = True
    
    # Find where the custom head starts.
    head_start_idx = len(model.layers)
    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.GlobalAveragePooling2D):
            head_start_idx = i
            break
            
    # Keep the lower backbone frozen and fine-tune only the upper part plus the head.
    n_total = len(model.layers)
    freeze_until = max(0, head_start_idx - n_unfreeze)
    
    for i, layer in enumerate(model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            # BatchNorm layers are left frozen to keep training more stable.
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"[fine-tune] Unfroze {trainable}/{n_total} total layers (including {n_total - head_start_idx} head layers)")
    return model


# ─────────────────────────────────────────────
# Training Function (2-phase)
# ─────────────────────────────────────────────

def train_transfer_model(model_name: str,
                         train_gen, valid_gen,
                         epochs_phase1: int = 20,
                         epochs_phase2: int = 30,
                         save_path: str = None,
                         plot_save_dir: str = PLOTS_DIR):
    """Train one transfer-learning model in a frozen phase and a fine-tuning phase."""
    ensure_dirs(PLOTS_DIR, CHECKPOINTS_DIR)

    model_builders = {
        "resnet50":     (build_resnet50,     RESNET50_PATH,     40),
        "mobilenet":    (build_mobilenet,    MOBILENET_PATH,    30),
        "efficientnet": (build_efficientnet, EFFICIENTNET_PATH, 25),
    }
    if model_name not in model_builders:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_builders)}")

    builder_fn, default_path, n_unfreeze = model_builders[model_name]
    if save_path is None:
        save_path = default_path
    ensure_dirs(os.path.dirname(save_path))

    print("\n" + "="*60)
    print(f"  PHASE 1: Training {model_name.upper()} head (backbone frozen)")
    print("="*60)

    model = builder_fn()
    model = _compile(model, lr=LEARNING_RATE)
    model.summary()

    h1 = model.fit(
        train_gen, validation_data=valid_gen,
        epochs=epochs_phase1, callbacks=_get_callbacks(save_path, "phase1"),
        verbose=1,
    )

    # ── Phase 2: Fine-tune ──────────────────
    print("\n" + "="*60)
    print(f"  PHASE 2: Fine-tuning {model_name.upper()} (top {n_unfreeze} layers unfrozen)")
    print("="*60)

    model = unfreeze_top_layers(model, n_unfreeze=n_unfreeze)
    model = _compile(model, lr=FINE_TUNE_LR)

    h2 = model.fit(
        train_gen, validation_data=valid_gen,
        epochs=epochs_phase2, callbacks=_get_callbacks(save_path, "phase2"),
        verbose=1,
    )

    # Save the final tuned model.
    model.save(save_path)
    print(f"\n[transfer] Final model saved: {save_path}")

    # Merge both histories so the training curve reads like one run.
    class CombinedHistory:
        """Small wrapper that combines two Keras history objects."""
        def __init__(self, h1, h2):
            self.history = {
                k: list(h1.history.get(k, [])) + list(h2.history.get(k, []))
                for k in set(list(h1.history) + list(h2.history))
            }

    combined = CombinedHistory(h1, h2)
    plot_path = os.path.join(plot_save_dir, f"{model_name}_history.png")
    plot_training_history(combined, model_name=model_name.upper(), save_path=plot_path)

    return model, h1, h2


# ─────────────────────────────────────────────
# Train all transfer models at once
# ─────────────────────────────────────────────

def train_all_transfer_models(train_gen, valid_gen,
                              epochs_p1: int = 15, epochs_p2: int = 20):
    """Train all three transfer-learning backbones one after another."""
    results = {}
    for name in ["resnet50", "mobilenet", "efficientnet"]:
        model, h1, h2 = train_transfer_model(
            name, train_gen, valid_gen,
            epochs_phase1=epochs_p1,
            epochs_phase2=epochs_p2,
        )
        results[name] = (model, h1, h2)
    return results


if __name__ == "__main__":
    from src.data_preprocessing import get_all_generators
    from src.utils import set_seed

    set_seed()
    train_gen, valid_gen, _ = get_all_generators()
    train_transfer_model("mobilenet", train_gen, valid_gen,
                         epochs_phase1=15, epochs_phase2=20)
