"""
gradcam.py — Grad-CAM & Guided Backprop Visualization
Explainability module for aerial classification models.

Supports:
  - Standard Grad-CAM (heatmap overlaid on original image)
  - Guided Grad-CAM
  - Multi-layer CAM comparison
  - Batch Grad-CAM for test set
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from PIL import Image
from pathlib import Path
from typing import Optional, List

from src.config import IMG_SIZE, CLASSES, PLOTS_DIR
from src.utils import load_and_preprocess_image, ensure_dirs


# ─────────────────────────────────────────────
# Core Grad-CAM
# ─────────────────────────────────────────────

def compute_gradcam(model: tf.keras.Model,
                    img_array: np.ndarray,
                    layer_name: str,
                    class_idx: Optional[int] = None) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for one image.

    Parameters
    ----------
    model      : trained Keras model
    img_array  : np.ndarray shape (1, H, W, 3), pixel values [0, 1]
    layer_name : name of last convolutional layer
    class_idx  : target class (None → predicted class)

    Returns
    -------
    np.ndarray : 2D heatmap normalized to [0, 1]
    """
    # Build a model outputting both conv features and predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        # For binary sigmoid: output is scalar → differentiate directly
        if predictions.shape[-1] == 1:
            score = predictions[:, 0]
        else:
            score = predictions[:, class_idx]

    # Gradient of class score w.r.t. last conv output
    grads = tape.gradient(score, conv_outputs)         # (1, H', W', C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_outputs = conv_outputs[0]                     # (H', W', C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H', W', 1)
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.4,
                    colormap: str = "jet") -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_img : np.ndarray (H, W, 3), uint8 [0,255] or float [0,1]
    heatmap      : np.ndarray 2D, values in [0, 1]
    alpha        : float — heatmap opacity
    colormap     : matplotlib colormap name

    Returns
    -------
    np.ndarray  : (H, W, 3) blended image, values in [0, 1]
    """
    if original_img.dtype != np.float32:
        original_img = original_img.astype(np.float32) / 255.0

    h, w = original_img.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize((w, h))
    ) / 255.0

    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]

    overlay = alpha * heatmap_colored + (1 - alpha) * original_img
    return np.clip(overlay, 0, 1)


# ─────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────

def visualize_gradcam(image_path: str,
                      model: tf.keras.Model,
                      layer_name: str,
                      alpha: float = 0.4,
                      save_path: Optional[str] = None,
                      show: bool = True) -> dict:
    """
    Full Grad-CAM visualization pipeline for a single image.

    Returns predicted label, confidence, and the heatmap array.
    """
    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    original = np.array(img_pil)
    img_array = load_and_preprocess_image(image_path, target_size=IMG_SIZE)

    # Predict
    prob = float(model.predict(img_array, verbose=0)[0][0])
    pred_idx  = int(prob >= 0.5)
    pred_label = CLASSES[pred_idx]
    confidence = prob if pred_idx == 1 else 1 - prob

    # Grad-CAM
    heatmap = compute_gradcam(model, img_array, layer_name, class_idx=pred_idx)
    blended = overlay_heatmap(original, heatmap, alpha=alpha)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Grad-CAM Visualization  |  Prediction: {pred_label.upper()}  "
        f"({confidence*100:.1f}% confidence)",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(img_pil.resize(IMG_SIZE))
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", interpolation="bilinear")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(blended)
    axes[2].set_title(f"Overlay (α={alpha})", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[gradcam] Saved: {save_path}")
    if show:
        plt.show()
    plt.close()

    return {
        "label":      pred_label,
        "confidence": round(confidence * 100, 2),
        "heatmap":    heatmap,
        "blended":    blended,
    }


def batch_gradcam(image_dir: str,
                  model: tf.keras.Model,
                  layer_name: str,
                  n_samples: int = 8,
                  alpha: float = 0.4,
                  save_path: Optional[str] = None):
    """
    Display Grad-CAM for n_samples images from a directory.
    Shows original, heatmap, and overlay in a grid.
    """
    exts = {".jpg", ".jpeg", ".png"}
    images = [f for f in Path(image_dir).iterdir() if f.suffix.lower() in exts]
    import random; random.shuffle(images)
    images = images[:n_samples]

    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n))
    if n == 1:
        axes = axes[None, :]  # ensure 2D

    fig.suptitle(f"Batch Grad-CAM — {os.path.basename(image_dir)}", fontsize=14, fontweight="bold")

    for row, img_path in enumerate(images):
        img_pil  = Image.open(img_path).convert("RGB")
        original = np.array(img_pil)
        img_array = load_and_preprocess_image(str(img_path), target_size=IMG_SIZE)

        prob = float(model.predict(img_array, verbose=0)[0][0])
        pred_idx   = int(prob >= 0.5)
        pred_label = CLASSES[pred_idx]
        conf       = prob if pred_idx == 1 else 1 - prob

        heatmap = compute_gradcam(model, img_array, layer_name, class_idx=pred_idx)
        blended = overlay_heatmap(original, heatmap, alpha=alpha)

        axes[row, 0].imshow(img_pil.resize(IMG_SIZE))
        axes[row, 0].set_ylabel(f"{img_path.name[:20]}", fontsize=8)
        axes[row, 0].set_title("Original", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap, cmap="jet")
        color = "#4CAF50" if pred_label == "bird" else "#f44336"
        axes[row, 1].set_title(
            f"Heatmap | Pred: {pred_label.upper()} ({conf*100:.0f}%)",
            fontsize=9, color=color
        )
        axes[row, 1].axis("off")

        axes[row, 2].imshow(blended)
        axes[row, 2].set_title("Overlay", fontsize=9)
        axes[row, 2].axis("off")

    plt.tight_layout()
    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[gradcam] Batch saved: {save_path}")
    plt.show()
    plt.close()


def multi_layer_comparison(image_path: str,
                            model: tf.keras.Model,
                            layer_names: List[str],
                            save_path: Optional[str] = None):
    """
    Compare Grad-CAM heatmaps across different convolutional layers
    to show how different depths of the network attend to the image.
    """
    img_pil = Image.open(image_path).convert("RGB")
    img_array = load_and_preprocess_image(image_path, target_size=IMG_SIZE)

    prob = float(model.predict(img_array, verbose=0)[0][0])
    pred_idx = int(prob >= 0.5)

    n = len(layer_names)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4.5))

    axes[0].imshow(img_pil.resize(IMG_SIZE))
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    for i, layer in enumerate(layer_names):
        try:
            heatmap = compute_gradcam(model, img_array, layer, class_idx=pred_idx)
            axes[i + 1].imshow(heatmap, cmap="jet")
            axes[i + 1].set_title(f"Layer: {layer}", fontsize=10)
        except Exception as e:
            axes[i + 1].text(0.5, 0.5, f"Error:\n{layer}\n{str(e)[:40]}",
                             ha="center", va="center", transform=axes[i+1].transAxes,
                             fontsize=9, color="red")
        axes[i + 1].axis("off")

    fig.suptitle(
        f"Multi-Layer Grad-CAM | Prediction: {CLASSES[pred_idx].upper()} ({prob*100:.1f}%)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# Auto-detect last conv layer
# ─────────────────────────────────────────────

def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Automatically find the name of the last Conv2D layer in a model.
    Useful when the layer name is unknown.
    """
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
    if last_conv is None:
        raise ValueError("No Conv2D layers found in model.")
    return last_conv


if __name__ == "__main__":
    import sys
    from src.config import CUSTOM_CNN_PATH, TRAIN_DIR

    if not os.path.exists(CUSTOM_CNN_PATH):
        print("[gradcam] Custom CNN not trained yet. Train it first:")
        print("          python scripts/train_all.py --models cnn")
        sys.exit(1)

    model = tf.keras.models.load_model(CUSTOM_CNN_PATH)

    # Auto-detect last conv layer
    layer = find_last_conv_layer(model)
    print(f"[gradcam] Using layer: {layer}")

    # Sample an image
    cls = "bird"
    sample = os.path.join(TRAIN_DIR, cls, os.listdir(os.path.join(TRAIN_DIR, cls))[0])

    visualize_gradcam(
        sample, model, layer,
        save_path=os.path.join(PLOTS_DIR, "gradcam_sample.png")
    )
