"""
utils.py — Shared utility functions for the Aerial Classification & Detection project.
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import tensorflow as tf
from src.config import CLASSES, PLOTS_DIR, REPORTS_DIR, SEED


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[utils] Seed set to {seed}")


# ─────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────
def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def count_images(directory: str) -> dict:
    """
    Count images per class in a directory structured as dir/class/images.

    Returns
    -------
    dict : {class_name: count}
    """
    counts = {}
    for cls in sorted(os.listdir(directory)):
        cls_path = os.path.join(directory, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
    return counts


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────
def plot_sample_images(directory: str, classes: list, n: int = 5,
                       save_path: str = None, title: str = "Sample Images"):
    """
    Plot n sample images per class from a dataset directory.
    """
    fig, axes = plt.subplots(len(classes), n, figsize=(n * 3, len(classes) * 3))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    for row, cls in enumerate(classes):
        cls_path = os.path.join(directory, cls)
        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        sampled = random.sample(images, min(n, len(images)))

        for col, img_file in enumerate(sampled):
            img = Image.open(os.path.join(cls_path, img_file)).convert("RGB")
            ax = axes[row][col] if len(classes) > 1 else axes[col]
            ax.imshow(img)
            ax.set_title(cls.upper(), fontsize=9, color="darkblue")
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Saved: {save_path}")
    plt.show()


def plot_class_distribution(split_counts: dict, save_path: str = None):
    """
    Bar chart of class distribution across TRAIN / VALID / TEST splits.

    Parameters
    ----------
    split_counts : dict
        e.g. {"TRAIN": {"bird": 1414, "drone": 1248}, "VALID": {...}, "TEST": {...}}
    """
    splits = list(split_counts.keys())
    classes = CLASSES
    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4CAF50", "#F44336"]

    for i, cls in enumerate(classes):
        counts = [split_counts[s].get(cls, 0) for s in splits]
        bars = ax.bar(x + i * width, counts, width, label=cls.upper(),
                      color=colors[i], alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Dataset Split", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Class Distribution Across Dataset Splits", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(splits, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Saved: {save_path}")
    plt.show()


def plot_training_history(history, model_name: str = "Model", save_path: str = None):
    """
    Plot accuracy and loss curves from Keras training history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=15, fontweight="bold")

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train Accuracy", color="#2196F3", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", color="#FF9800",
             linewidth=2, linestyle="--")
    ax1.set_title("Model Accuracy", fontsize=13)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Train Loss", color="#4CAF50", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss", color="#F44336",
             linewidth=2, linestyle="--")
    ax2.set_title("Model Loss", fontsize=13)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Saved: {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names: list = CLASSES,
                          model_name: str = "Model", save_path: str = None):
    """
    Plot a nicely formatted confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Raw Counts", "Normalized"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=[c.upper() for c in class_names],
                    yticklabels=[c.upper() for c in class_names],
                    ax=ax, linewidths=0.5, linecolor="white")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Saved: {save_path}")
    plt.show()


def plot_roc_curve(y_true, y_prob, model_name: str = "Model", save_path: str = None):
    """
    Plot ROC curve with AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2196F3", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Classifier")
    plt.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"{model_name} — ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return roc_auc


# ─────────────────────────────────────────────
# Metrics & Reporting
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, class_names: list = CLASSES) -> dict:
    """
    Compute classification metrics and return as a dictionary.
    """
    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   output_dict=True)
    return report


def save_report(report: dict, model_name: str, save_path: str = None):
    """
    Save classification report as a JSON file.
    """
    if save_path is None:
        ensure_dirs(REPORTS_DIR)
        save_path = os.path.join(REPORTS_DIR, f"{model_name}_report.json")
    with open(save_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"[utils] Report saved: {save_path}")


def plot_model_comparison(results: dict, save_path: str = None):
    """
    Bar chart comparing multiple models on accuracy, precision, recall, F1.

    Parameters
    ----------
    results : dict
        e.g. {"Custom CNN": {"accuracy": 0.91, "f1": 0.90, ...}, ...}
    """
    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ["#3F51B5", "#4CAF50", "#FF9800", "#F44336"]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Classification Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────
def load_and_preprocess_image(image_path: str, target_size=(224, 224)) -> np.ndarray:
    """
    Load and preprocess a single image for inference.

    Returns
    -------
    np.ndarray : shape (1, H, W, 3), normalized to [0, 1]
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def get_image_stats(directory: str) -> dict:
    """
    Compute mean pixel intensity and std across all images in a directory.
    (Walks subdirectories recursively.)
    """
    means, stds = [], []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    img = np.array(Image.open(os.path.join(root, f)).convert("RGB"),
                                   dtype=np.float32) / 255.0
                    means.append(img.mean())
                    stds.append(img.std())
                except Exception:
                    continue
    return {
        "mean": float(np.mean(means)),
        "std":  float(np.mean(stds)),
        "n_images": len(means)
    }
