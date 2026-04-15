"""Evaluation helpers for reports, plots, and model comparison."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)

from src.config import CLASSES, PLOTS_DIR, REPORTS_DIR, BEST_MODEL_PATH
from src.utils import (
    plot_confusion_matrix, plot_roc_curve,
    compute_metrics, save_report, plot_model_comparison, ensure_dirs
)


# ─────────────────────────────────────────────
# Single model evaluation
# ─────────────────────────────────────────────

def evaluate_model(model: tf.keras.Model,
                   test_gen,
                   model_name: str = "Model",
                   threshold: float = 0.5,
                   save_dir: str = PLOTS_DIR) -> dict:
    """Run the test set, compute metrics, and save the usual evaluation artifacts."""
    ensure_dirs(save_dir, REPORTS_DIR)

    # ── Get predictions ──────────────────────
    test_gen.reset()
    probs = model.predict(test_gen, verbose=1).ravel()
    preds = (probs >= threshold).astype(int)
    labels = test_gen.classes

    # ── Keras built-in evaluation ────────────
    test_gen.reset()
    loss, acc, *_ = model.evaluate(test_gen, verbose=0)
    print(f"\n[evaluate] {model_name} — Test Loss : {loss:.4f} | Test Acc : {acc:.4f}")

    # ── Sklearn metrics ───────────────────────
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)
    roc_auc   = roc_auc_score(labels, probs)

    results = {
        "model":     model_name,
        "accuracy":  float(acc),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "roc_auc":   float(roc_auc),
    }

    # ── Console report ────────────────────────
    print(f"\n{'─'*50}")
    print(f"  EVALUATION REPORT — {model_name}")
    print(f"{'─'*50}")
    print(classification_report(labels, preds, target_names=[c.upper() for c in CLASSES]))
    for k, v in results.items():
        if k != "model":
            print(f"  {k.upper():<12}: {v:.4f}")
    print(f"{'─'*50}\n")

    # ── Plots ─────────────────────────────────
    cm_path  = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_confusion.png")
    roc_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_roc.png")
    plot_confusion_matrix(labels, preds, class_names=CLASSES,
                          model_name=model_name, save_path=cm_path)
    plot_roc_curve(labels, probs, model_name=model_name, save_path=roc_path)

    # ── Save report ───────────────────────────
    report_dict = compute_metrics(labels, preds)
    report_dict["summary"] = results
    report_path = os.path.join(REPORTS_DIR,
                               f"{model_name.lower().replace(' ', '_')}_report.json")
    save_report(report_dict, model_name, save_path=report_path)

    return results


# ─────────────────────────────────────────────
# Compare multiple models
# ─────────────────────────────────────────────

def compare_models(model_results: dict, save_dir: str = PLOTS_DIR) -> pd.DataFrame:
    """Build a table and chart so multiple models are easy to compare."""
    ensure_dirs(save_dir, REPORTS_DIR)

    df = pd.DataFrame(model_results).T.reset_index()
    df.rename(columns={"index": "Model"}, inplace=True)
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "="*70)
    print("  MODEL COMPARISON TABLE")
    print("="*70)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print("="*70)

    # Save table
    table_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df.to_csv(table_path, index=False)
    print(f"[evaluate] Comparison table saved: {table_path}")

    # Plot comparison chart
    chart_path = os.path.join(save_dir, "model_comparison.png")
    plot_model_comparison(model_results, save_path=chart_path)

    # Determine best model
    best_name = df.iloc[0]["Model"]
    print(f"\n[evaluate] ✅ Best model: {best_name}  "
          f"(Accuracy={df.iloc[0]['accuracy']:.4f}, "
          f"F1={df.iloc[0]['f1']:.4f})")

    return df


# ─────────────────────────────────────────────
# Grad-CAM Visualization
# ─────────────────────────────────────────────

def make_gradcam_heatmap(img_array: np.ndarray,
                          model: tf.keras.Model,
                          last_conv_layer_name: str,
                          pred_index: int = None) -> np.ndarray:
    """Compute a Grad-CAM heatmap for one preprocessed image."""
    # Create model mapping input → (conv_output, final_prediction)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def display_gradcam(img_path: str, model: tf.keras.Model,
                    last_conv_layer: str, alpha: float = 0.4,
                    save_path: str = None):
    """Overlay the Grad-CAM heatmap on top of the original image."""
    from PIL import Image
    from src.utils import load_and_preprocess_image

    img_array = load_and_preprocess_image(img_path)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

    # Resize heatmap to original image size
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize((w, h))
    ) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]
    superimposed = heatmap_colored * alpha + img / 255.0 * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    axes[2].imshow(superimposed); axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.suptitle("Grad-CAM Visualization", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Grad-CAM saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# Select & save best model
# ─────────────────────────────────────────────

def select_best_model(model_results: dict,
                      model_objects: dict,
                      save_path: str = BEST_MODEL_PATH,
                      metric: str = "f1"):
    """Choose the top model using the requested metric and save it."""
    best_name = max(model_results, key=lambda n: model_results[n].get(metric, 0))
    best_model = model_objects[best_name]
    best_model.save(save_path)
    print(f"\n[evaluate] 🏆 Best model '{best_name}' saved to: {save_path}")
    print(f"           Metrics → {model_results[best_name]}")
    return best_name, best_model


if __name__ == "__main__":
    import tensorflow as tf
    from src.data_preprocessing import get_test_generator
    from src.config import CUSTOM_CNN_PATH

    test_gen = get_test_generator()
    model = tf.keras.models.load_model(CUSTOM_CNN_PATH)
    results = evaluate_model(model, test_gen, model_name="Custom CNN")
    print(results)
