"""
predict.py — Single-image and batch inference for trained classification models.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from src.config import IMG_SIZE, CLASSES, BEST_MODEL_PATH
from src.utils import load_and_preprocess_image


def load_model(model_path: str = BEST_MODEL_PATH) -> tf.keras.Model:
    """Load a saved Keras model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"[predict] Loaded model: {model_path}")
    return model


def predict_single(image_path: str,
                   model: tf.keras.Model,
                   threshold: float = 0.5) -> dict:
    """
    Run inference on a single image file.

    Parameters
    ----------
    image_path : str — path to image (.jpg / .png)
    model      : loaded tf.keras.Model
    threshold  : float — decision boundary

    Returns
    -------
    dict: {label, confidence, probability}
    """
    img_array = load_and_preprocess_image(image_path, target_size=IMG_SIZE)
    prob = float(model.predict(img_array, verbose=0)[0][0])
    label_idx = int(prob >= threshold)
    label = CLASSES[label_idx]
    confidence = prob if label_idx == 1 else 1 - prob

    return {
        "label":       label,
        "confidence":  round(confidence * 100, 2),
        "probability": round(prob, 6),
        "class_idx":   label_idx,
    }


def predict_batch(image_paths: list,
                  model: tf.keras.Model,
                  threshold: float = 0.5) -> list:
    """
    Run inference on a list of image paths.

    Returns
    -------
    list of result dicts (same format as predict_single)
    """
    results = []
    for path in image_paths:
        try:
            result = predict_single(path, model, threshold)
            result["file"] = os.path.basename(path)
            results.append(result)
        except Exception as e:
            results.append({"file": os.path.basename(path), "error": str(e)})
    return results


def predict_from_pil(pil_image: Image.Image,
                     model: tf.keras.Model,
                     threshold: float = 0.5) -> dict:
    """
    Run inference on a PIL Image object (for Streamlit integration).
    """
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label_idx = int(prob >= threshold)
    label = CLASSES[label_idx]
    confidence = prob if label_idx == 1 else 1 - prob

    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "probability": round(prob, 6),
        "class_idx":  label_idx,
        "raw_probs":  {CLASSES[0]: round(1 - prob, 6), CLASSES[1]: round(prob, 6)},
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path> [model_path]")
        sys.exit(1)
    img = sys.argv[1]
    mp  = sys.argv[2] if len(sys.argv) > 2 else BEST_MODEL_PATH
    m   = load_model(mp)
    res = predict_single(img, m)
    print(f"\n{'─'*40}")
    print(f"  Image     : {img}")
    print(f"  Prediction: {res['label'].upper()}")
    print(f"  Confidence: {res['confidence']}%")
    print(f"{'─'*40}")
