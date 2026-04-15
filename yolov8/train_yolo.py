"""
train_yolo.py — YOLOv8 Object Detection Training Script
Aerial Bird vs Drone Detection

Steps:
  1. Install ultralytics (pip install ultralytics)
  2. Prepare dataset with YOLOv8-format .txt labels
  3. Update data.yaml paths if needed
  4. Run: python yolov8/train_yolo.py
"""

import os
import sys
import yaml
import shutil
from pathlib import Path

# ── Add project root to path ────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    YOLO_MODEL_SIZE, YOLO_EPOCHS, YOLO_IMG_SIZE,
    YOLO_BATCH, YOLO_PATIENCE, YOLO_CONF_THRES,
    YOLO_IOU_THRES, YOLO_YAML, YOLO_RUNS_DIR
)

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
YAML_PATH   = str(ROOT / "yolov8" / "data.yaml")
RUNS_DIR    = str(ROOT / "yolov8" / "runs")
PROJECT_NAME = "aerial_detection"
RUN_NAME     = f"{YOLO_MODEL_SIZE}_bird_drone"


def verify_dataset():
    """
    Check that dataset directories and at least some images exist.
    Prints a summary and returns True if valid.
    """
    with open(YAML_PATH) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg.get("path", ROOT / "data" / "object_detection_dataset"))
    splits = {"train": cfg.get("train",""), "val": cfg.get("val",""), "test": cfg.get("test","")}

    print("\n[yolo] Dataset verification:")
    all_ok = True
    for split, rel_path in splits.items():
        img_dir   = data_root / rel_path
        label_dir = data_root / rel_path.replace("images", "labels")
        n_imgs   = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))) \
                   if img_dir.exists() else 0
        n_labels = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        status = "✅" if n_imgs > 0 else "⚠️  MISSING"
        print(f"  {split:<6} → images: {n_imgs:>4}  labels: {n_labels:>4}  {status}")
        if n_imgs == 0:
            all_ok = False

    print(f"  Classes : {cfg.get('names', [])}")
    return all_ok


def train():
    """
    Train YOLOv8 model for aerial object detection.
    """
    print("\n" + "=" * 60)
    print(f"  YOLOv8 TRAINING — {YOLO_MODEL_SIZE.upper()}")
    print("=" * 60)

    if not verify_dataset():
        print("\n[yolo] ⚠️  Dataset not fully populated.")
        print("       Please add images & labels to data/object_detection_dataset/")
        print("       Continuing with demo run anyway...\n")

    # Load pretrained YOLOv8 model
    model = YOLO(f"{YOLO_MODEL_SIZE}.pt")
    print(f"\n[yolo] Loaded: {YOLO_MODEL_SIZE}.pt")

    # ── Train ────────────────────────────────
    results = model.train(
        data      = YAML_PATH,
        epochs    = YOLO_EPOCHS,
        imgsz     = YOLO_IMG_SIZE,
        batch     = YOLO_BATCH,
        patience  = YOLO_PATIENCE,
        project   = RUNS_DIR,
        name      = RUN_NAME,
        exist_ok  = True,
        device    = "0" if _gpu_available() else "cpu",
        workers   = 4,
        # Augmentation
        augment   = True,
        mosaic    = 1.0,
        mixup     = 0.1,
        fliplr    = 0.5,
        degrees   = 15.0,
        translate = 0.1,
        scale     = 0.5,
        # Regularization
        dropout   = 0.0,
        weight_decay = 0.0005,
        # Logging
        plots     = True,
        save      = True,
        verbose   = True,
    )

    best_weights = Path(RUNS_DIR) / RUN_NAME / "weights" / "best.pt"
    print(f"\n[yolo] ✅ Training complete!")
    print(f"[yolo] Best weights: {best_weights}")
    return model, results


def validate(weights_path: str = None):
    """
    Validate the trained model on the validation split.
    """
    if weights_path is None:
        weights_path = str(Path(RUNS_DIR) / RUN_NAME / "weights" / "best.pt")

    if not os.path.exists(weights_path):
        print(f"[yolo] Weights not found: {weights_path}")
        return None

    model = YOLO(weights_path)
    metrics = model.val(data=YAML_PATH, imgsz=YOLO_IMG_SIZE, verbose=True)
    print(f"\n[yolo] Validation mAP@50   : {metrics.box.map50:.4f}")
    print(f"[yolo] Validation mAP@50-95: {metrics.box.map:.4f}")
    return metrics


def run_inference(source: str, weights_path: str = None,
                  conf: float = YOLO_CONF_THRES,
                  iou: float  = YOLO_IOU_THRES,
                  save_results: bool = True):
    """
    Run inference on an image, folder, or video.

    Parameters
    ----------
    source       : str — image path, folder, URL, or video path
    weights_path : str — path to best.pt (defaults to last trained)
    conf         : float — confidence threshold
    iou          : float — IoU threshold for NMS
    save_results : bool — save annotated output images

    Returns
    -------
    list of ultralytics Results objects
    """
    if weights_path is None:
        weights_path = str(Path(RUNS_DIR) / RUN_NAME / "weights" / "best.pt")
    if not os.path.exists(weights_path):
        print(f"[yolo] Weights not found: {weights_path}")
        return []

    model = YOLO(weights_path)
    results = model.predict(
        source   = source,
        conf     = conf,
        iou      = iou,
        imgsz    = YOLO_IMG_SIZE,
        save     = save_results,
        project  = RUNS_DIR,
        name     = f"{RUN_NAME}_inference",
        exist_ok = True,
        verbose  = True,
    )

    # Print detection summary
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls  = int(box.cls[0])
                conf_ = float(box.conf[0])
                xyxy  = box.xyxy[0].tolist()
                name  = model.names[cls]
                print(f"  Detected: {name:<8} conf={conf_:.2f}  bbox={[round(v,1) for v in xyxy]}")
        else:
            print(f"  No detections in: {r.path}")

    return results


def export_model(weights_path: str = None, export_format: str = "onnx"):
    """
    Export trained YOLOv8 model to ONNX / TFLite / CoreML for deployment.

    Parameters
    ----------
    export_format : str — "onnx", "tflite", "coreml", "torchscript"
    """
    if weights_path is None:
        weights_path = str(Path(RUNS_DIR) / RUN_NAME / "weights" / "best.pt")

    model = YOLO(weights_path)
    exported = model.export(format=export_format, imgsz=YOLO_IMG_SIZE)
    print(f"\n[yolo] Exported to {export_format}: {exported}")
    return exported


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 Aerial Detection")
    parser.add_argument("--mode", choices=["train","val","infer","export"],
                        default="train", help="Operation to perform")
    parser.add_argument("--source", type=str, default=None,
                        help="Image/folder for inference")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to .pt weights file")
    parser.add_argument("--format", type=str, default="onnx",
                        help="Export format (onnx/tflite/coreml)")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "val":
        validate(args.weights)
    elif args.mode == "infer":
        if args.source is None:
            print("[ERROR] --source required for inference mode")
        else:
            run_inference(args.source, args.weights)
    elif args.mode == "export":
        export_model(args.weights, args.format)
