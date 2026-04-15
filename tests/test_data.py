"""
tests/test_data.py — Unit tests for data preprocessing pipeline
Run: pytest tests/ -v
"""
import os
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_dataset_exists():
    """Dataset directories must exist with images."""
    from src.config import TRAIN_DIR, VALID_DIR, TEST_DIR, CLASSES

    for split_dir in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
        assert os.path.isdir(split_dir), f"Missing split dir: {split_dir}"
        for cls in CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            assert os.path.isdir(cls_dir), f"Missing class dir: {cls_dir}"
            images = [f for f in os.listdir(cls_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            assert len(images) > 0, f"No images in: {cls_dir}"


def test_count_images():
    """count_images should return correct counts."""
    from src.utils import count_images
    from src.config import TRAIN_DIR

    counts = count_images(TRAIN_DIR)
    assert "bird" in counts
    assert "drone" in counts
    assert counts["bird"] > 0
    assert counts["drone"] > 0


def test_load_preprocess_image():
    """load_and_preprocess_image should return correct shape and range."""
    from src.utils import load_and_preprocess_image
    from src.config import TRAIN_DIR, IMG_SIZE

    cls_dir = os.path.join(TRAIN_DIR, "bird")
    sample = os.path.join(cls_dir, os.listdir(cls_dir)[0])
    arr = load_and_preprocess_image(sample, target_size=IMG_SIZE)

    assert arr.ndim == 4, "Output must be 4D (batch)"
    assert arr.shape == (1, IMG_SIZE[0], IMG_SIZE[1], 3)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_generators():
    """Data generators should produce correct shapes and labels."""
    try:
        import tensorflow as tf
    except ImportError:
        import pytest; pytest.skip("TensorFlow not installed")

    from src.data_preprocessing import get_train_generator
    from src.config import IMG_SIZE, BATCH_SIZE

    gen = get_train_generator(batch_size=4)
    assert gen is not None
    assert gen.num_classes == 2

    images, labels = next(gen)
    assert images.ndim == 4
    assert images.shape[1:] == (IMG_SIZE[0], IMG_SIZE[1], 3)
    assert images.min() >= 0.0
    assert images.max() <= 1.01  # allow small float error
    assert all(l in [0, 1] for l in labels)


def test_detection_dataset_labels():
    """Detection dataset should have matching image/label pairs."""
    from src.config import DETECTION_DIR

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DETECTION_DIR, "images", split)
        lbl_dir = os.path.join(DETECTION_DIR, "labels", split)

        if not os.path.isdir(img_dir):
            continue

        images = {Path(f).stem for f in os.listdir(img_dir)
                  if f.lower().endswith((".jpg", ".png"))}
        labels = {Path(f).stem for f in os.listdir(lbl_dir)
                  if f.endswith(".txt")} if os.path.isdir(lbl_dir) else set()

        # Every image should have a label
        missing = images - labels
        assert len(missing) == 0, \
            f"Split '{split}': {len(missing)} images missing labels"


def test_yolo_label_format():
    """YOLOv8 label files must follow <class_id> <cx> <cy> <w> <h> format."""
    from src.config import DETECTION_DIR

    label_dir = os.path.join(DETECTION_DIR, "labels", "train")
    if not os.path.isdir(label_dir):
        return

    label_files = list(Path(label_dir).glob("*.txt"))
    assert len(label_files) > 0, "No label files found"

    for lf in label_files[:10]:  # spot-check first 10
        with open(lf) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 5, f"Invalid label line in {lf}: {line!r}"
            cls_id = int(parts[0])
            assert cls_id in (0, 1), f"Invalid class ID {cls_id} in {lf}"
            coords = [float(p) for p in parts[1:]]
            for v in coords:
                assert 0.0 <= v <= 1.0, f"Coord {v} out of [0,1] in {lf}"
