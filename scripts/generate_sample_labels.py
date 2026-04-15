"""
generate_sample_labels.py
Generates synthetic YOLOv8-format label files for sample images
and copies images into the object_detection_dataset structure.

Run from project root:
    python scripts/generate_sample_labels.py
"""

import os
import sys
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSIFICATION_DIR = ROOT / "data" / "classification_dataset"
DETECTION_DIR      = ROOT / "data" / "object_detection_dataset"

random.seed(42)


def generate_bbox():
    """Generate a realistic bounding box (normalized YOLO format)."""
    cx = random.uniform(0.2, 0.8)
    cy = random.uniform(0.2, 0.8)
    w  = random.uniform(0.08, 0.30)
    h  = random.uniform(0.08, 0.30)
    # Clamp to image boundary
    cx = max(w/2, min(1-w/2, cx))
    cy = max(h/2, min(1-h/2, cy))
    return cx, cy, w, h


def populate_detection_dataset():
    """
    Copy classification images into detection splits and
    generate corresponding YOLOv8 .txt label files.

    Class mapping:
        0 → bird
        1 → drone
    """
    # Gather all classified images with their class label
    all_images = []
    for split in ["TRAIN", "VALID", "TEST"]:
        for cls_idx, cls_name in enumerate(["bird", "drone"]):
            cls_dir = CLASSIFICATION_DIR / split / cls_name
            if not cls_dir.exists():
                continue
            for img_file in cls_dir.glob("*.jpg"):
                all_images.append((img_file, cls_idx))

    random.shuffle(all_images)
    total = len(all_images)
    n_train = int(total * 0.80)
    n_val   = int(total * 0.10)

    splits = {
        "train": all_images[:n_train],
        "val":   all_images[n_train:n_train + n_val],
        "test":  all_images[n_train + n_val:],
    }

    print(f"\n[labels] Generating detection dataset from {total} images...")
    for split_name, img_list in splits.items():
        img_out_dir = DETECTION_DIR / "images" / split_name
        lbl_out_dir = DETECTION_DIR / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path, cls_idx in img_list:
            # Copy image
            dest_img = img_out_dir / img_path.name
            shutil.copy(str(img_path), str(dest_img))

            # Generate label (1-2 bounding boxes per image)
            n_boxes = random.randint(1, 2)
            label_path = lbl_out_dir / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                for _ in range(n_boxes):
                    cx, cy, w, h = generate_bbox()
                    f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        n = len(img_list)
        print(f"  {split_name:<6}: {n} images + {n} label files ✅")

    print(f"\n[labels] Detection dataset ready at: {DETECTION_DIR}")


if __name__ == "__main__":
    populate_detection_dataset()
