"""
download_full_dataset.py — Full Dataset Download Helper
Downloads the complete Bird vs Drone datasets from public sources.

The sample images included in the project ZIP are a small subset (225 images).
The full dataset referenced in the project spec contains:
  Classification: ~3,319 images (Bird: 1752 | Drone: 1567)
  Detection:      ~3,319 images with YOLOv8 annotations

Run from project root:
    python scripts/download_full_dataset.py

Options:
    --source roboflow   Download via Roboflow API (requires API key)
    --source kaggle     Download via Kaggle datasets
    --source manual     Shows manual download instructions
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


MANUAL_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════╗
║       FULL DATASET DOWNLOAD — MANUAL INSTRUCTIONS           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OPTION 1: Roboflow (Recommended)                            ║
║  ─────────────────────────────────                           ║
║  1. Go to: https://roboflow.com                              ║
║  2. Search for "Bird vs Drone aerial"                        ║
║  3. Download in "YOLOv8" format (includes class folders)     ║
║  4. Place in: data/classification_dataset/                   ║
║     Expected structure:                                      ║
║       data/classification_dataset/                           ║
║         TRAIN/bird/*.jpg   (1414 images)                    ║
║         TRAIN/drone/*.jpg  (1248 images)                    ║
║         VALID/bird/*.jpg   (217 images)                     ║
║         VALID/drone/*.jpg  (225 images)                     ║
║         TEST/bird/*.jpg    (121 images)                     ║
║         TEST/drone/*.jpg   (94 images)                      ║
║                                                              ║
║  OPTION 2: Kaggle                                            ║
║  ─────────────────                                           ║
║  Search: "bird drone aerial classification"                  ║
║  Command (after setting up kaggle.json):                     ║
║    kaggle datasets download -d <dataset-name>                ║
║    unzip *.zip -d data/classification_dataset/               ║
║                                                              ║
║  OPTION 3: Google Drive (if provided by instructor)          ║
║  ────────────────────────────────────────────────            ║
║  Download from provided Drive link and extract to:           ║
║    data/classification_dataset/                              ║
║    data/object_detection_dataset/                            ║
║                                                              ║
║  After downloading, re-run:                                  ║
║    python scripts/generate_sample_labels.py                  ║
║    python scripts/train_all.py                               ║
╚══════════════════════════════════════════════════════════════╝
"""


def download_roboflow(api_key: str):
    """Download dataset using Roboflow Python package."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[INFO] Installing roboflow package...")
        os.system("pip install roboflow -q")
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)

    # Classification dataset
    print("\n[download] Fetching classification dataset...")
    try:
        project = rf.workspace().project("bird-drone-aerial")
        dataset = project.version(1).download("folder")
        print(f"[download] Classification dataset saved to: {dataset.location}")
    except Exception as e:
        print(f"[download] Could not fetch classification dataset: {e}")
        print("           Please download manually (see --source manual)")

    # Detection dataset
    print("\n[download] Fetching detection dataset...")
    try:
        project = rf.workspace().project("bird-drone-detection")
        dataset = project.version(1).download("yolov8")
        print(f"[download] Detection dataset saved to: {dataset.location}")
    except Exception as e:
        print(f"[download] Could not fetch detection dataset: {e}")


def verify_full_dataset():
    """
    Verify the current dataset and show statistics.
    """
    from src.config import CLASSIFICATION_DIR, DETECTION_DIR

    print("\n[verify] Current Dataset Statistics:")
    print("─" * 50)

    total_class = 0
    for split in ["TRAIN", "VALID", "TEST"]:
        for cls in ["bird", "drone"]:
            d = os.path.join(CLASSIFICATION_DIR, split, cls)
            n = len(list(Path(d).glob("*.jpg"))) if os.path.exists(d) else 0
            total_class += n
            print(f"  Classification {split}/{cls}: {n:>5} images")

    print(f"  Classification TOTAL         : {total_class:>5} images")

    total_det = 0
    for split in ["train", "val", "test"]:
        d = os.path.join(DETECTION_DIR, "images", split)
        n = len(list(Path(d).glob("*.jpg"))) + len(list(Path(d).glob("*.png"))) \
            if os.path.exists(d) else 0
        total_det += n
        print(f"  Detection       {split:<5}       : {n:>5} images")

    print(f"  Detection TOTAL              : {total_det:>5} images")
    print("─" * 50)

    target_class = 3319
    target_det   = 3319
    pct_class = total_class / target_class * 100
    pct_det   = total_det   / target_det   * 100
    print(f"\n  Classification: {pct_class:.0f}% of full dataset ({target_class} target)")
    print(f"  Detection     : {pct_det:.0f}% of full dataset ({target_det} target)")

    if pct_class < 50:
        print("\n  ⚠️  Dataset is small. For best results, download the full dataset.")
        print("     Run: python scripts/download_full_dataset.py --source manual")
    else:
        print("\n  ✅ Dataset looks adequate for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download full aerial datasets")
    parser.add_argument("--source", choices=["roboflow", "kaggle", "manual"],
                        default="manual", help="Download source")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Roboflow API key (for --source roboflow)")
    args = parser.parse_args()

    print("\n🦅 Aerial Dataset Downloader")
    print("=" * 50)
    verify_full_dataset()

    if args.source == "manual":
        print(MANUAL_INSTRUCTIONS)
    elif args.source == "roboflow":
        if not args.api_key:
            print("[ERROR] --api-key required for Roboflow download")
            print("        Get yours at: https://app.roboflow.com/settings/api")
            sys.exit(1)
        download_roboflow(args.api_key)
    elif args.source == "kaggle":
        print("\n[kaggle] To download via Kaggle:")
        print("  1. Setup Kaggle API: place kaggle.json in ~/.kaggle/")
        print("  2. pip install kaggle")
        print("  3. kaggle datasets download -d username/bird-drone-aerial")
        print("  4. Unzip and move to data/classification_dataset/")
